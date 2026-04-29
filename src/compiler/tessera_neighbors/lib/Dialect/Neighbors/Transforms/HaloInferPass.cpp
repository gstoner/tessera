//===- HaloInferPass.cpp — Halo width inference (Phase 7) ------------------===//
//
// Scans the IR for:
//   1. tessera.neighbors.neighbor.read  — collects the absolute Δ per axis
//   2. tessera.neighbors.stencil.define — collects taps (each a DeltaArray)
//   3. tessera.neighbors.stencil.apply  — propagates the tap-set max to the
//      halo.region that feeds the stencil
//
// Result: every tessera.neighbors.halo.region op (and every
// tessera.neighbors.stencil.apply op that has no explicit halo.region) gains:
//   "halo.width" = array of max(|Δ_i|) per axis          (I64ArrayAttr)
//   "halo.axes"  = the axis names from the topology       (StringAttr)
//
// This information drives StencilLowerPass to emit pack/exchange/unpack calls
// with the correct buffer widths.
//===----------------------------------------------------------------------===//

#include "tessera/Dialect/Neighbors/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>    // std::abs
#include <limits>
#include <string>
#include <vector>

using namespace mlir;

namespace tessera {
namespace neighbors {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the DeltaArray values from an attribute. Returns empty on failure.
static llvm::SmallVector<int64_t> getDeltaValues(Attribute attr) {
  // DeltaArrayAttr stores as DictionaryAttr {"values": [i64...]} or
  // as a raw I64ArrayAttr depending on how the op was built.
  if (!attr) return {};

  // Case 1: I64ArrayAttr (simple encoding)
  if (auto arr = attr.dyn_cast<ArrayAttr>()) {
    llvm::SmallVector<int64_t> out;
    for (auto el : arr) {
      if (auto ia = el.dyn_cast<IntegerAttr>())
        out.push_back(ia.getInt());
    }
    return out;
  }
  return {};
}

/// Update a per-axis max-|Δ| vector from a single delta array.
static void updateMaxDelta(llvm::SmallVector<int64_t> &maxAbs,
                           llvm::ArrayRef<int64_t> delta) {
  if (maxAbs.size() < delta.size())
    maxAbs.resize(delta.size(), 0);
  for (size_t i = 0; i < delta.size(); ++i)
    maxAbs[i] = std::max(maxAbs[i], std::abs(delta[i]));
}

/// Collect max-|Δ| across all taps in a stencil.define op.
static llvm::SmallVector<int64_t> maxDeltaFromStencil(Operation *stencilDef) {
  llvm::SmallVector<int64_t> maxAbs;
  auto taps = stencilDef->getAttrOfType<ArrayAttr>("taps");
  if (!taps) return maxAbs;
  for (Attribute tap : taps) {
    auto delta = getDeltaValues(tap);
    updateMaxDelta(maxAbs, delta);
  }
  return maxAbs;
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct HaloInferPass
    : public PassWrapper<HaloInferPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HaloInferPass)

  StringRef getArgument() const final { return "tessera-halo-infer"; }
  StringRef getDescription() const final {
    return "Infer halo widths from neighbor.read Δ-uses and stencil taps";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();
    OpBuilder builder(ctx);

    // ----------------------------------------------------------------
    // Pass 1: collect all stencil.define → max-|Δ| table
    // ----------------------------------------------------------------
    llvm::DenseMap<Operation *, llvm::SmallVector<int64_t>> stencilMaxDelta;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() ==
          "tessera.neighbors.stencil.define") {
        stencilMaxDelta[op] = maxDeltaFromStencil(op);
      }
    });

    // ----------------------------------------------------------------
    // Pass 2: annotate halo.region ops from neighbor.read Δ-uses
    // ----------------------------------------------------------------
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() !=
          "tessera.neighbors.halo.region")
        return WalkResult::advance();

      // Collect all neighbor.read ops that consume the result of this
      // halo.region (directly or transitively through the same block).
      llvm::SmallVector<int64_t> maxAbs;
      Value haloVal = op->getResult(0);
      for (Operation *user : haloVal.getUsers()) {
        if (user->getName().getStringRef() ==
            "tessera.neighbors.neighbor.read") {
          auto delta = getDeltaValues(user->getAttr("delta"));
          updateMaxDelta(maxAbs, delta);
        }
        // Also handle stencil.apply ops that use the halo
        if (user->getName().getStringRef() ==
            "tessera.neighbors.stencil.apply") {
          // Find the stencil operand (operand 0) definition
          if (auto *stDef = user->getOperand(0).getDefiningOp()) {
            auto it = stencilMaxDelta.find(stDef);
            if (it != stencilMaxDelta.end())
              updateMaxDelta(maxAbs, it->second);
          }
        }
      }

      if (maxAbs.empty()) return WalkResult::advance();

      // Build I64ArrayAttr for halo.width
      llvm::SmallVector<Attribute> widthAttrs;
      for (int64_t w : maxAbs)
        widthAttrs.push_back(builder.getI64IntegerAttr(w));
      op->setAttr("halo.width", builder.getArrayAttr(widthAttrs));
      return WalkResult::advance();
    });

    // ----------------------------------------------------------------
    // Pass 3: annotate stencil.apply ops that have no explicit halo.region
    // ----------------------------------------------------------------
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() !=
          "tessera.neighbors.stencil.apply")
        return WalkResult::advance();

      // Skip if already has halo.width (set via halo.region above)
      if (op->hasAttr("halo.width")) return WalkResult::advance();

      // Derive from the stencil.define operand
      if (auto *stDef = op->getOperand(0).getDefiningOp()) {
        auto it = stencilMaxDelta.find(stDef);
        if (it != stencilMaxDelta.end() && !it->second.empty()) {
          llvm::SmallVector<Attribute> widthAttrs;
          for (int64_t w : it->second)
            widthAttrs.push_back(builder.getI64IntegerAttr(w));
          op->setAttr("halo.width", builder.getArrayAttr(widthAttrs));
        }
      }
      return WalkResult::advance();
    });
  }
};

void registerHaloInferPass() {
  PassRegistration<HaloInferPass>();
}

} // namespace neighbors
} // namespace tessera
