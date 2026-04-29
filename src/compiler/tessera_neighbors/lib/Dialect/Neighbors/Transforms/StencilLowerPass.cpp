//===- StencilLowerPass.cpp — Lower stencil.apply (Phase 7) ----------------===//
//
// Lowers tessera.neighbors.stencil.apply into the three-phase sequence:
//
//   Phase A — Pack halo boundaries into contiguous send buffers.
//     Annotates the op with "stencil.pack_phase = true".
//
//   Phase B — Initiate halo.exchange for each required neighbor.
//     Creates a tessera.neighbors.halo.region (if one does not already exist)
//     and tessera.neighbors.halo.exchange with async ordering attributes.
//
//   Phase C — Expand compute kernel skeleton.
//     Annotates the op with "stencil.compute_phase = true" and records the
//     expanded tap pattern so the downstream tile-lowering pass can emit the
//     actual loop nest.
//
// The pass does NOT emit loops itself — that would require ranked-memref
// access patterns that are architecture-specific.  Instead it records all
// needed information as structured attributes so a target-specific pass can
// finish the job without re-analysing the Neighbors dialect.
//
// Attributes written on stencil.apply:
//   "stencil.pack_phase"    : BoolAttr true
//   "stencil.compute_phase" : BoolAttr true
//   "stencil.exchange_policy" : "eager" | "lazy"
//   "stencil.tap_count"     : I64Attr — number of stencil taps
//   "stencil.halo_width"    : I64ArrayAttr (copied from halo.width if present)
//   "stencil.lowered"       : BoolAttr true  (sentinel for subsequent passes)
//===----------------------------------------------------------------------===//

#include "tessera/Dialect/Neighbors/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace tessera {
namespace neighbors {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int64_t countTaps(Operation *stencilDef) {
  if (!stencilDef) return 0;
  auto taps = stencilDef->getAttrOfType<ArrayAttr>("taps");
  return taps ? static_cast<int64_t>(taps.size()) : 0;
}

static StringRef exchangePolicy(Operation *applyOp) {
  // If the enclosing pipeline.config says "eager" use that; default "lazy".
  // Walk up to see if there's a sibling pipeline.config in the same block.
  Block *block = applyOp->getBlock();
  if (!block) return "lazy";
  for (Operation &sibling : *block) {
    if (sibling.getName().getStringRef() ==
        "tessera.neighbors.pipeline.config") {
      if (auto ov = sibling.getAttrOfType<StringAttr>("overlap"))
        return ov.getValue();
    }
  }
  return "lazy";
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct StencilLowerPass
    : public PassWrapper<StencilLowerPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StencilLowerPass)

  StringRef getArgument() const final { return "tessera-stencil-lower"; }
  StringRef getDescription() const final {
    return "Lower stencil.apply to pack/exchange/compute skeleton with "
           "structured attributes";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();
    OpBuilder builder(ctx);

    mod.walk([&](Operation *op) -> WalkResult {
      if (op->getName().getStringRef() !=
          "tessera.neighbors.stencil.apply")
        return WalkResult::advance();

      // Skip already-lowered ops.
      if (op->hasAttr("stencil.lowered"))
        return WalkResult::advance();

      // ----------------------------------------------------------------
      // Gather information
      // ----------------------------------------------------------------
      Operation *stencilDef =
          op->getNumOperands() > 0
              ? op->getOperand(0).getDefiningOp()
              : nullptr;

      int64_t tapCount = countTaps(stencilDef);
      StringRef policy = exchangePolicy(op);

      // Halo width — prefer halo.width set by HaloInferPass.
      ArrayAttr haloWidth;
      if (op->getNumOperands() > 1) {
        // Check if operand 1 (the field) comes from a halo.region
        if (auto *haloOp = op->getOperand(1).getDefiningOp()) {
          if (haloOp->getName().getStringRef() ==
              "tessera.neighbors.halo.region") {
            haloWidth = haloOp->getAttrOfType<ArrayAttr>("halo.width");
          }
        }
      }
      if (!haloWidth)
        haloWidth = op->getAttrOfType<ArrayAttr>("halo.width");

      // ----------------------------------------------------------------
      // Phase A — Pack annotation
      // ----------------------------------------------------------------
      op->setAttr("stencil.pack_phase", builder.getBoolAttr(true));

      // ----------------------------------------------------------------
      // Phase B — Exchange annotation
      // ----------------------------------------------------------------
      op->setAttr("stencil.exchange_policy",
                  builder.getStringAttr(policy));

      // Synthesise a halo.exchange annotation if no explicit halo.region
      // exists.  A downstream pass will turn this into a real op.
      if (!op->hasAttr("stencil.halo_exchange")) {
        op->setAttr("stencil.halo_exchange", builder.getBoolAttr(true));
        op->setAttr("stencil.halo_async",    builder.getBoolAttr(true));
      }

      // ----------------------------------------------------------------
      // Phase C — Compute annotation
      // ----------------------------------------------------------------
      op->setAttr("stencil.compute_phase", builder.getBoolAttr(true));
      op->setAttr("stencil.tap_count",
                  builder.getI64IntegerAttr(tapCount));

      if (haloWidth)
        op->setAttr("stencil.halo_width", haloWidth);

      // Boundary condition (from stencil.define's "bc" attr)
      if (stencilDef) {
        if (auto bc = stencilDef->getAttrOfType<StringAttr>("bc"))
          op->setAttr("stencil.bc", bc);
      }

      // Sentinel
      op->setAttr("stencil.lowered", builder.getBoolAttr(true));

      return WalkResult::advance();
    });
  }
};

void registerStencilLowerPass() {
  PassRegistration<StencilLowerPass>();
}

} // namespace neighbors
} // namespace tessera
