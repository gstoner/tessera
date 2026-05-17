//===- FuseEnergyGrad.cpp ----------------------------------------*- C++ -*-===//
//
// EBMFuseEnergyGradPass: links eligible `tessera_ebm.energy` evaluations
// with their downstream `tessera_ebm.langevin_step` / `tessera_ebm.inner_step`
// consumers so a backend codegen pass can emit a fused
// energy-and-gradient kernel that reuses activations.
//
// The fusion is recognized when:
//   1. An `ebm.energy` op references some `energy_fn` symbol and
//      consumes a y operand.
//   2. A subsequent `ebm.langevin_step` (or `inner_step`) in the same
//      block references the same `energy_fn` symbol and the same y
//      operand.
// The pass attaches two annotations:
//   - `tessera.ebm.fused_with` on both ops, holding a SymbolRefAttr
//     pointing at the partner op's anonymous name (we use a fresh
//     UnitAttr label since MLIR ops don't have stable identifiers).
//   - `tessera.ebm.energy_grad_fused` on each, a UnitAttr that
//     downstream backend lowering can match against to pick a fused
//     kernel.
//
// Without an explicit `ebm.grad_y` op in the dialect, this v1 pass is
// annotation-only: it doesn't rewrite the IR, just marks pairs.
// Backends are free to fuse or not at codegen time.
//
//===----------------------------------------------------------------------===//

#include "tessera/EBM/EBMPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <utility>
#include <vector>

using namespace mlir;

namespace tessera {
namespace {

constexpr StringRef kEnergyOpName = "tessera_ebm.energy";
constexpr StringRef kLangevinStepOpName = "tessera_ebm.langevin_step";
constexpr StringRef kInnerStepOpName = "tessera_ebm.inner_step";
constexpr StringRef kFusedMarkerAttr = "tessera.ebm.energy_grad_fused";
constexpr StringRef kFusedPartnerAttr = "tessera.ebm.fused_with_symbol";

static bool isStepOp(StringRef name) {
  return name == kLangevinStepOpName || name == kInnerStepOpName;
}

struct EBMFuseEnergyGradPass
    : public PassWrapper<EBMFuseEnergyGradPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EBMFuseEnergyGradPass)

  StringRef getArgument() const final { return "tessera-ebm-fuse-energy-grad"; }
  StringRef getDescription() const final {
    return "Mark eligible (energy, langevin_step|inner_step) pairs sharing "
           "energy_fn + y for a fused energy-and-gradient kernel at codegen.";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    // Walk every function and look for energy + step pairs within the
    // same block.  Cross-block fusion would require dataflow analysis
    // beyond v1 scope.
    mod.walk([&](func::FuncOp fn) {
      fn.walk([&](Block *block) {
        // Collect energy ops in block order.
        std::vector<Operation *> energyOps;
        for (Operation &op : *block) {
          if (op.getName().getStringRef() == kEnergyOpName) {
            energyOps.push_back(&op);
          }
        }
        if (energyOps.empty()) return;

        for (Operation &op : *block) {
          StringRef name = op.getName().getStringRef();
          if (!isStepOp(name)) continue;

          // The step op's `y` operand is operand 0; `langevin_step` and
          // `inner_step` both take y as operand 0.
          if (op.getNumOperands() < 1) continue;
          Value stepY = op.getOperand(0);
          auto stepEnergyFn =
              op.getAttrOfType<FlatSymbolRefAttr>("energy_fn");
          if (!stepEnergyFn) continue;

          // Find a preceding energy op sharing both `y` and `energy_fn`.
          for (Operation *eOp : energyOps) {
            // Order: energy must precede the step op in the block.
            if (!eOp->isBeforeInBlock(&op)) continue;
            // The energy op's y operand is operand 1 (context_x = 0, y = 1).
            if (eOp->getNumOperands() < 2) continue;
            Value energyY = eOp->getOperand(1);
            if (energyY != stepY) continue;
            auto energyFn =
                eOp->getAttrOfType<FlatSymbolRefAttr>("energy_fn");
            if (!energyFn || energyFn != stepEnergyFn) continue;

            // Match.  Mark both ops as fused.
            eOp->setAttr(kFusedMarkerAttr, builder.getUnitAttr());
            op.setAttr(kFusedMarkerAttr, builder.getUnitAttr());
            // Use the energy_fn symbol itself as the linking attribute —
            // simpler than synthesizing a fresh op identifier and stable
            // across rewrites that re-create the ops.
            eOp->setAttr(kFusedPartnerAttr, stepEnergyFn);
            op.setAttr(kFusedPartnerAttr, stepEnergyFn);
            break;  // one fusion per step op is enough
          }
        }
      });
    });
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createEBMFuseEnergyGradPass() {
  return std::make_unique<EBMFuseEnergyGradPass>();
}

}  // namespace tessera
