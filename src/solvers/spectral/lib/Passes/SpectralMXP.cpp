//===- SpectralMXP.cpp -----------------------------------------*- C++ -*-===//
//
// SpectralMXPPass: insert mixed-precision scaling decisions onto legalized
// spectral exec ops.  Reads the plan's `elem_precision`, `acc_precision`,
// and `scaling` attributes; decides per-stage block-floating-point
// parameters and rescaling factors; attaches them as structured
// attributes that the target-IR lowering pass consumes.
//
// Decision table (locked here so it can be unit-tested via lit):
//
//   elem_precision   acc_precision   scaling                 → mxp.block_size
//   --------------   -------------   --------------          ------------------
//   fp8_e4m3 / fp8_e5m2   f32        blockfp_per_stage       32 (per stage)
//   fp16 / bf16           f32        blockfp_per_stage       64 (per stage)
//   f32                   f32        none                    0  (no rescale)
//   *                     *          mu_law                  -1 (companding)
//
// We also emit `tessera.mxp.guard_eps` = 1e-6 by default (configurable via
// future pipeline options) so the inverse transform can clamp denormals.
//
// The pass is annotation-only.  No new ops are created; the actual rescale
// arithmetic is materialized by LowerSpectralToTargetIRPass.
//
//===----------------------------------------------------------------------===//

#include "tessera/Spectral/SpectralPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace tessera {
namespace {

static int64_t blockSizeFor(StringRef elem, StringRef scaling) {
  if (scaling == "none")
    return 0;
  if (scaling == "mu_law")
    return -1;
  // default = blockfp_per_stage
  if (elem.starts_with("fp8"))
    return 32;
  if (elem == "fp16" || elem == "bf16")
    return 64;
  return 0;
}

struct SpectralMXPPass
    : public PassWrapper<SpectralMXPPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SpectralMXPPass)

  StringRef getArgument() const final { return "tessera-spectral-mxp"; }
  StringRef getDescription() const final {
    return "Insert block-FP / mu-law mixed-precision scaling decisions onto "
           "tessera_spectral exec ops.";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    mod.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "tessera_spectral.fft" && name != "tessera_spectral.ifft" &&
          name != "tessera_spectral.conv_fft")
        return WalkResult::advance();

      if (op->getNumOperands() < 1)
        return WalkResult::advance();
      Operation *planDef = op->getOperand(0).getDefiningOp();
      if (!planDef || planDef->getName().getStringRef() != "tessera_spectral.plan")
        return WalkResult::advance();

      auto elem = planDef->getAttrOfType<StringAttr>("elem_precision");
      auto acc = planDef->getAttrOfType<StringAttr>("acc_precision");
      auto scaling = planDef->getAttrOfType<StringAttr>("scaling");
      if (!elem || !acc || !scaling)
        return WalkResult::advance();

      int64_t bs = blockSizeFor(elem.getValue(), scaling.getValue());
      op->setAttr("tessera.mxp.block_size", builder.getI64IntegerAttr(bs));
      op->setAttr("tessera.mxp.elem_dtype", elem);
      op->setAttr("tessera.mxp.acc_dtype", acc);
      op->setAttr("tessera.mxp.scaling", scaling);
      op->setAttr("tessera.mxp.guard_eps",
                  builder.getF64FloatAttr(1e-6));
      // Number of stages to scale = number of radix stages.  Read back from
      // LegalizeSpectralPass output if available.
      if (auto stages = op->getAttrOfType<ArrayAttr>("tessera.spectral.stages"))
        op->setAttr("tessera.mxp.scale_blocks",
                    builder.getI64IntegerAttr(stages.size()));
      op->setAttr("tessera.mxp.legalized", builder.getUnitAttr());
      return WalkResult::advance();
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSpectralMXPPass() {
  return std::make_unique<SpectralMXPPass>();
}

} // namespace tessera
