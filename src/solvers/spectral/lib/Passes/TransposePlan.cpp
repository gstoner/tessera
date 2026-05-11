//===- TransposePlan.cpp ---------------------------------------*- C++ -*-===//
//
// SpectralTransposePlanPass: choose tile shape, vector width, and shared-
// memory padding for the intra-stage transposes that radix-N Stockham FFTs
// need between butterfly passes.
//
// For a 2D FFT along axes (a0, a1) we need a transpose between the two 1D
// passes; for an N-D FFT we need N-1 transposes.  Each transpose is tagged
// with:
//   tessera.transpose.tile_shape = [TILE_M, TILE_N]   (e.g., [64, 64])
//   tessera.transpose.pad        = +1 lane to avoid 32-way bank conflicts on
//                                  CUDA-class targets
//   tessera.transpose.vector_w   = matches the element width of the plan's
//                                  acc_precision (4 for f32, 8 for f16/bf16,
//                                  16 for fp8_*)
//
// The default tile is 64×64 with +1 padding (the well-known "skewed shared
// memory" idiom from Volta-era cuFFT/cuBLAS), shrinking automatically when
// the static length is smaller than the tile.
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

static int64_t vectorWidthFor(StringRef acc) {
  if (acc == "f32" || acc == "fp32")
    return 4;
  if (acc == "fp16" || acc == "bf16" || acc == "f16")
    return 8;
  if (acc.starts_with("fp8"))
    return 16;
  return 4;
}

static int64_t chooseTile(int64_t N, int64_t maxTile = 64) {
  if (N <= 0)
    return maxTile;
  // Clamp to a power-of-two ≤ maxTile that divides N if possible.
  for (int64_t t = maxTile; t >= 4; t /= 2)
    if (N % t == 0)
      return t;
  return std::min<int64_t>(N, maxTile);
}

struct TransposePlanPass
    : public PassWrapper<TransposePlanPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransposePlanPass)

  StringRef getArgument() const final {
    return "tessera-spectral-transpose-plan";
  }
  StringRef getDescription() const final {
    return "Choose tile shapes + bank-conflict padding for intra-stage "
           "transposes of multi-axis FFTs.";
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

      auto axes = planDef->getAttrOfType<ArrayAttr>("axes");
      if (!axes)
        return WalkResult::advance();

      // No transpose needed for 1D FFTs.
      if (axes.size() < 2) {
        op->setAttr("tessera.transpose.required",
                    builder.getBoolAttr(false));
        return WalkResult::advance();
      }

      // For multi-axis FFTs, emit one tile_shape per transpose (axes.size()-1
      // transposes total).
      auto perAxis = op->getAttrOfType<ArrayAttr>("tessera.spectral.per_axis_len");
      SmallVector<Attribute, 4> tileShapes;
      for (size_t i = 0; i + 1 < axes.size(); ++i) {
        int64_t N0 = -1, N1 = -1;
        if (perAxis && i + 1 < perAxis.size()) {
          if (auto a = dyn_cast<IntegerAttr>(perAxis[i]))
            N0 = a.getInt();
          if (auto b = dyn_cast<IntegerAttr>(perAxis[i + 1]))
            N1 = b.getInt();
        }
        int64_t tM = chooseTile(N0);
        int64_t tN = chooseTile(N1);
        tileShapes.push_back(builder.getI64ArrayAttr({tM, tN}));
      }
      op->setAttr("tessera.transpose.tile_shapes",
                  ArrayAttr::get(ctx, tileShapes));
      op->setAttr("tessera.transpose.pad", builder.getI64IntegerAttr(1));

      // Vector width inferred from acc dtype if we recorded it via MXP pass,
      // otherwise default to f32 width 4.
      StringRef accDtype = "f32";
      if (auto a = op->getAttrOfType<StringAttr>("tessera.mxp.acc_dtype"))
        accDtype = a.getValue();
      op->setAttr("tessera.transpose.vector_w",
                  builder.getI64IntegerAttr(vectorWidthFor(accDtype)));
      op->setAttr("tessera.transpose.required", builder.getBoolAttr(true));
      return WalkResult::advance();
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSpectralTransposePlanPass() {
  return std::make_unique<TransposePlanPass>();
}

} // namespace tessera
