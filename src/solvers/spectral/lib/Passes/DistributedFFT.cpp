//===- DistributedFFT.cpp --------------------------------------*- C++ -*-===//
//
// SpectralDistributedFFTPass: decomposes multi-dim FFTs into local 1D FFTs
// separated by all-to-all collectives.  The classic "pencil decomposition"
// strategy:
//
//   3D FFT of shape (X, Y, Z) on a (P_x, P_y) mesh:
//     1. local FFT along Z (full Z is on every rank)
//     2. all-to-all along the P_y axis to make Y full
//     3. local FFT along Y
//     4. all-to-all along the P_x axis to make X full
//     5. local FFT along X
//
// We annotate each fft / ifft op with:
//
//   tessera.dist.axis_split    : ArrayAttr<I64> — initial partition shape
//                                                 (one entry per FFT axis)
//   tessera.dist.transposes    : ArrayAttr<StrAttr> — sequence of
//                                  "all_to_all:<mesh_axis>" tokens, one
//                                  between every pair of consecutive FFT axes
//   tessera.dist.overlap_token : StrAttr — name of the CommQ overlap stream
//                                          to issue the all-to-alls on
//   tessera.dist.local_only    : BoolAttr — true when no collective needed
//                                           (1D FFT or single-rank mesh)
//
// Single-axis FFT or a missing mesh attribute → local_only = true and no
// transposes are emitted.
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

struct DistributedFFTPass
    : public PassWrapper<DistributedFFTPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DistributedFFTPass)

  StringRef getArgument() const final {
    return "tessera-spectral-distributed";
  }
  StringRef getDescription() const final {
    return "Decompose multi-axis FFTs into local 1D FFTs + all-to-all "
           "collectives (pencil decomposition).";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    // Read mesh spec.  Expected form on the module:
    //   tessera.mesh.axes = ["dp", "tp"] (or similar)
    ArrayAttr meshAxes;
    if (auto a = mod->getAttrOfType<ArrayAttr>("tessera.mesh.axes"))
      meshAxes = a;

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

      bool localOnly = axes.size() < 2 || !meshAxes || meshAxes.empty();
      if (localOnly) {
        op->setAttr("tessera.dist.local_only", builder.getBoolAttr(true));
        return WalkResult::advance();
      }

      // Initial partition: split the first FFT axis across the first mesh
      // axis, the second FFT axis across the second mesh axis, and so on.
      // Any remaining FFT axes are replicated (split = 1).
      SmallVector<Attribute, 4> split;
      for (size_t i = 0; i < axes.size(); ++i) {
        if (i < meshAxes.size())
          split.push_back(meshAxes[i]);
        else
          split.push_back(StringAttr::get(ctx, "*"));
      }
      op->setAttr("tessera.dist.axis_split", ArrayAttr::get(ctx, split));

      // One all-to-all between every pair of consecutive FFT axes, cycling
      // through mesh axes to reduce contention.
      SmallVector<Attribute, 4> transposes;
      for (size_t i = 0; i + 1 < axes.size(); ++i) {
        StringRef meshAxis = "dp";
        if (auto a = dyn_cast<StringAttr>(meshAxes[i % meshAxes.size()]))
          meshAxis = a.getValue();
        std::string tok = ("all_to_all:" + meshAxis).str();
        transposes.push_back(StringAttr::get(ctx, tok));
      }
      op->setAttr("tessera.dist.transposes",
                  ArrayAttr::get(ctx, transposes));
      op->setAttr("tessera.dist.overlap_token",
                  StringAttr::get(ctx, "comm_q_default"));
      op->setAttr("tessera.dist.local_only", builder.getBoolAttr(false));
      return WalkResult::advance();
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSpectralDistributedPass() {
  return std::make_unique<DistributedFFTPass>();
}

} // namespace tessera
