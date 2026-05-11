//===- LowerToTargetIR.cpp -------------------------------------*- C++ -*-===//
//
// LowerSpectralToTargetIRPass: dispatches each legalized tessera_spectral
// exec op to a target-IR call.  We don't materialize Tile IR ourselves
// (that lives in the Tile/Schedule dialect downstream); instead we
// annotate every fft/ifft/conv_fft op with:
//
//     tessera.target_ir.backend  : "cpu" | "nvidia" | "amd"
//     tessera.target_ir.call     : the C ABI symbol to emit at codegen,
//                                  e.g., @ts_stockham_radix4_scalar
//     tessera.target_ir.stage_calls : ArrayAttr of per-stage symbols
//                                     (one entry per radix stage)
//
// Symbol naming convention is locked to the StockhamRadix4 target hook files
// shipped under lib/TargetHooks/{CPU,NVIDIA,AMD}/:
//
//   CPU    → @ts_stockham_radix4_scalar     (extern "C", reference scalar)
//   NVIDIA → @ts_stockham_radix4_sm90       (CUDA kernel, SM90+)
//   AMD    → @ts_stockham_radix4_gfx94x     (HIP kernel)
//
// Conv-FFT collapses to plan→pad→fft→cmul→ifft→crop; we attach the same
// stage_calls but mark the op with `tessera.target_ir.composite = "conv_fft"`
// so the actual codegen pass emits the wrapper.
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

static StringRef stageSymbolFor(StringRef backend, int64_t radix) {
  // Today every radix routes through the radix-4 Stockham kernel; future
  // work adds radix-2/3/5/7 specializations.  Returning the same symbol
  // is intentional — codegen branches on radix at the call site.
  (void)radix;
  if (backend == "nvidia")
    return "ts_stockham_radix4_sm90";
  if (backend == "amd")
    return "ts_stockham_radix4_gfx94x";
  return "ts_stockham_radix4_scalar";
}

struct LowerToTargetIRPass
    : public PassWrapper<LowerToTargetIRPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToTargetIRPass)

  StringRef getArgument() const final {
    return "lower-spectral-to-target-ir";
  }
  StringRef getDescription() const final {
    return "Annotate tessera_spectral exec ops with target-IR call symbols "
           "(CPU StockhamRadix4 / NVIDIA SM90 / AMD gfx94x).";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    StringRef backend = "cpu";
    if (auto a = mod->getAttrOfType<StringAttr>("tessera.target"))
      backend = a.getValue();

    mod.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      bool isFFT = name == "tessera_spectral.fft";
      bool isIFFT = name == "tessera_spectral.ifft";
      bool isConv = name == "tessera_spectral.conv_fft";
      if (!isFFT && !isIFFT && !isConv)
        return WalkResult::advance();

      op->setAttr("tessera.target_ir.backend", StringAttr::get(ctx, backend));

      // Default top-level symbol = first stage.  Composite ops mark
      // themselves so codegen knows to wrap with pad/cmul/crop.
      SmallVector<Attribute, 8> stageCalls;
      if (auto stages =
              op->getAttrOfType<ArrayAttr>("tessera.spectral.stages")) {
        for (Attribute a : stages) {
          auto ia = dyn_cast<IntegerAttr>(a);
          if (!ia)
            continue;
          int64_t r = ia.getInt();
          if (r < 0)
            continue; // axis separator from LegalizeSpectralPass
          stageCalls.push_back(
              StringAttr::get(ctx, stageSymbolFor(backend, r)));
        }
      }
      if (stageCalls.empty())
        stageCalls.push_back(
            StringAttr::get(ctx, stageSymbolFor(backend, 4)));

      op->setAttr("tessera.target_ir.call", stageCalls.front());
      op->setAttr("tessera.target_ir.stage_calls",
                  ArrayAttr::get(ctx, stageCalls));
      if (isConv)
        op->setAttr("tessera.target_ir.composite",
                    StringAttr::get(ctx, "conv_fft"));
      op->setAttr("tessera.target_ir.lowered", builder.getUnitAttr());
      return WalkResult::advance();
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerSpectralToTargetIRPass() {
  return std::make_unique<LowerToTargetIRPass>();
}

} // namespace tessera
