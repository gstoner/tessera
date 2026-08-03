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
// shipped under lib/TargetHooks/{CPU,NVIDIA,AMD}/.  Each backend exposes a
// radix-4 and a radix-2 stage kernel (mixed-radix Stockham autosort):
//
//   CPU    → @ts_stockham_r4_cpu     / @ts_stockham_r2_cpu     (extern "C")
//   NVIDIA → @ts_stockham_r4_nvidia  / @ts_stockham_r2_nvidia  (CUDA)
//   AMD    → @ts_stockham_r4_amd     / @ts_stockham_r2_amd     (HIP)
//
// Symbols are arch-agnostic C ABI entry points; the ISA (gfx1151, sm_90/
// sm_120, …) is chosen at kernel compile/launch time, not encoded here.
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
  // Route each stage to the matching Stockham kernel shipped under
  // lib/TargetHooks/{CPU,NVIDIA,AMD}/StockhamRadix4.*.
  //
  // Radix 4 and radix 2 have hand-written butterflies; every other radix the
  // planner emits (3, 5, 7, 11, 13) goes to the generic radix-r stage.
  //
  // This used to map EVERY radix other than 4 to the radix-2 symbol, with a
  // comment saying "until radix-3/5/7 specializations land".  They have
  // landed, and until now nothing routed to them: a statically-shaped length
  // like 12 = 4x3 emitted a radix-4 call followed by a radix-2 call for a
  // stage that is radix 3, producing an incorrect transform through the
  // compiler path even though the runtime driver -- which factors N itself --
  // was correct. Direct driver tests could not see it.
  if (radix == 4) {
    if (backend == "nvidia") return "ts_stockham_r4_nvidia";
    if (backend == "amd") return "ts_stockham_r4_amd";
    return "ts_stockham_r4_cpu";
  }
  if (radix == 2) {
    if (backend == "nvidia") return "ts_stockham_r2_nvidia";
    if (backend == "amd") return "ts_stockham_r2_amd";
    return "ts_stockham_r2_cpu";
  }
  if (backend == "nvidia") return "ts_stockham_rn_nvidia";
  if (backend == "amd") return "ts_stockham_rn_amd";
  return "ts_stockham_rn_cpu";
}

/// Runtime driver symbol: factors the runtime N and launches the matching
/// per-stage kernels.  Used for dynamic-shape FFTs, where the radix sequence
/// is unknown at compile time and must not be fabricated.
static StringRef driverSymbolFor(StringRef backend) {
  if (backend == "nvidia")
    return "ts_fft_stockham_nvidia";
  if (backend == "amd")
    return "ts_fft_stockham_amd";
  return "ts_fft_stockham_cpu";
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
      // Name the D1 arbiter op-kind so the runtime dispatches this op through
      // the shared candidate arbiter (emit/spectral_candidates.py) instead of a
      // hard-wired symbol: the arbiter enumerates the Stockham lanes for
      // (op="spectral_fft", target) and picks the fastest in-budget one.
      op->setAttr("tessera.target_ir.arbiter_op",
                  StringAttr::get(ctx, "spectral_fft"));

      // Dynamic-shape FFTs cannot have a compile-time stage list (the
      // legalizer left it empty and set `dynamic_shape`).  Route the whole
      // op to the runtime driver symbol, which factors the runtime N and
      // launches the matching per-stage kernels.  Do NOT fall through to a
      // fabricated radix-4 stage — that was the old silent-miscompile.
      if (op->hasAttr("tessera.spectral.dynamic_shape")) {
        StringAttr drv = StringAttr::get(ctx, driverSymbolFor(backend));
        op->setAttr("tessera.target_ir.call", drv);
        op->setAttr("tessera.target_ir.stage_calls",
                    ArrayAttr::get(ctx, {drv}));
        op->setAttr("tessera.target_ir.dynamic", builder.getUnitAttr());
        if (isConv)
          op->setAttr("tessera.target_ir.composite",
                      StringAttr::get(ctx, "conv_fft"));
        op->setAttr("tessera.target_ir.lowered", builder.getUnitAttr());
        return WalkResult::advance();
      }

      // Bluestein lengths have NO stage list -- it is a different algorithm,
      // not a longer sequence of butterflies -- so they route to the driver
      // exactly as dynamic shapes do.  The legalizer signals this by emitting
      // an empty stage list for a statically-shaped op.
      if (op->hasAttr("tessera.spectral.bluestein")) {
        StringAttr drv = StringAttr::get(ctx, driverSymbolFor(backend));
        op->setAttr("tessera.target_ir.call", drv);
        op->setAttr("tessera.target_ir.stage_calls",
                    ArrayAttr::get(ctx, {drv}));
        op->setAttr("tessera.target_ir.bluestein", builder.getUnitAttr());
        if (isConv)
          op->setAttr("tessera.target_ir.composite",
                      StringAttr::get(ctx, "conv_fft"));
        op->setAttr("tessera.target_ir.lowered", builder.getUnitAttr());
        return WalkResult::advance();
      }

      // Static shape: one C ABI symbol per resolved radix stage.  Composite
      // ops mark themselves so codegen knows to wrap with pad/cmul/crop.
      //
      // The radix travels alongside the symbol: `ts_stockham_rn_*` takes it as
      // an argument (r4/r2 do not), so a consumer that builds the actual call
      // needs it and cannot recover it from the symbol name.
      SmallVector<Attribute, 8> stageCalls;
      SmallVector<Attribute, 8> stageRadices;
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
          stageRadices.push_back(builder.getI64IntegerAttr(r));
        }
      }
      // A static, legalized FFT always has at least one stage.  If we somehow
      // reach here with none (e.g. an unlegalized op), route to the runtime
      // driver rather than inventing a radix-4 stage.
      if (stageCalls.empty())
        stageCalls.push_back(
            StringAttr::get(ctx, driverSymbolFor(backend)));

      op->setAttr("tessera.target_ir.call", stageCalls.front());
      op->setAttr("tessera.target_ir.stage_calls",
                  ArrayAttr::get(ctx, stageCalls));
      if (!stageRadices.empty())
        op->setAttr("tessera.target_ir.stage_radices",
                    ArrayAttr::get(ctx, stageRadices));
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
