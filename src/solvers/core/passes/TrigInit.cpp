//===- TrigInit.cpp — pre-compute trig tables for spectral solver ops ---*- C++ -*-===//
//
// Finds tessera_solver.spectral and tessera.fft ops and attaches:
//   tessera_solver.trig_table_size  — table length needed (power-of-2)
//   tessera_solver.fft_plan         — plan descriptor string
//   tessera_solver.window_type      — "hann" | "hamming" | "none" (default)
//
// The trig_table_size is the smallest power of two ≥ the dominant signal
// length inferred from the result tensor type.
//
//===----------------------------------------------------------------------===//

#include "SolversPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include <cstdint>

using namespace mlir;

namespace {

/// Next power of two ≥ n (minimum 1).
static int64_t nextPow2(int64_t n) {
  if (n <= 1) return 1;
  int64_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

struct TrigInitPass
    : PassWrapper<TrigInitPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TrigInitPass)

  Option<std::string> windowType{
      *this, "window-type",
      llvm::cl::desc("FFT window function: none | hann | hamming"),
      llvm::cl::init(std::string("none"))};

  StringRef getArgument() const final { return "tessera-trig-init"; }
  StringRef getDescription() const final {
    return "Attach trig-table and FFT-plan attrs to spectral solver ops";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    mod.walk([&](Operation *op) {
      StringRef opName = op->getName().getStringRef();
      bool isSpectral = opName.contains("spectral") || opName.contains("fft") ||
                        opName.contains("dft") || opName.contains("dct");
      if (!isSpectral)
        return;

      // Infer signal length from first result type (if shaped).
      int64_t sigLen = 1024; // conservative default
      if (op->getNumResults() > 0) {
        if (auto shaped = op->getResult(0).getType().dyn_cast<ShapedType>()) {
          if (shaped.hasStaticShape()) {
            // Use the last (innermost) dimension as the signal length.
            auto shape = shaped.getShape();
            if (!shape.empty())
              sigLen = shape.back();
          }
        }
      }

      // Allow explicit override.
      if (auto attr = op->getAttrOfType<IntegerAttr>(
              "tessera_solver.signal_length"))
        sigLen = attr.getInt();

      int64_t tableSize = nextPow2(sigLen);

      op->setAttr("tessera_solver.trig_table_size",
                  IntegerAttr::get(IntegerType::get(ctx, 64), tableSize));
      op->setAttr("tessera_solver.fft_plan",
                  StringAttr::get(ctx,
                                  "fft_c2c_size" + std::to_string(tableSize)));
      op->setAttr("tessera_solver.window_type",
                  StringAttr::get(ctx, windowType));
      op->setAttr("tessera_solver.trig_initialized", UnitAttr::get(ctx));
    });
  }
};

} // namespace

namespace tessera {
namespace passes {
std::unique_ptr<Pass> createTrigInitPass() {
  return std::make_unique<TrigInitPass>();
}
} // namespace passes
} // namespace tessera
