//===- DistributedFFT.cpp.cpp -------------------------------------------*- C++ -*-===//
// Decompose into local FFT + all-to-all
//===----------------------------------------------------------------------===//
#include "tessera/Spectral/SpectralPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace tessera {
namespace distributedfft {
struct DistributedFFTPass : public PassWrapper<DistributedFFTPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DistributedFFTPass)
  void runOnOperation() override {
    // TODO: implement
  }
};
} // namespace
std::unique_ptr<mlir::Pass> createSpectralDistributedPass() {
  return std::make_unique<DistributedFFTPass>();
}
} // namespace tessera
