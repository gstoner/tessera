//===- SpectralMXP.cpp.cpp -------------------------------------------*- C++ -*-===//
// Insert mixed-precision scaling operators
//===----------------------------------------------------------------------===//
#include "tessera/Spectral/SpectralPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace tessera {
namespace spectralmxp {
struct SpectralMXPPass : public PassWrapper<SpectralMXPPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SpectralMXPPass)
  void runOnOperation() override {
    // TODO: implement
  }
};
} // namespace
std::unique_ptr<mlir::Pass> createSpectralMXPPass() {
  return std::make_unique<SpectralMXPPass>();
}
} // namespace tessera
