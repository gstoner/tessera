//===- Autotune.cpp.cpp -------------------------------------------*- C++ -*-===//
// Autotune radix/tile configs
//===----------------------------------------------------------------------===//
#include "tessera/Spectral/SpectralPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace tessera {
namespace autotune {
struct AutotunePass : public PassWrapper<AutotunePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AutotunePass)
  void runOnOperation() override {
    // TODO: implement
  }
};
} // namespace
std::unique_ptr<mlir::Pass> createSpectralAutotunePass() {
  return std::make_unique<AutotunePass>();
}
} // namespace tessera
