//===- LegalizeSpectral.cpp.cpp -------------------------------------------*- C++ -*-===//
// Lower tessera_spectral ops to Tile/Schedule IR
//===----------------------------------------------------------------------===//
#include "tessera/Spectral/SpectralPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace tessera {
namespace legalizespectral {
struct LegalizeSpectralPass : public PassWrapper<LegalizeSpectralPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeSpectralPass)
  void runOnOperation() override {
    // TODO: implement
  }
};
} // namespace
std::unique_ptr<mlir::Pass> createLegalizeSpectralPass() {
  return std::make_unique<LegalizeSpectralPass>();
}
} // namespace tessera
