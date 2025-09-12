
// ts-spectral-opt: minimal mlir-opt-style driver that registers TesseraSpectral
#include "tessera/Spectral/SpectralDialect.h"
#include "tessera/Spectral/SpectralPasses.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<tessera::spectral::SpectralDialect>();
  PassPipelineRegistration<> cleanup("tessera-spectral-cleanup", "No-op cleanup for now",
    [](OpPassManager &pm){});
  return failed(MlirOptMain(argc, argv, "ts-spectral-opt\n", registry));
}
