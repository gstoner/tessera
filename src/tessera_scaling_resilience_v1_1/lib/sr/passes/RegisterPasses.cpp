#include "tessera/sr/Passes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

void mlir::tessera::sr::registerPasses() {
  PassRegistration<struct InsertRecomputePass>();
  PassRegistration<struct OptimizerShardPass>();
  PassRegistration<struct ResilienceRestartPass>();
  PassRegistration<struct ExportDeploymentManifestPass>();
}