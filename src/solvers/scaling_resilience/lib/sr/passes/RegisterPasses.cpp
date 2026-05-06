#include "tessera/sr/Passes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

void mlir::tessera::sr::registerPasses() {
  registerPass([]() { return createInsertRecomputePass(); });
  registerPass([]() { return createOptimizerShardPass(); });
  registerPass([]() { return createResilienceRestartPass(); });
  registerPass([]() { return createExportDeploymentManifestPass(); });
}
