#include "tessera/ScalingPasses.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

void mlir::tessera::sr::registerSRPasses() {
  PassRegistration<mlir::tessera::sr::createInsertRecomputePass>();
  PassRegistration<mlir::tessera::sr::createOptimizerShardPass>();
  PassRegistration<mlir::tessera::sr::createResilienceRestartPass>();
  PassRegistration<mlir::tessera::sr::createExportDeploymentManifestPass>();
}