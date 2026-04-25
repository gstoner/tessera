#pragma once
#include "mlir/Pass/Pass.h"
namespace mlir { namespace tessera { namespace sr {
std::unique_ptr<Pass> createInsertRecomputePass();
std::unique_ptr<Pass> createOptimizerShardPass();
std::unique_ptr<Pass> createResilienceRestartPass();
std::unique_ptr<Pass> createExportDeploymentManifestPass();
void registerPasses();
}}}