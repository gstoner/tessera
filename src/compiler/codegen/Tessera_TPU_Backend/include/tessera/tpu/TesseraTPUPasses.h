#pragma once
#include "mlir/Pass/Pass.h"

namespace mlir {
class DialectRegistry;
class OpPassManager;
}

namespace tessera {
std::unique_ptr<mlir::Pass> createLowerTesseraToStableHLOPass();
std::unique_ptr<mlir::Pass> createAnnotateShardingPass();

void buildTesseraTPUBackendPipeline(mlir::OpPassManager &pm);
void registerTesseraTPUPasses();
void registerTesseraTPUBackendPasses();
void registerTesseraTPUBackendDialects(mlir::DialectRegistry &registry);
} // namespace tessera

std::unique_ptr<mlir::Pass> createLowerTesseraAttentionToStableHLOPass();
std::unique_ptr<mlir::Pass> createLowerTesseraConvToStableHLOPass();
std::unique_ptr<mlir::Pass> createExportShardyPass();
