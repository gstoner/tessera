#pragma once
#include "mlir/Pass/Pass.h"

namespace tessera {
std::unique_ptr<mlir::Pass> createLowerTesseraToStableHLOPass();
std::unique_ptr<mlir::Pass> createAnnotateShardingPass();

void registerTesseraTPUPasses();
} // namespace tessera

std::unique_ptr<mlir::Pass> createLowerTesseraAttentionToStableHLOPass();
std::unique_ptr<mlir::Pass> createLowerTesseraConvToStableHLOPass();
std::unique_ptr<mlir::Pass> createExportShardyPass();
