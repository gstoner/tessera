#pragma once
#include "mlir/Pass/Pass.h"

namespace tessera {
std::unique_ptr<mlir::Pass> createLowerTesseraToStableHLOPass();
std::unique_ptr<mlir::Pass> createAnnotateShardingPass();

void registerTesseraTPUPasses();
} // namespace tessera
