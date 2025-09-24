#pragma once
#include "mlir/Pass/Pass.h"
namespace tessera { namespace power {
std::unique_ptr<mlir::Pass> createLowerPowerToTilePass();
std::unique_ptr<mlir::Pass> createLowerPowerToTargetPass();
void registerPowerPasses();
}}