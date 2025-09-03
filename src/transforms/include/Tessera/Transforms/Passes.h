
#pragma once
#include "mlir/Pass/Pass.h"

namespace tessera {
std::unique_ptr<mlir::Pass> createCanonicalizeTesseraIRPass();
std::unique_ptr<mlir::Pass> createVerifyTesseraIRPass();
std::unique_ptr<mlir::Pass> createMigrateTesseraIRPass();
void registerTesseraPasses();
} // namespace tessera
