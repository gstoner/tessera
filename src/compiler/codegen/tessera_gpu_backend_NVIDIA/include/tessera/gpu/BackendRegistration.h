#pragma once

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"

namespace tessera {

void buildTesseraNVIDIABackendPipeline(mlir::OpPassManager &pm);
void registerTesseraNVIDIABackendPasses();
void registerTesseraNVIDIABackendDialects(mlir::DialectRegistry &registry);

} // namespace tessera
