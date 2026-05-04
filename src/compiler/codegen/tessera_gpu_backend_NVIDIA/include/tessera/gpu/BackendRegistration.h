#pragma once

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace tessera {

std::unique_ptr<mlir::Pass> createLowerTileToNVIDIAPass(int sm = 90);
std::unique_ptr<mlir::Pass> createLowerNVIDIAToNVVMPass();
void buildTesseraNVIDIABackendPipeline(mlir::OpPassManager &pm);
void buildTesseraHopperBackendPipeline(mlir::OpPassManager &pm);
void buildTesseraBlackwellBackendPipeline(mlir::OpPassManager &pm);
void registerTesseraNVIDIABackendPasses();
void registerTesseraNVIDIABackendDialects(mlir::DialectRegistry &registry);

} // namespace tessera
