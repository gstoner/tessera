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

// The pre-Target-IR WGMMA/TMA pipeline remains available for existing users.
// It intentionally has separate registration symbols: the typed Target IR
// pipeline owns the canonical NVIDIA registration entry points above.
void registerTesseraNVIDIALegacyBackendPasses();

} // namespace tessera
