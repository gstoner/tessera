#pragma once
#include "mlir/Pass/Pass.h"
namespace mlir {
class Pass;
class OpPassManager;
class DialectRegistry;
namespace tessera_rocm {
std::unique_ptr<mlir::Pass> createLowerTesseraTargetToROCDLPass();
std::unique_ptr<mlir::Pass> createLowerKernelABIPass();
void buildTesseraROCMBackendPipeline(mlir::OpPassManager &pm);
void registerTesseraROCMPasses();
void registerTesseraROCMDialects(mlir::DialectRegistry &registry);
void registerTesseraROCMBackendPasses();
void registerTesseraROCMBackendDialects(mlir::DialectRegistry &registry);
}}
