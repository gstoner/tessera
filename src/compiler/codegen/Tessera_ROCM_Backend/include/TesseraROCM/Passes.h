#pragma once
#include "mlir/Pass/Pass.h"
namespace mlir {
class Pass;
class OpPassManager;
class DialectRegistry;
namespace tessera_rocm {
std::unique_ptr<mlir::Pass> createLowerTileToROCMImpl();
std::unique_ptr<mlir::Pass> createLowerTileToROCMPass();
std::unique_ptr<mlir::Pass> createROCMWaveLdsPipelinePass();
std::unique_ptr<mlir::Pass> createROCMWaveLdsLegalityPass();
std::unique_ptr<mlir::Pass> createLowerTesseraToROCDLImpl();
std::unique_ptr<mlir::Pass> createLowerTesseraTargetToROCDLPass();
std::unique_ptr<mlir::Pass> createLowerKernelABIPass();
std::unique_ptr<mlir::Pass> createGenerateWMMAGemmKernelPass();
std::unique_ptr<mlir::Pass> createGenerateWMMAFlashAttnKernelPass();
std::unique_ptr<mlir::Pass> createGenerateWMMAFlashAttnBwdKernelPass();
std::unique_ptr<mlir::Pass> createGenerateWMMALinearAttnKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMSoftmaxKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMNormKernelPass();
std::unique_ptr<mlir::Pass> createLowerROCMAsyncCopyToLoopPass();
void buildTesseraROCMBackendPipeline(mlir::OpPassManager &pm);
void registerTesseraROCMPasses();
void registerTesseraROCMDialects(mlir::DialectRegistry &registry);
void registerTesseraROCMBackendPasses();
void registerTesseraROCMBackendDialects(mlir::DialectRegistry &registry);
}}
