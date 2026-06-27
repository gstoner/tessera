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
std::unique_ptr<mlir::Pass> createGenerateROCMActivationKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMSiluMulKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMAlibiKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMDeltaNetKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMRopeKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMSoftmaxKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMNormKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMReduceKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMArgReduceKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMUnaryKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMBinaryKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMCompareKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMLogicalKernelPass();
std::unique_ptr<mlir::Pass> createGenerateROCMBitwiseKernelPass();
std::unique_ptr<mlir::Pass> createLowerROCMAsyncCopyToLoopPass();
void buildTesseraROCMBackendPipeline(mlir::OpPassManager &pm);
void registerTesseraROCMPasses();
void registerTesseraROCMDialects(mlir::DialectRegistry &registry);
void registerTesseraROCMBackendPasses();
void registerTesseraROCMBackendDialects(mlir::DialectRegistry &registry);
}}
