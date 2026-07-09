#include "TesseraROCM/Passes.h"
#include "mlir/IR/Dialect.h"
#include "TesseraROCMDialect.h.inc"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinOps.h"
using namespace mlir;

namespace mlir::tessera_rocm {
std::unique_ptr<Pass> createLowerTesseraToROCDLImpl();
std::unique_ptr<Pass> createLowerTileToROCMImpl();

std::unique_ptr<Pass> createLowerTileToROCMPass() {
  return createLowerTileToROCMImpl();
}

std::unique_ptr<Pass> createLowerTesseraTargetToROCDLPass() {
  return createLowerTesseraToROCDLImpl();
}

void buildTesseraROCMBackendPipeline(OpPassManager &pm) {
  pm.addPass(createROCMWaveLdsPipelinePass());
  pm.addPass(createROCMWaveLdsLegalityPass());
  pm.addPass(createLowerTileToROCMPass());
  pm.addPass(createLowerKernelABIPass());
  pm.addPass(createLowerTesseraTargetToROCDLPass());
}

void registerTesseraROCMPasses() {
  registerPass([]() { return createROCMWaveLdsPipelinePass(); });
  registerPass([]() { return createROCMWaveLdsLegalityPass(); });
  registerPass([]() { return createLowerTileToROCMPass(); });
  registerPass([]() { return createLowerKernelABIPass(); });
  registerPass([]() { return createLowerTesseraTargetToROCDLPass(); });
  registerPass([]() { return createGenerateWMMAGemmKernelPass(); });
  registerPass([]() { return createGenerateWMMAFlashAttnKernelPass(); });
  registerPass([]() { return createGenerateWMMAFlashAttnBwdKernelPass(); });
  registerPass([]() { return createGenerateWMMALinearAttnKernelPass(); });
  registerPass([]() { return createGenerateROCMActivationKernelPass(); });
  registerPass([]() { return createGenerateROCMSiluMulKernelPass(); });
  registerPass([]() { return createGenerateROCMPointwiseLossKernelPass(); });
  registerPass([]() { return createGenerateROCMBinaryLossKernelPass(); });
  registerPass([]() { return createGenerateROCMPolicyLossKernelPass(); });
  registerPass([]() { return createGenerateROCMFpQuantKernelPass(); });
  registerPass([]() { return createGenerateROCMDequantGemmKernelPass(); });
  registerPass([]() { return createGenerateROCMDftKernelPass(); });
  registerPass([]() { return createGenerateROCMSpmmKernelPass(); });
  registerPass([]() { return createGenerateROCMSddmmKernelPass(); });
  registerPass([]() { return createGenerateROCMSelectiveSsmKernelPass(); });
  registerPass([]() { return createGenerateROCMCholeskyKernelPass(); });
  registerPass([]() { return createGenerateROCMTriSolveKernelPass(); });
  registerPass([]() { return createGenerateROCMLuKernelPass(); });
  registerPass([]() { return createGenerateROCMQrKernelPass(); });
  registerPass([]() { return createGenerateROCMSvdKernelPass(); });
  registerPass([]() { return createGenerateROCMOptimizerKernelPass(); });
  registerPass([]() { return createGenerateROCMPredicateKernelPass(); });
  registerPass([]() { return createGenerateROCMMoeKernelPass(); });
  registerPass([]() { return createGenerateROCMAlibiKernelPass(); });
  registerPass([]() { return createGenerateROCMDeltaNetKernelPass(); });
  registerPass([]() { return createGenerateROCMRopeKernelPass(); });
  registerPass([]() { return createGenerateROCMDSparkDraftBlockKernelPass(); });
  registerPass([]() { return createGenerateROCMMLAAbsorbDecodeKernelPass(); });
  registerPass([]() { return createGenerateROCMBlockSparseAttnKernelPass(); });
  registerPass([]() { return createGenerateROCMBlockSparseTopKKernelPass(); });
  registerPass([]() { return createGenerateROCMSoftmaxKernelPass(); });
  registerPass([]() { return createGenerateROCMNormKernelPass(); });
  registerPass([]() { return createGenerateROCMReduceKernelPass(); });
  registerPass([]() { return createGenerateROCMArgReduceKernelPass(); });
  registerPass([]() { return createGenerateROCMScanKernelPass(); });
  registerPass([]() { return createGenerateROCMUnaryKernelPass(); });
  registerPass([]() { return createGenerateROCMControlForKernelPass(); });
  registerPass([]() { return createGenerateROCMControlForGemvKernelPass(); });
  registerPass([]() { return createGenerateROCMControlForNormKernelPass(); });
  registerPass([]() { return createGenerateROCMControlForWmmaKernelPass(); });
  registerPass(
      []() { return createGenerateROCMControlForWmmaTileKernelPass(); });
  registerPass([]() { return createGenerateROCMControlScanKernelPass(); });
  registerPass([]() { return createGenerateROCMControlIfNormKernelPass(); });
  registerPass([]() { return createGenerateROCMControlScanGemvKernelPass(); });
  registerPass([]() { return createGenerateROCMControlScanRnnKernelPass(); });
  registerPass([]() { return createGenerateROCMControlWhileGemvKernelPass(); });
  registerPass([]() { return createGenerateROCMSpecAcceptKernelPass(); });
  registerPass([]() { return createGenerateROCMSpecAcceptSampleKernelPass(); });
  registerPass([]() { return createGenerateROCMSpecAcceptTreeSampleKernelPass(); });
  registerPass([]() { return createGenerateROCMBinaryKernelPass(); });
  registerPass([]() { return createGenerateROCMCompareKernelPass(); });
  registerPass([]() { return createGenerateROCMLogicalKernelPass(); });
  registerPass([]() { return createGenerateROCMBitwiseKernelPass(); });
  registerPass([]() { return createGenerateROCMWhereKernelPass(); });
  registerPass([]() { return createGenerateROCMPhiloxKernelPass(); });
  registerPass([]() { return createGenerateROCMGatherKernelPass(); });
  registerPass([]() { return createGenerateROCMSortKernelPass(); });
  registerPass([]() { return createGenerateROCMScatterKernelPass(); });
  registerPass([]() { return createGenerateROCMEbmLangevinKernelPass(); });
  registerPass([]() { return createGenerateROCMEbmAffineLangevinKernelPass(); });
  registerPass([]() { return createGenerateROCMEbmPartitionKernelPass(); });
  registerPass([]() { return createGenerateROCMEbmDecodeInitKernelPass(); });
  registerPass([]() { return createGenerateROCMEbmEnergyQuadraticKernelPass(); });
  registerPass([]() { return createGenerateROCMEbmEbtTinyKernelPass(); });
  registerPass([]() { return createGenerateROCMCliffordKernelPass(); });
  registerPass([]() { return createLowerROCMAsyncCopyToLoopPass(); });
  PassPipelineRegistration<> pipeline(
      "tessera-rocm-backend",
      "Lower Tessera ROCm target IR through ABI conversion and ROCDL",
      [](OpPassManager &pm) { buildTesseraROCMBackendPipeline(pm); });
  PassPipelineRegistration<> canonicalPipeline(
      "tessera-lower-to-rocm",
      "Canonical Tessera Target IR pipeline for ROCm artifacts",
      [](OpPassManager &pm) { buildTesseraROCMBackendPipeline(pm); });
}

void registerTesseraROCMDialects(DialectRegistry &registry) {
  registry.insert<TesseraROCMDialect>();
}

void registerTesseraROCMBackendPasses() { registerTesseraROCMPasses(); }

void registerTesseraROCMBackendDialects(DialectRegistry &registry) {
  registerTesseraROCMDialects(registry);
}
}
