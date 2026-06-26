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
  registerPass([]() { return createGenerateROCMAlibiKernelPass(); });
  registerPass([]() { return createGenerateROCMDeltaNetKernelPass(); });
  registerPass([]() { return createGenerateROCMRopeKernelPass(); });
  registerPass([]() { return createGenerateROCMSoftmaxKernelPass(); });
  registerPass([]() { return createGenerateROCMNormKernelPass(); });
  registerPass([]() { return createGenerateROCMReduceKernelPass(); });
  registerPass([]() { return createGenerateROCMUnaryKernelPass(); });
  registerPass([]() { return createGenerateROCMBinaryKernelPass(); });
  registerPass([]() { return createGenerateROCMCompareKernelPass(); });
  registerPass([]() { return createGenerateROCMLogicalKernelPass(); });
  registerPass([]() { return createGenerateROCMBitwiseKernelPass(); });
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
