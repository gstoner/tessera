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
  pm.addPass(createLowerTileToROCMPass());
  pm.addPass(createLowerKernelABIPass());
  pm.addPass(createLowerTesseraTargetToROCDLPass());
}

void registerTesseraROCMPasses() {
  registerPass([]() { return createLowerTileToROCMPass(); });
  registerPass([]() { return createLowerKernelABIPass(); });
  registerPass([]() { return createLowerTesseraTargetToROCDLPass(); });
  registerPass([]() { return createGenerateWMMAGemmKernelPass(); });
  registerPass([]() { return createGenerateWMMAFlashAttnKernelPass(); });
  registerPass([]() { return createGenerateWMMAFlashAttnBwdKernelPass(); });
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
