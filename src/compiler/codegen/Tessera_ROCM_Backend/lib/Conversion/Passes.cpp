#include "TesseraROCM/Passes.h"
#include "TesseraROCM/IR/TesseraROCMDialect.h.inc"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinOps.h"
using namespace mlir;

namespace mlir::tessera_rocm {
std::unique_ptr<Pass> createLowerTesseraToROCDLImpl();

std::unique_ptr<Pass> createLowerTesseraTargetToROCDLPass() {
  return createLowerTesseraToROCDLImpl();
}

void buildTesseraROCMBackendPipeline(OpPassManager &pm) {
  pm.addPass(createLowerKernelABIPass());
  pm.addPass(createLowerTesseraTargetToROCDLPass());
}

void registerTesseraROCMPasses() {
  PassRegistration<Pass>(
      "lower-tessera-kernel-abi",
      "Lower Tessera ROCm kernel signatures to the AMDGPU ABI",
      []() { return createLowerKernelABIPass(); });
  PassRegistration<Pass>(
      "lower-tessera-target-to-rocdl",
      "Lower Tessera ROCm target ops to LLVM/ROCDL",
      []() { return createLowerTesseraTargetToROCDLPass(); });
  PassPipelineRegistration<> pipeline(
      "tessera-rocm-backend",
      "Lower Tessera ROCm target IR through ABI conversion and ROCDL",
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
