#include "tessera/gpu/BackendRegistration.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  tessera::registerTesseraNVIDIABackendDialects(registry);
  tessera::registerTesseraNVIDIABackendPasses();
  // The upstream passes a fixture needs to take a generated gpu.func all the
  // way to an llvm.func in ONE invocation (philox_stamps_gpu_kernel_once):
  // a text round-trip through mlir-opt collapses the duplicate-attribute
  // defect that fixture exists to catch, so the lowering has to be reachable
  // from this driver, as convert-gpu-to-rocdl is from tessera-opt.
  mlir::registerConvertGpuOpsToNVVMOps();
  mlir::registerSCFToControlFlowPass();
  mlir::registerReconcileUnrealizedCastsPass();
  return failed(mlir::MlirOptMain(argc, argv, "tessera-nvidia-opt\n", registry));
}
