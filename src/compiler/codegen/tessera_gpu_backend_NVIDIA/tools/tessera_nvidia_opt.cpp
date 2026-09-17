#include "tessera/gpu/BackendRegistration.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
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
  // convert-gpu-to-nvvm lowers the dialects inside the kernel through the
  // ConvertToLLVMPatternInterface each dialect *promises*; without these
  // extensions the assertions-ON driver aborts with "checking for an
  // interface ... promised by dialect 'arith' but never implemented".
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  return failed(mlir::MlirOptMain(argc, argv, "tessera-nvidia-opt\n", registry));
}
