#include "tessera/gpu/BackendRegistration.h"

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
  return failed(mlir::MlirOptMain(argc, argv, "tessera-nvidia-opt\n", registry));
}
