#include "TesseraROCM/Passes.h"
#include "Tessera/Dialect/Tile/TileDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
int main(int argc, char **argv){
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::ROCDL::ROCDLDialect,
                  tessera::tile::TesseraTileDialect>();
  mlir::tessera_rocm::registerTesseraROCMDialects(registry);
  mlir::tessera_rocm::registerTesseraROCMPasses();
  return failed(mlir::MlirOptMain(argc, argv, "tessera-rocm-opt\n", registry));
}
