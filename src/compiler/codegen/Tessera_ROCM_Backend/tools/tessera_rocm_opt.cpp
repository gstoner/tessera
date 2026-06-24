#include "TesseraROCM/Passes.h"
#include "Tessera/Dialect/Tile/TileDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
int main(int argc, char **argv){
  mlir::DialectRegistry registry;
  // scf is registered so the wave/LDS pipeline + legality can be exercised on
  // software-pipelined loop bodies (scf.for) — that is the transform's real
  // setting; without it scf.for cannot even be parsed in custom form.
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::ROCDL::ROCDLDialect,
                  mlir::scf::SCFDialect,
                  tessera::tile::TesseraTileDialect>();
  mlir::tessera_rocm::registerTesseraROCMDialects(registry);
  mlir::tessera_rocm::registerTesseraROCMPasses();
  return failed(mlir::MlirOptMain(argc, argv, "tessera-rocm-opt\n", registry));
}
