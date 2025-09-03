#include "TesseraROCM/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"
int main(int argc, char **argv){
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  mlir::tessera_rocm::registerTesseraROCMPasses();
  return failed(mlir::MlirOptMain(argc, argv, "tessera-rocm-opt\n", registry));
}
