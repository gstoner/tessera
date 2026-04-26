#include "tessera/tpu/TesseraTPUPasses.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  tessera::registerTesseraTPUPasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "tessera-tpu-opt\n", registry));
}
