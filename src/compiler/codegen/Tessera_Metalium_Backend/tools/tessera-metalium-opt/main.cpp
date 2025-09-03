#include "mlir/Support/MlirOptMain.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

#include "Tessera/Target/Metalium/Passes.h"
#include "Tessera/Target/Metalium/TesseraMetaliumDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);  // standard MLIR dialects
  mlir::tessera::metalium::registerMetaliumPasses(registry);
  registry.insert<mlir::tessera::metalium::TesseraMetaliumDialect>();

  mlir::registerAllPasses();            // standard MLIR passes
  // Our pipeline "tessera-metalium" is registered in Passes.cpp

  return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "tessera-metalium-opt\n", registry)
  );
}
