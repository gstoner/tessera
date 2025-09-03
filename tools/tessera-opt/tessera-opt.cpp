
#include "Tessera/Transforms/Passes.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  tessera::registerTesseraPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  return failed(mlir::MlirOptMain(argc, argv, "tessera-opt\n", registry));
}
