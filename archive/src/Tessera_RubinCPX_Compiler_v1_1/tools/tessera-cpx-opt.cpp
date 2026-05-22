
#include "tessera/Target/NVRubinCPX/NVRubinCPX.h"
#include "tessera/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<tessera::target::NVRubinCPXDialect>();
  tessera::registerCPXPasses();
  tessera::registerCPXPipeline();
  return failed(mlir::MlirOptMain(argc, argv, "tessera-cpx-opt v1.1\n", registry));
}
