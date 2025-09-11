
#include "tessera/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  // TODO: registry.insert<tessera::target::NVRubinCPXDialect>();
  tessera::registerCPXPassPipelines();

  return failed(MlirOptMain(argc, argv, "tessera-cpx-opt driver\n", registry));
}
