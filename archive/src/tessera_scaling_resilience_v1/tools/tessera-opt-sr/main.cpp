#include "tessera/ScalingPasses.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include <iostream>

int main(int argc, char** argv) {
  mlir::MLIRContext ctx;
  mlir::tessera::sr::registerSRPasses();
  std::cout << "tessera-opt-sr (stub) ready. Use with MLIR tools to run passes.\n";
  return 0;
}