
#include "Tessera/IR/Dialects.h"
#include "Tessera/Transforms/Passes.h"
#include "SolversPasses.h"
#include "tessera/Dialect/Solver/SolverDialect.h"
#include "tessera/Solvers/LinalgPasses.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  tessera::registerTesseraPasses();
  tessera::passes::registerTesseraSolversPipeline();
  tessera::solver::registerTesseraLinalgSolverPipeline();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  tessera::registerTesseraDialects(registry);
  tessera::solver::registerTesseraLinalgSolverDialect(registry);

  return failed(mlir::MlirOptMain(argc, argv, "tessera-opt\n", registry));
}
