//===- tessera-opt (solver plugin excerpt) ------------------------------*- C++ -*-===//
#include "SolversPasses.h"
#include "tessera/Dialect/Solver/SolverDialect.h"
#include "tessera/Solvers/LinalgPasses.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  tessera::passes::registerTesseraSolversPipeline();
  tessera::solver::registerTesseraLinalgSolverPipeline();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  tessera::solver::registerTesseraLinalgSolverDialect(registry);
  return failed(mlir::MlirOptMain(argc, argv, "Tessera Opt with Solver", registry));
}
