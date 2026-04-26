//===- tessera-opt (solver plugin excerpt) ------------------------------*- C++ -*-===//
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  // TODO: register Tessera core + Solver dialect here.
  return failed(mlir::MlirOptMain(argc, argv, "Tessera Opt with Solver", registry));
}
