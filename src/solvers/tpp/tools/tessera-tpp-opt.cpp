//===- tessera-tpp-opt.cpp - standalone mlir-opt driver for TPP ----------===//
//
// Registers the TPP dialect + all space-time passes + the `-tpp-space-time`
// pipeline alias, plus the upstream dialects the test IR uses (func, arith,
// tensor), then hands off to MlirOptMain.  This is the driver the lit tests
// under test/TPP/ run against when the monorepo `tessera-opt` is not built.
//
//===----------------------------------------------------------------------===//

#include "tpp/InitTPP.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<func::FuncDialect, arith::ArithDialect,
                  tensor::TensorDialect>();
  tessera::tpp::registerAllTPP(registry); // dialect + passes + pipelines

  return failed(
      MlirOptMain(argc, argv, "tessera-tpp-opt (TPP space-time)\n", registry));
}
