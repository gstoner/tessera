
#include "Tessera/IR/Dialects.h"
#include "Tessera/Transforms/Passes.h"
#include "SolversPasses.h"
#include "tessera/Dialect/Solver/SolverDialect.h"
#include "tessera/Solvers/LinalgPasses.h"
#include "tessera/Dialect/Neighbors/IR/NeighborsDialect.h"
#include "tessera/Dialect/Neighbors/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  tessera::registerTesseraPasses();
  tessera::passes::registerTesseraSolversPipeline();
  tessera::solver::registerTesseraLinalgSolverPipeline();

  // Phase 7: Neighbors dialect passes (halo infer, stencil lower,
  // pipeline overlap, dynamic topology).
  tessera::neighbors::registerHaloInferPass();
  tessera::neighbors::registerStencilLowerPass();
  tessera::neighbors::registerPipelineOverlapPass();
  tessera::neighbors::registerDynamicTopologyPass();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  tessera::registerTesseraDialects(registry);
  tessera::solver::registerTesseraLinalgSolverDialect(registry);
  tessera::neighbors::registerNeighborsDialect(registry);

  return failed(mlir::MlirOptMain(argc, argv, "tessera-opt\n", registry));
}
