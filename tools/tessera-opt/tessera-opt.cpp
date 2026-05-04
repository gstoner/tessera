
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

#ifdef TESSERA_HAVE_APPLE_BACKEND
#include "Tessera/Target/Apple/Passes.h"
#include "Tessera/Target/Apple/TesseraAppleDialect.h"
#endif

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

#ifdef TESSERA_HAVE_APPLE_BACKEND
  // Phase 8: Apple Silicon Target IR pipelines
  // (tessera-lower-to-apple_cpu, tessera-lower-to-apple_gpu).
  tessera::apple::registerTesseraAppleBackendPipelines();
#endif

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  tessera::registerTesseraDialects(registry);
  tessera::solver::registerTesseraLinalgSolverDialect(registry);
  tessera::neighbors::registerNeighborsDialect(registry);

#ifdef TESSERA_HAVE_APPLE_BACKEND
  tessera::apple::registerTesseraAppleBackendDialects(registry);
#endif

  return failed(mlir::MlirOptMain(argc, argv, "tessera-opt\n", registry));
}
