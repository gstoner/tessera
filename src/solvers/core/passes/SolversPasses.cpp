//===- SolversPasses.cpp -------------------------------------------*- C++ -*-===//
// Pass registrations + pipeline alias: -tessera-solver-suite
//===-------------------------------------------------------------------------===//
#include "SolversPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace tessera {
namespace passes {

void buildTesseraSolverCorePipeline(OpPassManager &pm) {
  // Canonical solver stack: sparse inspection/preconditioning first, nonlinear
  // structure next, parameter/continuation planning, then RNG legalization.
  pm.addPass(createSparseInspectorPass());
  pm.addPass(createSparsePrecondPass());
  pm.addPass(createSparseSolverSpecializePass());

  pm.addPass(createTrigInitPass());
  pm.addPass(createNewtonAutodiffPass());
  pm.addPass(createPeriodicHaloPass());

  pm.addPass(createParamBatchPlanPass());
  pm.addPass(createContinuationGuardPass());
  pm.addPass(createImplicitLowerPass());

  pm.addPass(createRNGLegalizePass());
  pm.addPass(createRNGStreamAssignPass());
  pm.addPass(createRNGQMCPlanPass());
}

void registerTesseraSolverPasses() {
  registerPass([]() { return createSparseInspectorPass(); });
  registerPass([]() { return createSparsePrecondPass(); });
  registerPass([]() { return createSparseSolverSpecializePass(); });

  registerPass([]() { return createTrigInitPass(); });
  registerPass([]() { return createNewtonAutodiffPass(); });
  registerPass([]() { return createPeriodicHaloPass(); });

  registerPass([]() { return createParamBatchPlanPass(); });
  registerPass([]() { return createContinuationGuardPass(); });
  registerPass([]() { return createImplicitLowerPass(); });

  registerPass([]() { return createRNGLegalizePass(); });
  registerPass([]() { return createRNGStreamAssignPass(); });
  registerPass([]() { return createRNGQMCPlanPass(); });
}

void registerTesseraSolversPipeline() {
  registerTesseraSolverPasses();
  PassPipelineRegistration<> pipeline(
      "tessera-solver-suite",
      "Canonical Tessera solver stack over linalg-compatible solver IR",
      [](OpPassManager &pm) { buildTesseraSolverCorePipeline(pm); });
}

} // namespace passes
} // namespace tessera
