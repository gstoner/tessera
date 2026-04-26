//===- SolversPasses.cpp -------------------------------------------*- C++ -*-===//
// Pass registrations + pipeline alias: -tessera-solver-suite
//===-------------------------------------------------------------------------===//
#include "SolversPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace tessera {
namespace passes {

// --- Sparse ---
std::unique_ptr<Pass> createSparseInspectorPass() { return std::make_unique<Pass>(); }
std::unique_ptr<Pass> createSparsePrecondPass()   { return std::make_unique<Pass>(); }
std::unique_ptr<Pass> createSparseSolverSpecializePass() { return std::make_unique<Pass>(); }

// --- RNG ---
std::unique_ptr<Pass> createRNGLegalizePass()     { return std::make_unique<Pass>(); }
std::unique_ptr<Pass> createRNGStreamAssignPass() { return std::make_unique<Pass>(); }
std::unique_ptr<Pass> createRNGQMCPlanPass()      { return std::make_unique<Pass>(); }

// --- Trig / Nonlinear ---
std::unique_ptr<Pass> createTrigInitPass()        { return std::make_unique<Pass>(); }
std::unique_ptr<Pass> createNewtonAutodiffPass()  { return std::make_unique<Pass>(); }
std::unique_ptr<Pass> createPeriodicHaloPass()    { return std::make_unique<Pass>(); }

// --- Parametric ---
std::unique_ptr<Pass> createParamBatchPlanPass()  { return std::make_unique<Pass>(); }
std::unique_ptr<Pass> createContinuationGuardPass(){ return std::make_unique<Pass>(); }
std::unique_ptr<Pass> createImplicitLowerPass()   { return std::make_unique<Pass>(); }

void registerTesseraSolversPipeline() {
  PassPipelineRegistration<> pipeline(
    "tessera-solver-suite",
    "Sparse + Nonlinear + RNG legalization & specialization",
    [](OpPassManager &pm) {
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
    });
}

} // namespace passes
} // namespace tessera
