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
  PassRegistration<::mlir::Pass> sparseInspector(
      "tessera-solver-sparse-inspector",
      "Inspect sparse solver structure and attach solver metadata",
      []() { return createSparseInspectorPass(); });
  PassRegistration<::mlir::Pass> sparsePrecond(
      "tessera-solver-sparse-precond",
      "Plan sparse preconditioners for solver regions",
      []() { return createSparsePrecondPass(); });
  PassRegistration<::mlir::Pass> sparseSpecialize(
      "tessera-solver-sparse-specialize",
      "Specialize sparse solver kernels after inspection",
      []() { return createSparseSolverSpecializePass(); });

  PassRegistration<::mlir::Pass> trigInit(
      "tessera-solver-trig-init",
      "Initialize trigonometric solver state",
      []() { return createTrigInitPass(); });
  PassRegistration<::mlir::Pass> newtonAutodiff(
      "tessera-solver-newton-autodiff",
      "Prepare autodiff structure for Newton-style nonlinear solves",
      []() { return createNewtonAutodiffPass(); });
  PassRegistration<::mlir::Pass> periodicHalo(
      "tessera-solver-periodic-halo",
      "Materialize periodic halo solver boundaries",
      []() { return createPeriodicHaloPass(); });

  PassRegistration<::mlir::Pass> paramBatch(
      "tessera-solver-param-batch-plan",
      "Plan parameter batches for solver sweeps",
      []() { return createParamBatchPlanPass(); });
  PassRegistration<::mlir::Pass> continuation(
      "tessera-solver-continuation-guard",
      "Insert continuation guards for solver convergence",
      []() { return createContinuationGuardPass(); });
  PassRegistration<::mlir::Pass> implicitLower(
      "tessera-solver-implicit-lower",
      "Lower implicit solver forms toward linalg-compatible IR",
      []() { return createImplicitLowerPass(); });

  PassRegistration<::mlir::Pass> rngLegalize(
      "tessera-solver-rng-legalize",
      "Legalize solver RNG operations",
      []() { return createRNGLegalizePass(); });
  PassRegistration<::mlir::Pass> rngStream(
      "tessera-solver-rng-stream-assign",
      "Assign deterministic RNG streams for solver regions",
      []() { return createRNGStreamAssignPass(); });
  PassRegistration<::mlir::Pass> rngQmc(
      "tessera-solver-rng-qmc-plan",
      "Plan quasi-Monte Carlo RNG usage for solvers",
      []() { return createRNGQMCPlanPass(); });
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
