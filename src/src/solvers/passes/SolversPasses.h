//===- SolversPasses.h -----------------------------------------------*- C++ -*-===//
// Minimal pass declarations & pipeline alias for Tessera solvers.
// Integrate with tessera-opt by loading the plugin or linking the lib.
//===-------------------------------------------------------------------------===//
#pragma once
#include "mlir/Pass/Pass.h"

namespace tessera {
namespace passes {

std::unique_ptr<mlir::Pass> createSparseInspectorPass();
std::unique_ptr<mlir::Pass> createSparsePrecondPass();
std::unique_ptr<mlir::Pass> createSparseSolverSpecializePass();

std::unique_ptr<mlir::Pass> createRNGLegalizePass();
std::unique_ptr<mlir::Pass> createRNGStreamAssignPass();
std::unique_ptr<mlir::Pass> createRNGQMCPlanPass();

std::unique_ptr<mlir::Pass> createTrigInitPass();
std::unique_ptr<mlir::Pass> createNewtonAutodiffPass();
std::unique_ptr<mlir::Pass> createPeriodicHaloPass();

std::unique_ptr<mlir::Pass> createParamBatchPlanPass();
std::unique_ptr<mlir::Pass> createContinuationGuardPass();
std::unique_ptr<mlir::Pass> createImplicitLowerPass();

void registerTesseraSolversPipeline();

} // namespace passes
} // namespace tessera
