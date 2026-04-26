//===- IterativeRefinement.cpp - IR wrapper pass ------------------------*- C++ -*-===//
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace tessera { namespace solver {

struct IterativeRefinementPass : public mlir::PassWrapper<IterativeRefinementPass, mlir::OperationPass<mlir::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IterativeRefinementPass)
  void runOnOperation() override {
    // Wrap a solve region with residual compute and loop (maxIters, tol), using fp32 residuals.
    // TODO: Materialize a small canonical loop IR around tessera.solver.* ops.
  }
};

std::unique_ptr<mlir::Pass> createIterativeRefinementPass() {
  return std::make_unique<IterativeRefinementPass>();
}

}} // namespace tessera::solver
