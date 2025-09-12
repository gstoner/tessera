//===- SparseSolverSpecialize.cpp -------------------------------------------------------*- C++ -*-===//
#include "SolversPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace {
struct SparseSolverSpecializePass : PassWrapper<SparseSolverSpecializePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseSolverSpecializePass)
  StringRef getArgument() const final { return "sparsesolverspecialize"; }
  StringRef getDescription() const final { return "sparsesolverspecialize pass (stub)"; }
  void runOnOperation() override {
    // TODO: Real implementation
    // For now, do nothing.
  }
};
} // namespace

namespace tessera { namespace passes {
std::unique_ptr<Pass> createSparseSolverSpecializePass() { return std::make_unique<SparseSolverSpecializePass>(); }
}} // namespace tessera::passes
