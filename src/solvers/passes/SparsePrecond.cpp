//===- SparsePrecond.cpp -------------------------------------------------------*- C++ -*-===//
#include "SolversPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace {
struct SparsePrecondPass : PassWrapper<SparsePrecondPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparsePrecondPass)
  StringRef getArgument() const final { return "sparseprecond"; }
  StringRef getDescription() const final { return "sparseprecond pass (stub)"; }
  void runOnOperation() override {
    // TODO: Real implementation
    // For now, do nothing.
  }
};
} // namespace

namespace tessera { namespace passes {
std::unique_ptr<Pass> createSparsePrecondPass() { return std::make_unique<SparsePrecondPass>(); }
}} // namespace tessera::passes
