//===- ContinuationGuard.cpp -------------------------------------------------------*- C++ -*-===//
#include "SolversPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace {
struct ContinuationGuardPass : PassWrapper<ContinuationGuardPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ContinuationGuardPass)
  StringRef getArgument() const final { return "continuationguard"; }
  StringRef getDescription() const final { return "continuationguard pass (stub)"; }
  void runOnOperation() override {
    // TODO: Real implementation
    // For now, do nothing.
  }
};
} // namespace

namespace tessera { namespace passes {
std::unique_ptr<Pass> createContinuationGuardPass() { return std::make_unique<ContinuationGuardPass>(); }
}} // namespace tessera::passes
