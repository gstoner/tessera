//===- RNGStreamAssign.cpp -------------------------------------------------------*- C++ -*-===//
#include "SolversPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace {
struct RNGStreamAssignPass : PassWrapper<RNGStreamAssignPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RNGStreamAssignPass)
  StringRef getArgument() const final { return "rngstreamassign"; }
  StringRef getDescription() const final { return "rngstreamassign pass (stub)"; }
  void runOnOperation() override {
    // TODO: Real implementation
    // For now, do nothing.
  }
};
} // namespace

namespace tessera { namespace passes {
std::unique_ptr<Pass> createRNGStreamAssignPass() { return std::make_unique<RNGStreamAssignPass>(); }
}} // namespace tessera::passes
