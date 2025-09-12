//===- RNGQMCPlan.cpp -------------------------------------------------------*- C++ -*-===//
#include "SolversPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace {
struct RNGQMCPlanPass : PassWrapper<RNGQMCPlanPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RNGQMCPlanPass)
  StringRef getArgument() const final { return "rngqmcplan"; }
  StringRef getDescription() const final { return "rngqmcplan pass (stub)"; }
  void runOnOperation() override {
    // TODO: Real implementation
    // For now, do nothing.
  }
};
} // namespace

namespace tessera { namespace passes {
std::unique_ptr<Pass> createRNGQMCPlanPass() { return std::make_unique<RNGQMCPlanPass>(); }
}} // namespace tessera::passes
