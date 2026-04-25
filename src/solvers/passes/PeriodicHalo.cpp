//===- PeriodicHalo.cpp -------------------------------------------------------*- C++ -*-===//
#include "SolversPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace {
struct PeriodicHaloPass : PassWrapper<PeriodicHaloPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PeriodicHaloPass)
  StringRef getArgument() const final { return "periodichalo"; }
  StringRef getDescription() const final { return "periodichalo pass (stub)"; }
  void runOnOperation() override {
    // TODO: Real implementation
    // For now, do nothing.
  }
};
} // namespace

namespace tessera { namespace passes {
std::unique_ptr<Pass> createPeriodicHaloPass() { return std::make_unique<PeriodicHaloPass>(); }
}} // namespace tessera::passes
