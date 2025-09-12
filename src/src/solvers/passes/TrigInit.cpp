//===- TrigInit.cpp -------------------------------------------------------*- C++ -*-===//
#include "SolversPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace {
struct TrigInitPass : PassWrapper<TrigInitPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TrigInitPass)
  StringRef getArgument() const final { return "triginit"; }
  StringRef getDescription() const final { return "triginit pass (stub)"; }
  void runOnOperation() override {
    // TODO: Real implementation
    // For now, do nothing.
  }
};
} // namespace

namespace tessera { namespace passes {
std::unique_ptr<Pass> createTrigInitPass() { return std::make_unique<TrigInitPass>(); }
}} // namespace tessera::passes
