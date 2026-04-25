//===- NewtonAutodiff.cpp -------------------------------------------------------*- C++ -*-===//
#include "SolversPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace {
struct NewtonAutodiffPass : PassWrapper<NewtonAutodiffPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NewtonAutodiffPass)
  StringRef getArgument() const final { return "newtonautodiff"; }
  StringRef getDescription() const final { return "newtonautodiff pass (stub)"; }
  void runOnOperation() override {
    // TODO: Real implementation
    // For now, do nothing.
  }
};
} // namespace

namespace tessera { namespace passes {
std::unique_ptr<Pass> createNewtonAutodiffPass() { return std::make_unique<NewtonAutodiffPass>(); }
}} // namespace tessera::passes
