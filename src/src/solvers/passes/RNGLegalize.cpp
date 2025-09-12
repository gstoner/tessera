//===- RNGLegalize.cpp -------------------------------------------------------*- C++ -*-===//
#include "SolversPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace {
struct RNGLegalizePass : PassWrapper<RNGLegalizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RNGLegalizePass)
  StringRef getArgument() const final { return "rnglegalize"; }
  StringRef getDescription() const final { return "rnglegalize pass (stub)"; }
  void runOnOperation() override {
    // TODO: Real implementation
    // For now, do nothing.
  }
};
} // namespace

namespace tessera { namespace passes {
std::unique_ptr<Pass> createRNGLegalizePass() { return std::make_unique<RNGLegalizePass>(); }
}} // namespace tessera::passes
