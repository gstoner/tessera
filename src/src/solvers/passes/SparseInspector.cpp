//===- SparseInspector.cpp -------------------------------------------------------*- C++ -*-===//
#include "SolversPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace {
struct SparseInspectorPass : PassWrapper<SparseInspectorPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SparseInspectorPass)
  StringRef getArgument() const final { return "sparseinspector"; }
  StringRef getDescription() const final { return "sparseinspector pass (stub)"; }
  void runOnOperation() override {
    // TODO: Real implementation
    // For now, do nothing.
  }
};
} // namespace

namespace tessera { namespace passes {
std::unique_ptr<Pass> createSparseInspectorPass() { return std::make_unique<SparseInspectorPass>(); }
}} // namespace tessera::passes
