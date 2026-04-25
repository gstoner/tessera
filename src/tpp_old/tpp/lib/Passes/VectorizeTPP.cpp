
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace {
struct VectorizeTPP : public PassWrapper<VectorizeTPP, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "tpp-vectorize"; }
  StringRef getDescription() const final { return "Vectorize/Tile mapping"; }
  void runOnOperation() final {
    // TODO: implement
  }
};
} // namespace

std::unique_ptr<Pass> createVectorizeTPPPass() { return std::make_unique<VectorizeTPP>(); }
