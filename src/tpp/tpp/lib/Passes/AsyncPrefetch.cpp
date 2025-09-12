
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace {
struct AsyncPrefetch : public PassWrapper<AsyncPrefetch, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "tpp-async-prefetch"; }
  StringRef getDescription() const final { return "Insert async prefetch operations"; }
  void runOnOperation() final {
    // TODO: implement
  }
};
} // namespace

std::unique_ptr<Pass> createAsyncPrefetchPass() { return std::make_unique<AsyncPrefetch>(); }
