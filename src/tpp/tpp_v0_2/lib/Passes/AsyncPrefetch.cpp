
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace {
struct AsyncPrefetch : public PassWrapper<AsyncPrefetch, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "tpp-async-prefetch"; }
  StringRef getDescription() const final { return "Insert async prefetch operations (stub)"; }
  void runOnOperation() final {}
};
}
std::unique_ptr<Pass> createAsyncPrefetchPass(){ return std::make_unique<AsyncPrefetch>(); }
