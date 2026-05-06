
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
using namespace mlir;
namespace {
struct FuseStencilTime : public PassWrapper<FuseStencilTime, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "tpp-fuse-stencil-time"; }
  StringRef getDescription() const final { return "Fuse stencil ops within time.step regions"; }
  void runOnOperation() final {}
};
}
std::unique_ptr<Pass> createFuseStencilTimePass(){ return std::make_unique<FuseStencilTime>(); }
