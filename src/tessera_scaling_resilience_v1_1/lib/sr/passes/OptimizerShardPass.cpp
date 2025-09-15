#include "tessera/sr/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct OptimizerShardPass : public PassWrapper<OptimizerShardPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizerShardPass)
  StringRef getArgument() const final { return "tessera-optimizer-shard"; }
  StringRef getDescription() const final { return "Propagate optimizer state sharding annotations"; }
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([](Operation *op){
      if (op->getName().stripDialect() == "optimizer.shard") {
        op->setAttr("sr.sharded", UnitAttr::get(op->getContext()));
      }
    });
  }
};
}

std::unique_ptr<Pass> mlir::tessera::sr::createOptimizerShardPass() {
  return std::make_unique<OptimizerShardPass>();
}