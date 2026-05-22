#include "tessera/ScalingPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace {
struct OptimizerShardPass : public PassWrapper<OptimizerShardPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizerShardPass)
  StringRef getArgument() const final { return "tessera-optimizer-shard"; }
  StringRef getDescription() const final { return "Propagate optimizer state sharding and insert partitioned states"; }
  void runOnOperation() override {
    // TODO: materialize state slices & annotate collectives
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tessera::sr::createOptimizerShardPass() {
  return std::make_unique<OptimizerShardPass>();
}