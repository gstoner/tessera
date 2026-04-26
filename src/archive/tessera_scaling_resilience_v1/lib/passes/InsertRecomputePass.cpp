#include "tessera/ScalingPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace {
struct InsertRecomputePass : public PassWrapper<InsertRecomputePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertRecomputePass)
  StringRef getArgument() const final { return "tessera-insert-recompute"; }
  StringRef getDescription() const final { return "Insert recomputation around checkpoint regions"; }
  void runOnOperation() override {
    // TODO: scan for tessera_sr.checkpoint and add save/restore barriers
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tessera::sr::createInsertRecomputePass() {
  return std::make_unique<InsertRecomputePass>();
}