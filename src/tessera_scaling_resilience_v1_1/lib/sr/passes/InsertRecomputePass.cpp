#include "tessera/sr/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct InsertRecomputePass : public PassWrapper<InsertRecomputePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertRecomputePass)
  StringRef getArgument() const final { return "tessera-insert-recompute"; }
  StringRef getDescription() const final { return "Insert recomputation barriers around tessera_sr.checkpoint regions"; }
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    // Walk and tag: in a full impl we'd clone bodies and place save/restore.
    mod.walk([](Operation *op){
      if (op->getName().getStringRef() == "tessera_sr.checkpoint") {
        op->setAttr("sr.instrumented", UnitAttr::get(op->getContext()));
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tessera::sr::createInsertRecomputePass() {
  return std::make_unique<InsertRecomputePass>();
}