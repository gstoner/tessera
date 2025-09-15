#include "tessera/sr/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct ResilienceRestartPass : public PassWrapper<ResilienceRestartPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResilienceRestartPass)
  StringRef getArgument() const final { return "tessera-resilience-restart"; }
  StringRef getDescription() const final { return "Thread restart tokens across critical regions"; }
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([](Operation *op){
      if (op->getName().getStringRef() == "tessera_sr.resilience_region") {
        op->setAttr("sr.restart_token", StringAttr::get(op->getContext(), "t0"));
      }
    });
  }
};
}

std::unique_ptr<Pass> mlir::tessera::sr::createResilienceRestartPass() {
  return std::make_unique<ResilienceRestartPass>();
}