#include "tessera/ScalingPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace {
struct ResilienceRestartPass : public PassWrapper<ResilienceRestartPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResilienceRestartPass)
  StringRef getArgument() const final { return "tessera-resilience-restart"; }
  StringRef getDescription() const final { return "Thread restart tokens and wrap critical regions"; }
  void runOnOperation() override {
    // TODO: create async export/import ordering via tokens
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tessera::sr::createResilienceRestartPass() {
  return std::make_unique<ResilienceRestartPass>();
}