//===- LowerScheduleToTarget.cpp --------------------------------------------===//
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace tessera {
namespace schedule {

struct LowerScheduleToTargetPass : public PassWrapper<LowerScheduleToTargetPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerScheduleToTargetPass)
  void runOnOperation() override {
    // Map schedule.warp / schedule.pipe into async regions + barriers
    // Map policy(kind=persistent) into launch config: cta_per_sm=1 + tile queues
  }
};

std::unique_ptr<Pass> createLowerScheduleToTargetPass() {
  return std::make_unique<LowerScheduleToTargetPass>();
}

} // namespace schedule
} // namespace tessera
