//===- LowerScheduleToTarget.cpp (v1.1) ------------------------------------===//
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace tessera { namespace schedule {

struct LowerScheduleToTargetPass : public PassWrapper<LowerScheduleToTargetPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerScheduleToTargetPass)
  void runOnOperation() override {
    // Map warp/pipe/policy to async regions + barriers; persistent -> 1 CTA/SM.
  }
};

std::unique_ptr<Pass> createLowerScheduleToTargetPass() { return std::make_unique<LowerScheduleToTargetPass>(); }

}} // ns
