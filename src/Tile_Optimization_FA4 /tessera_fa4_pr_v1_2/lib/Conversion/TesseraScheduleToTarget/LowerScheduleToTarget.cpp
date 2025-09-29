//===- LowerScheduleToTarget.cpp (v1.2) ------------------------------------===//
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace tessera { namespace schedule {

struct LowerScheduleToTargetPass : public PassWrapper<LowerScheduleToTargetPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerScheduleToTargetPass)
  void runOnOperation() override {
    ModuleOp m = getOperation();
    // Pseudocode:
    // - For each schedule region:
    //   * Build async.execute regions per warp role
    //   * Insert async.await/barriers according to pipe edges
    //   * If policy=persistent: attach launch metadata "ctas_per_sm=1"
    //   * Create a tile-queue SSA value (opaque) threaded as async.token deps
  }
};

std::unique_ptr<Pass> createLowerScheduleToTargetPass() {
  return std::make_unique<LowerScheduleToTargetPass>();
}

}} // ns
