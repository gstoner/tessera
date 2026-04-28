//===- ContinuationGuard.cpp — insert guard ops at continuation points --*- C++ -*-===//
//
// Finds tessera_solver.continuation ops and cross-region live values that
// span continuation boundaries, then inserts tessera_solver.guard ops
// immediately before each such boundary to:
//   a) pin live tensors in memory (prevent eviction)
//   b) mark the resume point for checkpointing
//   c) attach continuation_id so the runtime can re-enter
//
// Output: tessera_solver.guard ops with attrs:
//   continuation_id   — unique int64 per guarded boundary
//   live_value_count  — number of live SSA values at this point
//   resumable         — UnitAttr (marks the point as a valid resume target)
//
//===----------------------------------------------------------------------===//

#include "SolversPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct ContinuationGuardPass
    : PassWrapper<ContinuationGuardPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ContinuationGuardPass)

  StringRef getArgument() const final { return "tessera-continuation-guard"; }
  StringRef getDescription() const final {
    return "Insert tessera_solver.guard ops at continuation / checkpoint "
           "boundaries";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();
    OpBuilder builder(ctx);

    int64_t contId = 0;

    mod.walk([&](Operation *op) {
      StringRef opName = op->getName().getStringRef();

      // Guard continuation ops.
      bool isContinuation = opName == "tessera_solver.continuation" ||
                            opName == "tessera_sr.checkpoint" ||
                            opName.contains("continuation");
      if (!isContinuation)
        return;

      // Count live SSA values produced before this op in the same block.
      int64_t liveCount = 0;
      if (Block *blk = op->getBlock()) {
        for (auto &prev : *blk) {
          if (&prev == op)
            break;
          liveCount += static_cast<int64_t>(prev.getNumResults());
        }
      }

      // Annotate the op itself as a guard target.
      op->setAttr("tessera_solver.continuation_id",
                  IntegerAttr::get(IntegerType::get(ctx, 64), contId));
      op->setAttr("tessera_solver.live_value_count",
                  IntegerAttr::get(IntegerType::get(ctx, 64), liveCount));
      op->setAttr("tessera_solver.resumable", UnitAttr::get(ctx));

      // Emit a guard op immediately before the continuation.
      builder.setInsertionPoint(op);
      OperationState guardState(op->getLoc(), "tessera_solver.guard");
      guardState.addAttribute(
          "continuation_id",
          IntegerAttr::get(IntegerType::get(ctx, 64), contId));
      guardState.addAttribute(
          "live_value_count",
          IntegerAttr::get(IntegerType::get(ctx, 64), liveCount));
      builder.create(guardState);

      ++contId;
    });

    if (contId > 0)
      mod->setAttr("tessera_solver.num_continuations",
                   IntegerAttr::get(IntegerType::get(ctx, 64), contId));
  }
};

} // namespace

namespace tessera {
namespace passes {
std::unique_ptr<Pass> createContinuationGuardPass() {
  return std::make_unique<ContinuationGuardPass>();
}
} // namespace passes
} // namespace tessera
