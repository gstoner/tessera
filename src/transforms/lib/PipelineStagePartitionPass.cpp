// PipelineStagePartitionPass.cpp — 2026-06-23
//
// True pipeline-stage partitioning. Until now PipelineStageInsertionPass only
// *consumed* a `tessera.pp_stage` / `tessera.layer={stage=k}` assignment that
// something else had to provide ("a true cost-model-driven program-partitioning
// pass — deferred to Phase 5"). This pass closes that gap: it assigns each op of
// a function body to one of `num_stages` pipeline stages with a cost-balanced,
// program-order-monotonic partition, then PipelineStageInsertionPass inserts the
// send/recv SSA rewrites and PipelineScheduleLegalityPass proves the 1F1B
// schedule is well-formed.
//
//   --tessera-pipeline-partition  (--num-stages N overrides the module attr)
//
// Cost model: heavy ops (matmul / gemm / batched_gemm / flash_attn /
// conv2d_nhwc) weigh 4, everything else 1. Stages are contiguous in program
// order, so the assignment is monotonic non-decreasing — defs precede uses, so
// no value flows backward across a stage boundary (a hard requirement for a
// legal forward pipeline). Greedy on the prefix cost: stage(op_i) =
// min(p-1, floor(prefixCostBefore_i / (totalCost / p))).
//
// Ops already carrying a stage are left untouched (respect an upstream
// assignment); a function with no partitionable ops is skipped. Empty-stage /
// too-few-ops conditions are reported by the legality gate, not here (this pass
// is a pure transform).

#include "Tessera/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

static int64_t readPlanNumStages(ModuleOp m, int64_t defaultVal) {
  if (auto plan = m->getAttrOfType<DictionaryAttr>("tessera.pipeline_plan"))
    if (auto v = plan.getAs<IntegerAttr>("num_stages"))
      return v.getInt();
  return defaultVal;
}

// Per-op cost — a small cost model so heavy compute dominates the balance.
static int64_t opCost(Operation *op) {
  StringRef n = op->getName().getStringRef();
  if (n == "tessera.matmul" || n == "tessera.gemm" ||
      n == "tessera.batched_gemm" || n == "tessera.flash_attn" ||
      n == "tessera.conv2d_nhwc")
    return 4;
  return 1;
}

static bool hasStage(Operation *op) {
  if (auto layer = op->getAttrOfType<DictionaryAttr>("tessera.layer"))
    if (layer.getAs<IntegerAttr>("stage"))
      return true;
  return (bool)op->getAttrOfType<IntegerAttr>("tessera.pp_stage");
}

struct PipelineStagePartitionPass
    : public PassWrapper<PipelineStagePartitionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PipelineStagePartitionPass)

  PipelineStagePartitionPass() = default;
  PipelineStagePartitionPass(const PipelineStagePartitionPass &o)
      : PassWrapper(o) {}

  Option<int> numStagesOpt{
      *this, "num-stages",
      llvm::cl::desc("Pipeline stage count (overrides module attr)"),
      llvm::cl::init(0)};

  StringRef getArgument() const override {
    return "tessera-pipeline-partition";
  }
  StringRef getDescription() const override {
    return "Cost-balanced, program-order-monotonic partition of each function "
           "into num_stages pipeline stages (emits tessera.pp_stage).";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder b(module.getContext());
    int64_t numStages =
        numStagesOpt > 0 ? numStagesOpt : readPlanNumStages(module, 1);
    if (numStages <= 1)
      return; // single stage — nothing to partition.

    module.walk([&](func::FuncOp func) {
      if (func.getBody().empty())
        return;
      Block &block = func.getBody().front();

      // Partitionable ops = body ops in program order, minus the terminator.
      // Skip the whole function if any op already carries a stage (respect an
      // upstream assignment).
      SmallVector<Operation *> ops;
      for (Operation &op : block) {
        if (op.hasTrait<OpTrait::IsTerminator>())
          continue;
        if (hasStage(&op))
          return; // already partitioned upstream.
        ops.push_back(&op);
      }
      if (ops.empty())
        return;

      int64_t totalCost = 0;
      for (Operation *op : ops)
        totalCost += opCost(op);
      // target = totalCost / numStages, computed in fixed point to avoid floats.
      // stage(op_i) = min(p-1, (prefixBefore_i * p) / totalCost).
      int64_t prefix = 0;
      for (Operation *op : ops) {
        int64_t stage = (prefix * numStages) / totalCost;
        if (stage >= numStages)
          stage = numStages - 1;
        op->setAttr("tessera.pp_stage", b.getI64IntegerAttr(stage));
        prefix += opCost(op);
      }
    });
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createPipelineStagePartitionPass() {
  return std::make_unique<PipelineStagePartitionPass>();
}
} // namespace tessera
