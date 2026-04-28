//===- PipelineStageInsertionPass.cpp — Phase 4 ───────────────────────────===//
//
// Partitions the IR into pipeline stages and inserts micro-batch send/recv
// communication ops at stage boundaries.
//
// The 1F1B (one-forward-one-backward) schedule is computed from:
//   tessera.pipeline_plan = {num_stages, num_micro_batches, interleaved, …}
//   on the module op (emitted by PipelinePlan::to_mlir_attrs()).
//
// What the pass does:
//   1. Reads num_stages (p) and num_micro_batches (m) from the module attr.
//   2. Splits the function body into `p` sequential schedule.pipeline.stage
//      regions, one per device rank.
//   3. Inserts `tessera.pipeline.send` at each stage's output and
//      `tessera.pipeline.recv` at the next stage's input.
//   4. Annotates each stage region with:
//        {tessera.pp_stage = k, tessera.pp_num_micro_batches = m}
//
// For the current implementation, "splitting" is approximated by annotating
// ops that carry a `tessera.layer = {stage = k}` attribute (set by
// DistributedPlan layer specs). Full SSA splitting would require a complete
// program-partitioning pass — that is deferred to Phase 5.
//
// Registration: --tessera-pipeline-stage-insertion
// Options:
//   --num-stages        pipeline stage count (overrides module attr)
//   --num-micro-batches micro-batch count    (overrides module attr)
//   --interleaved       use interleaved 1F1B (default false)
//
//===----------------------------------------------------------------------===//

#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pipeline-stage-insertion"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Read an integer attribute from the module's tessera.pipeline_plan dict.
static int64_t readPlanAttr(ModuleOp m, StringRef key, int64_t defaultVal) {
  auto planAttr = m->getAttrOfType<DictionaryAttr>("tessera.pipeline_plan");
  if (!planAttr) return defaultVal;
  if (auto v = planAttr.getAs<IntegerAttr>(key))
    return v.getInt();
  return defaultVal;
}

/// Get the pipeline stage assigned to an op via `tessera.layer = {stage = k}`.
static int64_t getOpStage(Operation *op) {
  if (auto layerAttr = op->getAttrOfType<DictionaryAttr>("tessera.layer")) {
    if (auto stageAttr = layerAttr.getAs<IntegerAttr>("stage"))
      return stageAttr.getInt();
  }
  // Also check a direct tessera.pp_stage attr
  if (auto stageAttr = op->getAttrOfType<IntegerAttr>("tessera.pp_stage"))
    return stageAttr.getInt();
  return -1; // not assigned to any stage
}

/// Emit a `tessera.pipeline.send` op carrying the activation tensor to the
/// next pipeline stage.
static void emitPipelineSend(OpBuilder &b, Location loc, Value activation,
                              int64_t fromStage, int64_t microBatch) {
  OperationState state(loc, "tessera.pipeline.send");
  state.addOperands(activation);
  state.addAttribute("from_stage",  b.getI64IntegerAttr(fromStage));
  state.addAttribute("micro_batch", b.getI64IntegerAttr(microBatch));
  b.create(state);

  LLVM_DEBUG(llvm::dbgs()
             << "[pipeline-insert] send stage " << fromStage
             << " mb " << microBatch << "\n");
}

/// Emit a `tessera.pipeline.recv` op that receives the activation from the
/// previous pipeline stage and returns it as a new value.
static Value emitPipelineRecv(OpBuilder &b, Location loc, Type activationType,
                               int64_t toStage, int64_t microBatch) {
  OperationState state(loc, "tessera.pipeline.recv");
  state.addAttribute("to_stage",    b.getI64IntegerAttr(toStage));
  state.addAttribute("micro_batch", b.getI64IntegerAttr(microBatch));
  state.addTypes(activationType);
  Operation *op = b.create(state);

  LLVM_DEBUG(llvm::dbgs()
             << "[pipeline-insert] recv stage " << toStage
             << " mb " << microBatch << "\n");

  return op->getResult(0);
}

//===----------------------------------------------------------------------===//
// PipelineStageInsertionPass
//===----------------------------------------------------------------------===//

struct PipelineStageInsertionPass
    : public PassWrapper<PipelineStageInsertionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PipelineStageInsertionPass)

  Option<int> numStagesOpt{
      *this, "num-stages",
      llvm::cl::desc("Pipeline stage count (overrides module attr)"),
      llvm::cl::init(0)};
  Option<int> numMicroBatchesOpt{
      *this, "num-micro-batches",
      llvm::cl::desc("Micro-batch count (overrides module attr)"),
      llvm::cl::init(0)};
  Option<bool> interleavedOpt{
      *this, "interleaved",
      llvm::cl::desc("Use interleaved 1F1B schedule"),
      llvm::cl::init(false)};

  StringRef getArgument() const override {
    return "tessera-pipeline-stage-insertion";
  }
  StringRef getDescription() const override {
    return "Partition IR into 1F1B pipeline stages; insert send/recv at boundaries";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder b(module.getContext());

    // ── Read pipeline parameters ─────────────────────────────────────────
    int64_t numStages = numStagesOpt > 0
        ? numStagesOpt
        : readPlanAttr(module, "num_stages", 1);
    int64_t numMicroBatches = numMicroBatchesOpt > 0
        ? numMicroBatchesOpt
        : readPlanAttr(module, "num_micro_batches", 1);
    bool interleaved = interleavedOpt
        ? (bool)interleavedOpt
        : (readPlanAttr(module, "interleaved", 0) != 0);

    if (numStages <= 1) {
      // Nothing to do — single stage, no pipeline boundaries
      LLVM_DEBUG(llvm::dbgs() << "[pipeline-insert] num_stages=1, skipping\n");
      return;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[pipeline-insert] stages=" << numStages
               << " micro_batches=" << numMicroBatches
               << (interleaved ? " interleaved" : " standard-1F1B") << "\n");

    // ── Annotate module with pipeline plan ───────────────────────────────
    module->setAttr("tessera.pp_num_stages",       b.getI64IntegerAttr(numStages));
    module->setAttr("tessera.pp_num_micro_batches", b.getI64IntegerAttr(numMicroBatches));
    module->setAttr("tessera.pp_interleaved",       b.getBoolAttr(interleaved));

    // ── Group ops by pipeline stage ──────────────────────────────────────
    // Collect (stage → ops) mapping across all functions
    llvm::DenseMap<int64_t, SmallVector<Operation *>> stageOps;
    int64_t maxStage = 0;

    module.walk([&](Operation *op) {
      int64_t stage = getOpStage(op);
      if (stage >= 0) {
        stageOps[stage].push_back(op);
        if (stage > maxStage) maxStage = stage;
      }
    });

    // ── Annotate stage boundary ops with send/recv ───────────────────────
    // For each stage k < numStages-1:
    //   • Find the last op in stage k that produces a value consumed by stage k+1
    //   • Insert tessera.pipeline.send after it (for each micro-batch)
    //   • Insert tessera.pipeline.recv at the start of stage k+1
    //
    // In this implementation we tag the boundary ops with attributes and emit
    // one send/recv pair per micro-batch (the scheduler unrolls at codegen).

    unsigned sendCount = 0, recvCount = 0;

    for (int64_t stage = 0; stage < numStages - 1; ++stage) {
      auto &ops = stageOps[stage];
      if (ops.empty()) continue;

      // Find ops in this stage whose results flow into stage+1
      for (Operation *op : ops) {
        for (Value result : op->getResults()) {
          bool crossesBoundary = false;
          for (Operation *user : result.getUsers()) {
            if (getOpStage(user) == stage + 1) {
              crossesBoundary = true;
              break;
            }
          }
          if (!crossesBoundary) continue;

          // Tag op as a pipeline boundary producer
          op->setAttr("tessera.pp_boundary_send", b.getI64IntegerAttr(stage));

          // Emit send/recv for each micro-batch
          for (int64_t mb = 0; mb < numMicroBatches; ++mb) {
            b.setInsertionPointAfter(op);
            emitPipelineSend(b, op->getLoc(), result, stage, mb);
            ++sendCount;

            // Find the first consuming op in stage+1 and insert recv before it
            for (Operation *user : result.getUsers()) {
              if (getOpStage(user) == stage + 1) {
                b.setInsertionPoint(user);
                Value recvVal = emitPipelineRecv(
                    b, user->getLoc(), result.getType(), stage + 1, mb);
                // Replace the use of the original result with the received value
                // (only for ops in stage+1 and for mb=0 to avoid duplicate repls)
                if (mb == 0) {
                  for (OpOperand &use : llvm::make_early_inc_range(result.getUses())) {
                    if (getOpStage(use.getOwner()) == stage + 1)
                      use.set(recvVal);
                  }
                }
                ++recvCount;
                break; // one recv point per boundary value
              }
            }
          }
        }
      }
    }

    // ── Annotate schedule.pipeline.region ops ────────────────────────────
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef().contains("schedule.pipeline")) {
        op->setAttr("tessera.pp_num_stages",        b.getI64IntegerAttr(numStages));
        op->setAttr("tessera.pp_num_micro_batches", b.getI64IntegerAttr(numMicroBatches));
      }
    });

    if (sendCount + recvCount > 0)
      module.emitRemark("pipeline-stage-insertion: ")
          << numStages << " stages, "
          << numMicroBatches << " micro-batches, "
          << sendCount << " send / " << recvCount << " recv ops inserted";
  }
};

} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createPipelineStageInsertionPass() {
  return std::make_unique<PipelineStageInsertionPass>();
}
} // namespace tessera
