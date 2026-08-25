//===- PipelineStageInsertionPass.cpp — Phase 4 ───────────────────────────===//
//
// Inserts send/recv communication at already materialized pipeline boundaries.
// The one schedule authority is the content-addressed Schedule Object emitted
// by PipelinePlan. Its digest and dependency steps survive in module IR; this
// pass consumes that carrier and never reconstructs a schedule from scalar
// options or a parallel tessera.pipeline_plan dictionary.
//
// Registration: --tessera-pipeline-stage-insertion
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
                              int64_t fromStage, int64_t microBatch,
                              StringAttr scheduleDigest) {
  OperationState state(loc, "tessera.pipeline.send");
  state.addOperands(activation);
  state.addAttribute("from_stage",  b.getI64IntegerAttr(fromStage));
  state.addAttribute("tessera.schedule_digest", scheduleDigest);
  state.addAttribute("micro_batch", b.getI64IntegerAttr(microBatch));
  b.create(state);

  LLVM_DEBUG(llvm::dbgs()
             << "[pipeline-insert] send stage " << fromStage
             << " mb " << microBatch << "\n");
}

/// Emit a `tessera.pipeline.recv` op that receives the activation from the
/// previous pipeline stage and returns it as a new value.
static Value emitPipelineRecv(OpBuilder &b, Location loc, Type activationType,
                               int64_t toStage, int64_t microBatch,
                               StringAttr scheduleDigest) {
  OperationState state(loc, "tessera.pipeline.recv");
  state.addAttribute("to_stage",    b.getI64IntegerAttr(toStage));
  state.addAttribute("tessera.schedule_digest", scheduleDigest);
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

  PipelineStageInsertionPass() = default;
  PipelineStageInsertionPass(const PipelineStageInsertionPass &other)
      : PassWrapper(other) {}


  StringRef getArgument() const override {
    return "tessera-pipeline-stage-insertion";
  }
  StringRef getDescription() const override {
    return "Partition IR into 1F1B pipeline stages; insert send/recv at boundaries";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder b(module.getContext());

    // Consume the materialized Schedule Object carrier. Scalar plan dicts and
    // pass-option overrides are intentionally not accepted as schedule data.
    auto scheduleDigest =
        module->getAttrOfType<StringAttr>("tessera.schedule_digest");
    auto scheduleSchema =
        module->getAttrOfType<StringAttr>("tessera.pipeline_schedule_schema");
    auto scheduleSteps =
        module->getAttrOfType<ArrayAttr>("tessera.pipeline_steps");
    auto numStagesAttr =
        module->getAttrOfType<IntegerAttr>("tessera.pp_num_stages");
    auto numMicroBatchesAttr =
        module->getAttrOfType<IntegerAttr>("tessera.pp_num_micro_batches");
    auto interleavedAttr =
        module->getAttrOfType<BoolAttr>("tessera.pp_interleaved");
    if (!scheduleDigest || scheduleDigest.getValue().size() != 64 ||
        !scheduleSchema ||
        scheduleSchema.getValue() != "tessera.pipeline_schedule.v1" ||
        !scheduleSteps || scheduleSteps.empty() || !numStagesAttr ||
        !numMicroBatchesAttr || !interleavedAttr) {
      module.emitError(
          "pipeline stage insertion requires one complete digest-bound "
          "tessera.pipeline_schedule.v1 carrier");
      signalPassFailure();
      return;
    }
    int64_t numStages = numStagesAttr.getInt();
    int64_t numMicroBatches = numMicroBatchesAttr.getInt();
    bool interleaved = interleavedAttr.getValue();

    if (numStages <= 1) {
      // Nothing to do — single stage, no pipeline boundaries
      LLVM_DEBUG(llvm::dbgs() << "[pipeline-insert] num_stages=1, skipping\n");
      return;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[pipeline-insert] stages=" << numStages
               << " micro_batches=" << numMicroBatches
               << (interleaved ? " interleaved" : " standard-1F1B") << "\n");

    // Stamp the same content identity on every owning function. The resource
    // vectors and reasoned edges remain in the out-of-band Schedule Object.
    module.walk([&](func::FuncOp func) {
      func->setAttr("tessera.schedule_digest", scheduleDigest);
    });

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

          // Tag op as a pipeline boundary producer and record the micro-batch
          // count as metadata. (Previously this emitted one send PER micro-batch
          // of the same single SSA `result`, but only mb==0 was ever wired to a
          // recv — the mb>0 sends were dead and inflated sendCount. The scaffold
          // models one boundary data-dependency per value; per-mb pipelining is
          // a scheduler concern driven by this attribute.)
          op->setAttr("tessera.pp_boundary_send", b.getI64IntegerAttr(stage));
          op->setAttr("tessera.pp_micro_batches", b.getI64IntegerAttr(numMicroBatches));

          // Emit exactly one send after the producer.
          b.setInsertionPointAfter(op);
          emitPipelineSend(b, op->getLoc(), result, stage, /*mb=*/0,
                           scheduleDigest);
          ++sendCount;

          // Insert one recv before the first stage+1 consumer and rewire the
          // boundary uses to it.
          for (Operation *user : result.getUsers()) {
            if (getOpStage(user) == stage + 1) {
              b.setInsertionPoint(user);
              Value recvVal = emitPipelineRecv(
                  b, user->getLoc(), result.getType(), stage + 1, /*mb=*/0,
                  scheduleDigest);
              for (OpOperand &use : llvm::make_early_inc_range(result.getUses())) {
                if (getOpStage(use.getOwner()) == stage + 1)
                  use.set(recvVal);
              }
              ++recvCount;
              break; // one recv point per boundary value
            }
          }
        }
      }
    }

    // ── Annotate schedule.pipeline.region ops ────────────────────────────
    // Preserve the owning Schedule Object identity on every pipeline region.
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef().contains("schedule.pipeline")) {
        op->setAttr("tessera.pp_num_stages", b.getI64IntegerAttr(numStages));
        op->setAttr("tessera.pp_num_micro_batches",
                    b.getI64IntegerAttr(numMicroBatches));
        op->setAttr("tessera.schedule_digest", scheduleDigest);
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
