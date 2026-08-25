// PipelineScheduleLegalityPass.cpp — 2026-06-23
//
// The 1F1B "schedule proof" — the verification half of the pipeline layer,
// sibling to the warp-spec legality gates. It checks that a partitioned +
// send/recv-inserted pipeline is a well-formed 1F1B schedule. Run it after
// PipelineStagePartition + PipelineStageInsertion.
//
//   --tessera-pipeline-schedule-legality
//
// Reads the digest-bound `tessera.pipeline_steps` carrier and its materialized
// scalar views. It never reconstructs steps from a `tessera.pipeline_plan`.
//
//   PP_MICRO_BATCHES_TOO_FEW
//     1F1B needs micro_batches >= num_stages to fill the pipe; interleaved 1F1B
//     needs micro_batches >= 2*num_stages (Decision #17). Too few starves the
//     steady state — the schedule is all warmup/cooldown bubble.
//
//   PP_EMPTY_STAGE
//     Every stage in [0, num_stages) must own at least one op (tessera.pp_stage
//     / tessera.layer). An empty stage means the partition produced fewer real
//     stages than declared — the send/recv chain has a hole.
//
//   PP_SEND_WITHOUT_RECV / PP_RECV_WITHOUT_SEND
//     The boundary comms must form a forward-adjacent chain: every
//     `tessera.pipeline.send {from_stage=k}` pairs with a
//     `tessera.pipeline.recv {to_stage=k+1}` and vice versa. A send with no
//     matching recv (or a recv with no matching send one stage back) is an
//     unpaired / stage-skipping comm — a deadlock or dropped activation.
//
//   PP_UNROUTED_CROSS_STAGE_VALUE
//     The send/recv value-rewrite-completeness proof: after insertion, NO value
//     may flow directly from a stage-k op to a different-stage op — every
//     cross-stage activation must go through send/recv. A surviving direct SSA
//     edge means the rewrite missed a boundary (e.g. a stage-skipping 0->2 edge,
//     which the adjacent-only insertion pass silently leaves unrouted).

#include "Tessera/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"


using namespace mlir;

namespace {


static int64_t opStage(Operation *op) {
  if (auto layer = op->getAttrOfType<DictionaryAttr>("tessera.layer"))
    if (auto s = layer.getAs<IntegerAttr>("stage"))
      return s.getInt();
  if (auto s = op->getAttrOfType<IntegerAttr>("tessera.pp_stage"))
    return s.getInt();
  return -1;
}


struct PipelineScheduleLegalityPass
    : public PassWrapper<PipelineScheduleLegalityPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PipelineScheduleLegalityPass)

  StringRef getArgument() const override {
    return "tessera-pipeline-schedule-legality";
  }
  StringRef getDescription() const override {
    return "Prove 1F1B legality against the materialized Schedule Object "
           "dependency carrier.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto scheduleDigest =
        module->getAttrOfType<StringAttr>("tessera.schedule_digest");
    auto scheduleSchema =
        module->getAttrOfType<StringAttr>("tessera.pipeline_schedule_schema");
    auto scheduleSteps =
        module->getAttrOfType<ArrayAttr>("tessera.pipeline_steps");
    auto numStagesAttr =
        module->getAttrOfType<IntegerAttr>("tessera.pp_num_stages");
    auto microBatchesAttr =
        module->getAttrOfType<IntegerAttr>("tessera.pp_num_micro_batches");
    auto interleavedAttr =
        module->getAttrOfType<BoolAttr>("tessera.pp_interleaved");
    if (!scheduleDigest || scheduleDigest.getValue().size() != 64 ||
        !scheduleSchema ||
        scheduleSchema.getValue() != "tessera.pipeline_schedule.v1" ||
        !scheduleSteps || scheduleSteps.empty() || !numStagesAttr ||
        !microBatchesAttr || !interleavedAttr) {
      module.emitError(
          "pipeline legality requires one complete digest-bound "
          "tessera.pipeline_schedule.v1 carrier");
      signalPassFailure();
      return;
    }
    // The carrier IS the executable authority, so its rows are validated
    // rather than counted: unique action ids, resolvable dependencies, an
    // acyclic dependency order, and agreement with the declared pipeline
    // dimensions. Previously only `!empty()` was checked, so a stale or
    // hand-edited carrier with duplicate ids, dangling dependencies, or a
    // cycle was accepted as authoritative (PR #626 review).
    {
      llvm::DenseSet<StringRef> seen;
      llvm::SmallVector<StringRef> order;
      llvm::DenseMap<StringRef, llvm::SmallVector<StringRef>> requires;
      int64_t declaredStages = numStagesAttr.getInt();
      int64_t declaredMicroBatches = microBatchesAttr.getInt();
      for (Attribute entry : scheduleSteps) {
        auto row = dyn_cast<DictionaryAttr>(entry);
        if (!row) {
          module.emitError("pipeline step carrier row is not a dictionary");
          signalPassFailure();
          return;
        }
        auto actionId = row.getAs<StringAttr>("action_id");
        auto stage = row.getAs<IntegerAttr>("stage");
        auto microBatch = row.getAs<IntegerAttr>("micro_batch");
        auto clock = row.getAs<IntegerAttr>("clock");
        auto dependsOn = row.getAs<ArrayAttr>("depends_on");
        if (!actionId || actionId.getValue().empty() || !stage ||
            !microBatch || !clock || !dependsOn) {
          module.emitError(
              "pipeline step carrier row requires action_id, stage, "
              "micro_batch, clock, and depends_on");
          signalPassFailure();
          return;
        }
        if (!seen.insert(actionId.getValue()).second) {
          module.emitError("duplicate pipeline action id '")
              << actionId.getValue() << "'";
          signalPassFailure();
          return;
        }
        if (clock.getInt() < 0 || microBatch.getInt() < 0 ||
            microBatch.getInt() >= declaredMicroBatches || stage.getInt() < 0) {
          module.emitError("pipeline step '")
              << actionId.getValue()
              << "' disagrees with the declared pipeline dimensions";
          signalPassFailure();
          return;
        }
        // Virtual stages run 0..num_stages*num_chunks-1; without chunking the
        // bound is num_stages.
        int64_t chunks = 1;
        if (auto chunkAttr =
                module->getAttrOfType<IntegerAttr>("tessera.pp_num_chunks"))
          chunks = std::max<int64_t>(1, chunkAttr.getInt());
        if (stage.getInt() >= declaredStages * chunks) {
          module.emitError("pipeline step '")
              << actionId.getValue() << "' names virtual stage "
              << stage.getInt() << " beyond the declared "
              << (declaredStages * chunks);
          signalPassFailure();
          return;
        }
        llvm::SmallVector<StringRef> deps;
        for (Attribute dependency : dependsOn) {
          auto dependencyId = dyn_cast<StringAttr>(dependency);
          if (!dependencyId) {
            module.emitError("pipeline dependency is not a string");
            signalPassFailure();
            return;
          }
          deps.push_back(dependencyId.getValue());
        }
        requires[actionId.getValue()] = deps;
        order.push_back(actionId.getValue());
      }
      // Dependencies must resolve, and the carrier's own order must be a
      // topological order (a producer listed after its consumer would let the
      // consumer issue first).
      llvm::DenseMap<StringRef, size_t> position;
      for (auto [index, id] : llvm::enumerate(order)) position[id] = index;
      for (StringRef id : order) {
        for (StringRef dependency : requires[id]) {
          auto found = position.find(dependency);
          if (found == position.end()) {
            module.emitError("pipeline action '")
                << id << "' depends on unknown action '" << dependency << "'";
            signalPassFailure();
            return;
          }
          if (found->second >= position[id]) {
            module.emitError("pipeline action '")
                << id << "' depends on '" << dependency
                << "', which the carrier orders no earlier (cycle or "
                   "producer-after-consumer)";
            signalPassFailure();
            return;
          }
        }
      }
    }

    int64_t numStages = numStagesAttr.getInt();
    if (numStages <= 1)
      return;
    int64_t microBatches = microBatchesAttr.getInt();
    bool interleaved = interleavedAttr.getValue();
    bool anyError = false;

    // ── Micro-batch fill contract (Decision #17) ──
    int64_t minMb = interleaved ? 2 * numStages : numStages;
    if (microBatches < minMb) {
      module.emitError("PP_MICRO_BATCHES_TOO_FEW: ")
          << (interleaved ? "interleaved " : "") << "1F1B over " << numStages
          << " stages needs num_micro_batches >= " << minMb << " (got "
          << microBatches << ") to fill the pipeline.";
      anyError = true;
    }

    // ── Per-stage occupancy + boundary comm collection ──
    llvm::DenseSet<int64_t> occupied;
    llvm::DenseSet<int64_t> sendStages;  // from_stage of each send
    llvm::DenseSet<int64_t> recvStages;  // to_stage of each recv
    module.walk([&](Operation *op) {
      int64_t s = opStage(op);
      if (s >= 0)
        occupied.insert(s);
      StringRef name = op->getName().getStringRef();
      if (name == "tessera.pipeline.send")
        if (auto k = op->getAttrOfType<IntegerAttr>("from_stage"))
          sendStages.insert(k.getInt());
      if (name == "tessera.pipeline.recv")
        if (auto k = op->getAttrOfType<IntegerAttr>("to_stage"))
          recvStages.insert(k.getInt());
    });

    // ── No empty stage ──
    for (int64_t s = 0; s < numStages; ++s)
      if (!occupied.contains(s)) {
        module.emitError("PP_EMPTY_STAGE: ")
            << "stage " << s << " of " << numStages
            << " owns no op — the partition has a hole in the send/recv chain.";
        anyError = true;
      }

    // ── Forward-adjacent send/recv pairing ──
    for (int64_t k : sendStages)
      if (!recvStages.contains(k + 1)) {
        module.emitError("PP_SEND_WITHOUT_RECV: ")
            << "a send from stage " << k << " has no matching recv at stage "
            << (k + 1) << " — a dropped activation / deadlock.";
        anyError = true;
      }
    for (int64_t j : recvStages)
      if (!sendStages.contains(j - 1)) {
        module.emitError("PP_RECV_WITHOUT_SEND: ")
            << "a recv at stage " << j << " has no matching send from stage "
            << (j - 1) << " — an unpaired / stage-skipping comm.";
        anyError = true;
      }

    // ── Value-rewrite completeness: no direct cross-stage SSA edge ──
    module.walk([&](Operation *op) {
      int64_t producerStage = opStage(op);
      if (producerStage < 0)
        return;
      for (Value result : op->getResults())
        for (Operation *user : result.getUsers()) {
          int64_t userStage = opStage(user);
          if (userStage >= 0 && userStage != producerStage) {
            op->emitOpError("PP_UNROUTED_CROSS_STAGE_VALUE: ")
                << "a value flows directly from stage " << producerStage
                << " to stage " << userStage
                << " without a send/recv — the boundary rewrite missed it "
                   "(e.g. a stage-skipping edge).";
            anyError = true;
          }
        }
    });

    if (anyError) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createPipelineScheduleLegalityPass() {
  return std::make_unique<PipelineScheduleLegalityPass>();
}
} // namespace tessera
