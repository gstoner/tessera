// PipelineScheduleLegalityPass.cpp — 2026-06-23
//
// The 1F1B "schedule proof" — the verification half of the pipeline layer,
// sibling to the warp-spec legality gates. It checks that a partitioned +
// send/recv-inserted pipeline is a well-formed 1F1B schedule. Run it after
// PipelineStagePartition + PipelineStageInsertion.
//
//   --tessera-pipeline-schedule-legality
//
// Reads the pipeline plan from the module: `tessera.pp_num_stages` /
// `tessera.pp_num_micro_batches` / `tessera.pp_interleaved` (set by the
// insertion pass) or the `tessera.pipeline_plan` dict. Invariants:
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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace {

static int64_t readPlan(ModuleOp m, StringRef ppKey, StringRef planKey,
                        int64_t defaultVal) {
  if (auto v = m->getAttrOfType<IntegerAttr>(ppKey))
    return v.getInt();
  if (auto plan = m->getAttrOfType<DictionaryAttr>("tessera.pipeline_plan"))
    if (auto v = plan.getAs<IntegerAttr>(planKey))
      return v.getInt();
  return defaultVal;
}

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
    return "1F1B schedule legality — micro-batch fill (Decision #17), no empty "
           "stage, and forward-adjacent send/recv pairing.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    int64_t numStages = readPlan(module, "tessera.pp_num_stages", "num_stages", 1);
    if (numStages <= 1)
      return; // no pipeline.

    int64_t microBatches =
        readPlan(module, "tessera.pp_num_micro_batches", "num_micro_batches", 1);
    bool interleaved = readPlan(module, "tessera.pp_interleaved", "interleaved",
                                0) != 0;
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

    if (anyError)
      signalPassFailure();
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createPipelineScheduleLegalityPass() {
  return std::make_unique<PipelineScheduleLegalityPass>();
}
} // namespace tessera
