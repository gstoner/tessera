// TilePipelineLegalityPass.cpp — C3 (2026-06-23)
//
// Cross-op companion to the C3 #tile.pipeline_state / #tile.barrier attribute
// verifiers (which only check per-attribute bounds). This pass enforces the two
// invariants that span multiple ops in a warp-specialized pipeline (TIRx review
// / COMPILER_AUDIT item C3):
//
//   1. PHASE ASYMMETRY — within one pipeline (keyed by `tile.pipeline`), the
//      *initial* producer-role op must carry phase = 1 and the initial
//      consumer-role op phase = 0. This asymmetry is what makes the first wait
//      fall through instead of deadlocking (the classic off-by-one ring bug).
//      Diagnostic: TILE_PIPELINE_PHASE_ASYMMETRY.
//
//   2. BARRIER-KIND CONSISTENCY — all ops that reference the same barrier
//      (keyed by `tile.barrier_id`) must agree on the #tile.barrier `kind`
//      (tma / tcgen05 / mbarrier). Mixing engine-signaled and thread-arrival
//      semantics on one barrier is a latent hang.
//      Diagnostic: TILE_PIPELINE_BARRIER_KIND_MISMATCH.
//
// Sibling to LayoutLegalityPass / TileBarrierReuseLegalityPass — one walk,
// stable codes, registered standalone as `--tessera-tile-pipeline-legality`.
// Together with C2 it is the acceptance gate for the warp-spec lowering work:
// once WarpSpecialization emits #tile.pipeline_state + typed #tile.barrier, this
// pass going green on the FA-4 fixture is the deadlock-freedom check.

#include "Tessera/Dialect/Tile/TileDialect.h"
#include "Tessera/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

struct TilePipelineLegality
    : public PassWrapper<TilePipelineLegality, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TilePipelineLegality)

  StringRef getArgument() const override {
    return "tessera-tile-pipeline-legality";
  }
  StringRef getDescription() const override {
    return "Tile IR pipeline legality (C3) — producer phase=1 / consumer "
           "phase=0 asymmetry and per-barrier kind consistency across ops.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool anyError = false;

    module.walk([&](func::FuncOp func) {
      // First op seen per (pipeline id, role), and first kind per barrier id.
      llvm::DenseMap<StringRef, Operation *> firstProducer, firstConsumer;
      llvm::DenseMap<StringRef, std::pair<StringRef, Operation *>> firstKind;

      func.walk<WalkOrder::PreOrder>([&](Operation *op) {
        // ── Invariant 1: pipeline-state phase asymmetry ──
        if (auto ps = op->getAttrOfType<tessera::tile::TilePipelineStateAttr>(
                "tile.pipeline_state")) {
          auto idAttr = op->getAttrOfType<StringAttr>("tile.pipeline");
          if (idAttr) {
            StringRef id = idAttr.getValue();
            StringRef role = ps.getRole();
            bool isProducer = role == "producer";
            auto &firstForRole = isProducer ? firstProducer : firstConsumer;
            if (!firstForRole.count(id)) {
              firstForRole[id] = op;
              int64_t want = isProducer ? 1 : 0;
              if (ps.getPhase() != want) {
                op->emitOpError("TILE_PIPELINE_PHASE_ASYMMETRY: initial ")
                    << role << " of pipeline \"" << id << "\" has phase "
                    << ps.getPhase() << " but must start at phase " << want
                    << " (producer=1 / consumer=0) so the first wait falls "
                       "through instead of deadlocking.";
                anyError = true;
              }
            }
          }
        }
        // ── Invariant 2: per-barrier kind consistency ──
        if (auto bar =
                op->getAttrOfType<tessera::tile::TileBarrierAttr>("tile.barrier")) {
          auto idAttr = op->getAttrOfType<StringAttr>("tile.barrier_id");
          if (idAttr) {
            StringRef id = idAttr.getValue();
            auto it = firstKind.find(id);
            if (it == firstKind.end()) {
              firstKind[id] = {bar.getKind(), op};
            } else if (it->second.first != bar.getKind()) {
              InFlightDiagnostic diag = op->emitOpError(
                  "TILE_PIPELINE_BARRIER_KIND_MISMATCH: barrier \"");
              diag << id << "\" is used with kind \"" << bar.getKind()
                   << "\" but was first declared with kind \""
                   << it->second.first << "\".";
              diag.attachNote(it->second.second->getLoc())
                  << "first use of barrier \"" << id << "\" here";
              anyError = true;
            }
          }
        }
      });
    });

    if (anyError)
      signalPassFailure();
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createTilePipelineLegalityPass() {
  return std::make_unique<TilePipelineLegality>();
}
} // namespace tessera
