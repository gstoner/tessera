//===- TileDataflowLegalityPass.cpp — derived Tile sync legality (P1a) ----===//
//
// CAKE Phase 1 §5.3 / W2.4's named `TileDataflowLegalityPass`, first increment
// (2026-08-15). Where the C2/C3/C6 legality passes match names and attributes,
// this pass DERIVES its facts from types and def-use chains — including across
// `scf.for` block-argument edges via TileValueProvenance.h — and fails closed
// on anything it cannot derive (Decision #30). It exists because the measured
// §5.1.1 result showed the canonical pipelined kernel shape (loop-carried
// barriers, tokens, and pipeline state) passes every existing gate silently.
//
// Checks (each with a stable diagnostic code, Decision #21):
//
//   TILE_WAIT_TOKEN_UNPAIRED
//     A `tile.mbarrier.wait` `$token` must resolve (across loop carries) to
//     `tile.mbarrier.arrive_expect_tx` results only.
//   TILE_WAIT_SLOT_MISMATCH
//     The resolved arrive and the wait must name the same slot — the §5.1
//     probe shape: arrive slot 1 / wait slot 0 never releases on hardware.
//   TILE_WAIT_BARRIER_DISAGREES
//     The wait's barrier and its token's arrive barrier must resolve to a
//     common `tile.mbarrier.init` root.
//   TILE_BARRIER_ORIGIN_UNRESOLVED
//     Every `!tile.mbarrier` operand anywhere must resolve to
//     `tile.mbarrier.init` roots — a barrier of unprovable origin is unproven,
//     not permitted (CAKE §5.5 exit gate 2).
//   TILE_PIPELINE_RING_STALE / TILE_PIPELINE_RING_UNDERIVED
//     A loop-carried `!tile.pipeline_state` must be yielded from a
//     `tile.pipeline_advance` chain rooted at that iter_arg. Yielding the
//     un-advanced state is the stale-ring bug (§5.1.1 experiment 2: it used
//     to pass silently); an underivable yield fails closed.
//   TILE_TMA_EXPECT_MISMATCH / TILE_TMA_DESC_ORIGIN_UNRESOLVED
//     A `tile.tma.copy_async` and the descriptor it reaches through SSA must
//     agree on the expected transaction bytes. This is the SSA-keyed form of
//     the arrival-count check — the §5.1 probe with one SSA barrier under two
//     `tile.barrier_id` strings escaped the string-keyed gate.
//
// Registered standalone as `--tessera-tile-dataflow-legality`. Pipeline wiring
// (alongside the C3/C6 passes) is deliberately deferred until the
// research/tile_sync gate and full lit suites are green under it.
//
//===----------------------------------------------------------------------===//

#include "TileValueProvenance.h"

#include "Tessera/Dialect/Tile/TileDialect.h"
#include "Tessera/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using tessera::provenance::Resolution;
using tessera::provenance::resolveThroughLoopCarry;

namespace {

// The transaction byte count an op declares: prefer the `expect_tx` integer
// attribute, else the typed #tile.barrier `expect`. Returns std::nullopt when
// the op declares neither.
static std::optional<int64_t> declaredExpect(Operation *op) {
  if (auto a = op->getAttrOfType<IntegerAttr>("expect_tx"))
    return a.getInt();
  if (auto bar =
          op->getAttrOfType<tessera::tile::TileBarrierAttr>("tile.barrier"))
    return bar.getExpect();
  return std::nullopt;
}

struct TileDataflowLegality
    : public PassWrapper<TileDataflowLegality, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileDataflowLegality)

  StringRef getArgument() const override {
    return "tessera-tile-dataflow-legality";
  }
  StringRef getDescription() const override {
    return "Derived Tile sync legality (CAKE Phase 1 §5.3 / W2.4): arrive→wait "
           "pairing with slot identity, barrier origin, pipeline-ring "
           "advancement, and SSA-keyed TMA expect agreement — all across "
           "scf.for block-argument edges, failing closed on the underivable.";
  }

  bool anyError = false;

  InFlightDiagnostic err(Operation *op, const Twine &message) {
    anyError = true;
    return op->emitOpError(message);
  }

  // ── D2: every !tile.mbarrier operand resolves to mbarrier.init roots. ──
  void checkBarrierOrigin(Operation *op) {
    for (Value operand : op->getOperands()) {
      if (!isa<tessera::tile::MBarrierType>(operand.getType()))
        continue;
      Resolution r = resolveThroughLoopCarry(operand);
      if (!r.complete || r.roots.empty()) {
        err(op, "TILE_BARRIER_ORIGIN_UNRESOLVED: this !tile.mbarrier operand "
                "cannot be resolved to a tile.mbarrier.init across its "
                "def-use/loop-carry chain; an unprovable barrier origin is "
                "unproven, not permitted");
        continue;
      }
      for (Value root : r.roots)
        if (!root.getDefiningOp<tessera::tile::MBarrierInitOp>())
          err(op, "TILE_BARRIER_ORIGIN_UNRESOLVED: a resolved barrier root is "
                  "not produced by tile.mbarrier.init");
    }
  }

  // ── D1: wait token pairing — arrive origin, slot identity, same barrier. ──
  void checkWaitTokenPairing(tessera::tile::MBarrierWaitOp wait) {
    Value token = wait.getToken();
    if (!token)
      return; // async-token dependency form; producer sync is C6's check
    Resolution tok = resolveThroughLoopCarry(token);
    if (!tok.complete || tok.roots.empty()) {
      err(wait, "TILE_WAIT_TOKEN_UNPAIRED: the !tile.mbarrier_token cannot be "
                "resolved to its arrive across the def-use/loop-carry chain");
      return;
    }
    auto waitSlot = wait.getOperation()->getAttrOfType<IntegerAttr>("slot");
    for (Value root : tok.roots) {
      auto arrive =
          root.getDefiningOp<tessera::tile::MBarrierArriveExpectTxOp>();
      if (!arrive) {
        err(wait, "TILE_WAIT_TOKEN_UNPAIRED: a resolved token root is not "
                  "produced by tile.mbarrier.arrive_expect_tx");
        continue;
      }
      auto arriveSlot =
          arrive.getOperation()->getAttrOfType<IntegerAttr>("slot");
      if (waitSlot && arriveSlot && waitSlot.getInt() != arriveSlot.getInt()) {
        InFlightDiagnostic d =
            err(wait, "TILE_WAIT_SLOT_MISMATCH: wait on slot ");
        d << waitSlot.getInt() << " consumes a token arrived on slot "
          << arriveSlot.getInt() << " — this wait never releases";
        d.attachNote(arrive.getLoc()) << "token arrives here";
      }
      // Same-barrier identity, derived through SSA on both sides.
      if (Value waitBar = wait.getBarrier()) {
        Resolution wb = resolveThroughLoopCarry(waitBar);
        Resolution ab = resolveThroughLoopCarry(arrive.getBarrier());
        if (wb.complete && ab.complete) {
          bool intersects = llvm::any_of(
              wb.roots, [&](Value v) { return ab.roots.contains(v); });
          if (!intersects) {
            InFlightDiagnostic d = err(
                wait, "TILE_WAIT_BARRIER_DISAGREES: the wait's barrier and "
                      "its token's arrive barrier resolve to different "
                      "tile.mbarrier.init roots");
            d.attachNote(arrive.getLoc()) << "token arrives on this barrier";
          }
        }
        // Incomplete resolution is already reported by checkBarrierOrigin.
      }
    }
  }

  // ── D3: a loop-carried pipeline_state ring must actually advance. ──
  void checkPipelineRing(scf::ForOp forOp) {
    Operation *terminator = forOp.getBody()->getTerminator();
    for (auto [idx, arg] :
         llvm::enumerate(forOp.getBody()->getArguments().drop_front())) {
      if (!isa<tessera::tile::PipelineStateType>(arg.getType()))
        continue;
      Value yielded = terminator->getOperand(idx);
      if (yielded == arg) {
        err(forOp, "TILE_PIPELINE_RING_STALE: the loop-carried "
                   "!tile.pipeline_state is yielded un-advanced — every "
                   "iteration re-uses the same stage/phase and the ring "
                   "never moves");
        continue;
      }
      // Chase the advance chain: yielded must be a pipeline_advance whose
      // state-operand chain reaches this iter_arg.
      Value cursor = yielded;
      bool reachedArg = false;
      while (auto adv =
                 cursor.getDefiningOp<tessera::tile::PipelineAdvanceOp>()) {
        cursor = adv.getInputs().front(); // verifier guarantees state first
        if (cursor == arg) {
          reachedArg = true;
          break;
        }
      }
      if (!reachedArg) {
        if (yielded.getDefiningOp<tessera::tile::PipelineAdvanceOp>())
          err(forOp,
              "TILE_PIPELINE_RING_STALE: the yielded !tile.pipeline_state "
              "advances a different state than this loop's iter_arg — the "
              "carried ring never moves");
        else
          err(forOp,
              "TILE_PIPELINE_RING_UNDERIVED: the yielded "
              "!tile.pipeline_state is not derived from tile.pipeline_advance "
              "on this loop's iter_arg; an unprovable ring fails closed");
      }
    }
  }

  // ── D4: SSA-keyed expect agreement between a copy and its descriptor. ──
  void checkCopyExpect(tessera::tile::TMACopyAsyncOp copy) {
    Resolution desc = resolveThroughLoopCarry(copy.getDescriptor());
    if (!desc.complete || desc.roots.empty()) {
      err(copy, "TILE_TMA_DESC_ORIGIN_UNRESOLVED: the copy's descriptor "
                "cannot be resolved to a tile.tma.descriptor across the "
                "def-use/loop-carry chain");
      return;
    }
    std::optional<int64_t> copyExpect = declaredExpect(copy);
    for (Value root : desc.roots) {
      auto setup = root.getDefiningOp<tessera::tile::TMADescriptorOp>();
      if (!setup) {
        err(copy, "TILE_TMA_DESC_ORIGIN_UNRESOLVED: a resolved descriptor "
                  "root is not produced by tile.tma.descriptor");
        continue;
      }
      std::optional<int64_t> descExpect = declaredExpect(setup);
      if (copyExpect && descExpect && *copyExpect != *descExpect) {
        InFlightDiagnostic d =
            err(copy, "TILE_TMA_EXPECT_MISMATCH: copy expects ");
        d << *copyExpect << " transaction bytes but its descriptor declares "
          << *descExpect << " — the wait on this barrier never releases";
        d.attachNote(setup.getLoc()) << "descriptor declared here";
      }
    }
  }

  // ── SO-2: logical roles are provenance, not string annotations. ─────────
  bool collectRoleKinds(Value value, Operation *owner,
                        llvm::SmallDenseSet<StringRef, 2> &kinds,
                        llvm::SmallDenseSet<StringRef, 4> &names) {
    Resolution resolution = resolveThroughLoopCarry(value);
    if (!resolution.complete || resolution.roots.empty()) {
      err(owner,
          "TILE_ROLE_RELATION_INVALID: !tile.role provenance cannot be "
          "resolved to tile.role declarations across its def-use/loop-carry "
          "chain");
      return false;
    }
    bool valid = true;
    for (Value root : resolution.roots) {
      auto declaration = root.getDefiningOp<tessera::tile::RoleOp>();
      if (!declaration) {
        err(owner,
            "TILE_ROLE_RELATION_INVALID: a resolved !tile.role root is not "
            "produced by tile.role");
        valid = false;
        continue;
      }
      kinds.insert(declaration.getKind());
      names.insert(declaration.getName());
    }
    return valid;
  }

  void checkPipelineRole(tessera::tile::PipelineInitOp pipeline) {
    Value logicalRole = pipeline.getLogicalRole();
    if (!logicalRole)
      return; // compatibility form; physical producers migrate independently.
    llvm::SmallDenseSet<StringRef, 2> kinds;
    llvm::SmallDenseSet<StringRef, 4> names;
    if (!collectRoleKinds(logicalRole, pipeline, kinds, names))
      return;
    if (kinds.size() != 1 || !kinds.contains(pipeline.getRole()))
      err(pipeline,
          "TILE_ROLE_RELATION_INVALID: pipeline role disagrees with its "
          "resolved logical role declaration");
  }

  void checkBarrierRoles(tessera::tile::MBarrierInitOp barrier) {
    if (barrier.getRoles().empty())
      return; // compatibility form during staged producer migration.
    llvm::SmallDenseSet<StringRef, 2> kinds;
    llvm::SmallDenseSet<StringRef, 4> names;
    bool valid = true;
    for (Value role : barrier.getRoles())
      valid &= collectRoleKinds(role, barrier, kinds, names);
    if (!valid)
      return;
    if (!kinds.contains("producer") || !kinds.contains("consumer"))
      err(barrier,
          "TILE_ROLE_RELATION_INVALID: a role-bearing barrier requires "
          "resolved producer and consumer role sets");
  }

  void runOnOperation() override {
    anyError = false;
    getOperation().walk([&](func::FuncOp func) {
      func.walk([&](Operation *op) {
        if (auto wait = dyn_cast<tessera::tile::MBarrierWaitOp>(op))
          checkWaitTokenPairing(wait);
        if (auto copy = dyn_cast<tessera::tile::TMACopyAsyncOp>(op))
          checkCopyExpect(copy);
        if (auto pipeline = dyn_cast<tessera::tile::PipelineInitOp>(op))
          checkPipelineRole(pipeline);
        if (auto barrier = dyn_cast<tessera::tile::MBarrierInitOp>(op))
          checkBarrierRoles(barrier);
        if (auto forOp = dyn_cast<scf::ForOp>(op))
          checkPipelineRing(forOp);
        checkBarrierOrigin(op);
      });
    });
    if (anyError)
      signalPassFailure();
  }
};

} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createTileDataflowLegalityPass() {
  return std::make_unique<TileDataflowLegality>();
}
} // namespace tessera
