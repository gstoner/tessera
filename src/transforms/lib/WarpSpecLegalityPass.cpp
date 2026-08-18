// WarpSpecLegalityPass.cpp — C6 (2026-06-23)
//
// Static diagnostics for warp-specialized Tile IR, distilled from the TIRx
// "Debugging Warp-Specialized Kernels" appendix (COMPILER_AUDIT item C6). These
// are the *structural* invariants that complement C3's phase-asymmetry check —
// they catch deadlocks/races at compile time instead of as device hangs:
//
//   WARPSPEC_INIT_UNDER_GUARD
//     A barrier-init op must run at CTA top level (the appendix: ".init() guard
//     becomes threadIdx.x < 1" — CTA thread 0 only). An init nested inside a
//     warp-role region never initializes for the other roles → immediate hang.
//
//   WARPSPEC_COLLECTIVE_IN_DIVERGENT_BRANCH
//     A collective (cta_sync / cluster_sync / tile_scheduler.next_tile) must not
//     sit inside a warp-role-guarded region: partial participation hangs the
//     barrier or leaves the scheduler inconsistent.
//
//   WARPSPEC_LOOP_COUNT_DISAGREE
//     Producer (TMA) and consumer (MMA) loops on one pipeline must agree on
//     trip count — the "MMA does K_TILES-1" serialization/deadlock signature.
//
//   WARPSPEC_MISSING_VISIBILITY_FENCE
//     A TMA store needs a prior visibility fence (fence.proxy_async /
//     commit_group) in its block, else the async engine may read stale shared
//     memory.
//
//   WARPSPEC_ARRIVAL_COUNT_MISMATCH
//     All #tile.barrier sites on one tile.barrier_id must agree on `expect` —
//     the init (setup_descriptor) declares the expected transaction/arrival
//     count and every arrive (copy_async) must match it, else the wait never
//     releases. Fed by NVTMADescriptorPass's typed #tile.barrier emission.
//
//   WARPSPEC_USE_AFTER_FREE
//     A buffer free (tile.buffer_free / tile.access="free") needs a prior
//     cta_sync in its block, so every warp has finished reading the buffer
//     before it is deallocated during writeback. Fed by WarpSpecialization's
//     writeback-dealloc epilogue (cta_sync + buffer_free per region buffer).
//
// Sibling to LayoutLegalityPass / TileBarrierReuseLegalityPass (C2) /
// TilePipelineLegalityPass (C3) — one walk, stable codes, registered standalone
// as `--tessera-warpspec-legality`. Together with C2+C3 it is the deadlock-
// freedom gate for the FA-4 warp-spec lowering once WarpSpecialization emits
// these markers.
//
// Warp divergence is declaration-driven: a "warp-role region" is a registered
// schedule.warp ancestor.  Tile operations used by these checks are registered
// classes except for the explicitly tracked pre-SM90 compatibility husks.
//
// All seven structural + lifetime invariants from the appendix are now checked.

#include "TileValueProvenance.h"
#include "TileRelationalLegality.h"

#include "Tessera/Dialect/Tile/TileDialect.h"
#include "Tessera/Transforms/Passes.h"
#include "tessera/ProgrammingModel/ScheduleDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <utility>

using namespace mlir;

namespace {

// An op is inside a divergent warp-role region only when its ancestor is the
// registered Schedule IR carrier.  Attribute ancestry was a legacy escape
// hatch: any unrelated op could acquire `tile.warp_role` and silently change
// legality, while a producer that forgot the marker failed open.
static bool isUnderWarpRoleGuard(Operation *op, Operation *funcOp) {
  for (Operation *p = op->getParentOp(); p && p != funcOp; p = p->getParentOp())
    if (isa<tessera::schedule::WarpOp>(p))
      return true;
  return false;
}

// P1a second increment (CAKE §5.3–§5.4, 2026-08-15): the six predicates are
// derived from registered op identity, not name substrings, and the
// unproduced attribute escape hatches (`tile.barrier_init`, `tile.collective`,
// `tile.tma_store`, `tile.fence`, `tile.buffer_free`) are deleted — no pass
// stamped them, so they were pure fail-open surface. The remaining exact-name
// matches are the SM<90 `cp_async` husk trio (a real producer in
// AsyncCopyLoweringPass) and the two collectives with no producer yet; each
// graduates to a typed check when its op is registered (Decision #29).
static bool isBarrierInit(Operation *op) {
  return isa<tessera::tile::MBarrierInitOp>(op);
}

static bool isCollective(Operation *op) {
  if (isa<tessera::tile::CtaSyncOp>(op))
    return true;
  StringRef n = op->getName().getStringRef();
  return n == "tile.cluster_sync" || n == "tile.tile_scheduler.next_tile";
}

static bool isTmaStore(Operation *op) {
  return isa<tessera::tile::TMAStoreOp>(op);
}

static bool isVisibilityFence(Operation *op) {
  if (isa<tessera::tile::FenceOp>(op))
    return true;
  return op->getName().getStringRef() == "tile.cp_async.commit_group";
}

static bool isBufferFree(Operation *op) {
  return isa<tessera::tile::DeallocOp>(op);
}

// A producer of async-staged data for a consumer mma: either a tile.async_copy /
// tile.tma.copy_async directly (pre-warpspec / standalone), or a producer-role
// schedule.warp region whose results carry the staged tiles + tokens across the
// boundary (post-warpspec). The mma's completion-token edge must point at one.
static bool isAsyncDataProducer(Operation *op) {
  StringRef n = op->getName().getStringRef();
  if (n == "tile.async_copy" || n == "tile.tma.copy_async")
    return true;
  if (auto warp = dyn_cast<tessera::schedule::WarpOp>(op))
    return warp.getRole() == "producer";
  return false;
}

struct WarpSpecLegality
    : public PassWrapper<WarpSpecLegality, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WarpSpecLegality)

  StringRef getArgument() const override { return "tessera-warpspec-legality"; }
  StringRef getDescription() const override {
    return "Warp-spec legality (C6) — barrier-init placement, collectives "
           "outside divergent branches, producer/consumer loop-count agreement, "
           "and TMA-store visibility fences.";
  }

  LogicalResult verify(ModuleOp module) {
    bool anyError = false;

    module.walk([&](func::FuncOp func) {
      Operation *funcOp = func.getOperation();

      // Invariants 3/4/5 + arrival-count — single program-order walk.
      llvm::DenseMap<StringRef, std::pair<int64_t, Operation *>> firstTrip;
      llvm::DenseMap<StringRef, std::pair<int64_t, Operation *>> firstExpect;
      func.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (op == funcOp)
          return;

        // arrival-count == init-count: per tile.barrier_id, all #tile.barrier
        // `expect` values must agree (the init declares the count).
        if (auto bar = op->getAttrOfType<tessera::tile::TileBarrierAttr>(
                "tile.barrier")) {
          if (auto idAttr = op->getAttrOfType<StringAttr>("tile.barrier_id")) {
            StringRef id = idAttr.getValue();
            int64_t e = bar.getExpect();
            auto it = firstExpect.find(id);
            if (it == firstExpect.end()) {
              firstExpect[id] = {e, op};
            } else if (it->second.first != e) {
              InFlightDiagnostic diag = op->emitOpError(
                  "WARPSPEC_ARRIVAL_COUNT_MISMATCH: barrier \"");
              diag << id << "\" arrives with expect " << e
                   << " but its init declared expect " << it->second.first
                   << " — the wait will never release.";
              diag.attachNote(it->second.second->getLoc())
                  << "barrier \"" << id << "\" init count here";
              anyError = true;
            }
          }
        }

        if (isBarrierInit(op) && isUnderWarpRoleGuard(op, funcOp)) {
          op->emitOpError(
              "WARPSPEC_INIT_UNDER_GUARD: barrier init must run at CTA top "
              "level (thread 0), not inside a warp-role-guarded region — it "
              "would never initialize for the other roles.");
          anyError = true;
        }

        if (isCollective(op) && isUnderWarpRoleGuard(op, funcOp)) {
          op->emitOpError(
              "WARPSPEC_COLLECTIVE_IN_DIVERGENT_BRANCH: a collective "
              "(cta_sync / cluster_sync / next_tile) must not sit inside a "
              "warp-role-guarded region — partial participation hangs.");
          anyError = true;
        }

        auto pipe = op->getAttrOfType<StringAttr>("tile.pipeline");
        auto trip = op->getAttrOfType<IntegerAttr>("tile.trip_count");
        if (pipe && trip) {
          StringRef id = pipe.getValue();
          int64_t n = trip.getInt();
          auto it = firstTrip.find(id);
          if (it == firstTrip.end()) {
            firstTrip[id] = {n, op};
          } else if (it->second.first != n) {
            InFlightDiagnostic diag = op->emitOpError(
                "WARPSPEC_LOOP_COUNT_DISAGREE: pipeline \"");
            diag << id << "\" loop trip count " << n
                 << " disagrees with the producer/consumer count "
                 << it->second.first
                 << " — a deadlock/serialization signature.";
            diag.attachNote(it->second.second->getLoc())
                << "pipeline \"" << id << "\" first trip count here";
            anyError = true;
          }
        }
      });

      // Invariants 6/7 — block-local ordering: a TMA store needs a prior
      // visibility fence; a buffer free needs a prior cta_sync (use-after-free).
      func.walk([&](Block *block) {
        bool fenceSeen = false;
        bool ctaSyncSeen = false;
        for (Operation &op : *block) {
          if (isVisibilityFence(&op))
            fenceSeen = true;
          if (isCollective(&op))
            ctaSyncSeen = true;
          if (isTmaStore(&op) && !fenceSeen) {
            op.emitOpError(
                "WARPSPEC_MISSING_VISIBILITY_FENCE: TMA store has no prior "
                "visibility fence (fence.proxy_async / commit_group) in its "
                "block — the async engine may read stale shared memory.");
            anyError = true;
          }
          if (isBufferFree(&op) && !ctaSyncSeen) {
            op.emitOpError(
                "WARPSPEC_USE_AFTER_FREE: buffer free has no prior cta_sync in "
                "its block — a warp may still be reading the buffer during "
                "writeback while it is deallocated.");
            anyError = true;
          }
        }
      });

      // Token synchronization (Phase C-NV) — the SSA half of arrival==init.
      // arrival==init (above) checks the mbarrier *byte count* via #tile.barrier.
      // This checks the *ordering* edge via the !tile.async_token, in two parts:
      //   (a) PRESENCE — a consumer tile.mma that reads a producer's async-staged
      //       tile must also read a completion token from that same producer, so
      //       the matrix op is gated on copy completion by SSA, not program order
      //       (WARPSPEC_MMA_NOT_TOKEN_SYNCED).
      //   (b) RETIREMENT — a token minted directly by a tile.async_copy /
      //       tile.tma.copy_async must be RETIRED by a prior tile.wait_async
      //       before the mma; merely *holding* an unwaited token is a held-but-
      //       unwaited race (WARPSPEC_MMA_TOKEN_NOT_RETIRED). This converges with
      //       the ROCm legality, which also requires retirement, not just
      //       presence. Warp-boundary tokens (schedule.warp results) and loop-
      //       carried tokens (block args) are retired structurally inside the
      //       producer / on the back-edge, so retirement is checked only for the
      //       direct-copy case this single-visit walk can prove.
      // Single ordered walk so "retired before this mma" is well-defined.
      llvm::SmallPtrSet<Value, 8> retiredTokens;
      func.walk<WalkOrder::PreOrder>([&](Operation *op) {
        StringRef n = op->getName().getStringRef();
        if (n == "tile.wait_async" || n == "tile.mbarrier.wait") {
          for (Value v : op->getOperands())
            if (isa<tessera::tile::AsyncTokenType>(v.getType()))
              retiredTokens.insert(v);
          return;
        }
        if (n != "tile.mma")
          return;
        // Per producer the mma reads from: did it read data, and a token?
        // Producers are resolved ACROSS scf.for block-argument edges (P1a /
        // CAKE §5.3): a staged tile arriving as an iter_arg resolves to the
        // async copy that produced it, so the canonical pipelined shape is no
        // longer invisible to this check (the measured §5.1.1 fail-open).
        llvm::DenseMap<Operation *, std::pair<bool, bool>> fromProducer;
        for (Value operand : op->getOperands()) {
          tessera::provenance::Resolution res =
              tessera::provenance::resolveThroughLoopCarry(operand);
          bool isToken = isa<tessera::tile::AsyncTokenType>(operand.getType());
          for (Value root : res.roots) {
            Operation *def = root.getDefiningOp();
            if (!def || !isAsyncDataProducer(def))
              continue;
            auto &flags = fromProducer[def];
            (isToken ? flags.second : flags.first) = true;
          }
          // (b) retirement, only for a token minted directly by a copy op
          // (a loop-carried token is retired structurally on the back edge;
          // the direct case is the one this single-visit walk can prove).
          Operation *def = operand.getDefiningOp();
          if (isToken && def) {
            StringRef dn = def->getName().getStringRef();
            bool directCopy =
                dn == "tile.async_copy" || dn == "tile.tma.copy_async";
            if (directCopy && !isa<BlockArgument>(operand) &&
                !retiredTokens.count(operand)) {
              op->emitOpError(
                  "WARPSPEC_MMA_TOKEN_NOT_RETIRED: tile.mma holds an async-copy "
                  "completion token that no prior tile.wait_async retired — the "
                  "copy is still in flight when the matrix op runs (held-but-"
                  "unwaited race). Add a tile.wait_async on the token before the "
                  "mma.");
              anyError = true;
            }
          }
        }
        for (auto &entry : fromProducer) {
          if (entry.second.first && !entry.second.second) {
            op->emitOpError(
                "WARPSPEC_MMA_NOT_TOKEN_SYNCED: tile.mma reads an async-staged "
                "tile from a producer but has no !tile.async_token completion "
                "edge to it — the matrix op is not gated on copy completion "
                "(WarpSpecialization mints this edge from the mma's data "
                "operands).");
            anyError = true;
          }
        }
      });
    });

    return failure(anyError);
  }

  void runOnOperation() override {
    if (failed(verify(getOperation())))
      signalPassFailure();
  }
};

} // namespace

namespace tessera {
LogicalResult verifyWarpSpecializationRelations(ModuleOp module) {
  WarpSpecLegality verifier;
  return verifier.verify(module);
}

std::unique_ptr<Pass> createWarpSpecLegalityPass() {
  return std::make_unique<WarpSpecLegality>();
}
} // namespace tessera
