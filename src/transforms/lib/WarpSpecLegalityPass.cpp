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
// Convention-driven (works on the value lane and unregistered husks). A "warp-
// role region" is any ancestor op carrying `tile.warp_role` / `tile.warp_guard`
// / `tile.wg_id`. Op classes are recognized by a marker attribute or op-name
// substring (so a fixture can use `tile.cta_sync` / `tile.mbarrier_init` / etc.).
//
// All seven structural + lifetime invariants from the appendix are now checked.

#include "Tessera/Dialect/Tile/TileDialect.h"
#include "Tessera/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <utility>

using namespace mlir;

namespace {

// An op is inside a warp-role-guarded region if any ancestor (up to the func)
// carries a warp-role marker.
static bool isUnderWarpRoleGuard(Operation *op, Operation *funcOp) {
  for (Operation *p = op->getParentOp(); p && p != funcOp; p = p->getParentOp())
    if (p->hasAttr("tile.warp_role") || p->hasAttr("tile.warp_guard") ||
        p->hasAttr("tile.wg_id"))
      return true;
  return false;
}

static bool isBarrierInit(Operation *op) {
  if (op->hasAttr("tile.barrier_init"))
    return true;
  StringRef n = op->getName().getStringRef();
  return n.contains("mbarrier") && n.contains("init");
}

static bool isCollective(Operation *op) {
  if (op->hasAttr("tile.collective"))
    return true;
  StringRef n = op->getName().getStringRef();
  return n.contains("cta_sync") || n.contains("cluster_sync") ||
         n.contains("next_tile");
}

static bool isTmaStore(Operation *op) {
  if (op->hasAttr("tile.tma_store"))
    return true;
  StringRef n = op->getName().getStringRef();
  return n.contains("tma") && n.contains("store");
}

static bool isVisibilityFence(Operation *op) {
  if (op->hasAttr("tile.fence"))
    return true;
  StringRef n = op->getName().getStringRef();
  return n.contains("fence") || n.contains("proxy_async") ||
         n.contains("commit_group");
}

static bool isBufferFree(Operation *op) {
  if (op->hasAttr("tile.buffer_free"))
    return true;
  if (auto a = op->getAttrOfType<StringAttr>("tile.access"))
    if (a.getValue() == "free")
      return true;
  StringRef n = op->getName().getStringRef();
  return n.contains("buffer_free") || n.contains("dealloc");
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

  void runOnOperation() override {
    ModuleOp module = getOperation();
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
    });

    if (anyError)
      signalPassFailure();
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createWarpSpecLegalityPass() {
  return std::make_unique<WarpSpecLegality>();
}
} // namespace tessera
