// TileBarrierReuseLegalityPass.cpp — C2 (2026-06-23)
//
// "Barriers are a layout-reuse correctness property, not a scheduling artifact."
// (TIRx review / COMPILER_AUDIT item C2.) The motivating case is FA-4's TMEM
// allocation aliased as an fp32 view (S/O) and an fp16 view (P): the barriers
// exist because each region is *reused* strictly after its prior consumer
// finishes. This pass turns that into a checkable rule on Tile IR carrying the
// C1 `#tile.layout` attribute:
//
//   For a given buffer, if two WRITE ops target overlapping STORAGE-axis
//   (m / tlane / tcol) footprints of their `#tile.layout` with NO intervening
//   barrier op, emit TILE_BARRIER_REUSE_MISSING_BARRIER on the second writer.
//
// This is LayoutLegalityPass's sibling — one walk, a stable diagnostic code,
// registered standalone as `--tessera-tile-barrier-reuse-legality`. It is the
// forcing function / acceptance gate for the typed-barrier + reuse work (C3):
// once WarpSpecialization emits real barriers, "does this pass go green on the
// FA-4 fixture?" becomes the correctness check.
//
// Op conventions (kept attribute-driven so it works on the value lane and on
// unregistered husks alike):
//   write op   : `tile.buf = #tile.buffer_ref<name, space, access="write">`,
//                `tile.layout = #tile.layout<...>`
//   barrier op : op name contains "mbarrier" / "wait_async" / "barrier", OR
//                carries a `tile.barrier` unit attribute. A barrier releases the
//                pending-write hazard for every buffer (conservative v1 scope).
//
// Diagnostic code (stable):
//   TILE_BARRIER_REUSE_MISSING_BARRIER
//     A buffer is written twice over overlapping storage footprints with no
//     intervening barrier — a producer/consumer race on the reused region.

#include "Tessera/Dialect/Tile/TileDialect.h"
#include "Tessera/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <utility>

using namespace mlir;

namespace {

// Storage axes — the ones that name a physical memory/TMEM location and can
// therefore alias. Placement on register/lane/warp/grid axes does not alias a
// shared storage region.
static bool isStorageAxis(StringRef ax) {
  return ax == "m" || ax == "tlane" || ax == "tcol";
}

// Linear footprint of a layout restricted to its storage-axis shard dims:
// [offset, offset + span) where span = 1 + sum (extent-1)*|stride|. Returns
// nullopt when the layout touches no storage axis (a pure register/lane
// fragment — no shared-storage hazard).
static std::optional<std::pair<int64_t, int64_t>>
storageFootprint(tessera::tile::TileLayoutAttr layout) {
  ArrayRef<int64_t> extents = layout.getShardExtents();
  ArrayRef<int64_t> strides = layout.getShardStrides();
  ArrayRef<StringAttr> axes = layout.getShardAxes();
  int64_t span = 0;
  bool anyStorage = false;
  for (auto [extent, stride, ax] : llvm::zip(extents, strides, axes)) {
    if (!isStorageAxis(ax.getValue()))
      continue;
    anyStorage = true;
    int64_t s = stride < 0 ? -stride : stride;
    span += (extent - 1) * s;
  }
  if (!anyStorage)
    return std::nullopt;
  int64_t lo = layout.getOffset();
  return std::make_pair(lo, lo + span + 1);
}

static bool overlaps(const std::pair<int64_t, int64_t> &a,
                     const std::pair<int64_t, int64_t> &b) {
  return a.first < b.second && b.first < a.second;
}

// A barrier op releases the pending-write hazard.
static bool isBarrierOp(Operation *op) {
  if (op->hasAttr("tile.barrier"))
    return true;
  StringRef name = op->getName().getStringRef();
  return name.contains("mbarrier") || name.contains("wait_async") ||
         name.contains("barrier");
}

struct PendingWrite {
  Operation *op;
  tessera::tile::TileLayoutAttr layout;
};

struct TileBarrierReuseLegality
    : public PassWrapper<TileBarrierReuseLegality, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileBarrierReuseLegality)

  StringRef getArgument() const override {
    return "tessera-tile-barrier-reuse-legality";
  }
  StringRef getDescription() const override {
    return "Tile IR barrier/layout-reuse legality (C2) — two writes to "
           "overlapping storage footprints of one buffer with no intervening "
           "barrier emit TILE_BARRIER_REUSE_MISSING_BARRIER.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool anyError = false;

    module.walk([&](func::FuncOp func) {
      // buffer name → most-recent unbarried write. Pre-order walk visits ops in
      // program order (parent before children, siblings in order).
      llvm::DenseMap<StringRef, PendingWrite> pending;

      func.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (op == func.getOperation())
          return;

        // A barrier clears every pending hazard (conservative v1 scope).
        if (isBarrierOp(op)) {
          pending.clear();
          return;
        }

        // Typed buffer reference: a write to a named buffer in a memory space.
        auto bufRef =
            op->getAttrOfType<tessera::tile::TileBufferRefAttr>("tile.buf");
        if (!bufRef || bufRef.getAccess() != "write")
          return;
        auto layout =
            op->getAttrOfType<tessera::tile::TileLayoutAttr>("tile.layout");
        if (!layout)
          return;

        StringRef buf = bufRef.getName();
        std::optional<std::pair<int64_t, int64_t>> fp = storageFootprint(layout);

        auto it = pending.find(buf);
        if (fp && it != pending.end()) {
          std::optional<std::pair<int64_t, int64_t>> prevFp =
              storageFootprint(it->second.layout);
          if (prevFp && overlaps(*prevFp, *fp)) {
            InFlightDiagnostic diag = op->emitOpError(
                "TILE_BARRIER_REUSE_MISSING_BARRIER: buffer \"");
            diag << buf
                 << "\" is written over a storage region that overlaps a prior "
                    "write, with no intervening barrier (mbarrier / wait_async). "
                    "A reused layout region needs a barrier to be race-free.";
            diag.attachNote(it->second.op->getLoc())
                << "previous write to buffer \"" << buf << "\" here";
            anyError = true;
          }
        }
        // Record this write as the buffer's current pending writer.
        pending[buf] = PendingWrite{op, layout};
      });
    });

    if (anyError)
      signalPassFailure();
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createTileBarrierReuseLegalityPass() {
  return std::make_unique<TileBarrierReuseLegality>();
}
} // namespace tessera
