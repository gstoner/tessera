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
// The local write/write check uses !tile.buffer allocation identity plus
// tile.layout footprints. Only registered completing waits/barriers clear a
// pending hazard; policy attributes and nonblocking polls do not. Allocation-
// scoped release and branch/loop-sensitive lifetime analysis remain follow-ups.

#include "Tessera/Dialect/Tile/TileDialect.h"
#include "Tessera/Transforms/Passes.h"
#include "TileRelationalLegality.h"

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
  // Memory-resident axes that can alias: `m` (linear / NVIDIA SMEM), `lds` (AMD
  // Local Data Share), `tlane`/`tcol` (NVIDIA TMEM). LDS reuse needs an
  // intervening barrier exactly as SMEM does — ROCm is first-class here.
  return ax == "m" || ax == "lds" || ax == "tlane" || ax == "tcol";
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

// Completion is an operation effect, never an attribute attached to a poll,
// arrival or unrelated operation. In particular try_wait does not complete.
static bool isBarrierOp(Operation *op) {
  return isa<tessera::tile::WaitAsyncOp, tessera::tile::MBarrierWaitOp,
             tessera::tile::CtaSyncOp, tessera::tile::SBarrierOp>(op);
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

  LogicalResult verify(ModuleOp module) {
    bool anyError = false;

    module.walk([&](func::FuncOp func) {
      // SSA allocation root → most-recent unbarried write.
      llvm::DenseMap<Value, PendingWrite> pendingSSA;

      func.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (op == func.getOperation())
          return;

        // A barrier clears every pending hazard (conservative v1 scope).
        if (isBarrierOp(op)) {
          pendingSSA.clear();
          return;
        }

        Value bufferRoot;
        for (Value operand : op->getOperands()) {
          if (!isa<tessera::tile::BufferType>(operand.getType()))
            continue;
          bufferRoot = operand;
          while (Operation *def = bufferRoot.getDefiningOp()) {
            Value next;
            for (Value candidate : def->getOperands())
              if (isa<tessera::tile::BufferType>(candidate.getType())) {
                next = candidate;
                break;
              }
            if (!next)
              break;
            bufferRoot = next;
          }
          break;
        }
        if (!bufferRoot)
          return;
        auto layout =
            op->getAttrOfType<tessera::tile::TileLayoutAttr>("tile.layout");
        if (!layout)
          return;

        std::optional<std::pair<int64_t, int64_t>> fp = storageFootprint(layout);
        PendingWrite *previous = nullptr;
        auto it = pendingSSA.find(bufferRoot);
        if (it != pendingSSA.end())
          previous = &it->second;
        if (fp && previous) {
          std::optional<std::pair<int64_t, int64_t>> prevFp =
              storageFootprint(previous->layout);
          if (prevFp && overlaps(*prevFp, *fp)) {
            InFlightDiagnostic diag = op->emitOpError(
                "TILE_BARRIER_REUSE_MISSING_BARRIER: buffer ");
            diag << "SSA allocation";
            diag << " is written over a storage region that overlaps a prior "
                    "write, with no intervening barrier (mbarrier / wait_async). "
                    "A reused layout region needs a barrier to be race-free.";
            diag.attachNote(previous->op->getLoc())
                << "previous write to the same allocation root here";
            anyError = true;
          }
        }
        // Record this write as the buffer's current pending writer.
        pendingSSA[bufferRoot] = PendingWrite{op, layout};
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
LogicalResult verifyTileBarrierReuseRelations(ModuleOp module) {
  TileBarrierReuseLegality verifier;
  return verifier.verify(module);
}

std::unique_ptr<Pass> createTileBarrierReuseLegalityPass() {
  return std::make_unique<TileBarrierReuseLegality>();
}
} // namespace tessera
