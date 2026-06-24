// ROCMWaveLdsPipeline.cpp — ROCm consumption of the shared Tile-IR contract.
//
// This is the AMD-native sibling of the NVIDIA warp-specialized path.  It
// reads/annotates shared Tile attributes (#tile.layout, #tile.buffer_ref,
// #tile.pipeline_depths, numeric_policy) but keeps synchronization in ROCm
// terms: waitcnt counters and LDS/wave intent, never TMA/mbarrier semantics.

#include "Tessera/Dialect/Tile/TileDialect.h"
#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>
#include <utility>

using namespace mlir;

namespace {

static bool isStorageAxis(StringRef axis) {
  return axis == "m" || axis == "lds" || axis == "tlane" || axis == "tcol";
}

// NVIDIA-only Tile constructs that the ROCm path must reject by name (they carry
// no #tile.barrier attr to discriminate on). The legality pass fails on these.
static bool isNvidiaOnlyTileOp(StringRef name) {
  return name.starts_with("tile.mbarrier.") ||
         name.starts_with("tile.mbarrier_") || name.starts_with("tile.tma.") ||
         name.starts_with("tile.tma_") || name.starts_with("tile.tmem.");
}

// A workgroup barrier — it drains *all* outstanding async work (vs. a targeted
// wait that retires one barrier id down to a threshold).
static bool isSBarrier(Operation *op) {
  if (op->getName().getStringRef() == "tile.async_copy")
    return false; // carries kind="waitcnt", not s_barrier.
  auto barrier =
      op->getAttrOfType<tessera::tile::TileBarrierAttr>("tile.barrier");
  return barrier && barrier.getKind() == "s_barrier";
}

// Trace an mma's operands back through SSA def-use to any reachable
// tile.async_copy, collecting their barrier ids. This is the *most precise*
// stage dependency: it names the exact copies whose results the mma consumes,
// independent of which unrelated stages are still being prefetched. Bounded
// walk; returns empty when the copy->mma link is carried by an LDS buffer
// (memref) rather than an SSA value — the common ROCm shape today, where the
// caller falls back to the most-recently-retired stage.
static void collectSsaCopyDeps(Operation *mma,
                               SmallVectorImpl<std::string> &ids) {
  SmallVector<Value, 8> worklist(mma->getOperands().begin(),
                                 mma->getOperands().end());
  llvm::SmallPtrSet<Operation *, 16> seen;
  llvm::SmallPtrSet<StringAttr, 8> emitted;
  unsigned guard = 0;
  while (!worklist.empty() && guard++ < 256) {
    Value v = worklist.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def || !seen.insert(def).second)
      continue;
    if (def->getName().getStringRef() == "tile.async_copy") {
      if (auto a = def->getAttrOfType<StringAttr>("tile.barrier_id"))
        if (emitted.insert(a).second)
          ids.push_back(a.getValue().str());
      continue; // stop at the copy boundary — do not walk its source operands.
    }
    for (Value o : def->getOperands())
      worklist.push_back(o);
  }
}

static bool hasLdsAxis(tessera::tile::TileLayoutAttr layout) {
  for (StringAttr axis : layout.getShardAxes())
    if (axis.getValue() == "lds")
      return true;
  return false;
}

static SmallVector<int64_t> tileExtents(Operation *op) {
  auto rows = op->getAttrOfType<IntegerAttr>("tile_rows");
  auto cols = op->getAttrOfType<IntegerAttr>("tile_cols");
  if (rows && cols)
    return {rows.getInt(), cols.getInt()};
  if (op->getNumResults() == 1)
    if (auto t = dyn_cast<RankedTensorType>(op->getResult(0).getType()))
      if (t.hasStaticShape() && t.getRank() == 2)
        return {t.getShape()[0], t.getShape()[1]};
  return {};
}

static void ensureLdsLayout(OpBuilder &builder, Operation *op) {
  if (op->hasAttr("tile.layout"))
    return;
  SmallVector<int64_t> extents = tileExtents(op);
  if (extents.empty())
    return;
  for (int64_t extent : extents)
    if (extent <= 0)
      return;

  SmallVector<int64_t> strides(extents.size(), 1);
  for (int i = static_cast<int>(extents.size()) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * extents[i + 1];

  SmallVector<StringAttr> axes;
  axes.reserve(extents.size());
  for (size_t i = 0, e = extents.size(); i < e; ++i)
    axes.push_back(builder.getStringAttr(i == 0 ? "lds" : "waveid"));

  op->setAttr("tile.layout",
              tessera::tile::TileLayoutAttr::get(
                  builder.getContext(), extents, strides, axes,
                  /*replicaCounts=*/{}, /*replicaStrides=*/{},
                  /*replicaAxes=*/{}, /*offset=*/0,
                  /*swizzle=*/tessera::tile::TileSwizzleAttr()));
}

static void ensureLdsBuffer(OpBuilder &builder, Operation *op,
                            unsigned ordinal) {
  if (op->hasAttr("tile.buf"))
    return;
  op->setAttr("tile.buf", tessera::tile::TileBufferRefAttr::get(
                              builder.getContext(),
                              ("rocm.lds." + std::to_string(ordinal)), "lds",
                              "write"));
}

static void ensurePipelineDepths(OpBuilder &builder, Operation *op) {
  if (op->hasAttr("tile.pipeline_depths"))
    return;
  op->setAttr("tile.pipeline_depths",
              tessera::tile::TilePipelineDepthsAttr::get(
                  builder.getContext(), /*q=*/1, /*kv=*/2, /*tmem=*/1));
}

static std::optional<std::pair<int64_t, int64_t>>
storageFootprint(tessera::tile::TileLayoutAttr layout) {
  ArrayRef<int64_t> extents = layout.getShardExtents();
  ArrayRef<int64_t> strides = layout.getShardStrides();
  ArrayRef<StringAttr> axes = layout.getShardAxes();
  int64_t span = 0;
  bool anyStorage = false;
  for (auto [extent, stride, axis] : llvm::zip(extents, strides, axes)) {
    if (!isStorageAxis(axis.getValue()))
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

static bool overlaps(const std::pair<int64_t, int64_t> &lhs,
                     const std::pair<int64_t, int64_t> &rhs) {
  return lhs.first < rhs.second && rhs.first < lhs.second;
}

struct ROCMWaveLdsPipelinePass
    : PassWrapper<ROCMWaveLdsPipelinePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ROCMWaveLdsPipelinePass)

  StringRef getArgument() const override { return "rocm-wave-lds-pipeline"; }
  StringRef getDescription() const override {
    return "Annotate shared Tile IR with ROCm wave/LDS/waitcnt intent.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tessera::tile::TesseraTileDialect>();
  }

  void runOnOperation() override {
    OpBuilder builder(&getContext());

    // Per-function, program-order FIFO of outstanding async barrier ids. A wait
    // retires the *oldest* (vmcnt drains in issue order); an s_barrier drains
    // all. This is the single place async ids are assigned — the lowering reads
    // the stamps, so it cannot regress to a "last token" model.
    getOperation().walk([&](func::FuncOp func) {
      unsigned ordinal = 0;
      SmallVector<std::string> outstanding;   // oldest first
      // The stage(s) retired by the most recent wait_async, or an empty vector
      // after an s_barrier drain. nullopt means "nothing has been waited yet on
      // this path". An mma consumes the most-recently-retired stage (the
      // prefetch->wait->compute idiom), NOT whatever is still outstanding — so
      // a live prefetch never gets mistaken for the mma's dependency.
      std::optional<SmallVector<std::string>> retiredCtx;

      // Stamp a precise tile.depends_on on an mma from the best available
      // signal: SSA value link first, else the just-retired stage.
      auto stampMmaDeps = [&](Operation *mma) {
        if (mma->hasAttr("tile.depends_on"))
          return;
        SmallVector<std::string> deps;
        collectSsaCopyDeps(mma, deps);
        bool haveContext = !deps.empty();
        if (deps.empty() && retiredCtx.has_value()) {
          deps.assign(retiredCtx->begin(), retiredCtx->end());
          haveContext = true; // includes the empty (post-s_barrier) case.
        }
        if (!haveContext)
          return; // no wait/SSA evidence — legality flags if copies are live.
        SmallVector<Attribute> attrs;
        for (const std::string &id : deps)
          attrs.push_back(builder.getStringAttr(id));
        mma->setAttr("tile.depends_on", builder.getArrayAttr(attrs));
      };

      func.walk<WalkOrder::PreOrder>([&](Operation *op) {
        StringRef name = op->getName().getStringRef();

        if (name == "tile.async_copy") {
          ensureLdsBuffer(builder, op, ordinal);
          ensureLdsLayout(builder, op);
          ensurePipelineDepths(builder, op);
          std::string id = "rocm.waitcnt." + std::to_string(ordinal);
          if (auto a = op->getAttrOfType<StringAttr>("tile.barrier_id"))
            id = a.getValue().str();
          else
            op->setAttr("tile.barrier_id", builder.getStringAttr(id));
          if (!op->hasAttr("tile.barrier"))
            op->setAttr("tile.barrier", tessera::tile::TileBarrierAttr::get(
                                            builder.getContext(), "waitcnt", 0));
          if (!op->hasAttr("tile.wait_counter"))
            op->setAttr("tile.wait_counter", builder.getStringAttr("vmcnt"));
          if (!op->hasAttr("tile.pipeline"))
            op->setAttr("tile.pipeline", builder.getStringAttr(
                                             "rocm.wave_lds." +
                                             std::to_string(ordinal)));
          outstanding.push_back(id);
          ++ordinal;
          return;
        }

        if (name == "tile.wait_async") {
          // Retire a stamped id if present, else the oldest outstanding, and
          // record it as the stage subsequent mmas depend on.
          std::string retired;
          if (auto a = op->getAttrOfType<StringAttr>("tile.barrier_id")) {
            retired = a.getValue().str();
            auto it = llvm::find(outstanding, retired);
            if (it != outstanding.end())
              outstanding.erase(it);
          } else if (!outstanding.empty()) {
            retired = outstanding.front();
            op->setAttr("tile.barrier_id", builder.getStringAttr(retired));
            outstanding.erase(outstanding.begin());
          }
          if (!retired.empty())
            retiredCtx = SmallVector<std::string>{retired};
          // Threshold = ids still outstanding for this counter after retiring.
          op->setAttr("tile.waitcnt_threshold",
                      builder.getI64IntegerAttr(
                          static_cast<int64_t>(outstanding.size())));
          if (!op->hasAttr("tile.wait_counter"))
            op->setAttr("tile.wait_counter", builder.getStringAttr("vmcnt"));
          return;
        }

        if (isSBarrier(op)) {
          outstanding.clear(); // workgroup barrier drains all.
          retiredCtx = SmallVector<std::string>{}; // empty == drained, no deps.
          return;
        }

        if (name == "tile.mma") {
          ensurePipelineDepths(builder, op);
          if (!op->hasAttr("tile.rocm_matrix_path"))
            op->setAttr("tile.rocm_matrix_path",
                        builder.getStringAttr("wmma_or_mfma_by_arch"));
          stampMmaDeps(op);
        }
      });
    });
  }
};

struct PendingWrite {
  Operation *op;
  tessera::tile::TileLayoutAttr layout;
};

struct ROCMWaveLdsLegalityPass
    : PassWrapper<ROCMWaveLdsLegalityPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ROCMWaveLdsLegalityPass)

  StringRef getArgument() const override { return "rocm-wave-lds-legality"; }
  StringRef getDescription() const override {
    return "Verify ROCm LDS double-buffering and waitcnt correctness.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tessera::tile::TesseraTileDialect>();
  }

  void runOnOperation() override {
    bool anyError = false;

    getOperation().walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (isNvidiaOnlyTileOp(name)) {
        op->emitOpError(
            "ROCM_WAVE_LDS_UNSUPPORTED_NV_CONSTRUCT: ROCm Tile lowering cannot "
            "consume NVIDIA-only Tile ops (tile.mbarrier.* / tile.tma.* / "
            "tile.tmem.*); use LDS / waitcnt / s_barrier contracts instead.");
        anyError = true;
      }

      auto barrier =
          op->getAttrOfType<tessera::tile::TileBarrierAttr>("tile.barrier");
      if (barrier && (barrier.getKind() == "tma" ||
                      barrier.getKind() == "tcgen05" ||
                      barrier.getKind() == "mbarrier")) {
        op->emitOpError(
            "ROCM_WAVE_LDS_UNSUPPORTED_BARRIER_KIND: ROCm cannot consume "
            "NVIDIA TMA/TCGen05/mbarrier completion semantics; use waitcnt or "
            "s_barrier.");
        anyError = true;
      }

      auto buf =
          op->getAttrOfType<tessera::tile::TileBufferRefAttr>("tile.buf");
      if (buf && buf.getSpace() == "tmem") {
        op->emitOpError(
            "ROCM_WAVE_LDS_UNSUPPORTED_TMEM: #tile.buffer_ref space=tmem is "
            "NVIDIA-only for this ROCm path.");
        anyError = true;
      }
    });

    getOperation().walk([&](func::FuncOp func) {
      // Per-id outstanding async work (issued, not yet retired), ordered oldest
      // first. A wait retires the id it names (or the oldest); an s_barrier
      // drains all. Replaces the function-global pendingAsync bool, so an mma
      // may run while *unrelated* prefetch ids remain outstanding (double
      // buffering).
      SmallVector<std::string> outstanding;
      // Same retired-stage model as the planner, so legality is precise even on
      // hand-written IR the planner never stamped: an mma depends on the stage
      // most recently *retired*, never on a live prefetch.
      std::optional<SmallVector<std::string>> retiredCtx;
      unsigned synth = 0;
      llvm::DenseMap<StringRef, PendingWrite> pendingLdsWrites;

      auto asyncIdOf = [&](Operation *op) -> std::string {
        if (auto a = op->getAttrOfType<StringAttr>("tile.barrier_id"))
          return a.getValue().str();
        return "rocm.async.synth." + std::to_string(synth++);
      };

      func.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (op == func.getOperation())
          return;
        StringRef name = op->getName().getStringRef();

        if (name == "tile.wait_async") {
          std::string retired;
          if (auto a = op->getAttrOfType<StringAttr>("tile.barrier_id")) {
            retired = a.getValue().str();
            auto it = llvm::find(outstanding, retired);
            if (it != outstanding.end())
              outstanding.erase(it);
          } else if (!outstanding.empty()) {
            retired = outstanding.front();
            outstanding.erase(outstanding.begin());
          }
          if (!retired.empty())
            retiredCtx = SmallVector<std::string>{retired};
          pendingLdsWrites.clear(); // a retired copy makes its LDS write safe.
          return;
        }
        if (isSBarrier(op)) {
          outstanding.clear(); // workgroup barrier drains all.
          retiredCtx = SmallVector<std::string>{}; // empty == drained, no deps.
          pendingLdsWrites.clear();
          return;
        }

        if (name == "tile.mma") {
          // Resolve the LDS stage(s) this mma consumes, in precedence order:
          //   1. explicit tile.depends_on  (frontend/planner-stated, exact)
          //   2. SSA value link to a tile.async_copy  (exact when present)
          //   3. the most-recently-retired stage  (prefetch->wait->compute)
          // A live prefetch that the mma does NOT consume is intentionally not
          // a dependency, so software-pipelined double buffering is legal.
          SmallVector<std::string> deps;
          bool haveContext = false;
          if (auto arr = op->getAttrOfType<ArrayAttr>("tile.depends_on")) {
            haveContext = true;
            for (Attribute a : arr)
              if (auto s = dyn_cast<StringAttr>(a))
                deps.push_back(s.getValue().str());
          } else {
            collectSsaCopyDeps(op, deps);
            if (!deps.empty()) {
              haveContext = true;
            } else if (retiredCtx.has_value()) {
              deps.assign(retiredCtx->begin(), retiredCtx->end());
              haveContext = true; // includes the empty post-s_barrier case.
            }
          }

          if (!haveContext) {
            // Nothing waited yet and no value link: if copies are in flight the
            // mma would read unfilled LDS — a genuine waitcnt hazard.
            if (!outstanding.empty()) {
              op->emitOpError(
                  "ROCM_WAVE_LDS_MISSING_WAITCNT: tile.mma runs with "
                  "outstanding global-to-LDS async copies and no completed "
                  "tile.wait_async / waitcnt(vmcnt) — the LDS stage it consumes "
                  "is not yet resident.");
              anyError = true;
            }
            return;
          }
          for (const std::string &d : deps)
            if (llvm::is_contained(outstanding, d)) {
              op->emitOpError("ROCM_WAVE_LDS_MISSING_WAITCNT: tile.mma depends "
                              "on barrier id '")
                  << d
                  << "' from an outstanding global-to-LDS async copy with no "
                     "intervening tile.wait_async / waitcnt(vmcnt).";
              anyError = true;
            }
          return;
        }

        if (name != "tile.async_copy")
          return;

        // Record the async copy as outstanding + run the C2-style LDS
        // write/write reuse check.
        outstanding.push_back(asyncIdOf(op));

        auto buf =
            op->getAttrOfType<tessera::tile::TileBufferRefAttr>("tile.buf");
        if (!buf || buf.getSpace() != "lds" || buf.getAccess() != "write")
          return;
        auto layout =
            op->getAttrOfType<tessera::tile::TileLayoutAttr>("tile.layout");
        if (!layout || !hasLdsAxis(layout))
          return;
        auto fp = storageFootprint(layout);
        auto it = pendingLdsWrites.find(buf.getName());
        if (fp && it != pendingLdsWrites.end()) {
          auto prev = storageFootprint(it->second.layout);
          if (prev && overlaps(*prev, *fp)) {
            InFlightDiagnostic diag = op->emitOpError(
                "ROCM_WAVE_LDS_OVERLAPPING_WRITE: LDS buffer \"");
            diag << buf.getName()
                 << "\" is written over an overlapping layout region with no "
                    "intervening waitcnt/barrier.";
            diag.attachNote(it->second.op->getLoc())
                << "previous write to the same LDS buffer";
            anyError = true;
          }
        }
        pendingLdsWrites[buf.getName()] = PendingWrite{op, layout};
      });
    });

    if (anyError)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createROCMWaveLdsPipelinePass() {
  return std::make_unique<ROCMWaveLdsPipelinePass>();
}

std::unique_ptr<mlir::Pass>
mlir::tessera_rocm::createROCMWaveLdsLegalityPass() {
  return std::make_unique<ROCMWaveLdsLegalityPass>();
}
