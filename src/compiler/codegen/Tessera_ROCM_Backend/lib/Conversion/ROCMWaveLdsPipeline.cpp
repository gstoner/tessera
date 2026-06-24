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

// Real ROCm synchronization — what legitimately retires outstanding async work.
// Deliberately NOT a name.contains("barrier") sniff: that wrongly accepted
// NVIDIA `tile.mbarrier.*` ops and let them clear the waitcnt hazard.
static bool isROCMBarrierOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (name == "tile.wait_async" || name == "tessera_rocm.wait")
    return true;
  auto barrier =
      op->getAttrOfType<tessera::tile::TileBarrierAttr>("tile.barrier");
  return barrier && barrier.getKind() == "s_barrier";
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
      size_t maxOutstanding = 0;
      // (mma, outstanding-snapshot) for single-stage depends_on inference.
      SmallVector<std::pair<Operation *, SmallVector<std::string>>> mmaSnaps;

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
          maxOutstanding = std::max(maxOutstanding, outstanding.size());
          ++ordinal;
          return;
        }

        if (name == "tile.wait_async") {
          // Retire a stamped id if present, else the oldest outstanding.
          if (auto a = op->getAttrOfType<StringAttr>("tile.barrier_id")) {
            auto it = llvm::find(outstanding, a.getValue().str());
            if (it != outstanding.end())
              outstanding.erase(it);
          } else if (!outstanding.empty()) {
            op->setAttr("tile.barrier_id",
                        builder.getStringAttr(outstanding.front()));
            outstanding.erase(outstanding.begin());
          }
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
          return;
        }

        if (name == "tile.mma") {
          ensurePipelineDepths(builder, op);
          if (!op->hasAttr("tile.rocm_matrix_path"))
            op->setAttr("tile.rocm_matrix_path",
                        builder.getStringAttr("wmma_or_mfma_by_arch"));
          mmaSnaps.push_back(
              {op, SmallVector<std::string>(outstanding.begin(),
                                            outstanding.end())});
        }
      });

      // depends_on inference: only for unambiguous single-stage IR (never more
      // than one async copy in flight). Multi-stage IR is left unstamped — the
      // legality pass requires an explicit tile.depends_on and fails otherwise.
      bool singleStage = maxOutstanding <= 1;
      for (auto &[mma, snap] : mmaSnaps) {
        if (mma->hasAttr("tile.depends_on") || !singleStage)
          continue;
        SmallVector<Attribute> ids;
        for (const std::string &id : snap)
          ids.push_back(builder.getStringAttr(id));
        mma->setAttr("tile.depends_on", builder.getArrayAttr(ids));
      }
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
          if (auto a = op->getAttrOfType<StringAttr>("tile.barrier_id")) {
            auto it = llvm::find(outstanding, a.getValue().str());
            if (it != outstanding.end())
              outstanding.erase(it);
          } else if (!outstanding.empty()) {
            outstanding.erase(outstanding.begin());
          }
          pendingLdsWrites.clear(); // a retired copy makes its LDS write safe.
          return;
        }
        if (isSBarrier(op)) {
          outstanding.clear(); // workgroup barrier drains all.
          pendingLdsWrites.clear();
          return;
        }

        if (name == "tile.mma") {
          SmallVector<std::string> deps;
          if (auto arr = op->getAttrOfType<ArrayAttr>("tile.depends_on")) {
            for (Attribute a : arr)
              if (auto s = dyn_cast<StringAttr>(a))
                deps.push_back(s.getValue().str());
          } else if (outstanding.size() == 1) {
            deps.push_back(outstanding.front()); // unambiguous single stage.
          } else if (outstanding.size() > 1) {
            op->emitOpError(
                "ROCM_WAVE_LDS_AMBIGUOUS_DEPENDENCY: tile.mma has multiple "
                "outstanding async copies and no explicit tile.depends_on; the "
                "LDS stage it consumes is ambiguous (multi-stage IR must carry "
                "tile.depends_on).");
            anyError = true;
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
