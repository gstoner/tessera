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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
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

// Give an async copy a !tile.async_token result — the SSA value its waits/mmas
// consume. If one already exists, return it. Otherwise rewrite the op in place
// (results are immutable, so recreate with the token appended, RAUW the original
// results, and erase the old op). `copy` is updated to the new op. This is the
// single place the ROCm path mints async tokens; threading them into the
// consuming wait_async / mma operands turns the copy→consumer dependency into a
// def-use edge the legality pass can check by SSA instead of program order.
static Value materializeAsyncToken(OpBuilder &builder, Operation *&copy) {
  auto tokTy = tessera::tile::AsyncTokenType::get(builder.getContext());
  for (Value r : copy->getResults())
    if (r.getType() == tokTy)
      return r;
  builder.setInsertionPoint(copy);
  SmallVector<Type> resultTypes(copy->getResultTypes().begin(),
                               copy->getResultTypes().end());
  resultTypes.push_back(tokTy);
  OperationState state(copy->getLoc(), copy->getName().getStringRef());
  state.addOperands(copy->getOperands());
  state.addTypes(resultTypes);
  state.addAttributes(copy->getAttrs());
  Operation *grown = builder.create(state);
  for (unsigned i = 0, e = copy->getNumResults(); i < e; ++i)
    copy->getResult(i).replaceAllUsesWith(grown->getResult(i));
  copy->erase();
  copy = grown;
  return grown->getResult(grown->getNumResults() - 1);
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

    // The planner is the single place that resolves async dependencies — and it
    // records them as SSA token edges, not program order. During an ordered walk
    // it decides, per consumer, which copy it depends on (explicit tile.depends_on
    // > SSA value link > most-recently-retired stage, the prefetch->wait->compute
    // idiom); afterwards it mints a !tile.async_token on each copy and threads it
    // into the operands of the wait_async that retires it and the mmas that
    // consume it. The legality pass then verifies the def-use edge instead of
    // re-deriving program order (which is what made a count-based guess able to
    // wrongly reject valid double buffering).
    getOperation().walk([&](func::FuncOp func) {
      unsigned ordinal = 0;
      SmallVector<std::string> outstanding;   // oldest first
      std::optional<SmallVector<std::string>> retiredCtx;

      // Recorded during the walk, applied after (mid-walk op recreation would
      // invalidate the walk). copies: (barrier id, the async copy that mints its
      // token). waitRetire / mmaConsumes: the id(s) each wait / mma consumes.
      SmallVector<std::pair<std::string, Operation *>> copies;
      SmallVector<std::pair<Operation *, std::string>> waitRetire;
      SmallVector<std::pair<Operation *, SmallVector<std::string>>> mmaConsumes;

      auto inferMmaDeps = [&](Operation *mma) -> SmallVector<std::string> {
        if (auto arr = mma->getAttrOfType<ArrayAttr>("tile.depends_on")) {
          SmallVector<std::string> ids;
          for (Attribute a : arr)
            if (auto s = dyn_cast<StringAttr>(a))
              ids.push_back(s.getValue().str());
          return ids;
        }
        SmallVector<std::string> ssa;
        collectSsaCopyDeps(mma, ssa);
        if (!ssa.empty())
          return ssa;
        if (retiredCtx.has_value())
          return SmallVector<std::string>(retiredCtx->begin(),
                                          retiredCtx->end());
        return {};
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
          copies.push_back({id, op});
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
          if (!retired.empty()) {
            retiredCtx = SmallVector<std::string>{retired};
            waitRetire.push_back({op, retired});
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
          retiredCtx = SmallVector<std::string>{}; // empty == drained, no deps.
          return;
        }

        if (name == "tile.mma") {
          ensurePipelineDepths(builder, op);
          if (!op->hasAttr("tile.rocm_matrix_path"))
            op->setAttr("tile.rocm_matrix_path",
                        builder.getStringAttr("wmma_or_mfma_by_arch"));
          // Resolve the stage(s) this mma depends on and record them for token
          // threading. The SSA token operand the threading adds below is the
          // source of truth (Phase D) — the planner no longer also stamps the
          // redundant tile.depends_on string. A frontend may still *provide*
          // tile.depends_on as an explicit input (inferMmaDeps consults it), and
          // the legality pass keeps a depends_on fallback for token-less IR.
          SmallVector<std::string> deps = inferMmaDeps(op);
          if (!deps.empty())
            mmaConsumes.push_back({op, deps});
          return;
        }
      });

      // Materialize the SSA token edges. Mint a token on each async copy, then
      // thread it into the wait_async that retires it and the mmas that consume
      // it. The token rides the ops' Variadic<AnyType> operand slots, so no op
      // signature changes; downstream lowering keys retirement on the SSA value.
      llvm::StringMap<Value> tokenById;
      for (auto &entry : copies) {
        Operation *copy = entry.second;
        tokenById[entry.first] = materializeAsyncToken(builder, copy);
      }
      for (auto &wr : waitRetire) {
        auto it = tokenById.find(wr.second);
        if (it != tokenById.end())
          wr.first->insertOperands(wr.first->getNumOperands(), {it->second});
      }
      for (auto &mc : mmaConsumes) {
        SmallVector<Value> toks;
        for (const std::string &id : mc.second) {
          auto it = tokenById.find(id);
          if (it != tokenById.end())
            toks.push_back(it->second);
        }
        if (!toks.empty())
          mc.first->insertOperands(mc.first->getNumOperands(), toks);
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
      // SSA token model: an async copy mints a !tile.async_token result; a
      // wait_async / s_barrier retires it; an mma's token operands name exactly
      // the stages it consumes. Legality is then a pure def-use check — every
      // token an mma consumes must already be retired — with NO program-order
      // re-derivation. The planner encoded the dependency as SSA, so a live
      // prefetch can never be mistaken for a dependency (the over-rejection the
      // old count-based guess produced is structurally impossible here). The
      // string `outstanding` set + pendingLdsWrites remain for the C2 LDS
      // write/write check and a conservative fallback on token-less IR.
      llvm::SmallPtrSet<Value, 8> outstandingTokens; // minted, not retired
      llvm::SmallPtrSet<Value, 8> retiredTokens;     // waited or drained
      SmallVector<std::string> outstanding;          // barrier ids (fallback)
      bool sawAnyWait = false;
      unsigned synth = 0;
      llvm::DenseMap<StringRef, PendingWrite> pendingLdsWrites;

      auto isToken = [](Value v) {
        return isa<tessera::tile::AsyncTokenType>(v.getType());
      };
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
          sawAnyWait = true;
          // Retire by SSA token (precise) and keep the string set consistent.
          for (Value operand : op->getOperands())
            if (isToken(operand)) {
              outstandingTokens.erase(operand);
              retiredTokens.insert(operand);
            }
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
          for (Value t : outstandingTokens) // workgroup barrier drains all.
            retiredTokens.insert(t);
          outstandingTokens.clear();
          outstanding.clear();
          pendingLdsWrites.clear();
          return;
        }

        if (name == "tile.mma") {
          // Precise path: the mma's token operands are exactly the stages it
          // consumes; each must already be retired. No program-order guess.
          bool hasTokenOperand = false;
          for (Value operand : op->getOperands())
            if (isToken(operand)) {
              hasTokenOperand = true;
              if (!retiredTokens.count(operand)) {
                op->emitOpError(
                    "ROCM_WAVE_LDS_MISSING_WAITCNT: tile.mma consumes an async "
                    "copy token with no intervening tile.wait_async / "
                    "waitcnt(vmcnt) — the LDS stage it reads is not resident.");
                anyError = true;
              }
            }
          if (hasTokenOperand)
            return;

          // Fallback for hand-written, token-less IR: trust an explicit
          // tile.depends_on; else flag only if copies are in flight and nothing
          // has been waited at all (never over-reject a waited double buffer).
          if (auto arr = op->getAttrOfType<ArrayAttr>("tile.depends_on")) {
            for (Attribute a : arr)
              if (auto s = dyn_cast<StringAttr>(a))
                if (llvm::is_contained(outstanding, s.getValue().str())) {
                  op->emitOpError(
                      "ROCM_WAVE_LDS_MISSING_WAITCNT: tile.mma depends on "
                      "barrier id '")
                      << s.getValue()
                      << "' from an outstanding global-to-LDS async copy with "
                         "no intervening tile.wait_async / waitcnt(vmcnt).";
                  anyError = true;
                }
            return;
          }
          if (!outstanding.empty() && !sawAnyWait) {
            op->emitOpError(
                "ROCM_WAVE_LDS_MISSING_WAITCNT: tile.mma runs with outstanding "
                "global-to-LDS async copies and no completed tile.wait_async / "
                "waitcnt(vmcnt) — the LDS stage it consumes is not resident.");
            anyError = true;
          }
          return;
        }

        if (name != "tile.async_copy")
          return;

        // Record the async copy: its token (precise) + barrier id (fallback) +
        // run the C2-style LDS write/write reuse check.
        for (Value r : op->getResults())
          if (isToken(r))
            outstandingTokens.insert(r);
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
