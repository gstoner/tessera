// ROCMWaveLdsPipeline.cpp — ROCm consumption of the shared Tile-IR contract.
//
// This is the AMD-native sibling of the NVIDIA warp-specialized path. Physical
// allocation and pipeline ownership are canonical SSA (`!tile.buffer` and
// `!tile.pipeline_state`). Allocation identity is exclusively SSA-owned.
// Physical synchronization remains ROCm-owned: waitcnt counters and LDS/wave
// intent, never TMA/mbarrier semantics.

#include "Tessera/Dialect/Tile/TileDialect.h"
#include "TesseraROCM/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
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
  // Existing ROCm async-copy fixtures and frontend paths commonly carry their
  // static staging shape on the destination memref instead of duplicating
  // tile_rows/tile_cols metadata. Preserve that route while making allocation
  // identity explicit.
  for (Value operand : op->getOperands())
    if (auto shaped = dyn_cast<ShapedType>(operand.getType());
        shaped && shaped.hasRank() && shaped.hasStaticShape() &&
        shaped.getRank() > 0)
      return SmallVector<int64_t>(shaped.getShape());
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

static Value findBufferOperand(Operation *op) {
  for (Value operand : op->getOperands())
    if (isa<tessera::tile::BufferType>(operand.getType()))
      return operand;
  return {};
}

static Value findPipelineStateOperand(Operation *op) {
  for (Value operand : op->getOperands())
    if (isa<tessera::tile::PipelineStateType>(operand.getType()))
      return operand;
  return {};
}

static Value allocationRoot(Value buffer) {
  while (buffer) {
    Operation *def = buffer.getDefiningOp();
    if (!def)
      break;
    Value parent;
    for (Value operand : def->getOperands())
      if (isa<tessera::tile::BufferType>(operand.getType())) {
        parent = operand;
        break;
      }
    if (!parent)
      break;
    buffer = parent;
  }
  return buffer;
}

static int64_t storageBits(Operation *op) {
  auto policy = op->getAttrOfType<DictionaryAttr>("numeric_policy");
  auto storage = policy ? policy.getAs<StringAttr>("storage") : StringAttr();
  if (!storage)
    return 16;
  StringRef value = storage.getValue();
  if (value == "f64" || value == "i64" || value == "u64")
    return 64;
  if (value == "f32" || value == "i32" || value == "u32")
    return 32;
  if (value == "f16" || value == "bf16" || value == "i16" ||
      value == "u16")
    return 16;
  if (value == "int4" || value == "i4" || value == "uint4" ||
      value == "u4")
    return 4;
  return 8;
}

static int64_t allocationBytes(Operation *op) {
  SmallVector<int64_t> extents = tileExtents(op);
  int64_t elements = 1;
  for (int64_t extent : extents) {
    if (extent <= 0)
      return 1;
    elements *= extent;
  }
  int64_t bits = elements * storageBits(op);
  return std::max<int64_t>((bits + 7) / 8, 1);
}

static Value createLdsAllocation(OpBuilder &builder, FunctionOpInterface func,
                                 Operation *copy) {
  auto layout =
      copy->getAttrOfType<tessera::tile::TileLayoutAttr>("tile.layout");
  if (!layout || func.getFunctionBody().empty())
    return {};
  builder.setInsertionPointToStart(&func.getFunctionBody().front());
  OperationState state(copy->getLoc(), tessera::tile::AllocOp::getOperationName());
  // "smem" is the shared Tile spelling; the ROCm target lowering maps it to
  // address-space-3 LDS and records that physical spelling on its target op.
  state.addAttribute("space", builder.getStringAttr("smem"));
  state.addAttribute("bytes",
                     builder.getI64IntegerAttr(allocationBytes(copy)));
  state.addAttribute("layout", layout);
  state.addAttribute("target", builder.getStringAttr("rocm"));
  state.addTypes(tessera::tile::BufferType::get(builder.getContext()));
  return builder.create(state)->getResult(0);
}

static Value createPipelineState(OpBuilder &builder, FunctionOpInterface func,
                                 StringRef role, int64_t phase) {
  if (func.getFunctionBody().empty())
    return {};
  builder.setInsertionPointToStart(&func.getFunctionBody().front());
  OperationState state(func.getLoc(),
                       tessera::tile::PipelineInitOp::getOperationName());
  state.addAttribute("depth", builder.getI64IntegerAttr(2));
  state.addAttribute("stage", builder.getI64IntegerAttr(0));
  state.addAttribute("phase", builder.getI64IntegerAttr(phase));
  state.addAttribute("role", builder.getStringAttr(role));
  state.addAttribute("target", builder.getStringAttr("rocm"));
  state.addTypes(
      tessera::tile::PipelineStateType::get(builder.getContext()));
  return builder.create(state)->getResult(0);
}

static Value advancePipelineState(OpBuilder &builder, Operation *anchor,
                                  Value state, ValueRange dependencies) {
  builder.setInsertionPointAfter(anchor);
  OperationState advance(
      anchor->getLoc(), tessera::tile::PipelineAdvanceOp::getOperationName());
  advance.addOperands(state);
  for (Value dependency : dependencies)
    if (dependency != state &&
        !isa<tessera::tile::PipelineStateType>(dependency.getType()))
      advance.addOperands(dependency);
  advance.addAttribute("target", builder.getStringAttr("rocm"));
  advance.addTypes(
      tessera::tile::PipelineStateType::get(builder.getContext()));
  return builder.create(advance)->getResult(0);
}

static void materializeDeallocs(OpBuilder &builder, FunctionOpInterface func,
                                ArrayRef<Value> buffers) {
  if (buffers.empty())
    return;
  for (Block &block : func.getFunctionBody()) {
    Operation *terminator = block.getTerminator();
    if (!terminator || terminator->getNumSuccessors() != 0)
      continue;
    builder.setInsertionPoint(terminator);
    for (Value buffer : buffers) {
      OperationState state(terminator->getLoc(),
                           tessera::tile::DeallocOp::getOperationName());
      state.addOperands(buffer);
      state.addAttribute("target", builder.getStringAttr("rocm"));
      builder.create(state);
    }
  }
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
    bool anyError = false;

    // The planner is the single place that resolves async dependencies — and it
    // records them as SSA token edges, not program order. During an ordered walk
    // it decides, per consumer, which copy it depends on (explicit tile.depends_on
    // > SSA value link > most-recently-retired stage, the prefetch->wait->compute
    // idiom); afterwards it mints a !tile.async_token on each copy and threads it
    // into the operands of the wait_async that retires it and the mmas that
    // consume it. The legality pass then verifies the def-use edge instead of
    // re-deriving program order (which is what made a count-based guess able to
    // wrongly reject valid double buffering).
    // Walk every function body — func.func AND gpu.func — so the compiler-
    // generated ROCm kernels (emitted as gpu.func) route through the same
    // wave/LDS planning as hand-tiled func.func IR (Fork A).
    getOperation().walk([&](FunctionOpInterface func) {
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

      // Materialize physical allocation identity before the async-token edges.
      // A pre-existing !tile.buffer wins. Otherwise each structured copy owns
      // a distinct allocation; sharing must be expressed by SSA def-use.
      llvm::StringMap<Value> bufferById;
      SmallVector<Value> createdBuffers;
      for (auto &entry : copies) {
        Operation *copy = entry.second;
        Value buffer = findBufferOperand(copy);
        if (!buffer) {
          buffer = createLdsAllocation(builder, func, copy);
          if (!buffer) {
            // An unshaped pointer copy names externally owned storage and
            // carries its byte count dynamically. The static tile.alloc
            // contract cannot represent that lifetime without inventing a
            // false size.
            continue;
          }
          createdBuffers.push_back(buffer);
          copy->insertOperands(copy->getNumOperands(), {buffer});
        }
        bufferById[entry.first] = allocationRoot(buffer);
      }

      // Materialize the SSA completion edges. Mint a token on each async copy,
      // then thread the token and the matching allocation handle into the
      // wait_async that retires it and the mmas that consume it. The token and
      // buffer ride the ops' Variadic<AnyType> operands, so downstream lowering
      // can consume ownership without changing the portable operation ABI.
      llvm::StringMap<Value> tokenById;
      for (auto &entry : copies) {
        Operation *copy = entry.second;
        tokenById[entry.first] = materializeAsyncToken(builder, copy);
        entry.second = copy;
      }
      for (auto &wr : waitRetire) {
        auto it = tokenById.find(wr.second);
        if (it != tokenById.end())
          wr.first->insertOperands(wr.first->getNumOperands(), {it->second});
        auto buffer = bufferById.find(wr.second);
        if (buffer != bufferById.end())
          wr.first->insertOperands(wr.first->getNumOperands(),
                                   {buffer->second});
      }
      for (auto &mc : mmaConsumes) {
        SmallVector<Value> dependencies;
        llvm::SmallPtrSet<Value, 8> seen;
        for (const std::string &id : mc.second) {
          auto it = tokenById.find(id);
          if (it != tokenById.end() && seen.insert(it->second).second)
            dependencies.push_back(it->second);
          auto buffer = bufferById.find(id);
          if (buffer != bufferById.end() &&
              seen.insert(buffer->second).second)
            dependencies.push_back(buffer->second);
        }
        if (!dependencies.empty())
          mc.first->insertOperands(mc.first->getNumOperands(), dependencies);
      }

      if (anyError)
        return;

      // Thread architecture-neutral producer/consumer pipeline state through
      // the AMD operations. The state values establish phase ownership; async
      // tokens establish completion; buffer handles establish allocation
      // identity. ROCm target lowering later consumes this proof and maps the
      // waits to vmcnt/s_barrier semantics.
      bool needsProducerState = llvm::any_of(copies, [](const auto &entry) {
        return !findPipelineStateOperand(entry.second);
      });
      bool needsConsumerState =
          llvm::any_of(waitRetire, [](const auto &entry) {
            return !findPipelineStateOperand(entry.first);
          }) ||
          llvm::any_of(mmaConsumes, [](const auto &entry) {
            return !findPipelineStateOperand(entry.first);
          });
      Value producerRoot;
      if (needsProducerState)
        producerRoot =
            createPipelineState(builder, func, "producer", /*phase=*/1);
      Value consumerRoot;
      if (needsConsumerState)
        consumerRoot =
            createPipelineState(builder, func, "consumer", /*phase=*/0);
      llvm::DenseMap<Block *, Value> producerByBlock;
      llvm::DenseMap<Block *, Value> consumerByBlock;
      SmallVector<Operation *> ordered;
      func.walk<WalkOrder::PreOrder>([&](Operation *op) {
        StringRef name = op->getName().getStringRef();
        if (name == "tile.async_copy" || name == "tile.wait_async" ||
            name == "tile.mma")
          ordered.push_back(op);
      });
      for (Operation *op : ordered) {
        // Shared GEMM/attention formation may already own and thread this
        // state. Preserve that chain instead of appending a competing ROCm
        // state machine.
        if (findPipelineStateOperand(op))
          continue;
        StringRef name = op->getName().getStringRef();
        Value root =
            name == "tile.async_copy" ? producerRoot : consumerRoot;
        if (!root)
          continue;
        auto &states = name == "tile.async_copy" ? producerByBlock
                                                  : consumerByBlock;
        Value &state = states[op->getBlock()];
        if (!state)
          state = root;
        op->insertOperands(op->getNumOperands(), {state});
        SmallVector<Value> dependencies(op->getOperands().begin(),
                                        op->getOperands().end());
        dependencies.append(op->getResults().begin(), op->getResults().end());
        state = advancePipelineState(builder, op, state, dependencies);
      }

      materializeDeallocs(builder, func, createdBuffers);
    });

    if (anyError)
      signalPassFailure();
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

      if (Value buffer = findBufferOperand(op)) {
        Value root = allocationRoot(buffer);
        if (auto alloc = root.getDefiningOp<tessera::tile::AllocOp>();
            alloc && alloc.getSpace() == "tmem") {
          op->emitOpError(
              "ROCm cannot consume a !tile.buffer allocated in tmem; use "
              "shared-memory allocation identity, which lowers to LDS");
          anyError = true;
        }
      }

      bool structuredLds = false;
      if (auto layout =
              op->getAttrOfType<tessera::tile::TileLayoutAttr>("tile.layout"))
        structuredLds = hasLdsAxis(layout);
      if (Value buffer = allocationRoot(findBufferOperand(op)))
        if (auto alloc = buffer.getDefiningOp<tessera::tile::AllocOp>())
          structuredLds = structuredLds || alloc.getSpace() == "smem";
      if (!structuredLds)
        return;

      bool hasBuffer = static_cast<bool>(findBufferOperand(op));
      bool hasTokenOperand = llvm::any_of(op->getOperands(), [](Value value) {
        return isa<tessera::tile::AsyncTokenType>(value.getType());
      });
      bool hasTokenResult = llvm::any_of(op->getResults(), [](Value value) {
        return isa<tessera::tile::AsyncTokenType>(value.getType());
      });
      bool hasPipelineState =
          llvm::any_of(op->getOperands(), [](Value value) {
            return isa<tessera::tile::PipelineStateType>(value.getType());
          });
      if (name == "tile.async_copy" &&
          (!hasBuffer || !hasTokenResult || !hasPipelineState)) {
        op->emitOpError(
            "structured ROCm LDS copy must own !tile.buffer, "
            "!tile.async_token, and !tile.pipeline_state SSA edges");
        anyError = true;
      }
      if ((name == "tile.wait_async" || name == "tile.mma") &&
          (!hasBuffer || !hasTokenOperand || !hasPipelineState)) {
        op->emitOpError(
            "structured ROCm LDS consumer must carry !tile.buffer, "
            "!tile.async_token, and !tile.pipeline_state SSA edges");
        anyError = true;
      }
    });

    getOperation().walk([&](FunctionOpInterface func) {
      // SSA token model: an async copy mints a !tile.async_token result; a
      // wait_async / s_barrier retires it; an mma's token operands name exactly
      // the stages it consumes. Legality is then a pure def-use check — every
      // token an mma consumes must already be retired — with NO program-order
      // re-derivation. The planner encoded the dependency as SSA, so a live
      // prefetch can never be mistaken for a dependency (the over-rejection the
      // old count-based guess produced is structurally impossible here). The
      // string `outstanding` set + pendingLdsWrites remain for the C2 LDS
      // write/write check and a conservative fallback on token-less IR.
      //
      // Loop handling: this is a single program-order PreOrder walk (it descends
      // into scf.for/while bodies but visits each body once). That is sound and
      // conservative for legality — an in-body copy/wait/mma chain is checked by
      // its SSA token results, and a loop-carried token arrives as a block
      // argument (handled above: assumed resident, never false-rejected). It is
      // intentionally NOT an iterative dataflow fixpoint; cross-iteration LDS
      // reuse hazards are the domain of the write/write check below.
      llvm::SmallPtrSet<Value, 8> outstandingTokens; // minted, not retired
      llvm::SmallPtrSet<Value, 8> retiredTokens;     // waited or drained
      SmallVector<std::string> outstanding;          // barrier ids (fallback)
      // Token-less fallback CONTRACT (intentional, see wave_lds_depends_on_
      // legality.mlir @double_buffer_inferred): a token-less / depends_on-less
      // mma is assumed to consume the MOST-RECENTLY-RETIRED stage (the
      // prefetch->wait->compute idiom). So the only hazard this fallback flags is
      // "copies in flight and NOTHING has ever been retired" — there is no
      // resident stage for the mma to consume yet. Once any wait has retired a
      // stage, a token-less mma is assumed to read it; a live prefetch issued
      // afterwards (the next-iteration stage) is NOT its dependency. This is
      // deliberately permissive: it does NOT try to disambiguate which stage a
      // bare mma reads (impossible token-less), and it does NOT need to — a
      // buffer that an unwaited prefetch clobbers is caught independently by the
      // LDS write/write (OVERLAPPING_WRITE) check below. Precise per-stage
      // checking requires threading a !tile.async_token or annotating
      // tile.depends_on (both handled above, before this fallback).
      bool sawAnyWait = false;
      unsigned synth = 0;
      llvm::DenseMap<Value, PendingWrite> pendingSsaLdsWrites;

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
          // Retire only the allocation named by the SSA wait edge. A token-less
          // wait has no precise allocation identity, so it keeps the
          // conservative "drain all" behavior.
          if (Value waitedBuffer =
                  allocationRoot(findBufferOperand(op)))
            pendingSsaLdsWrites.erase(waitedBuffer);
          else
            pendingSsaLdsWrites.clear();
          return;
        }
        if (isSBarrier(op)) {
          for (Value t : outstandingTokens) // workgroup barrier drains all.
            retiredTokens.insert(t);
          outstandingTokens.clear();
          outstanding.clear();
          pendingSsaLdsWrites.clear();
          return;
        }

        if (name == "tile.mma") {
          // Precise path: the mma's token operands are exactly the stages it
          // consumes; each must already be retired. No program-order guess.
          bool hasTokenOperand = false;
          for (Value operand : op->getOperands())
            if (isToken(operand)) {
              hasTokenOperand = true;
              // Loop-carried token (a block argument, e.g. an scf.for iter_arg):
              // its producer/retirement live on the back-edge, which this single-
              // visit walk cannot follow. Treat it as resident rather than false-
              // reject the pipelined loop; the in-body copy/wait/mma edges are
              // still checked precisely from their SSA results below.
              if (isa<BlockArgument>(operand))
                continue;
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

        auto layout =
            op->getAttrOfType<tessera::tile::TileLayoutAttr>("tile.layout");
        if (!layout || !hasLdsAxis(layout))
          return;
        auto fp = storageFootprint(layout);
        Value buffer = allocationRoot(findBufferOperand(op));
        if (!buffer)
          return;

        PendingWrite *previous = nullptr;
        auto it = pendingSsaLdsWrites.find(buffer);
        if (it != pendingSsaLdsWrites.end())
          previous = &it->second;
        if (fp && previous) {
          auto prev = storageFootprint(previous->layout);
          if (prev && overlaps(*prev, *fp)) {
            InFlightDiagnostic diag =
                op->emitOpError("ROCM_WAVE_LDS_OVERLAPPING_WRITE: LDS ");
            diag << "SSA allocation";
            diag << " is written over an overlapping layout region with no "
                    "intervening waitcnt/barrier.";
            diag.attachNote(previous->op->getLoc())
                << "previous write to the same LDS buffer";
            anyError = true;
          }
        }
        pendingSsaLdsWrites[buffer] = PendingWrite{op, layout};
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
