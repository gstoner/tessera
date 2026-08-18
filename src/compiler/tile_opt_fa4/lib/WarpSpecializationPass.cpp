//===- WarpSpecializationPass.cpp — Phase 3 ──────────────────────────────===//
//
// Assigns warp roles (producer / consumer) inside tile IR regions and
// synchronizes them through `!tile.pipeline_state` + `!tile.async_token`
// SSA chains (tile.pipeline_init / tile.pipeline_advance).
//
// Structural rules:
//   1. tile.async_copy ops → emitted in the PRODUCER warp region.
//   2. tile.mma + tessera_attn.* ops → emitted in the CONSUMER warp region.
//   3. Each warp region owns a `tile.pipeline_init` state value; cross-role
//      ordering flows through `tile.pipeline_advance` dependencies.
//
// This models the SM_90 Hopper warp-specialization programming model where:
//   - Producer warps issue TMA loads (cp.async.bulk.tensor) and signal mbarrier
//   - Consumer warps (warpgroup) wait on mbarrier, then run WGMMA
//
// Output IR structure:
//
//   schedule.warp role="producer" {tile.pipeline = ...} {
//     %ps = tile.pipeline_init {role = "producer", ...}
//     %tok = tile.async_copy(...)
//     %ps' = tile.pipeline_advance %ps, %tok
//   }
//   schedule.warp role="consumer" {tile.pipeline = ...} {
//     %ps = tile.pipeline_init {role = "consumer", ...}
//     tile.mma(...)
//     %ps' = tile.pipeline_advance %ps, ...
//   }
//
// (An earlier draft of this header described a tessera.queue.create/push/pop
// mechanism this pass never emitted; the queue dialect was deleted 2026-08-10
// under Decisions #29/#31.)
//
// Registration: --tessera-warp-specialization
//===----------------------------------------------------------------------===//

#include "Tessera/Dialect/Tile/TileDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <string>

using namespace mlir;

namespace tessera {

namespace {

// Give a producer async copy a !tile.async_token result — the SSA completion
// edge a consumer mma reads. Idempotent (returns an existing token result).
// Results are immutable, so recreate the op with the token appended, RAUW the
// original results, and erase the old; `copy` is updated to the new op so the
// caller can keep its producer bookkeeping consistent. The new op is inserted at
// the old op's position, preserving program order (and dominance over the mma).
static Value mintAsyncToken(OpBuilder &b, Operation *&copy) {
  auto tokTy = tile::AsyncTokenType::get(b.getContext());
  for (Value r : copy->getResults())
    if (r.getType() == tokTy)
      return r;
  b.setInsertionPoint(copy);
  SmallVector<Type> resultTypes(copy->getResultTypes().begin(),
                               copy->getResultTypes().end());
  resultTypes.push_back(tokTy);
  OperationState st(copy->getLoc(), copy->getName().getStringRef());
  st.addOperands(copy->getOperands());
  st.addTypes(resultTypes);
  st.addAttributes(copy->getAttrs());
  Operation *grown = b.create(st);
  for (unsigned i = 0, e = copy->getNumResults(); i < e; ++i)
    copy->getResult(i).replaceAllUsesWith(grown->getResult(i));
  copy->erase();
  copy = grown;
  return grown->getResult(grown->getNumResults() - 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Classify ops into producer (async copy) vs consumer (compute) buckets.
// ─────────────────────────────────────────────────────────────────────────────
static bool isProducerOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "tile.async_copy" || name == "tile.wait_async";
}

static bool isConsumerOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name.starts_with("tessera_attn.") || name == "tile.mma";
}

// Logical pipeline identity remains an extensible scheduling attribute.  Warp
// role is the required property of the registered schedule.warp op; do not
// duplicate it as the legacy tile.warp_role ancestor marker. Pipeline state is
// exclusively SSA: tile.pipeline_init/advance carry ownership and ordering.
static void stampPipelineMarkers(OpBuilder &b, Operation *warpOp,
                                 StringRef pipelineId) {
  warpOp->setAttr("tile.pipeline", b.getStringAttr(pipelineId));
}

static Value createLogicalRole(OpBuilder &b, Location loc, StringRef name,
                               StringRef kind, ArrayRef<StringRef> members) {
  OperationState state(loc, tile::RoleOp::getOperationName());
  state.addAttribute("name", b.getStringAttr(name));
  state.addAttribute("kind", b.getStringAttr(kind));
  SmallVector<Attribute> memberAttrs;
  llvm::transform(members, std::back_inserter(memberAttrs),
                  [&](StringRef member) { return b.getStringAttr(member); });
  state.addAttribute("members", b.getArrayAttr(memberAttrs));
  state.addTypes(tile::RoleType::get(b.getContext()));
  return b.create(state)->getResult(0);
}

static Value createPipelineState(OpBuilder &b, Location loc, Value logicalRole,
                                 StringRef role, int64_t phase) {
  OperationState state(loc, "tile.pipeline_init");
  state.addOperands(logicalRole);
  state.addTypes(tile::PipelineStateType::get(b.getContext()));
  state.addAttribute("depth", b.getI64IntegerAttr(2));
  state.addAttribute("stage", b.getI64IntegerAttr(0));
  state.addAttribute("phase", b.getI64IntegerAttr(phase));
  state.addAttribute("role", b.getStringAttr(role));
  return b.create(state)->getResult(0);
}

static Value advancePipelineState(OpBuilder &b, Location loc, Value state,
                                  ValueRange dependencies) {
  OperationState advance(loc, "tile.pipeline_advance");
  advance.addOperands(state);
  advance.addOperands(dependencies);
  advance.addTypes(tile::PipelineStateType::get(b.getContext()));
  return b.create(advance)->getResult(0);
}

// The staged-tile extents — from the op's tile_rows/tile_cols attrs (async_copy)
// or its rank-2 static result shape (mma accumulator). Empty if unknown.
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

// C1/C2 join: stamp a buffer write (tile.access="write" + tile.buffer + a
// row-major #tile.layout on the given storage axes) so TileBarrierReuseLegality
// (C2) runs on real lowering output and C1's layout vocabulary appears on the
// staged shared-memory / TMEM tiles. axisNames must match the tile rank (2).
static tile::TileLayoutAttr
layoutForExtents(OpBuilder &b, ArrayRef<int64_t> extents,
                 ArrayRef<StringRef> axisNames) {
  if (extents.empty() || extents.size() != axisNames.size() ||
      llvm::any_of(extents, [](int64_t extent) { return extent <= 0; })) {
    SmallVector<StringAttr> fallbackAxes = {b.getStringAttr("m")};
    return tile::TileLayoutAttr::get(
        b.getContext(), ArrayRef<int64_t>{1}, ArrayRef<int64_t>{1},
        fallbackAxes, /*replicaCounts=*/{}, /*replicaStrides=*/{},
        /*replicaAxes=*/{}, /*offset=*/0,
        /*swizzle=*/tile::TileSwizzleAttr());
  }
  SmallVector<int64_t> strides(extents.size(), 1);
  for (int i = static_cast<int>(extents.size()) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * extents[i + 1];
  SmallVector<StringAttr> axes;
  for (StringRef axis : axisNames)
    axes.push_back(b.getStringAttr(axis));
  return tile::TileLayoutAttr::get(
      b.getContext(), extents, strides, axes, /*replicaCounts=*/{},
      /*replicaStrides=*/{}, /*replicaAxes=*/{}, /*offset=*/0,
      /*swizzle=*/tile::TileSwizzleAttr());
}

static void attachWriteLayout(OpBuilder &b, Operation *op,
                              ArrayRef<int64_t> extents,
                              ArrayRef<StringRef> axisNames) {
  op->setAttr("tile.layout", layoutForExtents(b, extents, axisNames));
}

static Value createBufferAllocation(OpBuilder &b, Location loc,
                                    StringRef space,
                                    ArrayRef<int64_t> extents,
                                    ArrayRef<StringRef> axisNames,
                                    int64_t elementBytes) {
  int64_t bytes = elementBytes;
  for (int64_t extent : extents)
    if (extent > 0)
      bytes *= extent;
  OperationState alloc(loc, tile::AllocOp::getOperationName());
  alloc.addAttribute("space", b.getStringAttr(space));
  alloc.addAttribute("bytes", b.getI64IntegerAttr(std::max<int64_t>(bytes, 1)));
  alloc.addAttribute("layout", layoutForExtents(b, extents, axisNames));
  alloc.addTypes(tile::BufferType::get(b.getContext()));
  return b.create(alloc)->getResult(0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass
// ─────────────────────────────────────────────────────────────────────────────

struct WarpSpecializationPass
    : public PassWrapper<WarpSpecializationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WarpSpecializationPass)

  StringRef getArgument() const override {
    return "tessera-warp-specialization";
  }
  StringRef getDescription() const override {
    return "Assign producer/consumer warp roles; thread !tile.pipeline_state "
           "SSA chains between them";
  }
  // C3 join: the pass constructs typed buffer and !tile.pipeline_state SSA
  // values, so the Tile dialect must be loaded.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tessera::tile::TesseraTileDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getContext());

    SmallVector<Operation *> regionOps;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "schedule.mesh.region")
        regionOps.push_back(op);
    });

    unsigned regionIndex = 0;
    for (Operation *regionOp : regionOps) {
      Region &body = regionOp->getRegion(0);
      if (body.empty())
        continue;
      Block &entryBlock = body.front();
      Location loc = regionOp->getLoc();

      // Canonical streaming attention already carries producer/consumer
      // !tile.pipeline_state values through its KV scf.for. Splitting only the
      // entry block would strand the loop body across warp regions, so leave
      // this explicitly stateful loop intact for target lowering.
      bool hasStreamingAttention = false;
      regionOp->walk([&](scf::ForOp loop) {
        if (loop->hasAttr("tessera.streaming_attention"))
          hasStreamingAttention = true;
      });
      if (hasStreamingAttention)
        continue;

      SmallVector<Operation *> producerOps, consumerOps, otherOps;
      for (Operation &op : entryBlock) {
        if (op.hasTrait<OpTrait::IsTerminator>())
          continue; // leave the region terminator in place
        if (isProducerOp(&op))
          producerOps.push_back(&op);
        else if (isConsumerOp(&op))
          consumerOps.push_back(&op);
        else
          otherOps.push_back(&op);
      }
      if (producerOps.empty() || consumerOps.empty())
        continue;

      // One pipeline id per specialized region, shared by its producer +
      // consumer so C3 pairs them.
      std::string pipelineId = ("warpspec." + Twine(regionIndex++)).str();

      DenseSet<Operation *> prodSet(producerOps.begin(), producerOps.end());
      DenseSet<Operation *> consSet(consumerOps.begin(), consumerOps.end());

      // ── Auto-mint the async-completion token (Phase C-NV) ─────────────────
      // For each consumer mma, thread the !tile.async_token of every producer
      // async_copy whose result it reads as an explicit operand. This makes the
      // cross-region copy→mma *synchronization* dependency a first-class SSA
      // def-use edge (the prodCross threading below then carries it across the
      // warp boundary), not merely the data edge — the NV analogue of the ROCm
      // token edge. The dependency is read straight off the mma's data operands
      // (no program-order guess). Recreating a copy to add the token result is
      // reflected back into producerOps/prodSet so later region surgery is sound.
      for (Operation *c : consumerOps) {
        if (c->getName().getStringRef() != "tile.mma")
          continue;
        // Producer copies this mma reads via a *data* operand, and the tokens it
        // already consumes (so a pre-threaded edge is not duplicated).
        SmallVector<Operation *> dataDeps;
        DenseSet<Operation *> seen;
        DenseSet<Value> haveTokens;
        for (Value operand : c->getOperands()) {
          Operation *def = operand.getDefiningOp();
          if (!def || !prodSet.contains(def))
            continue;
          if (isa<tile::AsyncTokenType>(operand.getType())) {
            haveTokens.insert(operand);
          } else if (def->getName().getStringRef() == "tile.async_copy" &&
                     seen.insert(def).second) {
            dataDeps.push_back(def);
          }
        }
        SmallVector<Value> tokens;
        for (Operation *def : dataDeps) {
          Operation *grown = def;
          Value tok = mintAsyncToken(b, grown);
          if (grown != def) {
            prodSet.erase(def);
            prodSet.insert(grown);
            for (Operation *&p : producerOps)
              if (p == def)
                p = grown;
          }
          if (!haveTokens.contains(tok))
            tokens.push_back(tok);
        }
        if (!tokens.empty())
          c->insertOperands(c->getNumOperands(), tokens);
      }

      // ── Cross-boundary value flow ─────────────────────────────────────────
      // producer→consumer: producer results that consumer ops read.  These must
      // become parent-level SSA values (the producer warp region's results) so
      // they dominate the sibling consumer region.
      SmallVector<Value> prodCross;
      DenseSet<Value> prodSeen;
      for (Operation *c : consumerOps)
        for (Value v : c->getOperands())
          if (Operation *def = v.getDefiningOp())
            if (prodSet.contains(def) && prodSeen.insert(v).second)
              prodCross.push_back(v);

      // consumer→outside: consumer results used outside the consumer set (e.g.
      // the region terminator).  These become the consumer warp's results.
      SmallVector<Value> consCross;
      DenseSet<Value> consSeen;
      for (Operation *c : consumerOps)
        for (Value res : c->getResults())
          for (Operation *user : res.getUsers())
            if (!consSet.contains(user) && consSeen.insert(res).second)
              consCross.push_back(res);

      // ── Producer warp region (yields prodCross) ───────────────────────────
      b.setInsertionPointToStart(&entryBlock);
      std::string producerRoleName = pipelineId + ".producer";
      std::string consumerRoleName = pipelineId + ".consumer";
      Value producerRole = createLogicalRole(
          b, loc, producerRoleName, "producer",
          {StringRef("async_copy")});
      Value consumerRole = createLogicalRole(
          b, loc, consumerRoleName, "consumer",
          {StringRef("mma")});
      OperationState prodSt(loc, "schedule.warp");
      prodSt.addAttribute("role", b.getStringAttr("producer"));
      prodSt.addRegion();
      for (Value v : prodCross)
        prodSt.addTypes(v.getType());
      Operation *prodWarp = b.create(prodSt);
      stampPipelineMarkers(b, prodWarp, pipelineId);

      // Hoist the consumer-needed "other" ops (e.g. constants) above the warp
      // regions so they dominate both — they only depend on region-external
      // values, so they are safe to move to the top.
      for (Operation *o : otherOps) {
        bool dependsOnSplit = llvm::any_of(o->getOperands(), [&](Value v) {
          Operation *def = v.getDefiningOp();
          return def && (prodSet.contains(def) || consSet.contains(def));
        });
        if (!dependsOnSplit)
          o->moveBefore(prodWarp);
      }

      // Allocate physical buffers in the parent region so their SSA handles
      // dominate both warp regions and the post-sync deallocation epilogue.
      SmallVector<Value> regionBuffers;
      b.setInsertionPoint(prodWarp);
      for (Operation *producer : producerOps) {
        if (producer->getName().getStringRef() != "tile.async_copy")
          continue;
        SmallVector<int64_t> extents = tileExtents(producer);
        Value buffer = createBufferAllocation(
            b, producer->getLoc(), "smem", extents,
            {StringRef("m"), StringRef("m")}, /*elementBytes=*/2);
        producer->insertOperands(producer->getNumOperands(), buffer);
        regionBuffers.push_back(buffer);
      }
      for (Operation *consumer : consumerOps) {
        if (consumer->getName().getStringRef() != "tile.mma")
          continue;
        SmallVector<int64_t> extents = tileExtents(consumer);
        Value buffer = createBufferAllocation(
            b, consumer->getLoc(), "tmem", extents,
            {StringRef("tlane"), StringRef("tcol")}, /*elementBytes=*/4);
        consumer->insertOperands(consumer->getNumOperands(), buffer);
        regionBuffers.push_back(buffer);
      }

      Block *prodBody = b.createBlock(&prodWarp->getRegion(0));
      b.setInsertionPointToEnd(prodBody);
      Value producerState =
          createPipelineState(b, loc, producerRole, "producer", /*phase=*/1);
      // Each tile.async_copy stages into its own shared-memory tile (linear `m`
      // axis); distinct buffers, so C2 sees no aliasing on well-formed lowering.
      for (Operation *p : producerOps) {
        if (p->getName().getStringRef() == "tile.async_copy") {
          attachWriteLayout(b, p, tileExtents(p),
                            {StringRef("m"), StringRef("m")});
        }
        p->moveBefore(prodBody, prodBody->end());
        producerState = advancePipelineState(
            b, p->getLoc(), producerState, p->getResults());
      }
      OperationState prodYield(loc, "schedule.yield");
      prodYield.addOperands(prodCross);
      b.create(prodYield);

      // Rewire consumer uses of producer results to the producer warp results.
      for (auto [i, v] : llvm::enumerate(prodCross))
        v.replaceUsesWithIf(prodWarp->getResult(i), [&](OpOperand &use) {
          return consSet.contains(use.getOwner());
        });

      // ── Consumer warp region (yields consCross) ───────────────────────────
      b.setInsertionPointAfter(prodWarp);
      OperationState consSt(loc, "schedule.warp");
      consSt.addAttribute("role", b.getStringAttr("consumer"));
      consSt.addRegion();
      for (Value v : consCross)
        consSt.addTypes(v.getType());
      Operation *consWarp = b.create(consSt);
      stampPipelineMarkers(b, consWarp, pipelineId);

      Block *consBody = b.createBlock(&consWarp->getRegion(0));
      b.setInsertionPointToEnd(consBody);
      Value consumerState =
          createPipelineState(b, loc, consumerRole, "consumer", /*phase=*/0);
      // Each tile.mma writes its accumulator to a TMEM tile (tlane/tcol axes).
      for (Operation *c : consumerOps) {
        if (c->getName().getStringRef() == "tile.mma") {
          attachWriteLayout(b, c, tileExtents(c),
                            {StringRef("tlane"), StringRef("tcol")});
        }
        c->moveBefore(consBody, consBody->end());
        consumerState = advancePipelineState(
            b, c->getLoc(), consumerState, c->getResults());
      }
      OperationState consYield(loc, "schedule.yield");
      consYield.addOperands(consCross);
      b.create(consYield);

      // Rewire external uses of consumer results to the consumer warp results.
      for (auto [i, v] : llvm::enumerate(consCross))
        v.replaceUsesWithIf(consWarp->getResult(i), [&](OpOperand &use) {
          Operation *owner = use.getOwner();
          // Newly created pipeline-state ops also consume the value inside the
          // consumer region.  They must keep using the region-local producer;
          // the warp result is only defined outside this operation and cannot
          // dominate a use nested in its own body.
          return !consSet.contains(owner) && owner != consWarp &&
                 !consWarp->isProperAncestor(owner);
        });

      // ── Writeback-dealloc epilogue ────────────────────────────────────────
      // After the consumer warpgroup drains its accumulators, the region's
      // staged buffers are freed. A `tile.cta_sync` must precede the frees so
      // every warp has finished reading them (WARPSPEC_USE_AFTER_FREE, the C6
      // use-after-free invariant). Emitted before the region terminator.
      if (Operation *term = entryBlock.getTerminator()) {
        b.setInsertionPoint(term);
        b.create(OperationState(loc, "tile.cta_sync"));
        for (Value buffer : regionBuffers) {
          OperationState freeSt(loc, tile::DeallocOp::getOperationName());
          freeSt.addOperands(buffer);
          b.create(freeSt);
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createWarpSpecializationPass() {
  return std::make_unique<WarpSpecializationPass>();
}

} // namespace tessera
