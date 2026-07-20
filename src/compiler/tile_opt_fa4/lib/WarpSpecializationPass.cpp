//===- WarpSpecializationPass.cpp — Phase 3 ──────────────────────────────===//
//
// Assigns warp roles (producer / consumer) inside tile IR regions and inserts
// tessera.queue barriers between them.
//
// Structural rules:
//   1. tile.async_copy ops → emitted in the PRODUCER warp region.
//   2. tile.mma + tessera_attn.* ops → emitted in the CONSUMER warp region.
//   3. A tessera.queue.create / push / pop triple separates the two regions.
//
// This models the SM_90 Hopper warp-specialization programming model where:
//   - Producer warps issue TMA loads (cp.async.bulk.tensor) and signal mbarrier
//   - Consumer warps (warpgroup) wait on mbarrier, then run WGMMA
//
// Output IR structure:
//
//   schedule.warp {role = "producer"} {
//     tile.async_copy(...)
//     %q = tessera.queue.create
//     tessera.queue.push %q, %tile
//   }
//   schedule.warp {role = "consumer"} {
//     %q  = tessera.queue.create
//     %t  = tessera.queue.pop %q, %dep
//     tile.mma(%t, ...)
//   }
//
// Registration: --tessera-warp-specialization
//===----------------------------------------------------------------------===//

#include "Tessera/Dialect/Tile/TileDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <string>

using namespace mlir;

namespace tessera {

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Helper: create a schedule.warp region op with a given role attr.
// ─────────────────────────────────────────────────────────────────────────────
static Operation *createWarpRegion(OpBuilder &b, Location loc,
                                    StringRef role) {
  OperationState st(loc, "schedule.warp");
  st.addAttribute("role", b.getStringAttr(role));
  // Single-block region; caller fills it with ops.
  st.addRegion();
  return b.create(st);
}

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

// C3 join (2026-06-23): stamp the typed PipelineState + warp-role markers that
// TilePipelineLegalityPass (C3) and WarpSpecLegalityPass (C6) verify. The
// producer ring starts at phase 1, the consumer at phase 0 — the asymmetry that
// makes the first wait fall through (the off-by-one ring-deadlock fix). depth=2
// is the default double-buffer; stage 0 is the initial ring slot.
static void stampPipelineMarkers(OpBuilder &b, Operation *warpOp,
                                 StringRef pipelineId, StringRef role,
                                 int64_t phase) {
  warpOp->setAttr("tile.warp_role", b.getStringAttr(role));
  warpOp->setAttr("tile.pipeline", b.getStringAttr(pipelineId));
  warpOp->setAttr("tile.pipeline_state",
                  tile::TilePipelineStateAttr::get(b.getContext(), /*depth=*/2,
                                                   /*stage=*/0, phase, role));
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
static void stampBufferWrite(OpBuilder &b, Operation *op, const Twine &buffer,
                             StringRef space, ArrayRef<int64_t> extents,
                             ArrayRef<StringRef> axisNames) {
  // Typed buffer reference (name + space + access) replaces the old
  // tile.buffer/tile.access string pair.
  op->setAttr("tile.buf", tile::TileBufferRefAttr::get(
                              b.getContext(), buffer.str(), space, "write"));
  if (extents.empty() || extents.size() != axisNames.size())
    return; // buffer stamped; no layout when the shape is unknown.
  // Dynamic / placeholder dims (kDynamic, -1) can't form a legal layout — stamp
  // buffer identity only rather than crash the #tile.layout verifier.
  for (int64_t e : extents)
    if (e <= 0)
      return;
  SmallVector<int64_t> strides(extents.size(), 1);
  for (int i = static_cast<int>(extents.size()) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * extents[i + 1]; // row-major
  SmallVector<StringAttr> axes;
  for (StringRef a : axisNames)
    axes.push_back(b.getStringAttr(a));
  op->setAttr("tile.layout",
              tile::TileLayoutAttr::get(b.getContext(), extents, strides, axes,
                                        /*replicaCounts=*/{},
                                        /*replicaStrides=*/{},
                                        /*replicaAxes=*/{}, /*offset=*/0,
                                        /*swizzle=*/tile::TileSwizzleAttr()));
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
    return "Assign producer/consumer warp roles; insert tessera.queue barriers";
  }
  // C3 join: the pass now constructs #tile.pipeline_state, so the Tile dialect
  // must be loaded.
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
      OperationState prodSt(loc, "schedule.warp");
      prodSt.addAttribute("role", b.getStringAttr("producer"));
      prodSt.addRegion();
      for (Value v : prodCross)
        prodSt.addTypes(v.getType());
      Operation *prodWarp = b.create(prodSt);
      stampPipelineMarkers(b, prodWarp, pipelineId, "producer", /*phase=*/1);

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

      // Buffers allocated this region (name, space) — freed in the epilogue.
      SmallVector<std::pair<std::string, std::string>> regionBuffers;

      Block *prodBody = b.createBlock(&prodWarp->getRegion(0));
      b.setInsertionPointToEnd(prodBody);
      // Each tile.async_copy stages into its own shared-memory tile (linear `m`
      // axis); distinct buffers, so C2 sees no aliasing on well-formed lowering.
      unsigned smemIdx = 0;
      for (Operation *p : producerOps) {
        if (p->getName().getStringRef() == "tile.async_copy") {
          std::string name = (pipelineId + ".smem." + Twine(smemIdx++)).str();
          stampBufferWrite(b, p, name, "smem", tileExtents(p),
                           {StringRef("m"), StringRef("m")});
          regionBuffers.push_back({name, "smem"});
        }
        p->moveBefore(prodBody, prodBody->end());
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
      stampPipelineMarkers(b, consWarp, pipelineId, "consumer", /*phase=*/0);

      Block *consBody = b.createBlock(&consWarp->getRegion(0));
      b.setInsertionPointToEnd(consBody);
      // Each tile.mma writes its accumulator to a TMEM tile (tlane/tcol axes).
      unsigned accIdx = 0;
      for (Operation *c : consumerOps) {
        if (c->getName().getStringRef() == "tile.mma") {
          std::string name =
              (pipelineId + ".tmem.acc." + Twine(accIdx++)).str();
          stampBufferWrite(b, c, name, "tmem", tileExtents(c),
                           {StringRef("tlane"), StringRef("tcol")});
          regionBuffers.push_back({name, "tmem"});
        }
        c->moveBefore(consBody, consBody->end());
      }
      OperationState consYield(loc, "schedule.yield");
      consYield.addOperands(consCross);
      b.create(consYield);

      // Rewire external uses of consumer results to the consumer warp results.
      for (auto [i, v] : llvm::enumerate(consCross))
        v.replaceUsesWithIf(consWarp->getResult(i), [&](OpOperand &use) {
          Operation *owner = use.getOwner();
          return !consSet.contains(owner) && owner != consWarp;
        });

      // ── Writeback-dealloc epilogue ────────────────────────────────────────
      // After the consumer warpgroup drains its accumulators, the region's
      // staged buffers are freed. A `tile.cta_sync` must precede the frees so
      // every warp has finished reading them (WARPSPEC_USE_AFTER_FREE, the C6
      // use-after-free invariant). Emitted before the region terminator.
      if (Operation *term = entryBlock.getTerminator()) {
        b.setInsertionPoint(term);
        b.create(OperationState(loc, "tile.cta_sync"));
        for (const auto &[name, space] : regionBuffers) {
          OperationState freeSt(loc, "tile.buffer_free");
          freeSt.addAttribute("tile.buf", tile::TileBufferRefAttr::get(
                                              b.getContext(), name, space,
                                              "free"));
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
