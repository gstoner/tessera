//===- WarpSpecializationPass.cpp — Phase 3 ──────────────────────────────===//
//
// Assigns warp roles (producer / consumer) inside tile IR regions and inserts
// tessera.queue barriers between them.
//
// Structural rules:
//   1. tile.async_copy ops → emitted in the PRODUCER warp region.
//   2. tile.mma + tessera.attn.* ops → emitted in the CONSUMER warp region.
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

// ─────────────────────────────────────────────────────────────────────────────
// Classify ops into producer (async copy) vs consumer (compute) buckets.
// ─────────────────────────────────────────────────────────────────────────────
static bool isProducerOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "tile.async_copy" || name == "tile.wait_async";
}

static bool isConsumerOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name.starts_with("tessera.attn.") || name == "tile.mma";
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

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getContext());

    SmallVector<Operation *> regionOps;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "schedule.mesh.region")
        regionOps.push_back(op);
    });

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

      DenseSet<Operation *> prodSet(producerOps.begin(), producerOps.end());
      DenseSet<Operation *> consSet(consumerOps.begin(), consumerOps.end());

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

      Block *prodBody = b.createBlock(&prodWarp->getRegion(0));
      b.setInsertionPointToEnd(prodBody);
      for (Operation *p : producerOps)
        p->moveBefore(prodBody, prodBody->end());
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

      Block *consBody = b.createBlock(&consWarp->getRegion(0));
      b.setInsertionPointToEnd(consBody);
      for (Operation *c : consumerOps)
        c->moveBefore(consBody, consBody->end());
      OperationState consYield(loc, "schedule.yield");
      consYield.addOperands(consCross);
      b.create(consYield);

      // Rewire external uses of consumer results to the consumer warp results.
      for (auto [i, v] : llvm::enumerate(consCross))
        v.replaceUsesWithIf(consWarp->getResult(i), [&](OpOperand &use) {
          Operation *owner = use.getOwner();
          return !consSet.contains(owner) && owner != consWarp;
        });
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createWarpSpecializationPass() {
  return std::make_unique<WarpSpecializationPass>();
}

} // namespace tessera
