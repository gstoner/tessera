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
//   tessera.schedule.warp {role = "producer"} {
//     tile.async_copy(...)
//     %q = tessera.queue.create
//     tessera.queue.push %q, %tile
//   }
//   tessera.schedule.warp {role = "consumer"} {
//     %q  = tessera.queue.create
//     %t  = tessera.queue.pop %q, %dep
//     tile.mma(%t, ...)
//   }
//
// Registration: --tessera-warp-specialization
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace tessera {

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Helper: create a tessera.schedule.warp region op with a given role attr.
// ─────────────────────────────────────────────────────────────────────────────
static Operation *createWarpRegion(OpBuilder &b, Location loc,
                                    StringRef role) {
  OperationState st(loc, "tessera.schedule.warp");
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
  return name.startswith("tessera.attn.") || name == "tile.mma";
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

    mod.walk([&](Operation *regionOp) {
      // Only process schedule.mesh.region ops or functions containing tile IR.
      if (regionOp->getName().getStringRef() != "schedule.mesh.region")
        return;

      // Collect all ops in the region body.
      auto &body = regionOp->getRegion(0);
      if (body.empty())
        return;

      Block &entryBlock = body.front();
      SmallVector<Operation *> producerOps, consumerOps, otherOps;

      for (auto &op : llvm::make_early_inc_range(entryBlock)) {
        if (isProducerOp(&op))
          producerOps.push_back(&op);
        else if (isConsumerOp(&op))
          consumerOps.push_back(&op);
        else
          otherOps.push_back(&op);
      }

      // Nothing to specialise if we have no separation.
      if (producerOps.empty() || consumerOps.empty())
        return;

      // Build producer warp region.
      b.setInsertionPointToStart(&entryBlock);
      Location loc = regionOp->getLoc();

      Operation *prodWarp = createWarpRegion(b, loc, "producer");
      Block *prodBody = b.createBlock(&prodWarp->getRegion(0));
      b.setInsertionPointToEnd(prodBody);

      // Create a queue for handoff between producer and consumer.
      OperationState qState(loc, "tessera.queue.create");
      qState.addTypes(b.getType<NoneType>()); // opaque queue type placeholder
      Operation *qCreate = b.create(qState);
      Value q = qCreate->getResult(0);

      // Move producer ops into the producer warp region.
      for (Operation *pop : producerOps) {
        pop->moveBefore(b.getBlock(), b.getInsertionPoint());
      }

      // Push the last async result into the queue.
      if (!producerOps.empty()) {
        Operation *lastProd = producerOps.back();
        if (!lastProd->getResults().empty()) {
          OperationState pushState(loc, "tessera.queue.push");
          pushState.addOperands({q, lastProd->getResult(0)});
          pushState.addTypes(b.getType<NoneType>()); // TokenType placeholder
          b.create(pushState);
        }
      }
      // Terminate producer region.
      OperationState yieldP(loc, "schedule.yield");
      b.create(yieldP);

      // Build consumer warp region.
      b.setInsertionPointAfter(prodWarp);
      Operation *consWarp = createWarpRegion(b, loc, "consumer");
      Block *consBody = b.createBlock(&consWarp->getRegion(0));
      b.setInsertionPointToEnd(consBody);

      // Pop from the queue before compute ops.
      if (!producerOps.empty()) {
        Value sentinel = b.create<arith::ConstantIndexOp>(loc, 0)->getResult(0);
        OperationState popState(loc, "tessera.queue.pop");
        popState.addOperands({q, sentinel});
        if (!producerOps.back()->getResults().empty())
          popState.addTypes(producerOps.back()->getResult(0).getType());
        b.create(popState);
      }

      // Move consumer ops into the consumer warp region.
      for (Operation *cop : consumerOps)
        cop->moveBefore(b.getBlock(), b.getInsertionPoint());

      // Terminate consumer region.
      OperationState yieldC(loc, "schedule.yield");
      b.create(yieldC);
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createWarpSpecializationPass() {
  return std::make_unique<WarpSpecializationPass>();
}

} // namespace tessera
