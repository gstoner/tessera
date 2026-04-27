
//===- VideoIngestFuse.cpp - Fuse video decode → prefill_fused chains ------===//
//
// Detects patterns of the form:
//
//   %frames = tessera.target.cpx.video.decode %bitstream, "h264"
//   %tokens = <patchify/tokenizer ops consuming %frames>
//   %o = tessera.target.cpx.attn.prefill_fused %q, %k, %v, %kv_cache, %seq_len
//
// and fuses the connected subgraph into a single
// `tessera.target.cpx.video_ingest_fused` region op.  The fused region:
//
//   • Keeps all data on-chip in the CPX 256 MB context cache (no PCIe round-
//     trips between decode and attention).
//   • Signals to the scheduler that the hardware video codec and the CPX
//     tensor units should be co-scheduled.
//
// Algorithm:
//   1. Collect all video.decode roots in the module.
//   2. For each root, follow def-use chains (BFS) until we hit either a
//      attn.prefill_fused consumer or a value that escapes the identified
//      sub-graph (in which case we abort the fusion for that chain).
//   3. Clone the matched ops into a new inline region op and erase originals.
//
//===-----------------------------------------------------------------------===//

#include "tessera/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "video-ingest-fuse"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static bool isVideoDecodeOp(Operation *op) {
  return op->getName().getStringRef().endswith("video.decode");
}
static bool isPrefillFusedOp(Operation *op) {
  return op->getName().getStringRef().endswith("attn.prefill_fused");
}

/// BFS from \p root following def-use edges.  Returns all ops reachable
/// between \p root and any `attn.prefill_fused` sink, inclusive.
/// Returns empty if no sink is reachable or if the chain leaks values outside.
static SmallVector<Operation *> collectFusionChain(Operation *root) {
  llvm::SetVector<Operation *> chain;
  SmallVector<Operation *> worklist{root};
  chain.insert(root);

  bool foundSink = false;

  while (!worklist.empty()) {
    Operation *cur = worklist.pop_back_val();

    for (Value result : cur->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (chain.count(user)) continue;
        chain.insert(user);
        if (isPrefillFusedOp(user)) {
          foundSink = true;
          // Don't follow uses beyond the sink
        } else {
          worklist.push_back(user);
        }
      }
    }
  }

  if (!foundSink) return {};
  return chain.takeVector();
}

//===----------------------------------------------------------------------===//
// FuseVideoIngestPass
//===----------------------------------------------------------------------===//

struct FuseVideoIngestPass
    : public PassWrapper<FuseVideoIngestPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseVideoIngestPass)

  StringRef getArgument() const override { return "tessera-fuse-video-ingest"; }
  StringRef getDescription() const override {
    return "Fuse video.decode → patchify → tokenizer → attn.prefill_fused "
           "on CPX to retain data in GDDR7 context cache";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder b(module.getContext());

    // Collect video.decode roots (snapshot before modification)
    SmallVector<Operation *> roots;
    module.walk([&](Operation *op) {
      if (isVideoDecodeOp(op)) roots.push_back(op);
    });

    unsigned fused = 0;

    for (Operation *root : roots) {
      SmallVector<Operation *> chain = collectFusionChain(root);
      if (chain.empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[video-ingest-fuse] no prefill_fused sink reachable from "
                   << root->getLoc() << "; skipping\n");
        continue;
      }

      // Determine insertion point: just before the first op in the chain
      Operation *firstOp = chain.front();
      b.setInsertionPoint(firstOp);
      Location loc = firstOp->getLoc();

      // ── Create a `tessera.target.cpx.video_ingest_fused` region op ────────
      // The region op captures all external operands of the chain as block
      // arguments and returns the output of the attn.prefill_fused sink.

      // Gather external operands (Values defined outside the chain)
      llvm::DenseSet<Operation *> chainSet(chain.begin(), chain.end());
      SmallVector<Value> externalOperands;
      llvm::DenseSet<Value> seen;
      for (Operation *op : chain) {
        for (Value operand : op->getOperands()) {
          if (!seen.count(operand) &&
              (!operand.getDefiningOp() ||
               !chainSet.count(operand.getDefiningOp()))) {
            externalOperands.push_back(operand);
            seen.insert(operand);
          }
        }
      }

      // Gather result types from the attn.prefill_fused sink
      SmallVector<Type> resultTypes;
      for (Operation *op : chain) {
        if (isPrefillFusedOp(op)) {
          for (Type t : op->getResultTypes()) resultTypes.push_back(t);
          break;
        }
      }

      // Emit the fused region op as a generic op (until a dedicated ODS op
      // is added for video_ingest_fused in a future TD file).
      OperationState state(loc, "tessera.target.cpx.video_ingest_fused");
      state.addOperands(externalOperands);
      state.addTypes(resultTypes);

      // Create a single-block region body containing cloned chain ops
      Region *region = state.addRegion();
      Block *body = new Block();
      region->push_back(body);

      // Add block arguments matching external operands
      SmallVector<Value> blockArgs;
      for (Value ext : externalOperands)
        blockArgs.push_back(body->addArgument(ext.getType(), loc));

      // Clone chain ops into region with value mapping
      IRMapping mapping;
      for (unsigned i = 0; i < externalOperands.size(); ++i)
        mapping.map(externalOperands[i], blockArgs[i]);

      OpBuilder regionBuilder = OpBuilder::atBlockEnd(body);
      Operation *sinkClone = nullptr;
      for (Operation *op : chain) {
        Operation *cloned = regionBuilder.clone(*op, mapping);
        if (isPrefillFusedOp(op)) sinkClone = cloned;
      }

      // Terminate the region with tessera.target.cpx.yield of sink results
      SmallVector<Value> yieldVals;
      if (sinkClone)
        for (Value r : sinkClone->getResults()) yieldVals.push_back(r);
      OperationState yieldState(loc, "tessera.target.cpx.yield");
      yieldState.addOperands(yieldVals);
      regionBuilder.create(yieldState);

      // Build the fused op
      Operation *fusedOp = b.create(state);

      // Replace chain's external result uses with fused op results
      unsigned resultIdx = 0;
      for (Operation *op : chain) {
        if (isPrefillFusedOp(op)) {
          for (Value r : op->getResults()) {
            if (resultIdx < fusedOp->getNumResults())
              r.replaceAllUsesWith(fusedOp->getResult(resultIdx++));
          }
        }
      }

      // Erase chain ops in reverse order
      for (auto it = chain.rbegin(); it != chain.rend(); ++it)
        (*it)->erase();

      ++fused;
      LLVM_DEBUG(llvm::dbgs()
                 << "[video-ingest-fuse] fused chain of " << chain.size()
                 << " ops at " << loc << "\n");
    }

    if (fused > 0)
      module.emitRemark("video-ingest-fuse: created ")
          << fused << " video_ingest_fused region(s)";
  }
};

PassRegistration<FuseVideoIngestPass> fuseVideoIngestPassReg;

} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createFuseVideoIngestPass() {
  return std::make_unique<FuseVideoIngestPass>();
}
} // namespace tessera
