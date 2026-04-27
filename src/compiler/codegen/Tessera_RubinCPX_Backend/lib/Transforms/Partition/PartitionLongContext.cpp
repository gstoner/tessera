
//===- PartitionLongContext.cpp - Partition CPX/Rubin context graph --------===//
//
// Partitions the IR into:
//   • CPX segment  — prefill / context ops running on the CPX device
//   • Rubin segment — decode ops running on the main Rubin GPU
//
// Inserts transport bridges at the segment boundary:
//   • After each `attn.prefill_fused` op on the CPX side:
//       kv.export  result, "pcie+cx9", 33554432 (32 MiB default chunk)
//   • Before each op on the Rubin side that consumes a KV:
//       kv.import  token → dst
//
// Heuristic classification (until a proper CPX affinity attribute is added):
//   • An op is "CPX-affine" if:
//       – its name contains "attn.prefill_fused" or "kv.cache"
//       – its name contains "video.decode" or "video.encode"
//       – it is tagged with `{tessera.device = "cpx"}`
//   • All other ops are "Rubin-affine".
//
// For each kv.export token, the pass scans downstream use chains for ops that
// look like KV consumers (ops with "attn", "decode", or "kv" in their name
// on the Rubin side) and inserts a kv.import before the first such consumer.
//
//===-----------------------------------------------------------------------===//

#include "tessera/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "partition-longcontext"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Classification helpers
//===----------------------------------------------------------------------===//

static bool isCPXAffine(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (name.contains("attn.prefill_fused")) return true;
  if (name.contains("kv.cache"))           return true;
  if (name.contains("video.decode"))        return true;
  if (name.contains("video.encode"))        return true;
  if (name.contains("video_ingest_fused"))  return true;
  // Explicit device tag
  if (auto devAttr = op->getAttrOfType<StringAttr>("tessera.device"))
    return devAttr.getValue() == "cpx";
  return false;
}

static bool isKVConsumer(Operation *op) {
  // Rubin-side ops that need the KV tensor imported from CPX
  StringRef name = op->getName().getStringRef();
  return name.contains("attn") || name.contains("decode") ||
         name.contains("kv.import");
}

//===----------------------------------------------------------------------===//
// PartitionLongContextPass
//===----------------------------------------------------------------------===//

struct PartitionLongContextPass
    : public PassWrapper<PartitionLongContextPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartitionLongContextPass)

  StringRef getArgument() const override { return "tessera-partition-longcontext"; }
  StringRef getDescription() const override {
    return "Partition graph into CPX (context/prefill) and Rubin (decode) "
           "segments; insert kv.export/import bridges at the boundary";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder b(module.getContext());

    // ── Step 1: Tag all CPX-affine ops ─────────────────────────────────────
    module.walk([](Operation *op) {
      if (isCPXAffine(op))
        op->setAttr("tessera.device", StringAttr::get(op->getContext(), "cpx"));
    });

    // ── Step 2: Insert kv.export after each attn.prefill_fused ─────────────
    SmallVector<Operation *> prefillOps;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef().contains("attn.prefill_fused"))
        prefillOps.push_back(op);
    });

    SmallVector<std::pair<Value, Value>> exportTokens; // (original KV result, token)

    for (Operation *op : prefillOps) {
      if (op->getResults().empty()) continue;

      Value kvResult = op->getResult(0);
      Location loc = op->getLoc();

      // Insert kv.export immediately after this op
      b.setInsertionPointAfter(op);

      auto policyAttr  = b.getStringAttr("pcie+cx9");
      auto chunkBytes  = b.getI64IntegerAttr(32 * 1024 * 1024); // 32 MiB

      // The token type is i64 (opaque transport handle)
      Type tokenTy = b.getI64Type();

      OperationState exportState(loc, "tessera.target.cpx.kv.export");
      exportState.addOperands(kvResult);
      exportState.addAttribute("policy", policyAttr);
      exportState.addAttribute("chunk_bytes", chunkBytes);
      exportState.addTypes(tokenTy);
      exportState.addAttribute("tessera.device", b.getStringAttr("cpx"));

      Operation *exportOp = b.create(exportState);
      Value token = exportOp->getResult(0);
      exportTokens.push_back({kvResult, token});

      LLVM_DEBUG(llvm::dbgs()
                 << "[partition] inserted kv.export after attn.prefill_fused at "
                 << loc << "\n");
    }

    // ── Step 3: Insert kv.import before Rubin-side KV consumers ────────────
    // For each (kvResult, token) pair, find downstream ops on the Rubin side
    // that consume the kvResult and insert a kv.import before the first one.

    for (auto [kvResult, token] : exportTokens) {
      SmallVector<OpOperand *> rubinUses;
      for (OpOperand &use : kvResult.getUses()) {
        Operation *user = use.getOwner();
        if (!isCPXAffine(user) && isKVConsumer(user))
          rubinUses.push_back(&use);
      }

      if (rubinUses.empty()) continue;

      // Insert kv.import before the first Rubin consumer
      Operation *firstConsumer = rubinUses.front()->getOwner();
      b.setInsertionPoint(firstConsumer);
      Location loc = firstConsumer->getLoc();

      // kv.import(token, dst) — dst is the same memref (in-place import)
      OperationState importState(loc, "tessera.target.cpx.kv.import");
      importState.addOperands(token);
      importState.addOperands(kvResult); // dst: same buffer, Rubin-side view
      importState.addAttribute("tessera.device", b.getStringAttr("rubin"));

      Operation *importOp = b.create(importState);

      // Replace kvResult uses in Rubin consumers with the imported value
      // (if kv.import produces a result)
      if (!importOp->getResults().empty()) {
        Value imported = importOp->getResult(0);
        for (OpOperand *use : rubinUses)
          use->set(imported);
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "[partition] inserted kv.import before Rubin consumer at "
                 << loc << "\n");
    }

    // ── Step 4: Emit module-level partition summary ─────────────────────────
    unsigned cpxCount = 0, rubinCount = 0;
    module.walk([&](Operation *op) {
      if (auto devAttr = op->getAttrOfType<StringAttr>("tessera.device")) {
        if (devAttr.getValue() == "cpx")   ++cpxCount;
        if (devAttr.getValue() == "rubin") ++rubinCount;
      }
    });

    if (cpxCount + rubinCount > 0)
      module.emitRemark("partition-longcontext: ")
          << cpxCount << " CPX op(s), " << rubinCount << " Rubin op(s), "
          << exportTokens.size() << " kv.export/import bridge(s) inserted";
  }
};

PassRegistration<PartitionLongContextPass> partitionLongContextPassReg;

} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createPartitionLongContextPass() {
  return std::make_unique<PartitionLongContextPass>();
}
} // namespace tessera
