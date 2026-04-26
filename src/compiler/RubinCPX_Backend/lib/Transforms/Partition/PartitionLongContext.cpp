
#include "tessera/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

namespace {
struct PartitionLongContextPass
    : public PassWrapper<PartitionLongContextPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartitionLongContextPass)
  StringRef getArgument() const override { return "tessera-partition-longcontext"; }
  StringRef getDescription() const override {
    return "Partition graph into CPX (context/prefill) and Rubin (decode) segments and insert kv.export/import bridges";
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    // Heuristic: For demo, just scan for ops named 'tessera.target.cpx.attn.prefill_fused' and insert export/import.
    m.walk([&](Operation *op){
      if (op->getName().getStringRef().endswith("attn.prefill_fused")) {
        OpBuilder b(op);
        // Create a fake kv.export after the op to represent "KV handoff"
        auto loc = op->getLoc();
        Value anyOut = op->getResult(0);
        // For the skeleton we don't type-check strictly.
        auto policyAttr = b.getStringAttr("pcie+cx9");
        auto chunkBytes = b.getI64IntegerAttr(32*1024*1024);
        auto exportOp = b.create<Operation>(loc, OperationName("tessera.target.cpx.kv.export", m.getContext()),
                         ValueRange{anyOut}, NamedAttributeList{
                           NamedAttribute(b.getStringAttr("policy"), policyAttr),
                           NamedAttribute(b.getStringAttr("chunk_bytes"), chunkBytes)
                         }, /*regions=*/0);
        (void)exportOp;
      }
    });
  }
};
} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createPartitionLongContextPass() {
  return std::make_unique<PartitionLongContextPass>();
}
} // namespace tessera
