
#include "tessera/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Async/IR/Async.h"

using namespace mlir;
namespace {
struct InsertKVBridgeForPrefill : OpRewritePattern<Operation> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (!op->getName().getStringRef().equals("tessera.target.cpx.attn.prefill_fused"))
      return failure();

    Location loc = op->getLoc();
    Value out = op->getResult(0);

    // Create kv.export returning !async.token
    auto policy = rewriter.getStringAttr("pcie+cx9");
    auto bytes  = rewriter.getI64IntegerAttr(32*1024*1024);
    OperationState st(loc, "tessera.target.cpx.kv.export");
    st.addOperands(out);
    st.addAttribute("policy", policy);
    st.addAttribute("chunk_bytes", bytes);
    auto tokTy = async::TokenType::get(rewriter.getContext());
    st.addTypes(tokTy);
    Operation *exp = rewriter.create(st);

    // Create a placeholder kv.import consuming the token (no real dst in skeleton)
    OperationState ist(loc, "tessera.target.cpx.kv.import");
    ist.addOperands(exp->getResult(0));
    ist.addAttribute("dst_placeholder", rewriter.getUnitAttr());
    rewriter.create(ist);

    return success();
  }
};

struct PartitionLongContextPass
  : public PassWrapper<PartitionLongContextPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartitionLongContextPass)
  StringRef getArgument() const override { return "tessera-partition-longcontext"; }
  StringRef getDescription() const override { return "Insert KV export/import with async token"; }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<InsertKVBridgeForPrefill>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createPartitionLongContextPass() {
  return std::make_unique<PartitionLongContextPass>();
}
} // namespace tessera
