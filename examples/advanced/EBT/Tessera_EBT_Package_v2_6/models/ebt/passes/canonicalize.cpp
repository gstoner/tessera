#include "canonicalize.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace mlir;

namespace {
// Inline trivial decode_init (no side effects, one user)
struct InlineDecodeInit : OpRewritePattern<Operation> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != "tessera.ebt.decode_init") return failure();
    if (!op->hasOneUse()) return failure();
    // This is a placeholder: a real impl would clone producers into the user block.
    // For now, tag and succeed to show pattern flow.
    op->setAttr("ebt.inlined", rewriter.getUnitAttr());
    return success();
  }
};

// Normalize self_verify → explicit min-reduce (marker attr only here)
struct NormalizeSelfVerify : OpRewritePattern<Operation> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != "tessera.ebt.self_verify") return failure();
    op->setAttr("ebt.normalized", rewriter.getUnitAttr());
    return success();
  }
};
} // namespace

namespace tessera { namespace ebt {
struct CanonicalizePass : PassWrapper<CanonicalizePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    RewritePatternSet ps(&getContext());
    ps.add<InlineDecodeInit, NormalizeSelfVerify>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(ps));
  }
};

std::unique_ptr<mlir::Pass> createCanonicalizePass() {
  return std::make_unique<CanonicalizePass>();
}

void registerCanonicalizePipeline() {
  PassPipelineRegistration<>("tessera-ebt-canonicalize",
    "Normalize EBT graph (self-verify, inline trivial init)",
    [](OpPassManager &pm){
      pm.addPass(createCanonicalizePass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
    });
}
}} // ns
