#include "select_grad_path.h"
// TODO(mlir): include PatternRewriter, DialectRegistry, etc.
/*
struct GradSwapPattern : public OpRewritePattern<TesseraAutodiffGradYOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TesseraAutodiffGradYOp op, PatternRewriter& rewriter) const override {
    if (!preferJVP) return failure();
    // Find or declare @energy_bilinear_jvp and replace the grad op with it.
    auto jvp = rewriter.create<CallOp>(op.getLoc(), jvpFn, op.getOperands());
    jvp->setAttr("ebt.grad_path", rewriter.getStringAttr("jvp"));
    rewriter.replaceOp(op, jvp.getResults());
    return success();
  }
};
*/
namespace tessera { namespace ebt {
mlir::Pass* createSelectGradPathPass(bool /*preferJVP*/) { return nullptr; }
void registerSelectGradPathPipeline() {
  // PassPipelineRegistration<>("tessera-ebt-select-grad-path",
  //   "Swap autodiff grad with custom JVP when requested",
  //   [](OpPassManager& pm){ pm.addPass(createSelectGradPathPass(/*preferJVP*/true)); });
}
}} // ns
