#include "select_grad_path.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace mlir;

namespace {
struct SwapGradWithJVP : OpRewritePattern<Operation> {
  using OpRewritePattern::OpRewritePattern;
  bool preferJVP=false;
  SwapGradWithJVP(MLIRContext *ctx, bool p): OpRewritePattern(ctx), preferJVP(p) {}
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != "tessera.autodiff.grad_y") return failure();
    if (!preferJVP) {
      op->setAttr("ebt.grad_path", rewriter.getStringAttr("vjp"));
      return failure();
    }
    // Replace with a call to energy_bilinear_jvp (symbol must exist in module).
    op->setAttr("ebt.grad_path", rewriter.getStringAttr("jvp"));
    // Marker-only replacement: rename op to show swap in tests.
    OperationState st(op->getLoc(), "ebt.energy_bilinear_jvp");
    st.addOperands(op->getOperands());
    st.addTypes(op->getResultTypes());
    auto *rep = rewriter.create(st);
    rewriter.replaceOp(op, rep->getResults());
    return success();
  }
};
} // namespace

namespace tessera { namespace ebt {
struct SelectGradPathPass : PassWrapper<SelectGradPathPass, OperationPass<ModuleOp>> {
  bool preferJVP=false;
  SelectGradPathPass() = default;
  SelectGradPathPass(bool p): preferJVP(p) {}
  void runOnOperation() override {
    RewritePatternSet ps(&getContext());
    ps.add<SwapGradWithJVP>(&getContext(), preferJVP);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(ps));
  }
};
std::unique_ptr<mlir::Pass> createSelectGradPathPass(bool preferJVP) {
  return std::make_unique<SelectGradPathPass>(preferJVP);
}
void registerSelectGradPathPipeline() {
  PassPipelineRegistration<bool>(
    "tessera-ebt-select-grad-path","Swap autodiff grad with custom JVP",
    [](OpPassManager& pm, bool preferJVP){
      pm.addPass(createSelectGradPathPass(preferJVP));
    });
}
}} // ns
