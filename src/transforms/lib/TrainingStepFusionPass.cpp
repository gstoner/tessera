#include "Tessera/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace tessera {
namespace {

static bool matchEligibleLossBackward(
    Value predictionGradient, PatternRewriter &rewriter,
    Operation *&backward, StringRef &kind, FloatAttr &parameter) {
  backward = predictionGradient.getDefiningOp();
  if (!backward || predictionGradient != backward->getResult(0) ||
      !predictionGradient.hasOneUse() || backward->getNumOperands() != 3 ||
      backward->getNumResults() != 2)
    return false;
  StringRef name = backward->getName().getStringRef();
  parameter = rewriter.getF64FloatAttr(1.0);
  if (name == "tessera.loss.mse_backward") {
    kind = "mse";
  } else if (name == "tessera.loss.binary_cross_entropy_backward") {
    kind = "bce";
  } else if (name == "tessera.loss.regression_backward") {
    auto kindAttr = backward->getAttrOfType<StringAttr>("kind");
    if (!kindAttr ||
        (kindAttr.getValue() != "mae" &&
         kindAttr.getValue() != "huber" &&
         kindAttr.getValue() != "smooth_l1"))
      return false;
    kind = kindAttr.getValue();
    if (auto attr = backward->getAttrOfType<FloatAttr>("parameter"))
      parameter = attr;
  } else {
    return false;
  }
  return true;
}

struct FuseRegressionLossBackwardSGD final : RewritePattern {
  explicit FuseRegressionLossBackwardSGD(MLIRContext *context)
      : RewritePattern("tessera.sgd", /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *sgd,
                                PatternRewriter &rewriter) const override {
    if (sgd->getNumOperands() != 2 || sgd->getNumResults() != 1)
      return failure();
    Value predictionGradient = sgd->getOperand(1);
    Operation *backward = nullptr;
    StringRef kind;
    FloatAttr parameter;
    if (!matchEligibleLossBackward(
            predictionGradient, rewriter, backward, kind, parameter))
      return failure();

    auto reduction = backward->getAttrOfType<StringAttr>("reduction");
    auto lr = sgd->getAttrOfType<FloatAttr>("lr");
    if (!reduction || !lr)
      return failure();

    OperationState state(sgd->getLoc(), "tessera.training.loss_sgd");
    state.addOperands(
        {backward->getOperand(0), backward->getOperand(1),
         backward->getOperand(2), sgd->getOperand(0)});
    state.addTypes({sgd->getResult(0).getType(),
                    backward->getResult(1).getType()});
    state.addAttribute("kind", rewriter.getStringAttr(kind));
    state.addAttribute("parameter", parameter);
    state.addAttribute("reduction", reduction);
    state.addAttribute("lr", lr);
    Operation *fused = rewriter.create(state);

    rewriter.replaceAllUsesWith(sgd->getResult(0), fused->getResult(0));
    rewriter.replaceAllUsesWith(
        backward->getResult(1), fused->getResult(1));
    rewriter.eraseOp(sgd);
    rewriter.eraseOp(backward);
    return success();
  }
};

struct FuseLossBackwardAdamW final : RewritePattern {
  explicit FuseLossBackwardAdamW(MLIRContext *context)
      : RewritePattern("tessera.adamw", /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *adamw,
                                PatternRewriter &rewriter) const override {
    if (adamw->getNumOperands() != 4 || adamw->getNumResults() != 3)
      return failure();
    Operation *backward = nullptr;
    StringRef kind;
    FloatAttr parameter;
    if (!matchEligibleLossBackward(
            adamw->getOperand(1), rewriter, backward, kind, parameter))
      return failure();
    auto reduction = backward->getAttrOfType<StringAttr>("reduction");
    if (!reduction)
      return failure();
    auto f64 = [&](StringRef name, double fallback) {
      if (auto attr = adamw->getAttrOfType<FloatAttr>(name))
        return attr;
      return rewriter.getF64FloatAttr(fallback);
    };
    IntegerAttr step = adamw->getAttrOfType<IntegerAttr>("step");
    if (!step)
      step = rewriter.getI64IntegerAttr(1);

    OperationState state(adamw->getLoc(), "tessera.training.loss_adamw");
    state.addOperands({
        backward->getOperand(0), backward->getOperand(1),
        backward->getOperand(2), adamw->getOperand(0),
        adamw->getOperand(2), adamw->getOperand(3),
    });
    state.addTypes({
        adamw->getResult(0).getType(), adamw->getResult(1).getType(),
        adamw->getResult(2).getType(), backward->getResult(1).getType(),
    });
    state.addAttribute("kind", rewriter.getStringAttr(kind));
    state.addAttribute("parameter", parameter);
    state.addAttribute("reduction", reduction);
    state.addAttribute("lr", f64("lr", 1.0e-3));
    state.addAttribute("beta1", f64("beta1", 0.9));
    state.addAttribute("beta2", f64("beta2", 0.999));
    state.addAttribute("eps", f64("eps", 1.0e-8));
    state.addAttribute("weight_decay", f64("weight_decay", 0.01));
    state.addAttribute("step", step);
    Operation *fused = rewriter.create(state);

    for (unsigned index = 0; index < 3; ++index)
      rewriter.replaceAllUsesWith(
          adamw->getResult(index), fused->getResult(index));
    rewriter.replaceAllUsesWith(
        backward->getResult(1), fused->getResult(3));
    rewriter.eraseOp(adamw);
    rewriter.eraseOp(backward);
    return success();
  }
};

struct TrainingStepFusionPass final
    : PassWrapper<TrainingStepFusionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TrainingStepFusionPass)

  StringRef getArgument() const final {
    return "tessera-training-step-fusion";
  }
  StringRef getDescription() const final {
    return "Fuse eligible loss backward to optimizer update chains";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseRegressionLossBackwardSGD, FuseLossBackwardAdamW>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createTrainingStepFusionPass() {
  return std::make_unique<TrainingStepFusionPass>();
}

} // namespace tessera
