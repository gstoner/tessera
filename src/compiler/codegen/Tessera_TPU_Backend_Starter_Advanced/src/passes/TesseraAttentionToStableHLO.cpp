#include "tessera/tpu/TesseraTPUPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace {
/// Lower "tessera.flash_attn" (Q,K,V, scale, mask?, p_drop?) to a TPU-friendly StableHLO composite.
/// For v1 we emit a stablehlo.custom_call("flash_attention") with well-known attributes that
/// backend/XLA can pattern-match and fuse. This keeps masking & dropout semantics explicit.
struct LowerFlashAttentionPattern : OpRewritePattern<Operation> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != "tessera.flash_attn") return failure();
    if (op->getNumOperands() < 3 || op->getNumResults() != 1) return failure();

    SmallVector<Value> args;
    for (auto v : op->getOperands()) args.push_back(v);

    // Collect attributes if present
    DictionaryAttr attrs = op->getAttrDictionary();
    auto scaleAttr = attrs.get("scale");           // optional
    auto pdropAttr = attrs.get("dropout_p");       // optional
    auto causalAttr = attrs.get("causal");         // optional
    auto maskPresent = (op->getNumOperands() >= 4); // Q,K,V,(mask?)

    // The result type matches the original op's result.
    Type outTy = op->getResult(0).getType();

    // Build a stablehlo.custom_call with a convention:
    //   call_target_name = "tessera.flash_attention"
    //   attrs: {scale: f32, dropout_p: f32, causal: i1, has_mask: i1}
    SmallVector<NamedAttribute> ccAttrs;
    ccAttrs.push_back(rewriter.getNamedAttr("call_target_name",
                        rewriter.getStringAttr("tessera.flash_attention")));
    if (scaleAttr)   ccAttrs.push_back(rewriter.getNamedAttr("tessera.scale", scaleAttr));
    if (pdropAttr)   ccAttrs.push_back(rewriter.getNamedAttr("tessera.dropout_p", pdropAttr));
    if (causalAttr)  ccAttrs.push_back(rewriter.getNamedAttr("tessera.causal", causalAttr));
    ccAttrs.push_back(rewriter.getNamedAttr("tessera.has_mask",
                        rewriter.getBoolAttr(maskPresent)));

    auto cc = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), TypeRange{outTy}, args, rewriter.getDictionaryAttr(ccAttrs));

    rewriter.replaceOp(op, cc.getResults());
    return success();
  }
};

struct TesseraAttentionToStableHLOPass
    : PassWrapper<TesseraAttentionToStableHLOPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "tessera-lower-attention-to-stablehlo"; }
  StringRef getDescription() const override { return "Lower Tessera FlashAttention to StableHLO (custom_call form)."; }
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerFlashAttentionPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createLowerTesseraAttentionToStableHLOPass() {
  return std::make_unique<TesseraAttentionToStableHLOPass>();
}
} // namespace tessera
