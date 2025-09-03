
#include "Tessera/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
static inline bool isOp(Operation *op, StringRef name) {
  return op && op->getName().getStringRef() == name;
}
struct FuseMatmulBiasGELU : public RewritePattern {
  FuseMatmulBiasGELU(MLIRContext *ctx) : RewritePattern("tessera.gelu", 2, ctx) {}
  LogicalResult matchAndRewrite(Operation *gelu, PatternRewriter &rewriter) const override {
    if (gelu->getNumOperands() != 1) return failure();
    Operation *add = gelu->getOperand(0).getDefiningOp();
    if (!isOp(add, "tessera.add")) return failure();
    if (add->getNumOperands() != 2) return failure();
    Operation *mm = add->getOperand(0).getDefiningOp();
    Value bias = add->getOperand(1);
    if (!isOp(mm, "tessera.matmul")) return failure();
    OperationState st(gelu->getLoc(), "tessera.fused_epilogue");
    st.addOperands({mm->getOperand(0), mm->getOperand(1), bias});
    st.addTypes(gelu->getResult(0).getType());
    st.addAttribute("epilogue", rewriter.getI32IntegerAttr(2));
    st.addAttribute("has_bias", rewriter.getBoolAttr(true));
    Operation *f = rewriter.create(st);
    rewriter.replaceOp(gelu, f->getResults());
    return success();
  }
};
struct FuseConvRelu : public RewritePattern {
  FuseConvRelu(MLIRContext *ctx) : RewritePattern("tessera.relu", 2, ctx) {}
  LogicalResult matchAndRewrite(Operation *relu, PatternRewriter &rewriter) const override {
    if (relu->getNumOperands() != 1) return failure();
    Operation *conv = relu->getOperand(0).getDefiningOp();
    if (!isOp(conv, "tessera.conv2d_nhwc")) return failure();
    OperationState st(conv->getLoc(), "tessera.conv2d_nhwc");
    st.addOperands(conv->getOperands());
    st.addTypes(relu->getResult(0).getType());
    for (auto &named : conv->getAttrs())
      if (named.getName() != "epilogue")
        st.addAttribute(named.getName(), named.getValue());
    st.addAttribute("epilogue", rewriter.getI32IntegerAttr(1));
    Operation *c = rewriter.create(st);
    rewriter.replaceOp(relu, c->getResults());
    return success();
  }
};
struct DropoutZeroSimplify : public RewritePattern {
  DropoutZeroSimplify(MLIRContext *ctx) : RewritePattern("tessera.flash_attn", 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *attn, PatternRewriter &rewriter) const override {
    auto pAttr = attn->getAttrOfType<FloatAttr>("dropout_p");
    if (!pAttr || pAttr.getValueAsDouble() != 0.0) return failure();
    OperationState st(attn->getLoc(), attn->getName().getStringRef());
    st.addOperands(attn->getOperands());
    st.addTypes(attn->getResultTypes());
    for (auto &na : attn->getAttrs())
      if (na.getName() != "dropout_p")
        st.addAttribute(na.getName(), na.getValue());
    Operation *n = rewriter.create(st);
    rewriter.replaceOp(attn, n->getResults());
    return success();
  }
};
struct TransposeIntoMatmul : public RewritePattern {
  TransposeIntoMatmul(MLIRContext *ctx) : RewritePattern("tessera.matmul", 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *mm, PatternRewriter &rewriter) const override {
    if (mm->getNumOperands() != 2) return failure();
    Operation *aDef = mm->getOperand(0).getDefiningOp();
    Operation *bDef = mm->getOperand(1).getDefiningOp();
    bool transA = false, transB = false;
    Value a = mm->getOperand(0), b = mm->getOperand(1);
    if (isOp(aDef, "tessera.transpose")) { transA = true; a = aDef->getOperand(0); }
    if (isOp(bDef, "tessera.transpose")) { transB = true; b = bDef->getOperand(0); }
    if (!transA && !transB) return failure();
    OperationState st(mm->getLoc(), "tessera.matmul");
    st.addOperands({a, b});
    st.addTypes(mm->getResultTypes());
    auto getFlag = [&](StringRef n){ if (auto b = mm->getAttrOfType<BoolAttr>(n)) return b.getValue(); return false; };
    for (auto &na : mm->getAttrs())
      if (na.getName() != "transposeA" && na.getName() != "transposeB")
        st.addAttribute(na.getName(), na.getValue());
    st.addAttribute("transposeA", rewriter.getBoolAttr(transA || getFlag("transposeA")));
    st.addAttribute("transposeB", rewriter.getBoolAttr(transB || getFlag("transposeB")));
    Operation *n = rewriter.create(st);
    rewriter.replaceOp(mm, n->getResults());
    return success();
  }
};
struct Canon : public PassWrapper<Canon, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Canon)
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMatmulBiasGELU, FuseConvRelu, DropoutZeroSimplify, TransposeIntoMatmul>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

namespace tessera {
std::unique_ptr<Pass> createCanonicalizeTesseraIRPass() { return std::make_unique<Canon>(); }
} // namespace tessera
