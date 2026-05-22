
#include "tessera/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct MarkNVFP4OnMatmul : OpRewritePattern<Operation> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto name = op->getName().getStringRef();
    if (!name.contains("matmul") && !name.contains("attn"))
      return failure();
    if (op->hasAttr("tessera.nvfp4.enabled")) return failure();
    op->setAttr("tessera.nvfp4.enabled", rewriter.getUnitAttr());
    op->setAttr("tessera.nvfp4.accum", rewriter.getStringAttr("fp16")); // or fp32
    op->setAttr("tessera.nvfp4.tile", rewriter.getStringAttr("m128n128k256"));
    return success();
  }
};

struct NVFP4VectorizePass
  : public PassWrapper<NVFP4VectorizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NVFP4VectorizePass)
  StringRef getArgument() const override { return "tessera-vectorize-nvfp4"; }
  StringRef getDescription() const override { return "Mark matmul/attn with NVFP4 packing + accum attrs"; }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MarkNVFP4OnMatmul>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createNVFP4VectorizePass() {
  return std::make_unique<NVFP4VectorizePass>();
}
} // namespace tessera
