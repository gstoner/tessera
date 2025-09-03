#include "tessera/targets/cerebras/Passes.h"
#if HAVE_MLIR
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "CerebrasDialect.h.inc"
#include "CerebrasOps.h.inc"

using namespace mlir;

namespace {

// Reuse some simple folds here to ensure canonicalization can run standalone.
struct MemcpyNoOpFold2 : public OpRewritePattern<tessera::cerebras::MemcpyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tessera::cerebras::MemcpyOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getDst() == op.getSrc() &&
        op.getDst_space() == op.getSrc_space()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

struct RouteDedup2 : public OpRewritePattern<tessera::cerebras::RouteOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tessera::cerebras::RouteOp op,
                                PatternRewriter &rewriter) const override {
    Block *blk = op->getBlock();
    for (Operation &prev : *blk) {
      if (&prev == op) break;
      if (auto r = dyn_cast<tessera::cerebras::RouteOp>(&prev)) {
        if (r.getFrom() == op.getFrom() && r.getTo() == op.getTo() && r.getColor() == op.getColor()) {
          rewriter.eraseOp(op);
          return success();
        }
      }
    }
    return failure();
  }
};

struct CerebrasCanonicalize : public PassWrapper<CerebrasCanonicalize, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet canon(ctx);
    canon.add<MemcpyNoOpFold2, RouteDedup2>(ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(canon));
  }
};

} // namespace

namespace tessera { namespace cerebras {
std::unique_ptr<mlir::Pass> createCerebrasCanonicalizePass() {
  return std::make_unique<CerebrasCanonicalize>();
}
}} // namespace

#else
// No-MLIR build
#endif
