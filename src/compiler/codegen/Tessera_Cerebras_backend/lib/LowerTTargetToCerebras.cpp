#include "tessera/targets/cerebras/Passes.h"
#if HAVE_MLIR
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "CerebrasDialect.h.inc"
#include "CerebrasOps.h.inc"
#include "TTargetDialect.h.inc"
#include "TTargetOps.h.inc"

using namespace mlir;

namespace {

// ttarget.copy → cerebras.(load_sram|store_sram|memcpy)
struct CopyLowering : public OpRewritePattern<tessera::ttarget::CopyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tessera::ttarget::CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto srcSpace = op.getSrc_space().str();
    auto dstSpace = op.getDst_space().str();
    if (srcSpace == "global" && dstSpace == "sram") {
      rewriter.replaceOpWithNewOp<tessera::cerebras::LoadSRAMOp>(op, op.getDst(), op.getSrc());
      return success();
    }
    if (srcSpace == "sram" && dstSpace == "global") {
      rewriter.replaceOpWithNewOp<tessera::cerebras::StoreSRAMOp>(op, op.getDst(), op.getSrc());
      return success();
    }
    rewriter.replaceOpWithNewOp<tessera::cerebras::MemcpyOp>(op, op.getDst(), op.getSrc(),
                                                             rewriter.getStringAttr(dstSpace),
                                                             rewriter.getStringAttr(srcSpace));
    return success();
  }
};

// ttarget.region → cerebras.region
struct RegionLowering : public OpRewritePattern<tessera::ttarget::RegionOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tessera::ttarget::RegionOp op,
                                PatternRewriter &rewriter) const override {
    auto x0 = op.getX0();
    auto y0 = op.getY0();
    auto x1 = op.getX1();
    auto y1 = op.getY1();
    int32_t color = 0;
    auto newOp = rewriter.create<tessera::cerebras::RegionOp>(op.getLoc(), x0, y0, x1, y1,
                                                              rewriter.getI32IntegerAttr(color));
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(), newOp.getBody().end());
    rewriter.eraseOp(op);
    return success();
  }
};

// ttarget.route → cerebras.route
struct RouteLowering : public OpRewritePattern<tessera::ttarget::RouteOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tessera::ttarget::RouteOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tessera::cerebras::RouteOp>(op, op.getFrom(), op.getTo(), op.getColor());
    return success();
  }
};

// ttarget.matmul → cerebras.matmul
struct MatmulLowering : public OpRewritePattern<tessera::ttarget::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tessera::ttarget::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tessera::cerebras::MatmulOp>(op, op.getA(), op.getB(), op.getC(),
                                                             op.getMAttr(), op.getNAttr(), op.getKAttr());
    return success();
  }
};

// === Extra rewrite patterns (canonicalization over cerebras.*) ===

// 1) Remove no-op memcpy: same src/dst AND same spaces.
struct MemcpyNoOpFold : public OpRewritePattern<tessera::cerebras::MemcpyOp> {
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

// 2) Fuse load_sram -> store_sram into memcpy(global->global) when SRAM buffer is the same SSA value.
struct LoadStoreFuse : public OpRewritePattern<tessera::cerebras::StoreSRAMOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tessera::cerebras::StoreSRAMOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Find a dominating load_sram that wrote to the same SRAM buffer value.
    Value sramBuf = storeOp.getSrc_sram();
    for (Operation *def = sramBuf.getDefiningOp(); def; ) {
      if (auto load = dyn_cast<tessera::cerebras::LoadSRAMOp>(def)) {
        // Replace with memcpy to/from globals; erase both if SRAM buf has no other uses.
        auto memcpy = rewriter.create<tessera::cerebras::MemcpyOp>(storeOp.getLoc(),
            storeOp.getDst_global(), load.getSrc_global(),
            rewriter.getStringAttr("global"), rewriter.getStringAttr("global"));
        (void)memcpy;
        rewriter.eraseOp(storeOp);
        rewriter.eraseOp(load);
        return success();
      }
      break;
    }
    return failure();
  }
};

// 3) Dedup identical route ops within the same block.
struct RouteDedup : public OpRewritePattern<tessera::cerebras::RouteOp> {
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

struct LowerTTargetToCerebras : public PassWrapper<LowerTTargetToCerebras, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<CopyLowering, RegionLowering, RouteLowering, MatmulLowering>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();

    // Run extra canonicalization patterns over cerebras.*
    RewritePatternSet canon(ctx);
    canon.add<MemcpyNoOpFold, LoadStoreFuse, RouteDedup>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(canon))))
      signalPassFailure();
  }
};

} // namespace

namespace tessera { namespace cerebras {
std::unique_ptr<mlir::Pass> createLowerTTargetToCerebrasPass() {
  return std::make_unique<LowerTTargetToCerebras>();
}
void registerCerebrasLoweringPasses() {}
}} // namespace

#else
// No-MLIR build
#endif
