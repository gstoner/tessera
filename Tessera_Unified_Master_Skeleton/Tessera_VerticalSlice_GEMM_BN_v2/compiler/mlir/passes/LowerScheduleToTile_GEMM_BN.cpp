
//===- LowerScheduleToTile_GEMM_BN.cpp ---------------------------------*- C++ -*-===//
// Worked example: Lower to tile-level ops with epilogue preserved.
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace tessera {
namespace {
// Pseudo classes again
class TSched_GemmOp : public Op<TSched_GemmOp, OpTrait::ZeroRegion> {};
class TSched_BatchNormOp : public Op<TSched_BatchNormOp, OpTrait::ZeroRegion> {};
class TTile_GemmMMAOp : public Op<TTile_GemmMMAOp, OpTrait::ZeroRegion> {};
class TTile_BatchNormVecOp : public Op<TTile_BatchNormVecOp, OpTrait::ZeroRegion> {};

struct GemmTilePattern : public OpRewritePattern<TSched_GemmOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TSched_GemmOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    // Get attrs
    // In real code: op.getBm(), getBn(), getBk(), getEpilogue()
    auto bm = rewriter.getI32IntegerAttr(128);
    auto bn = rewriter.getI32IntegerAttr(128);
    auto bk = rewriter.getI32IntegerAttr(64);
    auto ep = rewriter.getStringAttr("none");
    auto resultType = op->getResult(0).getType();
    auto mma = rewriter.create<TTile_GemmMMAOp>(loc, resultType, A, B, bm, bn, bk, ep);
    rewriter.replaceOp(op, mma.getResults());
    return success();
  }
};

struct BNTilePattern : public OpRewritePattern<TSched_BatchNormOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TSched_BatchNormOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value X = op->getOperand(0);
    Value Mean = op->getOperand(1);
    Value Var = op->getOperand(2);
    Value Gamma = op->getOperand(3);
    Value Beta = op->getOperand(4);
    auto eps = rewriter.getF32FloatAttr(1e-5f);
    auto vecW = rewriter.getI32IntegerAttr(8);
    auto resultType = op->getResult(0).getType();
    auto bn = rewriter.create<TTile_BatchNormVecOp>(loc, resultType, X, Mean, Var, Gamma, Beta, eps, vecW);
    rewriter.replaceOp(op, bn.getResults());
    return success();
  }
};

struct LowerScheduleToTilePass : public PassWrapper<LowerScheduleToTilePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<GemmTilePattern, BNTilePattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> createLowerScheduleToTilePass() {
  return std::make_unique<LowerScheduleToTilePass>();
}
} // namespace tessera
