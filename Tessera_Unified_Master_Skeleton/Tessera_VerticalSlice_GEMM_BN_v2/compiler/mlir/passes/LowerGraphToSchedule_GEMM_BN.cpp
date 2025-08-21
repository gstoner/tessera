
//===- LowerGraphToSchedule_GEMM_BN.cpp -------------------------------*- C++ -*-===//
// Worked example: choose tile sizes for GEMM and vector width for BN.
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace tessera {
namespace {
// Pseudo op classes (normally generated from TableGen)
class TGraph_GemmOp : public Op<TGraph_GemmOp, OpTrait::ZeroRegion> {};
class TGraph_BatchNormOp : public Op<TGraph_BatchNormOp, OpTrait::ZeroRegion> {};
class TSched_GemmOp : public Op<TSched_GemmOp, OpTrait::ZeroRegion> {};
class TSched_BatchNormOp : public Op<TSched_BatchNormOp, OpTrait::ZeroRegion> {};

struct GemmSchedPattern : public OpRewritePattern<TGraph_GemmOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TGraph_GemmOp op, PatternRewriter &rewriter) const override {
    // Heuristic: pick a macro-tile based on N dimension (pretend we can query shaped type)
    // bm=128, bn=128, bk=64 works well for Tensor Cores in many cases.
    auto loc = op.getLoc();
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    // Determine epilogue from presence of C0 or attrs (simplified)
    StringAttr epilogue = rewriter.getStringAttr("none");
    if (op->getNumOperands() == 3) epilogue = rewriter.getStringAttr("bias");
    // Create tsched.gemm
    auto resultType = op->getResult(0).getType();
    auto bm = rewriter.getI32IntegerAttr(128);
    auto bn = rewriter.getI32IntegerAttr(128);
    auto bk = rewriter.getI32IntegerAttr(64);
    auto gemm = rewriter.create<TSched_GemmOp>(loc, resultType, A, B, bm, bn, bk, epilogue);
    rewriter.replaceOp(op, gemm.getResults());
    return success();
  }
};

struct BNSchedPattern : public OpRewritePattern<TGraph_BatchNormOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TGraph_BatchNormOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value X = op->getOperand(0);
    Value Mean = op->getOperand(1);
    Value Var = op->getOperand(2);
    Value Gamma = op->getOperand(3);
    Value Beta = op->getOperand(4);
    // Choose vector width: prefer 8, else 4, else 1 (toy logic)
    IntegerAttr vecW = rewriter.getI32IntegerAttr(8);
    // eps from attr (pretend we have it)
    auto eps = rewriter.getF32FloatAttr(1e-5f);
    auto resultType = op->getResult(0).getType();
    auto bn = rewriter.create<TSched_BatchNormOp>(loc, resultType,
          X, Mean, Var, Gamma, Beta, eps, vecW);
    rewriter.replaceOp(op, bn.getResults());
    return success();
  }
};

struct LowerGraphToSchedulePass : public PassWrapper<LowerGraphToSchedulePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<GemmSchedPattern, BNSchedPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> createLowerGraphToSchedulePass() {
  return std::make_unique<LowerGraphToSchedulePass>();
}
} // namespace tessera
