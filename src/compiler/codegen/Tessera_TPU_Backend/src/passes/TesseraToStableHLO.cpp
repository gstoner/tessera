#include "tessera/tpu/TesseraTPUPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// StableHLO headers (assumes include path provided)
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace {
struct LowerMatmulPattern : OpRewritePattern<Operation> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != "tessera.matmul") return failure();
    if (op->getNumOperands() < 2 || op->getNumResults() != 1) return failure();

    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    auto lhsTy = llvm::dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = llvm::dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsTy || !rhsTy) return failure();

    if (lhsTy.getRank() < 2 || rhsTy.getRank() < 2)
      return rewriter.notifyMatchFailure(op, "matmul operands must have rank >= 2");

    // Build dot_general dims for [..., M, K] x [..., K, N] -> [..., M, N].
    SmallVector<int64_t> lhsBatch, rhsBatch;
    int64_t lhsBatchRank = lhsTy.getRank() - 2;
    int64_t rhsBatchRank = rhsTy.getRank() - 2;
    if (lhsBatchRank != rhsBatchRank)
      return rewriter.notifyMatchFailure(op, "batched matmul ranks must match");
    for (int64_t i = 0; i < lhsBatchRank; ++i) {
      lhsBatch.push_back(i);
      rhsBatch.push_back(i);
    }
    SmallVector<int64_t> lhsContract{(int64_t)lhsTy.getRank()-1};
    SmallVector<int64_t> rhsContract{(int64_t)rhsTy.getRank()-2};
    auto dimNums = stablehlo::DotDimensionNumbersAttr::get(
        op->getContext(), lhsBatch, rhsBatch, lhsContract, rhsContract);

    // Precision config: BF16 inputs, F32 accums are XLA defaults; we leave it empty here.
    SmallVector<Attribute> precision;

    Type resultTy = op->getResult(0).getType();

    auto dot = rewriter.create<stablehlo::DotGeneralOp>(
        op->getLoc(), resultTy, lhs, rhs, dimNums,
        rewriter.getArrayAttr(precision));

    rewriter.replaceOp(op, dot.getResult());
    return success();
  }
};

struct LowerAddPattern : OpRewritePattern<Operation> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != "tessera.add") return failure();
    if (op->getNumOperands() != 2 || op->getNumResults() != 1) return failure();
    Value a = op->getOperand(0), b = op->getOperand(1);
    rewriter.replaceOpWithNewOp<stablehlo::AddOp>(op, a.getType(), a, b);
    return success();
  }
};

struct LowerTesseraToStableHLOPass
    : PassWrapper<LowerTesseraToStableHLOPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "tessera-lower-to-stablehlo"; }
  StringRef getDescription() const override { return "Lower Tessera ops to StableHLO (TPU-friendly)."; }
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerMatmulPattern, LowerAddPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace tessera {
std::unique_ptr<mlir::Pass> createLowerTesseraToStableHLOPass() {
  return std::make_unique<LowerTesseraToStableHLOPass>();
}
} // namespace tessera
