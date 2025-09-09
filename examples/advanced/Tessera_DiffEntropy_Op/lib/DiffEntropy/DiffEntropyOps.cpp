//===- DiffEntropyOps.cpp -----------------------------------------------===//
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "DiffEntropyOps.h"

using namespace mlir;
using namespace tessera::diffentropy;

namespace {
struct RangeEntropyCanonicalize : OpRewritePattern<RangeEntropySoftOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RangeEntropySoftOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: constant-fold log/exp combos when alpha is very large/small, etc.
    return failure();
  }
};
} // namespace

void RangeEntropySoftOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                     MLIRContext *ctx) {
  patterns.add<RangeEntropyCanonicalize>(ctx);
}

LogicalResult RangeEntropySoftOp::verify() {
  if (getAlpha() <= 0.0) return emitOpError("alpha must be > 0");
  if (!(getReduction() == "none" || getReduction() == "mean" || getReduction() == "sum"))
    return emitOpError("reduction must be one of none|mean|sum");
  return success();
}

LogicalResult AttnRowEntropyOp::verify() {
  if (!(getMode() == "scores" || getMode() == "probs"))
    return emitOpError("mode must be 'scores' or 'probs'");
  if (!(getReduction() == "none" || getReduction() == "mean" || getReduction() == "sum"))
    return emitOpError("reduction must be one of none|mean|sum");
  return success();
}
