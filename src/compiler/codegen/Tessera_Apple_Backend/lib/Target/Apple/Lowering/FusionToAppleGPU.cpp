//===- FusionToAppleGPU.cpp - Declarative fusion lowering -----*- C++ -*-===//

#include "Tessera/Target/Apple/FusionPattern.h"
#include "Tessera/Target/Apple/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace ::mlir;

namespace tessera::apple {
namespace {

constexpr FusionPattern kFusionPatterns[] = {
    {"native_sparse_attention", "tessera.native_sparse_attn_fused", 4,
     rewriteNativeSparseAttnFusion},
    {"mla_decode", "tessera.mla_decode_fused", 4,
     rewriteMLADecodeFusion},
    {"swiglu", "tessera.swiglu_fused", 4, rewriteSwigluFusion},
    {"matmul_softmax_matmul", "tessera.matmul", 3,
     rewriteMatmulSoftmaxMatmulFusion},
    {"matmul_softmax", "tessera.softmax", 2,
     rewriteMatmulSoftmaxFusion},
    {"matmul_gelu", "tessera.gelu", 2, rewriteMatmulGeluFusion},
    {"matmul_rmsnorm", "tessera.rmsnorm", 2,
     rewriteMatmulRMSNormFusion},
    {"matmul_rmsnorm_safe", "tessera.rmsnorm_safe", 2,
     rewriteMatmulRMSNormFusion},
};

class DeclarativeFusionRewrite final : public RewritePattern {
public:
  DeclarativeFusionRewrite(MLIRContext *ctx, const FusionPattern &pattern)
      : RewritePattern(pattern.rootOp, pattern.benefit, ctx), pattern(pattern) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    return pattern.rewrite(op, rewriter);
  }

private:
  const FusionPattern &pattern;
};

class LowerDeclarativeFusionsToAppleGPUPass final
    : public PassWrapper<LowerDeclarativeFusionsToAppleGPUPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LowerDeclarativeFusionsToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-declarative-fusions-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Lower the declarative Apple GPU fusion registry with one generic "
           "rewrite";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateDeclarativeAppleFusionPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

llvm::ArrayRef<FusionPattern> appleFusionPatterns() { return kFusionPatterns; }

void populateDeclarativeAppleFusionPatterns(RewritePatternSet &patterns) {
  for (const FusionPattern &pattern : appleFusionPatterns())
    patterns.add<DeclarativeFusionRewrite>(patterns.getContext(), pattern);
}

std::unique_ptr<Pass> createLowerDeclarativeFusionsToAppleGPUPass() {
  return std::make_unique<LowerDeclarativeFusionsToAppleGPUPass>();
}

} // namespace tessera::apple
