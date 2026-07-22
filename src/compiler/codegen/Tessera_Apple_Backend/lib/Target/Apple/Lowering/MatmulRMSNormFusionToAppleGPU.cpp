//===- MatmulRMSNormFusionToAppleGPU.cpp ---------------------------------===//
//
// Phase 8.4.7 — Apple GPU MSL fusion: matmul -> rmsnorm. Pattern matches:
//
//   %m = tessera.matmul     %A, %B    : (M, K) x (K, N) -> (M, N)
//   %o = tessera.rmsnorm[_safe] %m    : (M, N) -> (M, N)
//
// and replaces both with a single func.call into the Apple-GPU runtime
// shim's matmul_rmsnorm_f32 kernel. Handles both `tessera.rmsnorm` and
// `tessera.rmsnorm_safe` (numerically-safe variant); both lower to the
// same kernel with appropriate eps default.
//
// Constraints:
//   - rank-2 f32 inputs/output
//   - static shapes
//   - matmul result has exactly one use (the rmsnorm)
//   - N <= 256 (per-thread stack array bound)
//
// The eps attribute (or default 1e-5 / 1e-6 per op) is passed as an f32
// to the runtime symbol.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Target/Apple/Passes.h"
#include "Tessera/Target/Apple/EpilogueFusion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace ::mlir;

namespace tessera {
namespace apple {

namespace {

// Optimizing-Compiler Plan F2b — matmul -> rmsnorm lowers to the generic
// SYNTHESIZED epilogue kernel (epilogue + eps carried as region-descriptor
// attributes), retiring the per-epilogue hand-written matmul_rmsnorm_f32
// kernel.  The symbolic call uses the uniform (A,B,O,M,N,K) signature shared
// with matmul_gelu so a module with both fusions declares one consistent
// external symbol.
constexpr llvm::StringLiteral kSynthEpilogueF32Symbol =
    "tessera_apple_gpu_synth_matmul_epilogue_f32";



// Two patterns, one per rmsnorm variant. We can't use a float NTTP before
// C++20 (DefaultEps), so we share the body via a helper and instantiate two
// concrete subclasses below.
struct LowerMatmulRMSNormPatternBase : public RewritePattern {
  LowerMatmulRMSNormPatternBase(MLIRContext *ctx, StringRef mnemonic,
                                float defaultEps)
      : RewritePattern(mnemonic, /*benefit=*/2, ctx),
        defaultEps_(defaultEps) {}

  float defaultEps_;

  LogicalResult matchAndRewrite(Operation *normOp,
                                PatternRewriter &rewriter) const override {
    return tessera::apple::lowerMatmulEpilogueFusion(
        rewriter, normOp,
        {/*epilogueLabel=*/"rmsnorm", /*intentKernel=*/"matmul_rmsnorm",
         /*synthSymbol=*/kSynthEpilogueF32Symbol, /*allowHalfPrecision=*/false,
         /*requireAxisMinusOne=*/false, /*hasEps=*/true,
         /*defaultEps=*/defaultEps_});
  }
};

constexpr llvm::StringLiteral kRMSNormMnemonic = "tessera.rmsnorm";
constexpr llvm::StringLiteral kRMSNormSafeMnemonic = "tessera.rmsnorm_safe";

struct LowerMatmulRMSNormPattern : public LowerMatmulRMSNormPatternBase {
  LowerMatmulRMSNormPattern(MLIRContext *ctx)
      : LowerMatmulRMSNormPatternBase(ctx, kRMSNormMnemonic, /*eps=*/1.0e-5f) {}
};

struct LowerMatmulRMSNormSafePattern : public LowerMatmulRMSNormPatternBase {
  LowerMatmulRMSNormSafePattern(MLIRContext *ctx)
      : LowerMatmulRMSNormPatternBase(ctx, kRMSNormSafeMnemonic, /*eps=*/1.0e-6f) {}
};

struct LowerMatmulRMSNormFusionToAppleGPUPass
    : public PassWrapper<LowerMatmulRMSNormFusionToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMatmulRMSNormFusionToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-matmul-rmsnorm-fusion-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Fuse tessera.matmul -> tessera.rmsnorm[_safe] (rank-2, f32, N <= 256) "
           "into a single Apple GPU runtime call (custom MSL kernel)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // Default eps: 1e-5 for "tessera.rmsnorm", 1e-6 for "tessera.rmsnorm_safe"
    // — matches the python runtime defaults in tessera.runtime._runtime_cpu_op.
    patterns.add<LowerMatmulRMSNormPattern>(&getContext());
    patterns.add<LowerMatmulRMSNormSafePattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

LogicalResult rewriteMatmulRMSNormFusion(Operation *op,
                                         PatternRewriter &rewriter) {
  if (op->getName().getStringRef() == kRMSNormSafeMnemonic) {
    LowerMatmulRMSNormSafePattern pattern(op->getContext());
    return pattern.matchAndRewrite(op, rewriter);
  }
  LowerMatmulRMSNormPattern pattern(op->getContext());
  return pattern.matchAndRewrite(op, rewriter);
}

std::unique_ptr<Pass> createLowerMatmulRMSNormFusionToAppleGPUPass() {
  return std::make_unique<LowerMatmulRMSNormFusionToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
