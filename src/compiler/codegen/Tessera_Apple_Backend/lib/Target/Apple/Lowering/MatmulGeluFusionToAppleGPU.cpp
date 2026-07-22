//===- MatmulGeluFusionToAppleGPU.cpp ------------------------------------===//
//
// Phase 8.4.7 — Apple GPU MSL fusion for the MLP block activation:
// matmul -> gelu. Pattern matches a 2-op SSA chain
//
//   %m = tessera.matmul %A, %B           : (M, K) x (K, N) -> (M, N)
//   %o = tessera.gelu   %m               : (M, N) -> (M, N)
//
// and replaces both with a single func.call into the Apple-GPU runtime
// shim's matmul_gelu_f32 kernel. Mirrors the Phase 8.4.3 matmul -> softmax
// fusion structurally — just a different post-matmul postlude.
//
// Constraints (Phase 8.4.7):
//   - rank-2 f32 inputs/output
//   - static shapes
//   - matmul result has exactly one use (the gelu)
//   - N <= 256 (per-thread stack array bound)
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

// Optimizing-Compiler Plan F2a — matmul -> gelu lowers to the generic
// SYNTHESIZED epilogue kernel (the epilogue carried as a region descriptor
// attribute), retiring the per-epilogue hand-written matmul_gelu_f32 kernel.
constexpr llvm::StringLiteral kSynthEpilogueF32Symbol =
    "tessera_apple_gpu_synth_matmul_epilogue_f32";



struct LowerMatmulGeluFusionToAppleGPU : public RewritePattern {
  LowerMatmulGeluFusionToAppleGPU(MLIRContext *ctx)
      : RewritePattern("tessera.gelu", /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *geluOp,
                                PatternRewriter &rewriter) const override {
    return tessera::apple::lowerMatmulEpilogueFusion(
        rewriter, geluOp,
        {/*epilogueLabel=*/"gelu", /*intentKernel=*/"matmul_gelu",
         /*synthSymbol=*/kSynthEpilogueF32Symbol});
  }
};

struct LowerMatmulGeluFusionToAppleGPUPass
    : public PassWrapper<LowerMatmulGeluFusionToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMatmulGeluFusionToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-matmul-gelu-fusion-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Fuse tessera.matmul -> tessera.gelu (rank-2, f32, N <= 256) into "
           "a single Apple GPU runtime call (custom MSL kernel)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerMatmulGeluFusionToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

LogicalResult rewriteMatmulGeluFusion(Operation *op,
                                      PatternRewriter &rewriter) {
  LowerMatmulGeluFusionToAppleGPU pattern(op->getContext());
  return pattern.matchAndRewrite(op, rewriter);
}

std::unique_ptr<Pass> createLowerMatmulGeluFusionToAppleGPUPass() {
  return std::make_unique<LowerMatmulGeluFusionToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
