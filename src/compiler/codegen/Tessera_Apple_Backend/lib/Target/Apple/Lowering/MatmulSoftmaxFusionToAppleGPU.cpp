//===- MatmulSoftmaxFusionToAppleGPU.cpp - Fused matmul->softmax MSL ----===//
//
// Phase 8.4.3 — Apple GPU first multi-op MSL fusion.
//
// Pattern matches a 2-op SSA chain:
//
//   %m = tessera.matmul %A, %B          : tensor<MxKxf32> -> tensor<MxNxf32>
//   %o = tessera.softmax %m              : tensor<MxNxf32> -> tensor<MxNxf32>
//
// and replaces both with a single func.call into the Apple-GPU runtime
// shim:
//
//   tessera_apple_gpu_matmul_softmax_f32(A, B, O, M, N, K)
//
// Constraints:
//   - rank-2 f32 inputs/output
//   - static shapes
//   - softmax axis must be -1 (innermost) — default
//   - N <= 256 to fit the GPU kernel's per-thread stack accumulator
//   - matmul result has exactly one use (the softmax) — otherwise the
//     fusion would change observable semantics by skipping the matmul
//     intermediate that some other op consumes.
//
// The pass runs *before* the single-op matmul / softmax passes so the
// fused rewrite wins the pattern race. If the constraints don't match,
// the chain falls through to the per-op runtime path (still executable)
// or the artifact-only path (multi-op programs that aren't a recognized
// fusion).
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

// Optimizing-Compiler Plan F2 (catalog retirement) — matmul->softmax lowers to
// the generic synthesized epilogue kernel for ALL dtypes.  The synthesizer now
// covers large N (threadgroup-tiled) and half precision (f16 native I/O, bf16
// host-convert), so it subsumes matmul_softmax_{f32,f16,bf16} + their tiled
// variants.  The emission is symbolic (uniform (A,B,O,M,N,K) signature); the
// runtime picks the actual f32/f16/bf16 dispatch.
constexpr llvm::StringLiteral kSynthEpilogueF32Symbol =
    "tessera_apple_gpu_synth_matmul_epilogue_f32";



// We rewrite at the softmax — that lets us peek upward at its operand and
// decide whether the chain matches without first touching the matmul. If
// matched, both ops are replaced atomically.
struct LowerMatmulSoftmaxFusionToAppleGPU : public RewritePattern {
  LowerMatmulSoftmaxFusionToAppleGPU(MLIRContext *ctx)
      : RewritePattern("tessera.softmax", /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *softmaxOp,
                                PatternRewriter &rewriter) const override {
    return tessera::apple::lowerMatmulEpilogueFusion(
        rewriter, softmaxOp,
        {/*epilogueLabel=*/"softmax", /*intentKernel=*/"matmul_softmax",
         /*synthSymbol=*/kSynthEpilogueF32Symbol, /*allowHalfPrecision=*/true,
         /*requireAxisMinusOne=*/true});
  }
};

struct LowerMatmulSoftmaxFusionToAppleGPUPass
    : public PassWrapper<LowerMatmulSoftmaxFusionToAppleGPUPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMatmulSoftmaxFusionToAppleGPUPass)

  StringRef getArgument() const override {
    return "tessera-matmul-softmax-fusion-to-apple_gpu";
  }
  StringRef getDescription() const override {
    return "Fuse tessera.matmul -> tessera.softmax (rank-2, f32/f16/bf16, "
           "axis=-1, N <= 256) into a single Apple GPU runtime call "
           "(custom MSL kernel)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerMatmulSoftmaxFusionToAppleGPU>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerMatmulSoftmaxFusionToAppleGPUPass() {
  return std::make_unique<LowerMatmulSoftmaxFusionToAppleGPUPass>();
}

} // namespace apple
} // namespace tessera
