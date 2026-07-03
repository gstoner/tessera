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
#include "Tessera/Target/Apple/FusionChainUtils.h"
#include "Tessera/Target/Apple/LoweringUtils.h"

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
    if (softmaxOp->getNumOperands() < 1)
      return failure();
    Value softmaxIn = softmaxOp->getOperand(0);

    // Decision #19 — consume the compiler's fusion descriptor when present.
    bool descriptorDriven =
        tessera::apple::fusionDescriptorDriven(softmaxOp, "matmul_softmax");

    // axis: defaults to -1. Anything else falls out of fusion.
    int64_t axis = -1;
    if (auto attr = softmaxOp->getAttrOfType<IntegerAttr>("axis"))
      axis = attr.getInt();

    auto smTy = dyn_cast<RankedTensorType>(softmaxIn.getType());
    if (!smTy || smTy.getRank() != 2)
      return rewriter.notifyMatchFailure(softmaxOp, "fusion: rank-2 only");
    Type smElem = smTy.getElementType();
    StringRef symbol = kSynthEpilogueF32Symbol;   // synthesized for all dtypes
    bool isSynth = true;
    if (!(smElem.isF32() || smElem.isF16() || smElem.isBF16())) {
      return rewriter.notifyMatchFailure(
          softmaxOp, "fusion: f32, f16, or bf16 only in Phase 8.4.4.2");
    }
    if (axis != -1 && axis != smTy.getRank() - 1)
      return rewriter.notifyMatchFailure(softmaxOp, "fusion: axis must be -1");

    // The softmax input must be the result of a matmul, with no other uses.
    auto matmulOr = tessera::apple::walkChainProducer(
        rewriter, softmaxOp, softmaxIn, "tessera.matmul", descriptorDriven);
    if (failed(matmulOr))
      return failure();
    Operation *matmulOp = *matmulOr;
    if (matmulOp->getNumOperands() < 2)
      return failure();
    Value lhs = matmulOp->getOperand(0);
    Value rhs = matmulOp->getOperand(1);

    auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsTy || !rhsTy || lhsTy.getRank() != 2 || rhsTy.getRank() != 2)
      return rewriter.notifyMatchFailure(softmaxOp, "fusion: matmul inputs not rank-2");
    // The matmul element type must match the softmax (and softmax-result)
    // element type. Mixed-dtype chains fall out of fusion.
    if (lhsTy.getElementType() != smElem || rhsTy.getElementType() != smElem)
      return rewriter.notifyMatchFailure(
          softmaxOp, "fusion: matmul element types must match softmax dtype");
    if (lhsTy.isDynamicDim(0) || lhsTy.isDynamicDim(1) ||
        rhsTy.isDynamicDim(0) || rhsTy.isDynamicDim(1))
      return rewriter.notifyMatchFailure(softmaxOp, "fusion: requires static shapes");

    int64_t M = lhsTy.getDimSize(0);
    int64_t K = lhsTy.getDimSize(1);
    int64_t N = rhsTy.getDimSize(1);
    if (rhsTy.getDimSize(0) != K)
      return rewriter.notifyMatchFailure(softmaxOp, "fusion: matmul K mismatch");
    if (N > 256)
      return rewriter.notifyMatchFailure(
          softmaxOp, "fusion: GPU kernel limited to N <= 256");

    Location loc = softmaxOp->getLoc();
    ModuleOp mod = softmaxOp->getParentOfType<ModuleOp>();

    auto aMemTy = MemRefType::get({M, K}, smElem);
    auto bMemTy = MemRefType::get({K, N}, smElem);
    auto oMemTy = MemRefType::get({M, N}, smElem);
    auto outTensorTy = RankedTensorType::get({M, N}, smElem);

    SmallVector<NamedAttribute> desc;
    desc.push_back(rewriter.getNamedAttr(
        "tessera.fusion.kernel",
        rewriter.getStringAttr(isSynth ? "synth_matmul_epilogue"
                                       : "matmul_softmax")));
    if (isSynth)
      desc.push_back(rewriter.getNamedAttr("tessera.fusion.epilogue",
                                           rewriter.getStringAttr("softmax")));
    desc.push_back(rewriter.getNamedAttr(
        "tessera.fusion.source",
        rewriter.getStringAttr(descriptorDriven ? "descriptor"
                                                : "rediscovered")));

    rewriter.setInsertionPoint(matmulOp);
    Value result = tessera::common::emitFusionCall(
        rewriter, loc, mod, symbol, {{lhs, aMemTy}, {rhs, bMemTy}}, oMemTy,
        outTensorTy, {M, N, K}, desc);

    // Replace the softmax op (chain consumer) with the fused result. Then
    // erase the matmul (now dead because its only use was softmax).
    rewriter.replaceOp(softmaxOp, result);
    rewriter.eraseOp(matmulOp);
    return success();
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
