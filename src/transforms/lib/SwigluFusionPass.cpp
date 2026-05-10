//===- SwigluFusionPass.cpp - Match SwiGLU 3-op chain ---------*- C++ -*-===//
//
// Phase 8.4.8 (SwiGLU Performance Plan, Stage 2b).
//
// Recognizes the SwiGLU MLP-block 3-op chain
//
//     %gate   = tessera.matmul(%x, %W_gate)
//     %up     = tessera.matmul(%x, %W_up)
//     %hidden = tessera.silu_mul(%gate, %up)
//     %out    = tessera.matmul(%hidden, %W_down)
//
// and rewrites it to a single
//
//     %out = tessera.swiglu_fused(%x, %W_gate, %W_up, %W_down)
//
// Backends with a fused MLP-block kernel (Apple GPU MSL, NVIDIA WGMMA
// epilogue, ROCm MFMA epilogue) lower the fused op directly; backends
// without one fall back to the original 4-op chain via a simple inverse
// expansion in their lowering pipelines.
//
// Pattern shape mirrors CanonicalizeTesseraIR.cpp's existing fusion
// patterns (FuseMatmulBiasGELU, FuseConvRelu) — see Architecture
// Decision #19: hardware-free Target IR sits below Schedule IR, so this
// fusion runs at the Graph IR layer and is target-agnostic.
//
//===----------------------------------------------------------------------===//

#include "Tessera/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

static inline bool isOp(Operation *op, StringRef name) {
  return op && op->getName().getStringRef() == name;
}

/// Rooted at the *terminal* tessera.matmul (the W_down GEMM). Walking up
/// from the terminal anchors the rewrite on a single root, which is what
/// the GreedyPatternRewriteDriver expects.
struct FuseSwiGLUChain : public RewritePattern {
  FuseSwiGLUChain(MLIRContext *ctx)
      : RewritePattern("tessera.matmul", /*benefit=*/3, ctx) {}

  LogicalResult matchAndRewrite(Operation *downMatmul,
                                PatternRewriter &rewriter) const override {
    if (downMatmul->getNumOperands() != 2)
      return failure();

    // Operand 0 of the down matmul must be the silu_mul output.
    Operation *siluMul = downMatmul->getOperand(0).getDefiningOp();
    if (!isOp(siluMul, "tessera.silu_mul"))
      return failure();
    if (siluMul->getNumOperands() != 2)
      return failure();
    // Single-use of the silu_mul output keeps the rewrite safe — if anything
    // else consumes `%hidden`, we'd lose that consumer when we replace the
    // root.
    if (!siluMul->getResult(0).hasOneUse())
      return failure();

    // silu_mul(%gate, %up) — %gate from W_gate matmul, %up from W_up matmul.
    Operation *gateMatmul = siluMul->getOperand(0).getDefiningOp();
    Operation *upMatmul = siluMul->getOperand(1).getDefiningOp();
    if (!isOp(gateMatmul, "tessera.matmul") ||
        !isOp(upMatmul, "tessera.matmul"))
      return failure();
    if (gateMatmul->getNumOperands() != 2 || upMatmul->getNumOperands() != 2)
      return failure();

    // Both gate and up matmuls must consume the SAME %x (operand 0). If not,
    // the chain isn't a SwiGLU block — bail out.
    if (gateMatmul->getOperand(0) != upMatmul->getOperand(0))
      return failure();

    // Single-use on the intermediate gate/up outputs guards against more
    // exotic graphs where %gate or %up feeds something besides the
    // silu_mul.
    if (!gateMatmul->getResult(0).hasOneUse() ||
        !upMatmul->getResult(0).hasOneUse())
      return failure();

    // None of the three matmuls may carry an epilogue attribute or a
    // non-default transpose flag — those need their own variants of the
    // fused op. `transposeA`/`transposeB` are `DefaultValuedAttr` on
    // Tessera_MatmulOp so we explicitly check for the truthy case.
    auto isPlainMatmul = [](Operation *mm) -> bool {
      if (mm->getAttr("epilogue") || mm->getAttr("has_bias"))
        return false;
      auto truthyBool = [&](StringRef name) {
        if (auto a = mm->getAttrOfType<BoolAttr>(name))
          return a.getValue();
        return false;
      };
      if (truthyBool("transposeA") || truthyBool("transposeB"))
        return false;
      return true;
    };
    if (!isPlainMatmul(gateMatmul) || !isPlainMatmul(upMatmul) ||
        !isPlainMatmul(downMatmul))
      return failure();

    Value x = gateMatmul->getOperand(0);
    Value wGate = gateMatmul->getOperand(1);
    Value wUp = upMatmul->getOperand(1);
    Value wDown = downMatmul->getOperand(1);

    OperationState st(downMatmul->getLoc(), "tessera.swiglu_fused");
    st.addOperands({x, wGate, wUp, wDown});
    st.addTypes(downMatmul->getResultTypes());
    Operation *fused = rewriter.create(st);
    rewriter.replaceOp(downMatmul, fused->getResults());
    // The greedy driver removes %hidden / %gate / %up automatically once
    // they have no remaining users — `hasOneUse()` above is what makes that
    // possible.
    return success();
  }
};

struct SwigluFusion
    : public PassWrapper<SwigluFusion, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SwigluFusion)
  StringRef getArgument() const override { return "tessera-swiglu-fusion"; }
  StringRef getDescription() const override {
    return "Fuse the SwiGLU 3-op chain (matmul → matmul → silu_mul "
           "→ matmul) into tessera.swiglu_fused";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseSwiGLUChain>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createSwigluFusionPass() {
  return std::make_unique<SwigluFusion>();
}
} // namespace tessera
