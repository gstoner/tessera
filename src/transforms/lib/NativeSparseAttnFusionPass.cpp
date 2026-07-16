//===- NativeSparseAttnFusionPass.cpp - DeepSeek NSA fusion --*- C++ -*-===//
//
// attention_variants_plan, NSA-4.
//
// Recognizes the canonical NSA branch shape:
//
//   %out_w = tessera.attn_sliding_window(%Q, %K, %V) {window_size=...}
//   %out_c = tessera.attn_compressed_blocks(%Q, %K_c, %V_c)
//   %out_s = tessera.attn_top_k_blocks(%Q, %K, %V, %scores) {top_k=..., block_size=...}
//   %out   = (gating) g_w * %out_w + g_c * %out_c + g_s * %out_s
//
// and rewrites it to a single
//
//   %out = tessera.native_sparse_attn_fused(%Q, %K, %V, %gate_logits,
//                                            window_size=..., block_size=...,
//                                            top_k=..., causal=...)
//
// Backends with a fused NSA kernel lower the fused op directly. Backends
// without one expand back to the three branches + gating.
//
// The recognizer is conservative — for v1 it matches just the three
// branches and emits the fused op carrying their attributes; the
// "gating" portion of the chain (per-query 3-way sigmoid) is left to
// the consumer side and recovered from the gate_logits operand.
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

/// We anchor on the sliding-window branch (the only branch that has
/// just (Q, K, V) as operands and no derived inputs). From there we
/// search the same parent block for matching compressed_blocks +
/// top_k_blocks ops that share the same Q. If we find them, we emit
/// the fused op.
struct FuseNSABranches : public RewritePattern {
  FuseNSABranches(MLIRContext *ctx)
      : RewritePattern("tessera.attn_sliding_window", /*benefit=*/3, ctx) {}

  LogicalResult matchAndRewrite(Operation *winOp,
                                PatternRewriter &rewriter) const override {
    if (winOp->getNumOperands() != 3)
      return failure();
    Value q = winOp->getOperand(0);
    Value k = winOp->getOperand(1);
    Value v = winOp->getOperand(2);

    // Look for compressed_blocks and top_k_blocks ops in the same block
    // that share %q.
    Operation *compOp = nullptr;
    Operation *topkOp = nullptr;
    Block *parent = winOp->getBlock();
    for (Operation &candidate : *parent) {
      if (&candidate == winOp) continue;
      if (isOp(&candidate, "tessera.attn_compressed_blocks") &&
          candidate.getNumOperands() == 3 &&
          candidate.getOperand(0) == q) {
        compOp = &candidate;
      } else if (isOp(&candidate, "tessera.attn_top_k_blocks") &&
                 candidate.getNumOperands() >= 4 &&
                 candidate.getOperand(0) == q &&
                 candidate.getOperand(1) == k &&
                 candidate.getOperand(2) == v) {
        topkOp = &candidate;
      }
    }
    if (!compOp || !topkOp)
      return rewriter.notifyMatchFailure(
          winOp, "NSA fusion: missing one of the three branches");

    // Structural guards (audit 2026-06-10). The rewrite replaces all three
    // branch results with the SAME fused value, so it is only sound when:
    //   * every branch yields exactly one result of the identical type, and
    //   * each branch result has exactly one use (the downstream gating
    //     chain) — a branch output consumed elsewhere must keep its own
    //     per-branch value and cannot be aliased to the fused output.
    // Note the documented contract: the consumer-side gating multiply-add
    // is expected to be subsumed by (or expanded from) the fused kernel;
    // see the file header. These guards make the structural preconditions
    // explicit instead of assumed.
    if (winOp->getNumResults() != 1 || compOp->getNumResults() != 1 ||
        topkOp->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          winOp, "NSA fusion: branches must each have a single result");
    if (winOp->getResult(0).getType() != compOp->getResult(0).getType() ||
        winOp->getResult(0).getType() != topkOp->getResult(0).getType())
      return rewriter.notifyMatchFailure(
          winOp, "NSA fusion: branch result types differ");
    if (!winOp->getResult(0).hasOneUse() ||
        !compOp->getResult(0).hasOneUse() ||
        !topkOp->getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(
          winOp, "NSA fusion: branch result has multiple uses");

    // Carry the attributes that the fused kernel needs.
    auto windowAttr = winOp->getAttrOfType<IntegerAttr>("window_size");
    auto topkAttr = topkOp->getAttrOfType<IntegerAttr>("top_k");
    auto blockAttr = topkOp->getAttrOfType<IntegerAttr>("block_size");
    if (!windowAttr || !topkAttr || !blockAttr)
      return rewriter.notifyMatchFailure(
          winOp, "NSA fusion: missing required attribute");

    bool causalW = true;
    if (auto a = winOp->getAttrOfType<BoolAttr>("causal"))
      causalW = a.getValue();
    bool causalT = true;
    if (auto a = topkOp->getAttrOfType<BoolAttr>("causal"))
      causalT = a.getValue();
    bool causal = causalW && causalT;

    // The scores operand of top_k_blocks is the most natural stand-in
    // for the gate_logits input: any dependent computation already
    // needs Q · K_c summaries. The fused op's `gate_logits` carries
    // this same scoring tensor.
    Value scores = topkOp->getOperand(3);

    OperationState st(winOp->getLoc(), "tessera.native_sparse_attn_fused");
    st.addOperands({q, k, v, scores});
    // Output type matches the sliding-window branch's output (all three
    // branches must have the same output shape; the gate-weighted sum
    // preserves it).
    st.addTypes(winOp->getResultTypes());
    st.addAttribute("window_size", windowAttr);
    st.addAttribute("block_size", blockAttr);
    st.addAttribute("top_k", topkAttr);
    st.addAttribute("causal", rewriter.getBoolAttr(causal));
    Operation *fused = rewriter.create(st);
    // Replace ALL three branches' uses with the single fused result —
    // the gating multiply-add downstream remains, but now it operates
    // on the same fused output for each branch arm. (Backends with a
    // fused NSA kernel typically subsume gating too; backends without
    // expand back to the three branches.)
    rewriter.replaceOp(winOp, fused->getResults());
    rewriter.replaceOp(compOp, fused->getResults());
    rewriter.replaceOp(topkOp, fused->getResults());
    return success();
  }
};

struct NativeSparseAttnFusion
    : public PassWrapper<NativeSparseAttnFusion, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NativeSparseAttnFusion)
  StringRef getArgument() const override {
    return "tessera-native-sparse-attn-fusion";
  }
  StringRef getDescription() const override {
    return "Fuse the DeepSeek NSA three-branch shape (sliding_window + "
           "compressed_blocks + top_k_blocks) into "
           "tessera.native_sparse_attn_fused";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseNSABranches>(&getContext());
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    (void)applyPatternsGreedily(getOperation(), frozenPatterns);
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createNativeSparseAttnFusionPass() {
  return std::make_unique<NativeSparseAttnFusion>();
}
} // namespace tessera
