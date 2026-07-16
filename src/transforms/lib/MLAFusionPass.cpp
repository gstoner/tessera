//===- MLAFusionPass.cpp - Match MLA decode 4-op chain --------*- C++ -*-===//
//
// attention_variants_plan, MLA-1.
//
// Recognizes the DeepSeek-style Multi-Latent Attention decode chain:
//
//   %c = tessera.latent_kv_compress(%x, %W_dkv)
//   %K = tessera.latent_kv_expand_k(%c, %W_uk)
//   %V = tessera.latent_kv_expand_v(%c, %W_uv)
//   %O = tessera.flash_attn(%Q, %K, %V) [head_dim=..., causal=...]
//
// and rewrites it to a single
//
//   %O = tessera.mla_decode_fused(%x, %W_dkv, %W_uk, %W_uv, %Q,
//                                  scale=..., causal=...)
//
// Backends with a fused FlashMLA-style absorb-K kernel (Hopper / future
// Blackwell) lower the fused op directly without ever materializing the
// full K/V matrices — the win is that the latent `%c` is the only thing
// stored in cache, and `W_uk`/`W_uv` get absorbed into the score kernel.
// Backends without a fused kernel expand the op back into the 4-op chain.
//
// Mirrors the structure of SwigluFusionPass.cpp.
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

/// Rooted at the *terminal* tessera.flash_attn (the score-and-output
/// step). Walking up from the terminal anchors the rewrite on a single
/// root, which the GreedyPatternRewriteDriver expects.
struct FuseMLADecodeChain : public RewritePattern {
  FuseMLADecodeChain(MLIRContext *ctx)
      : RewritePattern("tessera.flash_attn", /*benefit=*/3, ctx) {}

  LogicalResult matchAndRewrite(Operation *attn,
                                PatternRewriter &rewriter) const override {
    // flash_attn: (Q, K_or_cache, V, ...). v1 MLA fusion only matches
    // the rank-4 (Q, K, V) variant; the cache-input form is for paged
    // FA decoding and is its own future fusion.
    if (attn->getNumOperands() < 3)
      return failure();
    Value q = attn->getOperand(0);
    Value k = attn->getOperand(1);
    Value v = attn->getOperand(2);

    Operation *expandK = k.getDefiningOp();
    Operation *expandV = v.getDefiningOp();
    if (!isOp(expandK, "tessera.latent_kv_expand_k") ||
        !isOp(expandV, "tessera.latent_kv_expand_v"))
      return rewriter.notifyMatchFailure(
          attn, "MLA fusion: K/V must come from latent_kv_expand_k/v");
    if (!k.hasOneUse() || !v.hasOneUse())
      return rewriter.notifyMatchFailure(
          attn, "MLA fusion: expanded K/V must be single-use");

    if (expandK->getNumOperands() != 2 || expandV->getNumOperands() != 2)
      return failure();

    // Both expanders must consume the same %c (the compressed latent).
    Value cK = expandK->getOperand(0);
    Value cV = expandV->getOperand(0);
    if (cK != cV)
      return rewriter.notifyMatchFailure(
          attn, "MLA fusion: expand_k and expand_v must share %c");
    Value wUk = expandK->getOperand(1);
    Value wUv = expandV->getOperand(1);

    Operation *compress = cK.getDefiningOp();
    if (!isOp(compress, "tessera.latent_kv_compress"))
      return rewriter.notifyMatchFailure(
          attn, "MLA fusion: %c must come from latent_kv_compress");
    if (compress->getNumOperands() != 2)
      return failure();

    Value x = compress->getOperand(0);
    Value wDkv = compress->getOperand(1);

    // Carry the flash_attn's scale + causal attributes onto the fused
    // op so backends can reproduce the same semantics. Audit 2026-06-10:
    // numeric_policy (storage/accum coupling, Decision #15a) is carried
    // from the flash_attn — the attention step dominates the fused
    // kernel's numerics; the compress/expand GEMMs inherit it.
    OperationState st(attn->getLoc(), "tessera.mla_decode_fused");
    st.addOperands({x, wDkv, wUk, wUv, q});
    st.addTypes(attn->getResultTypes());
    if (auto sc = attn->getAttrOfType<FloatAttr>("scale"))
      st.addAttribute("scale", sc);
    if (auto causal = attn->getAttrOfType<BoolAttr>("causal"))
      st.addAttribute("causal", causal);
    if (Attribute np = attn->getAttr("numeric_policy"))
      st.addAttribute("numeric_policy", np);
    Operation *fused = rewriter.create(st);
    rewriter.replaceOp(attn, fused->getResults());
    // The greedy driver removes %K / %V / %c automatically once they
    // have no remaining users.
    return success();
  }
};

struct MLAFusion
    : public PassWrapper<MLAFusion, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MLAFusion)
  StringRef getArgument() const override { return "tessera-mla-fusion"; }
  StringRef getDescription() const override {
    return "Fuse the DeepSeek MLA decode chain (latent_kv_compress → "
           "latent_kv_expand_k/v → flash_attn) into tessera.mla_decode_fused";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMLADecodeChain>(&getContext());
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    (void)applyPatternsGreedily(getOperation(), frozenPatterns);
  }
};

} // namespace

namespace tessera {
std::unique_ptr<Pass> createMLAFusionPass() {
  return std::make_unique<MLAFusion>();
}
} // namespace tessera
