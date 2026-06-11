// Lookahead Sparse Attention (LSA) — experimental composite attention policy.
// LookaheadSparseAttnExpandPass is a compiler-visibility slot: it preserves the
// semantic op and marks it for the sparse-attention backend lane, without
// claiming a fused kernel. See docs/audit/domain/archive/lsa_scope.md.
//
// RUN: %tessera_strict_opt %s -tessera-lookahead-sparse-attn-expand | FileCheck %s

func.func @lookahead_sparse_visible(
    %q: tensor<2x3x16x16xf32>, %k: tensor<2x3x16x16xf32>,
    %v: tensor<2x3x16x16xf32>) -> tensor<2x3x16x16xf32> {
  // CHECK-LABEL: func.func @lookahead_sparse_visible
  // CHECK: tessera.lookahead_sparse_attention
  // CHECK-SAME: tessera.reasoning.compiler_visible = true
  // CHECK-SAME: tessera.reasoning.family = "lookahead_sparse"
  // CHECK-SAME: tessera.reasoning.variant = "lookahead_sparse_attention"
  %0 = "tessera.lookahead_sparse_attention"(%q, %k, %v)
      {window_size = 6 : i64, block_size = 4 : i64, tau = 64 : i64,
       threshold = 5.000000e-01 : f64, causal = true}
      : (tensor<2x3x16x16xf32>, tensor<2x3x16x16xf32>,
         tensor<2x3x16x16xf32>) -> tensor<2x3x16x16xf32>
  return %0 : tensor<2x3x16x16xf32>
}
