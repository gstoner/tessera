// MiniMax Sparse Attention (MSA, arXiv:2606.13392). MSAExpandPass is a
// compiler-visibility slot: it preserves the three semantic MSA ops (Index
// Branch / Top-k selector / exact block-sparse Main Branch) and marks them for
// the block-sparse backend lane, without claiming a fused kernel. The ops carry
// block_size/top_k/force_local_block/causal as ODS attributes. See docs/msa.md.
//
// RUN: %tessera_strict_opt %s -tessera-msa-expand | FileCheck %s

func.func @msa_visible(
    %q: tensor<1x8x16x8xf32>, %k: tensor<1x2x16x8xf32>,
    %v: tensor<1x2x16x8xf32>, %scores: tensor<1x2x16x4xf32>)
    -> (tensor<1x2x16x4xf32>, tensor<1x2x16x2xi64>, tensor<1x8x16x8xf32>) {
  // CHECK-LABEL: func.func @msa_visible

  // CHECK: tessera.msa_index_scores
  // CHECK-SAME: tessera.reasoning.compiler_visible = true
  // CHECK-SAME: tessera.reasoning.family = "minimax_sparse"
  // CHECK-SAME: tessera.reasoning.variant = "msa_index_scores"
  %s = "tessera.msa_index_scores"(%q, %k) {block_size = 4 : i64}
      : (tensor<1x8x16x8xf32>, tensor<1x2x16x8xf32>) -> tensor<1x2x16x4xf32>

  // CHECK: tessera.msa_select_blocks
  // CHECK-SAME: tessera.reasoning.compiler_visible = true
  // CHECK-SAME: tessera.reasoning.family = "minimax_sparse"
  // CHECK-SAME: tessera.reasoning.variant = "msa_select_blocks"
  %ids = "tessera.msa_select_blocks"(%scores)
      {top_k = 2 : i64, block_size = 4 : i64, force_local_block = true, causal = true}
      : (tensor<1x2x16x4xf32>) -> tensor<1x2x16x2xi64>

  // CHECK: tessera.msa_sparse_attention
  // CHECK-SAME: tessera.reasoning.compiler_visible = true
  // CHECK-SAME: tessera.reasoning.family = "minimax_sparse"
  // CHECK-SAME: tessera.reasoning.variant = "msa_sparse_attention"
  %o = "tessera.msa_sparse_attention"(%q, %k, %v)
      {block_size = 4 : i64, top_k = 2 : i64, force_local_block = true, causal = true}
      : (tensor<1x8x16x8xf32>, tensor<1x2x16x8xf32>,
         tensor<1x2x16x8xf32>) -> tensor<1x8x16x8xf32>

  return %s, %ids, %o
      : tensor<1x2x16x4xf32>, tensor<1x2x16x2xi64>, tensor<1x8x16x8xf32>
}
