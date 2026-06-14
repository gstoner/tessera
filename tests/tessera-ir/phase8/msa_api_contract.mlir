// MiniMax Sparse Attention (MSA) API/IR contract.
//
// Locks the Phase 0-2 compiler-visible chain together:
//   1. Index Branch scores KV blocks per GQA group.
//   2. Top-k selector emits block ids.
//   3. Sparse Main Branch is an exact block-sparse attention op.
//
// RUN: %tessera_strict_opt %s -tessera-msa-expand | FileCheck %s

func.func @msa_api_contract(
    %q: tensor<2x8x32x16xf32>, %k: tensor<2x2x32x16xf32>,
    %v: tensor<2x2x32x16xf32>)
    -> (tensor<2x2x32x4xf32>, tensor<2x2x32x3xi64>, tensor<2x8x32x16xf32>) {
  // CHECK-LABEL: func.func @msa_api_contract

  // CHECK:      %[[S:.*]] = tessera.msa_index_scores
  // CHECK-SAME: block_size = 8 : i64
  // CHECK-SAME: scale = 2.500000e-01 : f64
  // CHECK-SAME: tessera.reasoning.family = "minimax_sparse"
  %scores = "tessera.msa_index_scores"(%q, %k)
      {block_size = 8 : i64, scale = 2.500000e-01 : f64}
      : (tensor<2x8x32x16xf32>, tensor<2x2x32x16xf32>) -> tensor<2x2x32x4xf32>

  // CHECK:      %[[IDS:.*]] = tessera.msa_select_blocks %[[S]]
  // CHECK-SAME: block_size = 8 : i64
  // CHECK-SAME: causal = false
  // CHECK-SAME: force_local_block = false
  // CHECK-SAME: tessera.reasoning.variant = "msa_select_blocks"
  // CHECK-SAME: top_k = 3 : i64
  %ids = "tessera.msa_select_blocks"(%scores)
      {top_k = 3 : i64, block_size = 8 : i64, force_local_block = false, causal = false}
      : (tensor<2x2x32x4xf32>) -> tensor<2x2x32x3xi64>

  // CHECK:      tessera.msa_sparse_attention
  // CHECK-SAME: block_size = 8 : i64
  // CHECK-SAME: causal = false
  // CHECK-SAME: force_local_block = false
  // CHECK-SAME: scale = 2.500000e-01 : f64
  // CHECK-SAME: tessera.reasoning.variant = "msa_sparse_attention"
  // CHECK-SAME: top_k = 3 : i64
  %o = "tessera.msa_sparse_attention"(%q, %k, %v)
      {block_size = 8 : i64, top_k = 3 : i64, force_local_block = false,
       causal = false, scale = 2.500000e-01 : f64}
      : (tensor<2x8x32x16xf32>, tensor<2x2x32x16xf32>,
         tensor<2x2x32x16xf32>) -> tensor<2x8x32x16xf32>

  return %scores, %ids, %o
      : tensor<2x2x32x4xf32>, tensor<2x2x32x3xi64>, tensor<2x8x32x16xf32>
}
