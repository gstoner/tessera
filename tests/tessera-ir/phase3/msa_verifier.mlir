// RUN: tessera-opt %s -split-input-file -verify-diagnostics | FileCheck %s
//
// MiniMax Sparse Attention (MSA, arXiv:2606.13392) Graph IR verifiers.
// The MSA-specific contract on top of the shared verifyAttentionQKV shape
// checks: GQA divisibility (Hq % Hkv == 0), KV-block divisibility
// (Sk % block_size == 0), and top_k <= num_blocks. See docs/msa.md.
//
// One positive case + three negative cases.

// ─────────────────────────────────────────────────────────────────────────
// POSITIVE: GQA Hq=8 / Hkv=2 (8 % 2 == 0), Sk=16 / block_size=4 (16 % 4 == 0),
// top_k=2 <= num_blocks=4.
// ─────────────────────────────────────────────────────────────────────────
// CHECK-LABEL: func.func @msa_valid
// CHECK: tessera.msa_sparse_attention
func.func @msa_valid(
    %q: tensor<1x8x16x8xf32>, %k: tensor<1x2x16x8xf32>,
    %v: tensor<1x2x16x8xf32>) -> tensor<1x8x16x8xf32> {
  %o = "tessera.msa_sparse_attention"(%q, %k, %v)
      {block_size = 4 : i64, top_k = 2 : i64, force_local_block = true, causal = true}
      : (tensor<1x8x16x8xf32>, tensor<1x2x16x8xf32>,
         tensor<1x2x16x8xf32>) -> tensor<1x8x16x8xf32>
  return %o : tensor<1x8x16x8xf32>
}

// -----

// NEGATIVE: GQA non-divisible (Hq=4, Hkv=3).
func.func @msa_bad_gqa(
    %q: tensor<1x4x16x8xf32>, %k: tensor<1x3x16x8xf32>,
    %v: tensor<1x3x16x8xf32>) -> tensor<1x4x16x8xf32> {
  // expected-error @+1 {{requires Hq % Hkv == 0 (GQA grouping); got Hq=4, Hkv=3}}
  %o = "tessera.msa_sparse_attention"(%q, %k, %v)
      {block_size = 4 : i64, top_k = 2 : i64, force_local_block = true, causal = true}
      : (tensor<1x4x16x8xf32>, tensor<1x3x16x8xf32>,
         tensor<1x3x16x8xf32>) -> tensor<1x4x16x8xf32>
  return %o : tensor<1x4x16x8xf32>
}

// -----

// NEGATIVE: top_k (5) > num_blocks=Sk/block_size (16/4 == 4).
func.func @msa_bad_topk(
    %q: tensor<1x2x16x8xf32>, %k: tensor<1x2x16x8xf32>,
    %v: tensor<1x2x16x8xf32>) -> tensor<1x2x16x8xf32> {
  // expected-error @+1 {{top_k must be <= num_blocks=Sk/block_size; got top_k=5, num_blocks=4}}
  %o = "tessera.msa_sparse_attention"(%q, %k, %v)
      {block_size = 4 : i64, top_k = 5 : i64, force_local_block = true, causal = true}
      : (tensor<1x2x16x8xf32>, tensor<1x2x16x8xf32>,
         tensor<1x2x16x8xf32>) -> tensor<1x2x16x8xf32>
  return %o : tensor<1x2x16x8xf32>
}

// -----

// NEGATIVE: msa_select_blocks top_k (5) > num_blocks (scores dim 3 == 4).
func.func @msa_select_bad_topk(%scores: tensor<1x2x16x4xf32>) -> tensor<1x2x16x5xi64> {
  // expected-error @+1 {{top_k must be <= num_blocks (scores dim 3); got top_k=5, num_blocks=4}}
  %ids = "tessera.msa_select_blocks"(%scores)
      {top_k = 5 : i64, block_size = 4 : i64, force_local_block = true, causal = true}
      : (tensor<1x2x16x4xf32>) -> tensor<1x2x16x5xi64>
  return %ids : tensor<1x2x16x5xi64>
}
