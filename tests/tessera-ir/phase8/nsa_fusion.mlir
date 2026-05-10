// RUN: tessera-opt %s --tessera-native-sparse-attn-fusion | FileCheck %s

// attention_variants_plan, NSA-4 — DeepSeek NSA fusion recognizer.
//
// The pass collapses the three NSA branches (sliding_window +
// compressed_blocks + top_k_blocks) sharing the same Q into a single
// tessera.native_sparse_attn_fused op carrying their attributes.
//
// All three branch outputs are returned so they stay alive through the
// rewrite (otherwise the greedy driver's DCE would drop the unused
// branches before the pattern can match).

func.func @nsa_collapses(%Q: tensor<1x2x16x4xf32>,
                          %K: tensor<1x2x16x4xf32>,
                          %V: tensor<1x2x16x4xf32>,
                          %Kc: tensor<1x2x4x4xf32>,
                          %Vc: tensor<1x2x4x4xf32>,
                          %scores: tensor<1x2x16x4xf32>)
    -> (tensor<1x2x16x4xf32>, tensor<1x2x16x4xf32>, tensor<1x2x16x4xf32>) {
  // CHECK-LABEL: func.func @nsa_collapses
  // CHECK:       tessera.native_sparse_attn_fused
  // CHECK-NOT:   tessera.attn_sliding_window
  // CHECK-NOT:   tessera.attn_compressed_blocks
  // CHECK-NOT:   tessera.attn_top_k_blocks
  %ow = "tessera.attn_sliding_window"(%Q, %K, %V) {window_size = 4 : i64, causal = true}
      : (tensor<1x2x16x4xf32>, tensor<1x2x16x4xf32>, tensor<1x2x16x4xf32>) -> tensor<1x2x16x4xf32>
  %oc = "tessera.attn_compressed_blocks"(%Q, %Kc, %Vc)
      : (tensor<1x2x16x4xf32>, tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>) -> tensor<1x2x16x4xf32>
  %os = "tessera.attn_top_k_blocks"(%Q, %K, %V, %scores)
        {top_k = 2 : i64, block_size = 4 : i64, causal = true}
      : (tensor<1x2x16x4xf32>, tensor<1x2x16x4xf32>, tensor<1x2x16x4xf32>, tensor<1x2x16x4xf32>)
      -> tensor<1x2x16x4xf32>
  return %ow, %oc, %os : tensor<1x2x16x4xf32>, tensor<1x2x16x4xf32>, tensor<1x2x16x4xf32>
}
