// RUN: tessera-opt %s --tessera-mla-fusion | FileCheck %s

// attention_variants_plan, MLA-1 — DeepSeek MLA decode fusion recognizer.
//
// Verifies that the Schedule IR pass collapses the canonical 4-op chain
//   compress → expand_k → expand_v → flash_attn
// into a single tessera.mla_decode_fused, and that it declines to match
// when the chain shape differs (different %c feeding expand_k vs
// expand_v, expanded K/V used by extra consumers, etc.).

// Positive: canonical MLA decode chain collapses to mla_decode_fused.
func.func @mla_collapses(%x: tensor<8x16xf32>,
                          %Wdkv: tensor<16x32xf32>,
                          %Wuk: tensor<32x16xf32>,
                          %Wuv: tensor<32x16xf32>,
                          %Q: tensor<1x8x16xf32>) -> tensor<1x8x16xf32> {
  // CHECK-LABEL: func.func @mla_collapses
  // CHECK:       tessera.mla_decode_fused
  // CHECK-NOT:   tessera.latent_kv_compress
  // CHECK-NOT:   tessera.latent_kv_expand_k
  // CHECK-NOT:   tessera.latent_kv_expand_v
  %c = "tessera.latent_kv_compress"(%x, %Wdkv) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %K = "tessera.latent_kv_expand_k"(%c, %Wuk) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<1x8x16xf32>
  %V = "tessera.latent_kv_expand_v"(%c, %Wuv) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<1x8x16xf32>
  %O = "tessera.flash_attn"(%Q, %K, %V) {head_dim = 16 : i64, causal = false}
      : (tensor<1x8x16xf32>, tensor<1x8x16xf32>, tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
  return %O : tensor<1x8x16xf32>
}

// Negative #1: expand_k and expand_v consume different latents — not a
// real MLA chain. Pattern must decline.
func.func @two_latents_no_fusion(%x1: tensor<8x16xf32>,
                                  %x2: tensor<8x16xf32>,
                                  %Wdkv: tensor<16x32xf32>,
                                  %Wuk: tensor<32x16xf32>,
                                  %Wuv: tensor<32x16xf32>,
                                  %Q: tensor<1x8x16xf32>) -> tensor<1x8x16xf32> {
  // CHECK-LABEL: func.func @two_latents_no_fusion
  // CHECK:       tessera.latent_kv_compress
  // CHECK:       tessera.latent_kv_compress
  // CHECK:       tessera.flash_attn
  // CHECK-NOT:   tessera.mla_decode_fused
  %c1 = "tessera.latent_kv_compress"(%x1, %Wdkv) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %c2 = "tessera.latent_kv_compress"(%x2, %Wdkv) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %K = "tessera.latent_kv_expand_k"(%c1, %Wuk) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<1x8x16xf32>
  %V = "tessera.latent_kv_expand_v"(%c2, %Wuv) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<1x8x16xf32>
  %O = "tessera.flash_attn"(%Q, %K, %V) {head_dim = 16 : i64, causal = false}
      : (tensor<1x8x16xf32>, tensor<1x8x16xf32>, tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
  return %O : tensor<1x8x16xf32>
}

// Negative #2: expanded K has a second consumer beyond flash_attn.
// Fusing would lose that consumer's input. Pattern must decline.
func.func @k_with_extra_user(%x: tensor<8x16xf32>,
                              %Wdkv: tensor<16x32xf32>,
                              %Wuk: tensor<32x16xf32>,
                              %Wuv: tensor<32x16xf32>,
                              %Q: tensor<1x8x16xf32>)
    -> (tensor<1x8x16xf32>, tensor<1x8x16xf32>) {
  // CHECK-LABEL: func.func @k_with_extra_user
  // CHECK:       tessera.latent_kv_expand_k
  // CHECK-NOT:   tessera.mla_decode_fused
  %c = "tessera.latent_kv_compress"(%x, %Wdkv) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %K = "tessera.latent_kv_expand_k"(%c, %Wuk) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<1x8x16xf32>
  %V = "tessera.latent_kv_expand_v"(%c, %Wuv) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<1x8x16xf32>
  %O = "tessera.flash_attn"(%Q, %K, %V) {head_dim = 16 : i64, causal = false}
      : (tensor<1x8x16xf32>, tensor<1x8x16xf32>, tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
  return %O, %K : tensor<1x8x16xf32>, tensor<1x8x16xf32>
}

// numeric_policy propagation (audit 2026-06-10, Decision #15a): the
// flash_attn's numeric_policy must survive onto the fused op — the
// attention step dominates the fused kernel's numerics.
func.func @mla_propagates_numeric_policy(%x: tensor<8x16xf32>,
                                          %Wdkv: tensor<16x32xf32>,
                                          %Wuk: tensor<32x16xf32>,
                                          %Wuv: tensor<32x16xf32>,
                                          %Q: tensor<1x8x16xf32>) -> tensor<1x8x16xf32> {
  // CHECK-LABEL: func.func @mla_propagates_numeric_policy
  // CHECK:       tessera.mla_decode_fused
  // CHECK-SAME:  numeric_policy = {accum = "fp32", storage = "bf16"}
  %c = "tessera.latent_kv_compress"(%x, %Wdkv) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %K = "tessera.latent_kv_expand_k"(%c, %Wuk) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<1x8x16xf32>
  %V = "tessera.latent_kv_expand_v"(%c, %Wuv) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<1x8x16xf32>
  %O = "tessera.flash_attn"(%Q, %K, %V)
      {head_dim = 16 : i64, causal = false,
       numeric_policy = {storage = "bf16", accum = "fp32"}}
      : (tensor<1x8x16xf32>, tensor<1x8x16xf32>, tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
  return %O : tensor<1x8x16xf32>
}
