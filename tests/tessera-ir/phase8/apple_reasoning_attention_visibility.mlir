// Stage 12 — Lightning / Delta / Kimi / Hybrid passes are compiler-visible,
// not executable Apple GPU lowerings.
//
// RUN: %tessera_strict_opt %s -tessera-hybrid-attn-expand -tessera-lightning-attn-fusion -tessera-delta-attn-chunking | FileCheck %s

func.func @reasoning_family_visible(
    %q: tensor<1x2x8x4xf32>, %k: tensor<1x2x8x4xf32>,
    %v: tensor<1x2x8x4xf32>)
    -> (tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>,
        tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>) {
  // CHECK-LABEL: func.func @reasoning_family_visible
  // CHECK-NOT: tessera_apple.gpu.kernel_call
  // CHECK: tessera.lightning_attention
  // CHECK-SAME: tessera.reasoning.compiler_visible = true
  // CHECK-SAME: tessera.reasoning.family = "lightning"
  // CHECK-SAME: tessera.reasoning.variant = "lightning_attention"
  %light = "tessera.lightning_attention"(%q, %k, %v)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
      : (tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>,
         tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>

  // CHECK: tessera.gated_deltanet
  // CHECK-SAME: tessera.reasoning.compiler_visible = true
  // CHECK-SAME: tessera.reasoning.family = "delta"
  // CHECK-SAME: tessera.reasoning.variant = "gated_deltanet"
  %delta = "tessera.gated_deltanet"(%q, %k, %v)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>}>
      : (tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>,
         tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>

  // CHECK: tessera.kimi_delta_attention
  // CHECK-SAME: tessera.reasoning.compiler_visible = true
  // CHECK-SAME: tessera.reasoning.family = "delta"
  // CHECK-SAME: tessera.reasoning.variant = "kimi_delta_attention"
  %kimi = "tessera.kimi_delta_attention"(%q, %k, %v)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>}>
      : (tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>,
         tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>

  // CHECK: tessera.hybrid_attention
  // CHECK-SAME: pattern = "kimi_kda_mla"
  // CHECK-SAME: tessera.reasoning.compiler_visible = true
  // CHECK-SAME: tessera.reasoning.family = "hybrid"
  // CHECK-SAME: tessera.reasoning.variant = "kimi_kda_mla"
  %hybrid = tessera.hybrid_attention %q, %k, %v
      {pattern = "kimi_kda_mla", layer_index = 1 : i64}
      : (tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>,
         tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>

  return %light, %delta, %kimi, %hybrid
      : tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>,
        tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>
}
