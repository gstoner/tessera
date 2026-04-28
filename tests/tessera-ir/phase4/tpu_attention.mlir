// RUN: tessera-opt --tessera-tpu-attention-lowering %s | FileCheck %s

// Test: TPU attention lowering pass converts tessera.flash_attn ops into
// stablehlo composite ops suitable for JAX/XLA TPU code generation.
// Each flash_attn op becomes a stablehlo.composite with name
// "tessera.flash_attention" and a decomposition attribute.

module attributes {
  tessera.tpu_target = {
    generation = "v4",
    mxu_tile = 128,
    dtype = "bf16",
    mesh = {"dp" = 4, "tp" = 2}
  }
} {

  // CHECK-LABEL: func.func @attention_forward
  func.func @attention_forward(
      %q: tensor<4x8x128x128xbf16>,
      %k: tensor<4x8x128x128xbf16>,
      %v: tensor<4x8x128x128xbf16>
  ) -> tensor<4x8x128x128xbf16> {

    // Before lowering: tessera.flash_attn custom op
    // After lowering: stablehlo.composite wrapping flash attention
    // CHECK: stablehlo.composite
    // CHECK-SAME: "tessera.flash_attention"
    // CHECK-SAME: scale
    %scale = arith.constant 0.088388 : bf16
    %out = "tessera.flash_attn"(%q, %k, %v, %scale) {
      tessera.tpu_target = "v4",
      tessera.mxu_tile = 128 : i64,
      causal = true
    } : (tensor<4x8x128x128xbf16>, tensor<4x8x128x128xbf16>,
         tensor<4x8x128x128xbf16>, bf16) -> tensor<4x8x128x128xbf16>

    return %out : tensor<4x8x128x128xbf16>
  }

  // CHECK-LABEL: func.func @attention_with_mask
  func.func @attention_with_mask(
      %q:    tensor<2x4x64x128xbf16>,
      %k:    tensor<2x4x64x128xbf16>,
      %v:    tensor<2x4x64x128xbf16>,
      %mask: tensor<2x4x64x64xi1>
  ) -> tensor<2x4x64x128xbf16> {

    // Masked variant should also produce a composite with mask arg.
    // CHECK: stablehlo.composite
    // CHECK-SAME: "tessera.flash_attention"
    // CHECK-SAME: causal
    %scale = arith.constant 0.125 : bf16
    %out = "tessera.flash_attn"(%q, %k, %v, %scale, %mask) {
      tessera.tpu_target = "v4",
      tessera.mxu_tile = 128 : i64,
      causal = false,
      has_mask = true
    } : (tensor<2x4x64x128xbf16>, tensor<2x4x64x128xbf16>,
         tensor<2x4x64x128xbf16>, bf16, tensor<2x4x64x64xi1>
        ) -> tensor<2x4x64x128xbf16>

    return %out : tensor<2x4x64x128xbf16>
  }

  // Non-TPU-target ops should pass through unchanged.
  // CHECK-LABEL: func.func @passthrough
  func.func @passthrough(%x: tensor<128x256xbf16>) -> tensor<128x256xbf16> {
    // CHECK-NOT: stablehlo.composite
    return %x : tensor<128x256xbf16>
  }
}
