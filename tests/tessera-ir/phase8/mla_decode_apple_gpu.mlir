// REQUIRES: tessera-apple-backend
//
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-mla-fusion,tessera-lower-to-apple_gpu-runtime)' | FileCheck %s

// attention_variants_plan, MLA-2 — Apple GPU MLA decode lowering.
//
// The runtime pipeline composes the SwiGLU + MLA fusion passes ahead of
// per-op lowering, so a chain that arrives as the canonical 4-op MLA
// pattern collapses through MLAFusionPass first, then this lowering
// emits a `func.call @tessera_apple_gpu_mla_decode_f32` with the right
// pointer + dim ABI.

// CHECK-DAG: func.func private @tessera_apple_gpu_mla_decode_f32(i64, i64, i64, i64, i64, i64, i32, i32, i32, i32, i32, i32)

// f32 happy path — chain collapses then lowers to the runtime call.
func.func @mla_decode_f32(%x: tensor<8x16xf32>,
                           %Wdkv: tensor<16x32xf32>,
                           %Wuk: tensor<32x16xf32>,
                           %Wuv: tensor<32x16xf32>,
                           %Q: tensor<1x8x16xf32>) -> tensor<1x8x16xf32> {
  // CHECK-LABEL: func.func @mla_decode_f32
  // CHECK:       call @tessera_apple_gpu_mla_decode_f32
  // CHECK-NOT:   tessera.mla_decode_fused
  // CHECK-NOT:   tessera.latent_kv_compress
  %c = "tessera.latent_kv_compress"(%x, %Wdkv) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %K = "tessera.latent_kv_expand_k"(%c, %Wuk) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<1x8x16xf32>
  %V = "tessera.latent_kv_expand_v"(%c, %Wuv) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<1x8x16xf32>
  %O = "tessera.flash_attn"(%Q, %K, %V) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {head_dim = 16 : i64, causal = false}
      : (tensor<1x8x16xf32>, tensor<1x8x16xf32>, tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
  return %O : tensor<1x8x16xf32>
}
