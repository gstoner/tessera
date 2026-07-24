// REQUIRES: tessera-apple-backend
//
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu-runtime)' | FileCheck %s

// attention_variants_plan, LA-2 — Apple GPU MSL linear-attention forward.
//
// The lowering pass replaces a rank-4 f32 `tessera.linear_attn` op with a
// `func.call @tessera_apple_gpu_linear_attn_f32(...)`. Out-of-envelope
// inputs (wrong rank, dynamic shape, D_qk * D_v > 256) stay as
// `tessera.linear_attn` for downstream passes / runtime fallback.

// CHECK-DAG: func.func private @tessera_apple_gpu_linear_attn_f32(i64, i64, i64, i64, i32, i32, i32, i32, i32, i32, i32)

// f32 happy path: D_qk = 4, D_v = 4 → state = 16, well under 256 cap.
func.func @linear_attn_f32_happy(%Q: tensor<2x2x4x4xf32>,
                                  %K: tensor<2x2x4x4xf32>,
                                  %V: tensor<2x2x4x4xf32>) -> tensor<2x2x4x4xf32> {
  // CHECK-LABEL: func.func @linear_attn_f32_happy
  // CHECK:       call @tessera_apple_gpu_linear_attn_f32
  // CHECK-NOT:   tessera.linear_attn
  %0 = "tessera.linear_attn"(%Q, %K, %V) {causal = true, feature_map = "identity"}
      : (tensor<2x2x4x4xf32>, tensor<2x2x4x4xf32>, tensor<2x2x4x4xf32>)
      -> tensor<2x2x4x4xf32>
  return %0 : tensor<2x2x4x4xf32>
}

// State exceeds 256 fp32 cap → falls through.
func.func @linear_attn_state_too_large(%Q: tensor<1x1x4x32xf32>,
                                        %K: tensor<1x1x4x32xf32>,
                                        %V: tensor<1x1x4x32xf32>) -> tensor<1x1x4x32xf32> {
  // CHECK-LABEL: func.func @linear_attn_state_too_large
  // CHECK:       tessera.linear_attn
  // CHECK-NOT:   call @tessera_apple_gpu_linear_attn_f32
  %0 = "tessera.linear_attn"(%Q, %K, %V) {causal = true, feature_map = "identity"}
      : (tensor<1x1x4x32xf32>, tensor<1x1x4x32xf32>, tensor<1x1x4x32xf32>)
      -> tensor<1x1x4x32xf32>
  return %0 : tensor<1x1x4x32xf32>
}

// Non-causal stays in IR (v1 MSL kernel is causal-only).
func.func @linear_attn_non_causal(%Q: tensor<1x1x2x2xf32>,
                                   %K: tensor<1x1x2x2xf32>,
                                   %V: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32> {
  // CHECK-LABEL: func.func @linear_attn_non_causal
  // CHECK:       tessera.linear_attn
  %0 = "tessera.linear_attn"(%Q, %K, %V) {causal = false, feature_map = "identity"}
      : (tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>)
      -> tensor<1x1x2x2xf32>
  return %0 : tensor<1x1x2x2xf32>
}
