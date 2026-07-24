// REQUIRES: tessera-apple-backend
//
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu-runtime)' --allow-unregistered-dialect | FileCheck %s

// Phase 8.4.5 — 3-op MSL fusion: matmul -> softmax -> matmul (full
// attention block). Verifies that the runtime pipeline collapses a 3-op
// SSA chain into a single func.call into the fused runtime symbol. The
// 3-op pass runs BEFORE the 2-op pass so the longer chain wins.

// Runtime declarations: one per dtype. CHECK-DAG since order is
// implementation-defined.
// CHECK-DAG: func.func private @tessera_apple_gpu_matmul_softmax_matmul_f32(i64, i64, i64, i64, i32, i32, i32, i32)
// CHECK-DAG: func.func private @tessera_apple_gpu_matmul_softmax_matmul_f16(i64, i64, i64, i64, i32, i32, i32, i32)
// CHECK-DAG: func.func private @tessera_apple_gpu_matmul_softmax_matmul_bf16(i64, i64, i64, i64, i32, i32, i32, i32)

func.func @attn_block_f32(%A: tensor<8x16xf32>, %B: tensor<16x32xf32>, %C: tensor<32x16xf32>) -> tensor<8x16xf32> {
  // CHECK-LABEL: func.func @attn_block_f32
  // CHECK:       call @tessera_apple_gpu_matmul_softmax_matmul_f32
  // CHECK-NOT:   tessera.matmul
  // CHECK-NOT:   tessera.softmax
  // CHECK-NOT:   call @tessera_apple_gpu_synth_matmul_epilogue_f32
  // CHECK-NOT:   call @tessera_apple_gpu_mps_matmul_f32
  %m1 = "tessera.matmul"(%A, %B) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %p  = "tessera.softmax"(%m1) : (tensor<8x32xf32>) -> tensor<8x32xf32>
  %o  = "tessera.matmul"(%p, %C) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %o : tensor<8x16xf32>
}

func.func @attn_block_f16(%A: tensor<8x16xf16>, %B: tensor<16x32xf16>, %C: tensor<32x16xf16>) -> tensor<8x16xf16> {
  // CHECK-LABEL: func.func @attn_block_f16
  // CHECK:       call @tessera_apple_gpu_matmul_softmax_matmul_f16
  %m1 = "tessera.matmul"(%A, %B) : (tensor<8x16xf16>, tensor<16x32xf16>) -> tensor<8x32xf16>
  %p  = "tessera.softmax"(%m1) : (tensor<8x32xf16>) -> tensor<8x32xf16>
  %o  = "tessera.matmul"(%p, %C) : (tensor<8x32xf16>, tensor<32x16xf16>) -> tensor<8x16xf16>
  return %o : tensor<8x16xf16>
}

func.func @attn_block_bf16(%A: tensor<8x16xbf16>, %B: tensor<16x32xbf16>, %C: tensor<32x16xbf16>) -> tensor<8x16xbf16> {
  // CHECK-LABEL: func.func @attn_block_bf16
  // CHECK:       call @tessera_apple_gpu_matmul_softmax_matmul_bf16
  %m1 = "tessera.matmul"(%A, %B) : (tensor<8x16xbf16>, tensor<16x32xbf16>) -> tensor<8x32xbf16>
  %p  = "tessera.softmax"(%m1) : (tensor<8x32xbf16>) -> tensor<8x32xbf16>
  %o  = "tessera.matmul"(%p, %C) : (tensor<8x32xbf16>, tensor<32x16xbf16>) -> tensor<8x16xbf16>
  return %o : tensor<8x16xbf16>
}

// Negative case: extra use of the softmax intermediate prevents the 3-op
// fusion. The 2-op fusion still fires for matmul -> softmax; the second
// matmul stays standalone.

// CHECK-LABEL: func.func @attn_block_extra_softmax_use
// CHECK:       call @tessera_apple_gpu_synth_matmul_epilogue_f32
// CHECK-NOT:   call @tessera_apple_gpu_matmul_softmax_matmul_f32

func.func @attn_block_extra_softmax_use(%A: tensor<8x16xf32>, %B: tensor<16x32xf32>, %C: tensor<32x16xf32>) -> (tensor<8x16xf32>, tensor<8x32xf32>) {
  %m1 = "tessera.matmul"(%A, %B) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %p  = "tessera.softmax"(%m1) : (tensor<8x32xf32>) -> tensor<8x32xf32>
  %o  = "tessera.matmul"(%p, %C) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %o, %p : tensor<8x16xf32>, tensor<8x32xf32>
}

// Negative case: N > 256 (softmax intermediate too wide for the per-thread
// stack array) falls out of the 3-op fusion. The 2-op fusion also has the
// same N constraint and rejects, so each op falls through to its standalone
// runtime symbol — that's still correct, just not maximally fused.

// CHECK-LABEL: func.func @attn_block_n_too_big
// CHECK-NOT:   call @tessera_apple_gpu_matmul_softmax_matmul
// CHECK-NOT:   call @tessera_apple_gpu_synth_matmul_epilogue_f32

func.func @attn_block_n_too_big(%A: tensor<2x4xf32>, %B: tensor<4x512xf32>, %C: tensor<512x16xf32>) -> tensor<2x16xf32> {
  %m1 = "tessera.matmul"(%A, %B) : (tensor<2x4xf32>, tensor<4x512xf32>) -> tensor<2x512xf32>
  %p  = "tessera.softmax"(%m1) : (tensor<2x512xf32>) -> tensor<2x512xf32>
  %o  = "tessera.matmul"(%p, %C) : (tensor<2x512xf32>, tensor<512x16xf32>) -> tensor<2x16xf32>
  return %o : tensor<2x16xf32>
}
