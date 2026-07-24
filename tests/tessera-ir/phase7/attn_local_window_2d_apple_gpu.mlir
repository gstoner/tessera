// REQUIRES: tessera-apple-backend
//
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu-runtime)' | FileCheck %s

// ============================================================================
// Sub-2 — Apple GPU lowering for tessera.attn_local_window_2d.
//
// The pass replaces the Graph IR op with a func.call into the Apple GPU
// runtime symbol ``tessera_apple_gpu_attn_local_window_2d_f32`` (declared
// at module level as a private extern), passing four i64 pointers
// (Q, K, V, O) and seven i32 dims/window scalars.
// ============================================================================

// CHECK: func.func private @tessera_apple_gpu_attn_local_window_2d_f32
// CHECK-LABEL: func @attn_local_window_2d_static_f32
// CHECK:       memref.alloc() : memref<2x4x8x8x16xf32>
// CHECK:       call @tessera_apple_gpu_attn_local_window_2d_f32
// The Graph IR op is fully replaced (no surviving tessera.attn_local_window_2d).
// CHECK-NOT:   tessera.attn_local_window_2d
func.func @attn_local_window_2d_static_f32(
    %q: tensor<2x4x8x8x16xf32>,
    %k: tensor<2x4x8x8x16xf32>,
    %v: tensor<2x4x8x8x16xf32>
) -> tensor<2x4x8x8x16xf32> {
  %o = tessera.attn_local_window_2d %q, %k, %v {window = [1, 1]} :
      (tensor<2x4x8x8x16xf32>, tensor<2x4x8x8x16xf32>, tensor<2x4x8x8x16xf32>)
      -> tensor<2x4x8x8x16xf32>
  return %o : tensor<2x4x8x8x16xf32>
}

// ============================================================================
// Out-of-envelope inputs stay in IR.
// ============================================================================
//
// bf16 is not yet wired in the Apple GPU runtime — pass falls through
// and leaves the op for the host reference / future lowering.

// CHECK-LABEL: func @attn_local_window_2d_bf16_falls_through
// CHECK:       tessera.attn_local_window_2d
// CHECK-NOT:   call @tessera_apple_gpu_attn_local_window_2d_f32
func.func @attn_local_window_2d_bf16_falls_through(
    %q: tensor<1x1x4x4x8xbf16>,
    %k: tensor<1x1x4x4x8xbf16>,
    %v: tensor<1x1x4x4x8xbf16>
) -> tensor<1x1x4x4x8xbf16> {
  %o = tessera.attn_local_window_2d %q, %k, %v {window = [1, 1]} :
      (tensor<1x1x4x4x8xbf16>, tensor<1x1x4x4x8xbf16>, tensor<1x1x4x4x8xbf16>)
      -> tensor<1x1x4x4x8xbf16>
  return %o : tensor<1x1x4x4x8xbf16>
}

// ============================================================================
// Asymmetric window — (rh, rw) is correctly threaded into the call.
// ============================================================================

// CHECK-LABEL: func @attn_local_window_2d_asymmetric
// 2 — rh attribute on the runtime call.
// CHECK-DAG:   arith.constant 2 : i32
// 3 — rw attribute on the runtime call.
// CHECK-DAG:   arith.constant 3 : i32
// CHECK:       call @tessera_apple_gpu_attn_local_window_2d_f32
func.func @attn_local_window_2d_asymmetric(
    %q: tensor<1x2x6x6x8xf32>,
    %k: tensor<1x2x6x6x8xf32>,
    %v: tensor<1x2x6x6x8xf32>
) -> tensor<1x2x6x6x8xf32> {
  %o = tessera.attn_local_window_2d %q, %k, %v {window = [2, 3]} :
      (tensor<1x2x6x6x8xf32>, tensor<1x2x6x6x8xf32>, tensor<1x2x6x6x8xf32>)
      -> tensor<1x2x6x6x8xf32>
  return %o : tensor<1x2x6x6x8xf32>
}
