// REQUIRES: tessera-apple-backend
//
// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu-runtime)' | FileCheck %s

// Phase 8.4.8 — Apple GPU SwiGLU fused MLP-block kernel.
//
// The Schedule IR fusion recognizer (Stage 2b) collapses the
// `matmul → silu_mul → matmul` chain into `tessera.swiglu_fused`. This
// fixture exercises the Apple GPU lowering pipeline directly on the
// fused op (the form the apple_gpu pipeline sees in practice once
// Stage 2b has run on the Graph IR layer above).

// Runtime declarations for all three dtype variants (CHECK-DAG since order
// is implementation-defined). All three sit at the top of the lowered
// module before any function body, so they all need to live before the
// first CHECK-LABEL — otherwise FileCheck would only search forward from
// the last match and miss the prior decls.
// CHECK-DAG: func.func private @tessera_apple_gpu_swiglu_f32(i64, i64, i64, i64, i64, i32, i32, i32, i32)
// CHECK-DAG: func.func private @tessera_apple_gpu_swiglu_f16(i64, i64, i64, i64, i64, i32, i32, i32, i32)
// CHECK-DAG: func.func private @tessera_apple_gpu_swiglu_bf16(i64, i64, i64, i64, i64, i32, i32, i32, i32)

// f32 happy path: H = 32, K_out = 16 — both within the 256 cap.
func.func @swiglu_f32(%x: tensor<8x16xf32>,
                      %Wg: tensor<16x32xf32>,
                      %Wu: tensor<16x32xf32>,
                      %Wd: tensor<32x16xf32>) -> tensor<8x16xf32> {
  // CHECK-LABEL: func.func @swiglu_f32
  // CHECK:       call @tessera_apple_gpu_swiglu_f32
  // CHECK-NOT:   tessera.swiglu_fused
  // CHECK-NOT:   tessera.silu_mul
  // CHECK-NOT:   tessera.matmul
  %y = "tessera.swiglu_fused"(%x, %Wg, %Wu, %Wd)
      : (tensor<8x16xf32>, tensor<16x32xf32>, tensor<16x32xf32>, tensor<32x16xf32>)
      -> tensor<8x16xf32>
  return %y : tensor<8x16xf32>
}

func.func @swiglu_f16(%x: tensor<4x8xf16>,
                      %Wg: tensor<8x16xf16>,
                      %Wu: tensor<8x16xf16>,
                      %Wd: tensor<16x8xf16>) -> tensor<4x8xf16> {
  // CHECK-LABEL: func.func @swiglu_f16
  // CHECK:       call @tessera_apple_gpu_swiglu_f16
  %y = "tessera.swiglu_fused"(%x, %Wg, %Wu, %Wd)
      : (tensor<4x8xf16>, tensor<8x16xf16>, tensor<8x16xf16>, tensor<16x8xf16>)
      -> tensor<4x8xf16>
  return %y : tensor<4x8xf16>
}

func.func @swiglu_bf16(%x: tensor<4x8xbf16>,
                       %Wg: tensor<8x16xbf16>,
                       %Wu: tensor<8x16xbf16>,
                       %Wd: tensor<16x8xbf16>) -> tensor<4x8xbf16> {
  // CHECK-LABEL: func.func @swiglu_bf16
  // CHECK:       call @tessera_apple_gpu_swiglu_bf16
  %y = "tessera.swiglu_fused"(%x, %Wg, %Wu, %Wd)
      : (tensor<4x8xbf16>, tensor<8x16xbf16>, tensor<8x16xbf16>, tensor<16x8xbf16>)
      -> tensor<4x8xbf16>
  return %y : tensor<4x8xbf16>
}

// H > 256 must NOT lower to the fused MSL kernel — falls through. The fused
// op stays in the IR (no per-op pattern claims `tessera.swiglu_fused`),
// which is exactly the contract: the runtime path then handles it via
// fallback decomposition.
func.func @swiglu_too_wide(%x: tensor<2x4xf32>,
                           %Wg: tensor<4x512xf32>,
                           %Wu: tensor<4x512xf32>,
                           %Wd: tensor<512x4xf32>) -> tensor<2x4xf32> {
  // CHECK-LABEL: func.func @swiglu_too_wide
  // CHECK:       tessera.swiglu_fused
  // CHECK-NOT:   call @tessera_apple_gpu_swiglu
  %y = "tessera.swiglu_fused"(%x, %Wg, %Wu, %Wd)
      : (tensor<2x4xf32>, tensor<4x512xf32>, tensor<4x512xf32>, tensor<512x4xf32>)
      -> tensor<2x4xf32>
  return %y : tensor<2x4xf32>
}
