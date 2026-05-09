// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu-runtime)' --allow-unregistered-dialect | FileCheck %s

// Phase 8.4.4.2 — fp16 / bf16 dtype variants for the fused matmul -> softmax
// chain and for flash_attn. Both kernels use mixed-precision: half/bfloat
// I/O at the boundary, fp32 per-thread accumulators internally. The
// lowering passes pick the runtime symbol by input element type.

// Runtime declarations: one per (kernel, dtype) pair.
// CHECK-DAG: func.func private @tessera_apple_gpu_matmul_softmax_f32(i64, i64, i64, i32, i32, i32)
// CHECK-DAG: func.func private @tessera_apple_gpu_matmul_softmax_f16(i64, i64, i64, i32, i32, i32)
// CHECK-DAG: func.func private @tessera_apple_gpu_matmul_softmax_bf16(i64, i64, i64, i32, i32, i32)
// CHECK-DAG: func.func private @tessera_apple_gpu_flash_attn_f32(i64, i64, i64, i64, i32, i32, i32, i32, f32, i32)
// CHECK-DAG: func.func private @tessera_apple_gpu_flash_attn_f16(i64, i64, i64, i64, i32, i32, i32, i32, f32, i32)
// CHECK-DAG: func.func private @tessera_apple_gpu_flash_attn_bf16(i64, i64, i64, i64, i32, i32, i32, i32, f32, i32)

// Fused matmul -> softmax chains, one per dtype.

func.func @fused_f32(%A: tensor<8x16xf32>, %B: tensor<16x32xf32>) -> tensor<8x32xf32> {
  // CHECK-LABEL: func.func @fused_f32
  // CHECK:       call @tessera_apple_gpu_matmul_softmax_f32
  // CHECK-NOT:   tessera.matmul
  // CHECK-NOT:   tessera.softmax
  %m = "tessera.matmul"(%A, %B) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %o = "tessera.softmax"(%m) : (tensor<8x32xf32>) -> tensor<8x32xf32>
  return %o : tensor<8x32xf32>
}

func.func @fused_f16(%A: tensor<8x16xf16>, %B: tensor<16x32xf16>) -> tensor<8x32xf16> {
  // CHECK-LABEL: func.func @fused_f16
  // CHECK:       call @tessera_apple_gpu_matmul_softmax_f16
  // CHECK-NOT:   tessera.matmul
  // CHECK-NOT:   tessera.softmax
  %m = "tessera.matmul"(%A, %B) : (tensor<8x16xf16>, tensor<16x32xf16>) -> tensor<8x32xf16>
  %o = "tessera.softmax"(%m) : (tensor<8x32xf16>) -> tensor<8x32xf16>
  return %o : tensor<8x32xf16>
}

func.func @fused_bf16(%A: tensor<8x16xbf16>, %B: tensor<16x32xbf16>) -> tensor<8x32xbf16> {
  // CHECK-LABEL: func.func @fused_bf16
  // CHECK:       call @tessera_apple_gpu_matmul_softmax_bf16
  // CHECK-NOT:   tessera.matmul
  // CHECK-NOT:   tessera.softmax
  %m = "tessera.matmul"(%A, %B) : (tensor<8x16xbf16>, tensor<16x32xbf16>) -> tensor<8x32xbf16>
  %o = "tessera.softmax"(%m) : (tensor<8x32xbf16>) -> tensor<8x32xbf16>
  return %o : tensor<8x32xbf16>
}

// Standalone flash_attn, one per dtype.

func.func @flash_f32(%Q: tensor<2x8x16xf32>, %K: tensor<2x8x16xf32>, %V: tensor<2x8x16xf32>) -> tensor<2x8x16xf32> {
  // CHECK-LABEL: func.func @flash_f32
  // CHECK:       call @tessera_apple_gpu_flash_attn_f32
  %O = "tessera.flash_attn"(%Q, %K, %V) {causal = false, head_dim = 16 : i64} : (tensor<2x8x16xf32>, tensor<2x8x16xf32>, tensor<2x8x16xf32>) -> tensor<2x8x16xf32>
  return %O : tensor<2x8x16xf32>
}

func.func @flash_f16(%Q: tensor<2x8x16xf16>, %K: tensor<2x8x16xf16>, %V: tensor<2x8x16xf16>) -> tensor<2x8x16xf16> {
  // CHECK-LABEL: func.func @flash_f16
  // CHECK:       call @tessera_apple_gpu_flash_attn_f16
  %O = "tessera.flash_attn"(%Q, %K, %V) {causal = false, head_dim = 16 : i64} : (tensor<2x8x16xf16>, tensor<2x8x16xf16>, tensor<2x8x16xf16>) -> tensor<2x8x16xf16>
  return %O : tensor<2x8x16xf16>
}

func.func @flash_bf16(%Q: tensor<2x8x16xbf16>, %K: tensor<2x8x16xbf16>, %V: tensor<2x8x16xbf16>) -> tensor<2x8x16xbf16> {
  // CHECK-LABEL: func.func @flash_bf16
  // CHECK:       call @tessera_apple_gpu_flash_attn_bf16
  %O = "tessera.flash_attn"(%Q, %K, %V) {causal = false, head_dim = 16 : i64} : (tensor<2x8x16xbf16>, tensor<2x8x16xbf16>, tensor<2x8x16xbf16>) -> tensor<2x8x16xbf16>
  return %O : tensor<2x8x16xbf16>
}

// Mixed-dtype operands fall out of the fusion. The matmul stays as
// tessera.matmul (its operand types differ from the result, so the matmul
// pass also rejects it), but the softmax (with f32 input) still gets
// lowered to its standalone f32 runtime symbol.

// CHECK-LABEL: func.func @fused_mixed_dtypes
// CHECK:       tessera.matmul
// CHECK-NOT:   call @tessera_apple_gpu_matmul_softmax

func.func @fused_mixed_dtypes(%A: tensor<8x16xf16>, %B: tensor<16x32xf32>) -> tensor<8x32xf32> {
  %m = "tessera.matmul"(%A, %B) : (tensor<8x16xf16>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %o = "tessera.softmax"(%m) : (tensor<8x32xf32>) -> tensor<8x32xf32>
  return %o : tensor<8x32xf32>
}
