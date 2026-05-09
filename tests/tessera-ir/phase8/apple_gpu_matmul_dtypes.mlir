// RUN: tessera-opt %s --pass-pipeline='builtin.module(tessera-lower-to-apple_gpu-runtime)' --allow-unregistered-dialect | FileCheck %s

// Phase 8.4.4 — Apple GPU fp16 + bf16 matmul. Verifies that the runtime
// pipeline picks the right runtime symbol based on the matmul's input
// element type:
//   f32  -> tessera_apple_gpu_mps_matmul_f32   (Phase 8.3, native MPS)
//   f16  -> tessera_apple_gpu_mps_matmul_f16   (Phase 8.4.4, native MPS)
//   bf16 -> tessera_apple_gpu_mps_matmul_bf16  (Phase 8.4.4, fp32 conversion)
//
// Each runtime symbol shares the same i64×3 + i32×3 ABI shape — the element
// type is encoded in the symbol name, not the signature.

// CHECK-DAG: func.func private @tessera_apple_gpu_mps_matmul_f32(i64, i64, i64, i32, i32, i32)
// CHECK-DAG: func.func private @tessera_apple_gpu_mps_matmul_f16(i64, i64, i64, i32, i32, i32)
// CHECK-DAG: func.func private @tessera_apple_gpu_mps_matmul_bf16(i64, i64, i64, i32, i32, i32)

func.func @gemm_f32(%A: tensor<8x16xf32>, %B: tensor<16x32xf32>) -> tensor<8x32xf32> {
  // CHECK-LABEL: func.func @gemm_f32
  // CHECK:       call @tessera_apple_gpu_mps_matmul_f32
  // CHECK-NOT:   tessera.matmul
  %C = "tessera.matmul"(%A, %B) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %C : tensor<8x32xf32>
}

func.func @gemm_f16(%A: tensor<8x16xf16>, %B: tensor<16x32xf16>) -> tensor<8x32xf16> {
  // CHECK-LABEL: func.func @gemm_f16
  // CHECK:       call @tessera_apple_gpu_mps_matmul_f16
  // CHECK-NOT:   tessera.matmul
  %C = "tessera.matmul"(%A, %B) : (tensor<8x16xf16>, tensor<16x32xf16>) -> tensor<8x32xf16>
  return %C : tensor<8x32xf16>
}

func.func @gemm_bf16(%A: tensor<8x16xbf16>, %B: tensor<16x32xbf16>) -> tensor<8x32xbf16> {
  // CHECK-LABEL: func.func @gemm_bf16
  // CHECK:       call @tessera_apple_gpu_mps_matmul_bf16
  // CHECK-NOT:   tessera.matmul
  %C = "tessera.matmul"(%A, %B) : (tensor<8x16xbf16>, tensor<16x32xbf16>) -> tensor<8x32xbf16>
  return %C : tensor<8x32xbf16>
}

// Negative case: mismatched element types fall back to the artifact path.

// CHECK-LABEL: func.func @gemm_mixed_dtypes
// CHECK:       tessera.matmul

func.func @gemm_mixed_dtypes(%A: tensor<8x16xf32>, %B: tensor<16x32xf16>) -> tensor<8x32xf32> {
  %C = "tessera.matmul"(%A, %B) : (tensor<8x16xf32>, tensor<16x32xf16>) -> tensor<8x32xf32>
  return %C : tensor<8x32xf32>
}
