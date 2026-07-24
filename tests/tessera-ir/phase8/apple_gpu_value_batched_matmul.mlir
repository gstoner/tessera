// Apple Value Target IR Sprint 8 — rank-3 batched matmul executes on the Apple
// GPU value lane for f32/f16/bf16, lowering to the dtype-specific MPSGraph bmm
// symbol via tessera_apple.gpu.kernel_call. Each stays a single
// tile.batched_gemm -> gpu.kernel_call (no husk, no scf.for, no tile leftover).
//
// REQUIRES: tessera-apple-backend
//
// RUN: %tessera_strict_opt %s -tessera-lower-to-apple_gpu-full | FileCheck %s

// CHECK-LABEL: func.func @bmm_f32
// CHECK: tessera_apple.gpu.kernel_call
// CHECK-SAME: op_kind = "batched_gemm"
// CHECK-SAME: symbol = "tessera_apple_gpu_bmm_f32"
// CHECK-NOT: ub.poison
// CHECK-NOT: scf.for
// CHECK-NOT: tile.batched_gemm
func.func @bmm_f32(%a: tensor<2x4x8xf32>, %b: tensor<2x8x16xf32>) -> tensor<2x4x16xf32> {
  %0 = tessera.batched_gemm %a, %b : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
  return %0 : tensor<2x4x16xf32>
}

// CHECK-LABEL: func.func @bmm_f16
// CHECK: tessera_apple.gpu.kernel_call
// CHECK-SAME: symbol = "tessera_apple_gpu_bmm_f16"
// CHECK-SAME: : (tensor<2x4x8xf16>, tensor<2x8x16xf16>) -> tensor<2x4x16xf16>
func.func @bmm_f16(%a: tensor<2x4x8xf16>, %b: tensor<2x8x16xf16>) -> tensor<2x4x16xf16> {
  %0 = tessera.batched_gemm %a, %b : (tensor<2x4x8xf16>, tensor<2x8x16xf16>) -> tensor<2x4x16xf16>
  return %0 : tensor<2x4x16xf16>
}

// CHECK-LABEL: func.func @bmm_bf16
// CHECK: tessera_apple.gpu.kernel_call
// CHECK-SAME: symbol = "tessera_apple_gpu_bmm_bf16"
// CHECK-SAME: : (tensor<2x4x8xbf16>, tensor<2x8x16xbf16>) -> tensor<2x4x16xbf16>
func.func @bmm_bf16(%a: tensor<2x4x8xbf16>, %b: tensor<2x8x16xbf16>) -> tensor<2x4x16xbf16> {
  %0 = tessera.batched_gemm %a, %b : (tensor<2x4x8xbf16>, tensor<2x8x16xbf16>) -> tensor<2x4x16xbf16>
  return %0 : tensor<2x4x16xbf16>
}
