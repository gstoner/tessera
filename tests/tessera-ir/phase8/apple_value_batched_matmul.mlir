// Apple Value Target IR Sprint 6 — fp32 rank-3 batched matmul executes as a
// single Accelerate batched-GEMM value call. The value `-full` pipeline
// preserves the dense batched contraction as one tile op (no scf.for) and
// lowers it to a tessera_apple.cpu.call carrying
// tessera_apple_cpu_gemm_f32_batched.
//
// RUN: tessera-opt %s -tessera-lower-to-apple_cpu-full | FileCheck %s

// CHECK-LABEL: func.func @batched_matmul_value
// CHECK: tessera_apple.cpu.call
// CHECK-SAME: abi = "cblas_sgemm_batched_loop"
// CHECK-SAME: op_kind = "batched_gemm"
// CHECK-SAME: symbol = "tessera_apple_cpu_gemm_f32_batched"
// CHECK-SAME: : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
// CHECK-NOT: ub.poison
// CHECK-NOT: scf.for
// CHECK-NOT: tile.batched_gemm
func.func @batched_matmul_value(%a: tensor<2x4x8xf32>, %b: tensor<2x8x16xf32>) -> tensor<2x4x16xf32> {
  %0 = tessera.batched_gemm %a, %b : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
  return %0 : tensor<2x4x16xf32>
}
