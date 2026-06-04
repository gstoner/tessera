// Apple Value Target IR Sprint 5 — fp32 rank-2 matmul executes as a single
// Accelerate GEMM value call. The value `-full` pipeline preserves the dense
// contraction as one tile op (no scf.for) and lowers it to a
// tessera_apple.cpu.call carrying tessera_apple_cpu_gemm_f32 — the first
// non-linalg executable value op.
//
// RUN: %tessera_strict_opt %s -tessera-lower-to-apple_cpu-full | FileCheck %s

// CHECK-LABEL: func.func @matmul_value
// CHECK: tessera_apple.cpu.call
// CHECK-SAME: abi = "cblas_sgemm"
// CHECK-SAME: op_kind = "matmul"
// CHECK-SAME: symbol = "tessera_apple_cpu_gemm_f32"
// CHECK-SAME: : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
// CHECK-NOT: ub.poison
// CHECK-NOT: scf.for
// CHECK-NOT: tile.matmul
func.func @matmul_value(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %0 = tessera.matmul %a, %b : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}
