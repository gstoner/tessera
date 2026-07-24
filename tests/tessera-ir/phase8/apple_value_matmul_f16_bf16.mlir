// Apple Value Target IR Sprint 7 — rank-2 f16 / bf16 matmul executes on the CPU
// value lane, lowering to the dtype-specific Accelerate/BNNS GEMM symbol. Each
// stays a single tile.matmul -> tessera_apple.cpu.call (no scf.for, no husk).
//
// REQUIRES: tessera-apple-backend
//
// RUN: %tessera_strict_opt %s -tessera-lower-to-apple_cpu-full | FileCheck %s

// CHECK-LABEL: func.func @matmul_f16
// CHECK: tessera_apple.cpu.call
// CHECK-SAME: abi = "bnns_matmul_f16"
// CHECK-SAME: op_kind = "matmul"
// CHECK-SAME: symbol = "tessera_apple_cpu_gemm_f16"
// CHECK-SAME: : (tensor<4x8xf16>, tensor<8x16xf16>) -> tensor<4x16xf16>
// CHECK-NOT: ub.poison
// CHECK-NOT: scf.for
// CHECK-NOT: tile.matmul
func.func @matmul_f16(%a: tensor<4x8xf16>, %b: tensor<8x16xf16>) -> tensor<4x16xf16> {
  %0 = tessera.matmul %a, %b : (tensor<4x8xf16>, tensor<8x16xf16>) -> tensor<4x16xf16>
  return %0 : tensor<4x16xf16>
}

// CHECK-LABEL: func.func @matmul_bf16
// CHECK: tessera_apple.cpu.call
// CHECK-SAME: abi = "bnns_matmul_bf16"
// CHECK-SAME: symbol = "tessera_apple_cpu_gemm_bf16"
// CHECK-SAME: : (tensor<4x8xbf16>, tensor<8x16xbf16>) -> tensor<4x16xbf16>
func.func @matmul_bf16(%a: tensor<4x8xbf16>, %b: tensor<8x16xbf16>) -> tensor<4x16xbf16> {
  %0 = tessera.matmul %a, %b : (tensor<4x8xbf16>, tensor<8x16xbf16>) -> tensor<4x16xbf16>
  return %0 : tensor<4x16xbf16>
}
