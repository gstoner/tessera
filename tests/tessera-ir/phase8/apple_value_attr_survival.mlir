// Apple Value Target IR Sprint 3 — linalg semantic attrs survive the value-mode
// lowering (Graph IR tessera.* -> tessera_apple.cpu.call) end-to-end.
//
// The CPU value executor reads `lower`/`trans`/`unit_diag`/`full_matrices` from
// the emitted cpu.call to parameterize the LAPACK ABI, so the `-full` pipeline
// must copy these optional attrs from the source op onto the value call. This
// pins that survival via the real value-mode pipeline.
//
// RUN: tessera-opt %s -tessera-lower-to-apple_cpu-full | FileCheck %s

// tri_solve carries lower/trans/unit_diag; they must reach the value call.
// CHECK-LABEL: func.func @tri_solve_attrs
// CHECK: tessera_apple.cpu.call
// CHECK-SAME: lower = false
// CHECK-SAME: op_kind = "tri_solve"
// CHECK-SAME: symbol = "tessera_apple_cpu_tri_solve_f32"
// CHECK-SAME: trans = true
// CHECK-SAME: unit_diag = true
// CHECK-NOT: ub.poison
// CHECK-NOT: tile.tri_solve
func.func @tri_solve_attrs(%a: tensor<4x4xf32>, %b: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %0 = tessera.tri_solve %a, %b {lower = false, trans = true, unit_diag = true}
      : (tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// cholesky carries the `lower` factor selector.
// CHECK-LABEL: func.func @cholesky_attr
// CHECK: tessera_apple.cpu.call
// CHECK-SAME: lower = true
// CHECK-SAME: op_kind = "cholesky"
// CHECK-SAME: symbol = "tessera_apple_cpu_cholesky_f32"
func.func @cholesky_attr(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = tessera.cholesky %a {lower = true} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
