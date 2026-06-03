// L-series linalg family (LF1–LF5): the 5 next linalg members each lower
// through the table-driven full Graph→Schedule→Tile→Target Apple spine to a
// tessera_apple.cpu.call (value op) naming the Accelerate LAPACK C ABI symbol.
//
// RUN: tessera-opt -tessera-lower-to-apple_cpu-full --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @tri_solve
// CHECK: tessera_apple.cpu.call
// CHECK-SAME: abi = "lapack_strtrs"
// CHECK-SAME: symbol = "tessera_apple_cpu_tri_solve_f32"
func.func @tri_solve(%a: tensor<4x4xf32>, %b: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %0 = tessera.tri_solve %a, %b : (tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// CHECK-LABEL: func.func @cholesky_solve
// CHECK: tessera_apple.cpu.call
// CHECK-SAME: abi = "lapack_spotrs"
// CHECK-SAME: symbol = "tessera_apple_cpu_cholesky_solve_f32"
func.func @cholesky_solve(%a: tensor<4x4xf32>, %b: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %0 = tessera.cholesky_solve %a, %b : (tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// CHECK-LABEL: func.func @lu
// CHECK: tessera_apple.cpu.call
// CHECK-SAME: abi = "lapack_sgetrf"
// CHECK-SAME: symbol = "tessera_apple_cpu_lu_f32"
func.func @lu(%a: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4xi32>) {
  %lu, %p = tessera.lu %a : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4xi32>)
  return %lu, %p : tensor<4x4xf32>, tensor<4xi32>
}

// CHECK-LABEL: func.func @qr
// CHECK: tessera_apple.cpu.call
// CHECK-SAME: abi = "lapack_sgeqrf"
// CHECK-SAME: symbol = "tessera_apple_cpu_qr_f32"
func.func @qr(%a: tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4x4xf32>) {
  %q, %r = tessera.qr %a : (tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4x4xf32>)
  return %q, %r : tensor<6x4xf32>, tensor<4x4xf32>
}

// CHECK-LABEL: func.func @svd
// CHECK: tessera_apple.cpu.call
// CHECK-SAME: abi = "lapack_sgesvd"
// CHECK-SAME: symbol = "tessera_apple_cpu_svd_f32"
func.func @svd(%a: tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>) {
  %u, %s, %v = tessera.svd %a : (tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>)
  return %u, %s, %v : tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>
}
