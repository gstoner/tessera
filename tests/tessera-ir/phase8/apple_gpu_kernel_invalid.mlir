// E2E-REAL-6 Apple GPU linalg (2026-11-04): an opted-in Apple GPU module fails
// closed on any semantic the Metal kernels do not implement, instead of letting
// the delegate compute a different result.
// RUN: not tessera-opt --tessera-graph-to-schedule --split-input-file %s 2>&1 | FileCheck %s

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple_gpu", tessera.launch_bindings = ["a", "l"]} {
  func.func @upper_cholesky(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tessera.cholesky %a {lower = false} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}

// CHECK: error: Apple GPU native kernel: cholesky implements lower = true into the matrix shape only

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple_gpu", tessera.launch_bindings = ["a", "b", "x"]} {
  func.func @upper_cholesky_solve(%a: tensor<4x4xf32>, %b: tensor<4x2xf32>) -> tensor<4x2xf32> {
    %0 = tessera.cholesky_solve %a, %b {lower = false} : (tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
    return %0 : tensor<4x2xf32>
  }
}

// CHECK: error: Apple GPU native kernel: cholesky_solve implements lower = true only

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple_gpu", tessera.launch_bindings = ["a", "u", "s", "v"]} {
  func.func @wide_svd(%a: tensor<4x6xf32>) -> (tensor<4x4xf32>, tensor<4xf32>, tensor<4x6xf32>) {
    %u, %s, %v = tessera.svd %a : (tensor<4x6xf32>) -> (tensor<4x4xf32>, tensor<4xf32>, tensor<4x6xf32>)
    return %u, %s, %v : tensor<4x4xf32>, tensor<4xf32>, tensor<4x6xf32>
  }
}

// CHECK: error: Apple GPU native kernel: svd requires a tall matrix (M >= N)

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple_gpu", tessera.launch_bindings = ["a", "u", "s", "v"]} {
  func.func @full_svd(%a: tensor<6x4xf32>) -> (tensor<6x6xf32>, tensor<4xf32>, tensor<4x4xf32>) {
    %u, %s, %v = tessera.svd %a {full_matrices = true} : (tensor<6x4xf32>) -> (tensor<6x6xf32>, tensor<4xf32>, tensor<4x4xf32>)
    return %u, %s, %v : tensor<6x6xf32>, tensor<4xf32>, tensor<4x4xf32>
  }
}

// CHECK: error: Apple GPU native kernel: svd implements full_matrices = false only

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple_gpu", tessera.launch_bindings = ["a", "l"]} {
  func.func @f16_cholesky(%a: tensor<4x4xf16>) -> tensor<4x4xf16> {
    %0 = tessera.cholesky %a : (tensor<4x4xf16>) -> tensor<4x4xf16>
    return %0 : tensor<4x4xf16>
  }
}

// CHECK: error: Apple GPU native kernel: requires static positive f32 operands within the int32 ABI

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["a", "l"]} {
  func.func @wrong_arch(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tessera.cholesky %a : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}

// CHECK: error: Apple GPU native kernel: requires the apple_gpu Apple GPU target

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple_gpu", tessera.launch_bindings = ["a", "l"]} {
  func.func @unknown_policy(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tessera.cholesky %a {tessera.tile_q = 4 : i64} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}

// CHECK: error: Apple GPU native kernel: has an unsupported policy attribute 'tessera.tile_q'
