// E2E-REAL-6 Apple CPU factorizations (2026-11-04): an opted-in Apple CPU
// module fails closed on any semantic the Accelerate/LAPACK C ABI does not
// implement, instead of letting the delegate compute a different result.
// RUN: not tessera-opt --tessera-graph-to-schedule --split-input-file %s 2>&1 | FileCheck %s

module attributes {tessera.target = "apple_cpu", tessera.arch = "apple_silicon_cpu", tessera.launch_bindings = ["a", "l"]} {
  func.func @upper_cholesky(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tessera.cholesky %a {lower = false} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}

// CHECK: error: Apple CPU native kernel: cholesky implements lower = true into the matrix shape only

// -----

module attributes {tessera.target = "apple_cpu", tessera.arch = "apple_silicon_cpu", tessera.launch_bindings = ["a", "q", "r"]} {
  func.func @full_qr(%a: tensor<5x3xf32>) -> (tensor<5x3xf32>, tensor<3x3xf32>) {
    %q, %r = tessera.qr %a {full_matrices = true} : (tensor<5x3xf32>) -> (tensor<5x3xf32>, tensor<3x3xf32>)
    return %q, %r : tensor<5x3xf32>, tensor<3x3xf32>
  }
}

// CHECK: error: Apple CPU native kernel: qr implements full_matrices = false only

// -----

module attributes {tessera.target = "apple_cpu", tessera.arch = "apple_silicon_cpu", tessera.launch_bindings = ["a", "q", "r"]} {
  func.func @wide_qr(%a: tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<5x5xf32>) {
    %q, %r = tessera.qr %a : (tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<5x5xf32>)
    return %q, %r : tensor<3x5xf32>, tensor<5x5xf32>
  }
}

// CHECK: error: Apple CPU native kernel: qr requires M >= N and reduced

// -----

module attributes {tessera.target = "apple_cpu", tessera.arch = "apple_silicon_cpu", tessera.launch_bindings = ["a", "l"]} {
  func.func @batched_cholesky(%a: tensor<2x4x4xf32>) -> tensor<2x4x4xf32> {
    %0 = tessera.cholesky %a : (tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
    return %0 : tensor<2x4x4xf32>
  }
}

// CHECK: error: Apple CPU native kernel: cholesky requires a square rank-2 f32 matrix

// -----

module attributes {tessera.target = "apple_cpu", tessera.arch = "apple_silicon_cpu", tessera.launch_bindings = ["a", "b", "c"]} {
  func.func @f16_batched(%a: tensor<2x7x9xf16>, %b: tensor<2x9x5xf16>) -> tensor<2x7x5xf16> {
    %0 = tessera.batched_gemm %a, %b : (tensor<2x7x9xf16>, tensor<2x9x5xf16>) -> tensor<2x7x5xf16>
    return %0 : tensor<2x7x5xf16>
  }
}

// CHECK: error: Apple CPU native kernel: batched_gemm requires rank-3 f32 operands and result

// -----

module attributes {tessera.target = "apple_cpu", tessera.launch_bindings = ["a", "b", "x"]} {
  func.func @no_arch(%a: tensor<4x4xf32>, %b: tensor<4x3xf32>) -> tensor<4x3xf32> {
    %0 = tessera.cholesky_solve %a, %b : (tensor<4x4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }
}

// CHECK: error: Apple CPU native kernel: requires the apple_silicon_cpu Apple CPU target

// -----

module attributes {tessera.target = "apple_cpu", tessera.arch = "apple_silicon_cpu", tessera.launch_bindings = ["a", "u", "s", "v"]} {
  func.func @policy_svd(%a: tensor<5x3xf32>) -> (tensor<5x3xf32>, tensor<3xf32>, tensor<3x3xf32>) {
    %u, %s, %v = tessera.svd %a {tessera.tile_hint = 4 : i64} : (tensor<5x3xf32>) -> (tensor<5x3xf32>, tensor<3xf32>, tensor<3x3xf32>)
    return %u, %s, %v : tensor<5x3xf32>, tensor<3xf32>, tensor<3x3xf32>
  }
}

// CHECK: error: Apple CPU native kernel: has an unsupported policy attribute 'tessera.tile_hint'
