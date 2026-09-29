// E2E-REAL-6 x86 native kernel contract: what it refuses (Decision #21a).
// RUN: tessera-opt --split-input-file --tessera-graph-to-schedule --verify-diagnostics %s

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512", tessera.launch_bindings = ["a", "b", "o"]} {
  func.func @unknown_policy(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
    // expected-error @+1 {{x86 native kernel has an unsupported policy attribute 'alpha'}}
    %o = tessera.add %a, %b {alpha = 2.0 : f64} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %o : tensor<8xf32>
  }
}

// -----

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512", tessera.launch_bindings = ["m", "o"]} {
  func.func @upper_cholesky(%m: tensor<3x3xf32>) -> tensor<3x3xf32> {
    // expected-error @+1 {{x86 native cholesky implements lower = true into the matrix shape only}}
    %o = tessera.cholesky %m {lower = false} : (tensor<3x3xf32>) -> tensor<3x3xf32>
    return %o : tensor<3x3xf32>
  }
}

// -----

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512", tessera.launch_bindings = ["p", "t", "o"]} {
  func.func @reduced_loss(%p: tensor<8xf32>, %t: tensor<8xf32>) -> tensor<f32> {
    // expected-error @+1 {{x86 native pointwise loss requires same-shape f32 operands and reduction = "none"}}
    %o = tessera.loss.mse %p, %t {reduction = "mean"} : (tensor<8xf32>, tensor<8xf32>) -> tensor<f32>
    return %o : tensor<f32>
  }
}

// -----

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512", tessera.launch_bindings = ["a", "o"]} {
  func.func @dynamic(%a: tensor<?xf32>) -> tensor<?xf32> {
    // expected-error @+1 {{x86 native kernel requires static positive-extent operands}}
    %o = tessera.exp %a : (tensor<?xf32>) -> tensor<?xf32>
    return %o : tensor<?xf32>
  }
}

// -----

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512", tessera.launch_bindings = ["a", "b", "o"]} {
  func.func @wrong_storage(%a: tensor<8xi32>, %b: tensor<8xi32>) -> tensor<8xi32> {
    // expected-error @+1 {{x86 native elementwise requires same-shape operands of its ABI storage}}
    %o = tessera.add %a, %b : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
    return %o : tensor<8xi32>
  }
}
