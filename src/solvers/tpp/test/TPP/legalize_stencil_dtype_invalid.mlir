// RUN: tessera-opt %s -tpp-legalize-space-time -verify-diagnostics

func.func @integer_coefficients(
    %x: tensor<8x8xf32>, %k: tensor<3x3xi32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{stencil coefficient tensor must be a ranked floating-point tensor}}
  %y = "tpp.stencil.apply"(%x, %k) {scheme = "central", order = 2 : i64, spacing = [1.0 : f64, 1.0 : f64]} : (tensor<8x8xf32>, tensor<3x3xi32>) -> tensor<8x8xf32>
  return %y : tensor<8x8xf32>
}

func.func @mismatched_coefficient_dtype(
    %x: tensor<8x8xf32>, %k: tensor<3x3xf64>) -> tensor<8x8xf32> {
  // expected-error @+1 {{stencil coefficient dtype must match the field element dtype}}
  %y = "tpp.stencil.apply"(%x, %k) {scheme = "central", order = 2 : i64, spacing = [1.0 : f64, 1.0 : f64]} : (tensor<8x8xf32>, tensor<3x3xf64>) -> tensor<8x8xf32>
  return %y : tensor<8x8xf32>
}

func.func @mismatched_coefficient_rank(
    %x: tensor<8x8xf32>, %k: tensor<3xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{stencil coefficient tensor rank must match the field rank}}
  %y = "tpp.stencil.apply"(%x, %k) {scheme = "central", order = 2 : i64, spacing = [1.0 : f64, 1.0 : f64]} : (tensor<8x8xf32>, tensor<3xf32>) -> tensor<8x8xf32>
  return %y : tensor<8x8xf32>
}
