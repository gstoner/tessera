// RUN: %tessera_strict_opt %s -split-input-file -verify-diagnostics -o /dev/null

func.func @score_combine_bad_delta_shape(
    %base: tensor<2x4xf32>, %delta: tensor<2x3xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{score_combine base/delta shapes must match}}
  %guided = "tessera.score_combine"(%base, %delta) {gamma = 7.500000e-01 : f64}
      : (tensor<2x4xf32>, tensor<2x3xf32>) -> tensor<2x4xf32>
  return %guided : tensor<2x4xf32>
}

// -----

func.func @score_combine_bad_result_shape(
    %base: tensor<2x4xf32>, %delta: tensor<2x4xf32>) -> tensor<2x3xf32> {
  // expected-error @+1 {{score_combine result shapes must match}}
  %guided = "tessera.score_combine"(%base, %delta) {gamma = 7.500000e-01 : f64}
      : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x3xf32>
  return %guided : tensor<2x3xf32>
}

// -----

func.func @score_combine_bad_dtype(
    %base: tensor<2x4xi32>, %delta: tensor<2x4xi32>) -> tensor<2x4xi32> {
  // expected-error @+1 {{expects floating tensor operands and result}}
  %guided = "tessera.score_combine"(%base, %delta) {gamma = 7.500000e-01 : f64}
      : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi32>
  return %guided : tensor<2x4xi32>
}
