// RUN: tessera-opt %s -split-input-file -verify-diagnostics

module {
  func.func @valid_dynamic(
      %prediction: tensor<?x5xf16>, %target: tensor<?x5xf16>) -> tensor<f16> {
    %loss = "tessera.loss.mse"(%prediction, %target) :
        (tensor<?x5xf16>, tensor<?x5xf16>) -> tensor<f16>
    return %loss : tensor<f16>
  }
}

// -----

module {
  func.func @bad_shape(
      %prediction: tensor<2x5xf32>, %target: tensor<3x5xf32>) -> tensor<f32> {
    // expected-error @+1 {{mse prediction/target shapes must match}}
    %loss = "tessera.loss.mse"(%prediction, %target) :
        (tensor<2x5xf32>, tensor<3x5xf32>) -> tensor<f32>
    return %loss : tensor<f32>
  }
}

// -----

module {
  func.func @bad_reduction_result(
      %prediction: tensor<2x5xf32>, %target: tensor<2x5xf32>)
      -> tensor<2x5xf32> {
    // expected-error @+1 {{mean/sum reduction result must be rank-0 tensor}}
    %loss = "tessera.loss.mse"(%prediction, %target) :
        (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<2x5xf32>
    return %loss : tensor<2x5xf32>
  }
}

// -----

module {
  func.func @bad_backward_cotangent(
      %prediction: tensor<2x5xf32>, %target: tensor<2x5xf32>,
      %cotangent: tensor<2x5xf32>) ->
      (tensor<2x5xf32>, tensor<2x5xf32>) {
    // expected-error @+1 {{mean/sum reduction result must be rank-0 tensor}}
    %dp, %dt = "tessera.loss.mse_backward"(
        %prediction, %target, %cotangent) :
        (tensor<2x5xf32>, tensor<2x5xf32>, tensor<2x5xf32>) ->
        (tensor<2x5xf32>, tensor<2x5xf32>)
    return %dp, %dt : tensor<2x5xf32>, tensor<2x5xf32>
  }
}
