// RUN: tessera-opt --tessera-to-linalg %s | FileCheck %s
//
// MSE is a first-class Graph operation. Mean reduction uses runtime element
// counts, and its paired backward broadcasts the scalar cotangent while
// producing equal-and-opposite prediction/target gradients.

module {
  // CHECK-LABEL: func.func @dynamic_mean
  // CHECK: linalg.generic
  // CHECK-COUNT-2: linalg.reduce
  // CHECK: tensor.dim
  // CHECK: arith.divf
  // CHECK-NOT: tessera.loss.mse
  func.func @dynamic_mean(%prediction: tensor<?x5xf32>,
                          %target: tensor<?x5xf32>) -> tensor<f32> {
    %loss = "tessera.loss.mse"(%prediction, %target)
        {reduction = "mean"} :
        (tensor<?x5xf32>, tensor<?x5xf32>) -> tensor<f32>
    return %loss : tensor<f32>
  }

  // CHECK-LABEL: func.func @dynamic_mean_backward
  // CHECK: tensor.dim
  // CHECK: arith.divf
  // CHECK: linalg.generic
  // CHECK: arith.subf
  // CHECK: arith.mulf
  // CHECK: arith.negf
  // CHECK-NOT: tessera.loss.mse_backward
  func.func @dynamic_mean_backward(
      %prediction: tensor<?x5xf32>, %target: tensor<?x5xf32>,
      %cotangent: tensor<f32>) -> (tensor<?x5xf32>, tensor<?x5xf32>) {
    %dp, %dt = "tessera.loss.mse_backward"(
        %prediction, %target, %cotangent) {reduction = "mean"} :
        (tensor<?x5xf32>, tensor<?x5xf32>, tensor<f32>) ->
        (tensor<?x5xf32>, tensor<?x5xf32>)
    return %dp, %dt : tensor<?x5xf32>, tensor<?x5xf32>
  }

  // CHECK-LABEL: func.func @none_backward
  // CHECK: linalg.generic
  // CHECK-NOT: linalg.reduce
  func.func @none_backward(
      %prediction: tensor<2x3xf32>, %target: tensor<2x3xf32>,
      %cotangent: tensor<2x3xf32>) ->
      (tensor<2x3xf32>, tensor<2x3xf32>) {
    %dp, %dt = "tessera.loss.mse_backward"(
        %prediction, %target, %cotangent) {reduction = "none"} :
        (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) ->
        (tensor<2x3xf32>, tensor<2x3xf32>)
    return %dp, %dt : tensor<2x3xf32>, tensor<2x3xf32>
  }

  // Low-precision storage does all loss arithmetic, accumulation, and dynamic
  // mean scaling in fp32, truncating only at the Graph result boundary.
  // CHECK-LABEL: func.func @bf16_mean
  // CHECK: arith.extf {{.*}} : bf16 to f32
  // CHECK: arith.subf
  // CHECK: arith.mulf
  // CHECK-COUNT-2: linalg.reduce
  // CHECK: arith.divf
  // CHECK: arith.truncf {{.*}} : f32 to bf16
  func.func @bf16_mean(%prediction: tensor<?x7xbf16>,
                       %target: tensor<?x7xbf16>) -> tensor<bf16> {
    %loss = "tessera.loss.mse"(%prediction, %target)
        {reduction = "mean"} :
        (tensor<?x7xbf16>, tensor<?x7xbf16>) -> tensor<bf16>
    return %loss : tensor<bf16>
  }

  // CHECK-LABEL: func.func @f16_mean_backward
  // CHECK: arith.divf
  // CHECK: arith.extf {{.*}} : f16 to f32
  // CHECK: arith.mulf
  // CHECK: arith.truncf {{.*}} : f32 to f16
  func.func @f16_mean_backward(
      %prediction: tensor<?x7xf16>, %target: tensor<?x7xf16>,
      %cotangent: tensor<f16>) -> (tensor<?x7xf16>, tensor<?x7xf16>) {
    %dp, %dt = "tessera.loss.mse_backward"(
        %prediction, %target, %cotangent) {reduction = "mean"} :
        (tensor<?x7xf16>, tensor<?x7xf16>, tensor<f16>) ->
        (tensor<?x7xf16>, tensor<?x7xf16>)
    return %dp, %dt : tensor<?x7xf16>, tensor<?x7xf16>
  }
}
