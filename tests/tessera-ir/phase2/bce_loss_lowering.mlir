// RUN: tessera-opt --tessera-to-linalg %s | FileCheck %s

module {
  func.func @bce_dynamic(%z: tensor<?x7xf32>,
                         %t: tensor<?x7xf32>) -> tensor<f32> {
    %loss = "tessera.loss.binary_cross_entropy"(%z, %t)
        {reduction = "mean"} :
        (tensor<?x7xf32>, tensor<?x7xf32>) -> tensor<f32>
    return %loss : tensor<f32>
  }
  // CHECK-LABEL: func.func @bce_dynamic
  // CHECK-NOT: tessera.loss.binary_cross_entropy
  // CHECK: linalg.generic
  // CHECK: math.absf
  // CHECK: math.log1p
  // CHECK: tensor.dim

  func.func @bce_backward(%z: tensor<?x7xf32>, %t: tensor<?x7xf32>,
                          %dy: tensor<f32>)
      -> (tensor<?x7xf32>, tensor<?x7xf32>) {
    %dz, %dt = "tessera.loss.binary_cross_entropy_backward"(%z, %t, %dy)
        {reduction = "sum"} :
        (tensor<?x7xf32>, tensor<?x7xf32>, tensor<f32>) ->
        (tensor<?x7xf32>, tensor<?x7xf32>)
    return %dz, %dt : tensor<?x7xf32>, tensor<?x7xf32>
  }
  // CHECK-LABEL: func.func @bce_backward
  // CHECK-NOT: tessera.loss.binary_cross_entropy_backward
  // CHECK: math.exp
  // CHECK: arith.select
}
