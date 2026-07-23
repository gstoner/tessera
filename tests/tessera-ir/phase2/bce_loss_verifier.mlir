// RUN: tessera-opt %s -split-input-file -verify-diagnostics

module {
  func.func @valid(%z: tensor<?x7xf32>, %t: tensor<?x7xf32>) -> tensor<f32> {
    %loss = "tessera.loss.binary_cross_entropy"(%z, %t) :
        (tensor<?x7xf32>, tensor<?x7xf32>) -> tensor<f32>
    return %loss : tensor<f32>
  }
}

// -----

module {
  func.func @bad_shape(%z: tensor<3xf32>, %t: tensor<4xf32>) -> tensor<f32> {
    // expected-error @+1 {{binary cross entropy logits/target shapes must match}}
    %loss = "tessera.loss.binary_cross_entropy"(%z, %t) :
        (tensor<3xf32>, tensor<4xf32>) -> tensor<f32>
    return %loss : tensor<f32>
  }
}
