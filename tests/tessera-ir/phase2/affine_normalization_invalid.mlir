// RUN: tessera-opt %s -verify-diagnostics

module {
  func.func @bad_rms_gamma(%x: tensor<2x5xf32>, %g: tensor<4xf32>)
      -> tensor<2x5xf32> {
    // expected-error @+1 {{gamma length must match the last input dim}}
    %y = "tessera.rmsnorm"(%x, %g) :
        (tensor<2x5xf32>, tensor<4xf32>) -> tensor<2x5xf32>
    return %y : tensor<2x5xf32>
  }

  func.func @bad_layer_beta_rank(
      %x: tensor<2x5xf32>, %g: tensor<5xf32>, %b: tensor<1x5xf32>)
      -> tensor<2x5xf32> {
    // expected-error @+1 {{beta must be a rank-1 channel tensor}}
    %y = "tessera.layer_norm"(%x, %g, %b) :
        (tensor<2x5xf32>, tensor<5xf32>, tensor<1x5xf32>) -> tensor<2x5xf32>
    return %y : tensor<2x5xf32>
  }

  func.func @dynamic_broadcast_without_shape(%x: tensor<?xf32>)
      -> tensor<?x?xf32> {
    // expected-error @+1 {{dynamic result requires a shape_like operand}}
    %y = "tessera.broadcast_in_dim"(%x) {broadcast_dimensions = [0]} :
        (tensor<?xf32>) -> tensor<?x?xf32>
    return %y : tensor<?x?xf32>
  }
}
