// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// Native unweighted normalization adjoints share one explicit rank-reduced
// statistics carrier.  No host custom-adjoint call remains, including when a
// leading extent is dynamic.

module {
  func.func @rmsnorm(%x: tensor<2x5xf32>) -> tensor<2x5xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.rmsnorm"(%x) {eps = 1.0e-5 : f64} :
        (tensor<2x5xf32>) -> tensor<2x5xf32>
    return %y : tensor<2x5xf32>
  }
  // CHECK-LABEL: func.func @rmsnorm__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: tessera.normalization_stats
  // CHECK-SAME: centered = false
  // CHECK: tessera.reduce
  // CHECK: tessera.broadcast_in_dim

  func.func @layer_norm(%x: tensor<2x5xf32>) -> tensor<2x5xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.layer_norm"(%x) {eps = 1.0e-5 : f64} :
        (tensor<2x5xf32>) -> tensor<2x5xf32>
    return %y : tensor<2x5xf32>
  }
  // CHECK-LABEL: func.func @layer_norm__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: tessera.normalization_stats
  // CHECK: tessera.reduce
  // CHECK: tessera.broadcast_in_dim

  func.func @dynamic_layer_norm(%x: tensor<?x5xf32>) -> tensor<?x5xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.layer_norm"(%x) :
        (tensor<?x5xf32>) -> tensor<?x5xf32>
    return %y : tensor<?x5xf32>
  }
  // CHECK-LABEL: func.func @dynamic_layer_norm__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: tessera.normalization_stats
}
