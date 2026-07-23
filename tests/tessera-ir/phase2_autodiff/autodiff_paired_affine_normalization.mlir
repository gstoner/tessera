// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s

module {
  func.func @rmsnorm_affine(%x: tensor<?x5xf32>, %gamma: tensor<5xf32>)
      -> tensor<?x5xf32> attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.rmsnorm"(%x, %gamma) :
        (tensor<?x5xf32>, tensor<5xf32>) -> tensor<?x5xf32>
    return %y : tensor<?x5xf32>
  }
  // CHECK-LABEL: func.func @rmsnorm_affine__bwd
  // CHECK-SAME: -> (tensor<?x5xf32>, tensor<5xf32>)
  // CHECK: tessera.normalization_stats
  // CHECK: tessera.broadcast_in_dim
  // CHECK: kind = "sum"
  // CHECK-NOT: tessera.custom_adjoint_call

  func.func @layernorm_affine(
      %x: tensor<?x5xf32>, %gamma: tensor<5xf32>, %beta: tensor<5xf32>)
      -> tensor<?x5xf32> attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.layer_norm"(%x, %gamma, %beta) :
        (tensor<?x5xf32>, tensor<5xf32>, tensor<5xf32>) -> tensor<?x5xf32>
    return %y : tensor<?x5xf32>
  }
  // CHECK-LABEL: func.func @layernorm_affine__bwd
  // CHECK-SAME: -> (tensor<?x5xf32>, tensor<5xf32>, tensor<5xf32>)
  // CHECK: tessera.normalization_stats
  // CHECK: tessera.broadcast_in_dim
  // CHECK-COUNT-2: kind = "sum"
  // CHECK-NOT: tessera.custom_adjoint_call
}
