// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// Phase 5 activation cohort: static SiLU/GELU are native Graph adjoints;
// dynamic shapes and ReLU retain explicit placeholders.

module {
  func.func @silu(%x: tensor<2x3xf32>) -> tensor<2x3xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.silu"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %y : tensor<2x3xf32>
  }
  // CHECK-LABEL: func.func @silu__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: tessera.sigmoid
  // CHECK: tessera.mul

  func.func @gelu(%x: tensor<2x3xf32>) -> tensor<2x3xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.gelu"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %y : tensor<2x3xf32>
  }
  // CHECK-LABEL: func.func @gelu__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: tessera.tanh
  // CHECK: tessera.mul

  func.func @dynamic_silu(%x: tensor<?x3xf32>) -> tensor<?x3xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.silu"(%x) : (tensor<?x3xf32>) -> tensor<?x3xf32>
    return %y : tensor<?x3xf32>
  }
  // CHECK-LABEL: func.func @dynamic_silu__bwd
  // CHECK: tessera.custom_adjoint_call "silu"

  func.func @relu(%x: tensor<2x3xf32>) -> tensor<2x3xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.relu"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %y : tensor<2x3xf32>
  }
  // CHECK-LABEL: func.func @relu__bwd
  // CHECK: tessera.custom_adjoint_call "relu"
}
