// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// Phase 5 reduction cohort: sum/mean own native Graph adjoints, while max/min
// and a dynamic-extent mean retain explicit reference fallbacks.

module {
  func.func @sum(%x: tensor<2x3xf32>) -> tensor<2xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.reduce"(%x) {kind = "sum", axis = 1 : i64} :
        (tensor<2x3xf32>) -> tensor<2xf32>
    return %y : tensor<2xf32>
  }

  // CHECK-LABEL: func.func @sum__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: tessera.unsqueeze
  // CHECK: tessera.broadcast

  func.func @mean(%x: tensor<2x3xf32>) -> tensor<2xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.reduce"(%x) {kind = "mean", axis = 1 : i64} :
        (tensor<2x3xf32>) -> tensor<2xf32>
    return %y : tensor<2xf32>
  }

  // CHECK-LABEL: func.func @mean__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: tessera.unsqueeze
  // CHECK: tessera.broadcast
  // CHECK: arith.constant
  // CHECK: tessera.mul

  func.func @max(%x: tensor<2x3xf32>) -> tensor<2xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.reduce"(%x) {kind = "max", axis = 1 : i64} :
        (tensor<2x3xf32>) -> tensor<2xf32>
    return %y : tensor<2xf32>
  }

  // CHECK-LABEL: func.func @max__bwd
  // CHECK: tessera.custom_adjoint_call "max"

  func.func @dynamic_mean(%x: tensor<2x?xf32>) -> tensor<2xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.reduce"(%x) {kind = "mean", axis = 1 : i64} :
        (tensor<2x?xf32>) -> tensor<2xf32>
    return %y : tensor<2xf32>
  }

  // CHECK-LABEL: func.func @dynamic_mean__bwd
  // CHECK: tessera.custom_adjoint_call "mean"
}
