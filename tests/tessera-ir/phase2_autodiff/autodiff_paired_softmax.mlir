// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// Phase 5 normalization cohort: softmax backward is native Graph IR for both
// static and dynamic shapes because its dot reduction needs no shape constant.

module {
  func.func @softmax(%x: tensor<2x3xf32>) -> tensor<2x3xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.softmax"(%x) {axis = 1 : i64} :
        (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %y : tensor<2x3xf32>
  }
  // CHECK-LABEL: func.func @softmax__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: tessera.softmax
  // CHECK: tessera.reduce
  // CHECK: tessera.unsqueeze
  // CHECK: tessera.broadcast
  // CHECK: tessera.sub

  func.func @dynamic_softmax(%x: tensor<?x3xf32>) -> tensor<?x3xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.softmax"(%x) {axis = -1 : i64} :
        (tensor<?x3xf32>) -> tensor<?x3xf32>
    return %y : tensor<?x3xf32>
  }
  // CHECK-LABEL: func.func @dynamic_softmax__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: tessera.reduce
  // CHECK: tessera.broadcast
}
