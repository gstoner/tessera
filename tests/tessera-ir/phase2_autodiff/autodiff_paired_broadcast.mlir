// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// Phase 5 tensor-algebra cohort: a static broadcast adjoint sums all expanded
// axes in descending order, then restores explicit singleton dimensions.

module {
  func.func @broadcast(%x: tensor<1x3xf32>) -> tensor<2x4x3xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.broadcast"(%x) :
        (tensor<1x3xf32>) -> tensor<2x4x3xf32>
    return %y : tensor<2x4x3xf32>
  }

  // CHECK-LABEL: func.func @broadcast__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: tessera.reduce{{.*}}axis = 1{{.*}}kind = "sum"
  // CHECK: tessera.reduce{{.*}}axis = 0{{.*}}kind = "sum"
  // CHECK: tessera.reshape
  // CHECK: return {{.*}} : tensor<1x3xf32>

  func.func @dynamic_broadcast(%x: tensor<?x1x3xf32>)
      -> tensor<?x4x3xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.broadcast"(%x) :
        (tensor<?x1x3xf32>) -> tensor<?x4x3xf32>
    return %y : tensor<?x4x3xf32>
  }

  // CHECK-LABEL: func.func @dynamic_broadcast__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: tessera.reduce{{.*}}axis = 1{{.*}}kind = "sum"
  // CHECK: tessera.reshape
  // CHECK: return {{.*}} : tensor<?x1x3xf32>
}
