// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s
//
// Phase 5 tensor-algebra cohort: add and multiply emit native Graph IR
// adjoints. No tessera.custom_adjoint_call may remain in the paired backward.

module {
  func.func @binary(%x: tensor<3x5xf32>, %y: tensor<3x5xf32>) -> tensor<3x5xf32>
      attributes {tessera.autodiff = "reverse"} {
    %sum = "tessera.add"(%x, %y) :
        (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
    %product = "tessera.mul"(%sum, %y) :
        (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
    return %product : tensor<3x5xf32>
  }

  // CHECK-LABEL: func.func @binary__bwd
  // CHECK-NOT: tessera.custom_adjoint_call
  // CHECK: tessera.mul
  // CHECK: arith.addf
  // CHECK: return {{.*}} : tensor<3x5xf32>, tensor<3x5xf32>
}
