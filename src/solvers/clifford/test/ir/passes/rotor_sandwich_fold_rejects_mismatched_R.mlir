// RUN: ts-clifford-opt --tessera-clifford-rotor-sandwich-fold %s | FileCheck %s
//
// Reject the fold when the reverse's source differs from the inner
// product's left operand. The chain `gp(gp(R, x), reverse(S))` with
// R ≠ S is NOT a rotor sandwich and must survive as-is.

module {
  func.func @not_a_sandwich(
      %R : tensor<8xf32>, %S : tensor<8xf32>,
      %x : tensor<8xf32>) -> tensor<8xf32> {
    %inner = "tessera_clifford.geo_product"(%R, %x)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %Sdag = "tessera_clifford.reverse"(%S)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>) -> tensor<8xf32>
    %y = "tessera_clifford.geo_product"(%inner, %Sdag)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %y : tensor<8xf32>
  }
}

// CHECK-NOT: tessera_clifford.rotor_sandwich
// All three original ops survive.
// CHECK: tessera_clifford.geo_product
// CHECK: tessera_clifford.reverse
// CHECK: tessera_clifford.geo_product
