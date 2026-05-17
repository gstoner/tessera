// RUN: ts-clifford-opt --tessera-clifford-rotor-sandwich-fold %s | FileCheck %s
//
// RotorSandwichFold: gp(gp(R, x), reverse(R)) → rotor_sandwich(R, x).
// The fused op carries `tessera.clifford.from_chain_fold` so a
// diagnostic can trace it back to the original chain.

module {
  func.func @three_op_sandwich(
      %R : tensor<8xf32>, %x : tensor<8xf32>) -> tensor<8xf32> {
    %inner = "tessera_clifford.geo_product"(%R, %x)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %Rdag = "tessera_clifford.reverse"(%R)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>) -> tensor<8xf32>
    %y = "tessera_clifford.geo_product"(%inner, %Rdag)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %y : tensor<8xf32>
  }
}

// CHECK: tessera_clifford.rotor_sandwich
// CHECK-SAME: tessera.clifford.from_chain_fold
