// RUN: ts-clifford-opt --tessera-clifford-pipeline %s | FileCheck %s
//
// End-to-end pipeline: annotate → rotor-sandwich-fold → grade-fusion →
// expand-product-table. A three-op rotor-sandwich chain fuses to a
// single high-level op (which then survives for GA9 backend lowering),
// while any remaining geo_products lower to arith.mulf/addf.

module {
  func.func @sandwich_and_extra_product(
      %R : tensor<8xf32>, %x : tensor<8xf32>,
      %a : tensor<8xf32>, %b : tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
    // A rotor sandwich written as three primitive ops — should fold.
    %inner = "tessera_clifford.geo_product"(%R, %x)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %Rdag = "tessera_clifford.reverse"(%R)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>) -> tensor<8xf32>
    %sandwich = "tessera_clifford.geo_product"(%inner, %Rdag)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>

    // A standalone product that should lower to arith.
    %extra = "tessera_clifford.geo_product"(%a, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %sandwich, %extra : tensor<8xf32>, tensor<8xf32>
  }
}

// The chain folded into a rotor_sandwich (survives for GA9 backend
// kernel lowering) with the chain-fold trace marker.
// CHECK: tessera_clifford.rotor_sandwich
// CHECK-SAME: tessera.clifford.from_chain_fold
// The standalone product lowered to arith.
// CHECK: arith.mulf
// CHECK: tensor.from_elements
