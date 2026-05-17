// RUN: ts-clifford-opt --tessera-clifford-expand-product-table %s | FileCheck %s
//
// Same lowering on Cl(1,3) — the Minkowski signature. The Cayley table
// here includes the sign flips from the q=3 spatial generators
// squaring to -1, which the pass picks up via arith.subf accumulations
// for those table entries.

module {
  func.func @cl13_geo_product(
      %a : tensor<16xf32>, %b : tensor<16xf32>) -> tensor<16xf32> {
    %r = "tessera_clifford.geo_product"(%a, %b)
        { algebra = [1, 3, 0], dtype = "fp32" }
        : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    return %r : tensor<16xf32>
  }
}

// CHECK-NOT: tessera_clifford.geo_product
// The Minkowski signature produces arith.subf accumulations for the
// negative-square generators (e2, e3, e4).
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: tensor.from_elements
