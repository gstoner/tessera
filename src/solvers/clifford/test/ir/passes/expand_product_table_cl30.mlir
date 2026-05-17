// RUN: ts-clifford-opt --tessera-clifford-expand-product-table %s | FileCheck %s
//
// ExpandProductTable lowers `tessera_clifford.geo_product` to an unrolled
// sequence of arith.mulf + arith.addf / arith.subf driven by the
// compile-time-known Cl(3,0) Cayley table.
//
// For Cl(3,0) the table has up to 64 non-zero entries; the output
// tensor has 8 coefficients (one per basis blade). Each output
// coefficient is built up via tensor.extract + arith.mulf + (add/sub).

module {
  func.func @cl30_geo_product(
      %a : tensor<8xf32>, %b : tensor<8xf32>) -> tensor<8xf32> {
    %r = "tessera_clifford.geo_product"(%a, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %r : tensor<8xf32>
  }
}

// After expansion: no more clifford.geo_product, only arith + tensor ops.
// CHECK-NOT: tessera_clifford.geo_product
// CHECK: tensor.extract
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: tensor.from_elements
