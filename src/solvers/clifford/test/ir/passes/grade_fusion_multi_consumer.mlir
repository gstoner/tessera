// RUN: ts-clifford-opt --tessera-clifford-grade-fusion %s | FileCheck %s
//
// When the same geo_product is consumed by two grade ops requesting
// different grade slices, the fused output_grades attribute is the
// union — so a downstream ExpandProductTable still emits the correct
// (joint) set of coefficients.

module {
  func.func @scalar_and_bivector(
      %a : tensor<8xf32>, %b : tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
    %gp = "tessera_clifford.geo_product"(%a, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %g0 = "tessera_clifford.grade"(%gp)
        { grades = [0], algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>) -> tensor<8xf32>
    %g2 = "tessera_clifford.grade"(%gp)
        { grades = [2], algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>) -> tensor<8xf32>
    return %g0, %g2 : tensor<8xf32>, tensor<8xf32>
  }
}

// CHECK: tessera_clifford.geo_product
// CHECK-SAME: tessera.clifford.output_grades = [0, 2]
// CHECK-NOT: tessera_clifford.grade
