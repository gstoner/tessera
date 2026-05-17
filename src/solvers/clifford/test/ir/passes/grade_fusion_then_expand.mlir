// RUN: ts-clifford-opt --tessera-clifford-grade-fusion --tessera-clifford-expand-product-table %s | FileCheck %s
//
// Composed: GradeFusion attaches output_grades, then
// ExpandProductTable emits only the slice of the Cayley table that
// contributes to the requested grades.
//
// For Cl(3,0) with output_grades=[2], the table has only 6 contributing
// (i, j) pairs (the bivector coefficients e12, e13, e23). The total
// arith op count drops dramatically vs. the full expansion (which would
// emit ~64 mul-adds).

module {
  func.func @grade2_only_fast(
      %a : tensor<8xf32>, %b : tensor<8xf32>) -> tensor<8xf32> {
    %gp = "tessera_clifford.geo_product"(%a, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %g2 = "tessera_clifford.grade"(%gp)
        { grades = [2], algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>) -> tensor<8xf32>
    return %g2 : tensor<8xf32>
  }
}

// CHECK-NOT: tessera_clifford.geo_product
// CHECK-NOT: tessera_clifford.grade
// CHECK: tensor.extract
// CHECK: arith.mulf
// CHECK: tensor.from_elements
