// RUN: ts-clifford-opt --tessera-clifford-grade-fusion %s | FileCheck %s
//
// Two grade ops asking for DIFFERENT slices of the same geo_product must both
// survive.
//
// This fixture previously asserted the opposite — that the two were folded away
// and the product annotated with the union `output_grades = [0, 2]`. That is
// unsound, and the union is where it goes wrong: `output_grades` restricts the
// PRODUCT, and ExpandProductTable emits one shared value holding every result
// grade in the set (see its `wantGrade` mask). Folding both consumers into it
// makes `%g0` and `%g2` the same SSA value, so the caller's scalar part comes
// back carrying bivector coefficients and vice versa — each consumer receives
// the other's grades. A union is only correct when the consumers are summed,
// which is not what a pair of separate projections means.
//
// The fusion is still applied whenever it is sound: a product whose consumers
// all agree on the grade set (grade_fusion_basic.mlir) folds as before.

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

// The product keeps its full result set — no output_grades restriction.
// CHECK: tessera_clifford.geo_product
// CHECK-NOT: tessera.clifford.output_grades
// Both projections remain, each with its own grade.
// CHECK: tessera_clifford.grade
// CHECK-SAME: grades [0]
// CHECK: tessera_clifford.grade
// CHECK-SAME: grades [2]
