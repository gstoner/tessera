// RUN: ts-clifford-opt --tessera-clifford-grade-fusion %s | FileCheck %s
//
// GradeFusion: `grade(2, geo_product(a, b))` rewrites to a geo_product
// annotated with `tessera.clifford.output_grades = [2]`, and the
// grade op is replaced with the geo_product result.

module {
  func.func @bivector_only(
      %a : tensor<8xf32>, %b : tensor<8xf32>) -> tensor<8xf32> {
    %gp = "tessera_clifford.geo_product"(%a, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %grade2 = "tessera_clifford.grade"(%gp)
        { grades = [2], algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>) -> tensor<8xf32>
    return %grade2 : tensor<8xf32>
  }
}

// After fusion: no more clifford.grade; the geo_product carries the attr.
// CHECK-NOT: tessera_clifford.grade
// CHECK: tessera_clifford.geo_product
// CHECK-SAME: tessera.clifford.output_grades = [2]
