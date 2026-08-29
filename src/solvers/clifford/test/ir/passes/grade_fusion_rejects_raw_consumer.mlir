// RUN: ts-clifford-opt --tessera-clifford-grade-fusion %s | FileCheck %s
//
// A geo_product that is ALSO consumed unprojected must not be restricted.
// `output_grades` prunes the product itself, so folding the grade op here would
// hand the raw consumer a product missing every grade outside [2] — a silent
// wrong value for a consumer that never asked for a projection.

module {
  func.func @raw_and_projected(
      %a : tensor<8xf32>, %b : tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
    %gp = "tessera_clifford.geo_product"(%a, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    // Unprojected consumer: wants every grade of the product.
    %raw = "tessera_clifford.geo_product"(%gp, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %g2 = "tessera_clifford.grade"(%gp)
        { grades = [2], algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>) -> tensor<8xf32>
    return %raw, %g2 : tensor<8xf32>, tensor<8xf32>
  }
}

// The first product keeps every grade, and the projection is not folded away.
// CHECK-NOT: tessera.clifford.output_grades
// CHECK: tessera_clifford.grade
// CHECK-SAME: grades [2]
