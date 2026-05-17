// RUN: ts-clifford-opt --tessera-clifford-annotate-algebra %s | FileCheck %s
//
// AnnotateAlgebraPass walks every clifford op, validates the signature
// against the v1 allow-list (Cl(3,0) and Cl(1,3) only), and attaches
// derived metadata.  Allow-listed signatures get
// `tessera.clifford.canonical`; out-of-allow-list signatures get a
// warning and no canonical marker.

module {
  func.func @cl30_canonical(
      %a : tensor<8xf32>, %b : tensor<8xf32>) -> tensor<8xf32> {
    %r = "tessera_clifford.geo_product"(%a, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %r : tensor<8xf32>
  }
}

// CHECK: tessera_clifford.geo_product
// CHECK-SAME: tessera.clifford.allow_listed = true
// CHECK-SAME: tessera.clifford.canonical
// CHECK-SAME: tessera.clifford.dim = 8
