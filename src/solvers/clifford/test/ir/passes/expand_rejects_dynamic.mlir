// RUN: not ts-clifford-opt --tessera-clifford-expand-product-table %s 2>&1 | FileCheck %s
//
// `not`: since 2026-09-16 the pass fails when it refuses an op, instead of
// printing the diagnostic and exiting 0.
//
// A dynamic *coefficient* axis fails closed, and always will: that axis indexes
// the algebra's compile-time Cayley table, so a runtime extent has no meaning
// there. Dynamic *leading* axes are a ragged batch and are lowered (see
// expand_ragged_batch.mlir) -- this fixture used to refuse those too.

module {
  func.func @cl30_dynamic_coefficients(
      %a : tensor<4x?xf32>, %b : tensor<4x?xf32>) -> tensor<4x?xf32> {
    %r = "tessera_clifford.geo_product"(%a, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<4x?xf32>, tensor<4x?xf32>) -> tensor<4x?xf32>
    return %r : tensor<4x?xf32>
  }
}

// CHECK: coefficient axis is static
// CHECK: tessera_clifford.geo_product
