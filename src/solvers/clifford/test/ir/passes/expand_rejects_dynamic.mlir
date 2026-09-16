// RUN: ts-clifford-opt --tessera-clifford-expand-product-table %s 2>&1 | FileCheck %s
//
// Dynamic extents fail closed: the batched lowering needs static leading
// axes for its loop bounds and a static coefficient axis for the table.
// The op is left in place with a diagnostic, never half-lowered.

module {
  func.func @cl30_dynamic(
      %a : tensor<?x8xf32>, %b : tensor<?x8xf32>) -> tensor<?x8xf32> {
    %r = "tessera_clifford.geo_product"(%a, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<?x8xf32>, tensor<?x8xf32>) -> tensor<?x8xf32>
    return %r : tensor<?x8xf32>
  }
}

// CHECK: dynamic or unranked operands are not lowered
// CHECK: tessera_clifford.geo_product
