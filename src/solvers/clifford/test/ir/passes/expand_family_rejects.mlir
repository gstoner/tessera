// RUN: not ts-clifford-opt --tessera-clifford-expand-product-table %s 2>&1 | FileCheck %s
//
// `not`: since 2026-09-16 the pass fails when it refuses an op, instead of
// printing the diagnostic and exiting 0.
//
// Scalar forms must be typed [..., 1]; a mistyped result is refused, never
// silently reshaped. The op stays in place.

module {
  func.func @inner_mistyped(%a : tensor<4x8xf32>, %b : tensor<4x8xf32>) -> tensor<4x8xf32> {
    %i = "tessera_clifford.inner"(%a, %b) { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    return %i : tensor<4x8xf32>
  }
}

// CHECK: scalar-form result must be typed 'tensor<4x1xf32>'
// CHECK: tessera_clifford.inner
