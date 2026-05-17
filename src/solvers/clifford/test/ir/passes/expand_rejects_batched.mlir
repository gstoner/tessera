// RUN: ts-clifford-opt --tessera-clifford-expand-product-table %s 2>&1 | FileCheck %s
//
// V1 ExpandProductTable handles rank-1 static tensors only; batched
// (rank > 1) operands are explicitly out-of-scope. The pass emits a
// warning and skips the op (leaving it unlowered for a follow-on pass).

module {
  func.func @cl30_batched(
      %a : tensor<32x8xf32>, %b : tensor<32x8xf32>) -> tensor<32x8xf32> {
    %r = "tessera_clifford.geo_product"(%a, %b)
        { algebra = [3, 0, 0], dtype = "fp32" }
        : (tensor<32x8xf32>, tensor<32x8xf32>) -> tensor<32x8xf32>
    return %r : tensor<32x8xf32>
  }
}

// CHECK: batched (rank > 1) operands are pending a follow-on sprint
// The op stays in the IR (not lowered).
// CHECK: tessera_clifford.geo_product
