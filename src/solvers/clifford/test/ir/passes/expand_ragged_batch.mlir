// RUN: ts-clifford-opt --tessera-clifford-expand-product-table %s | FileCheck %s
//
// A ragged batch: the leading extent is a runtime value, so the loop bound comes
// from `tensor.dim` on the operand instead of a constant, and the result tensor
// is allocated with that dynamic size. The coefficient axis stays static, since
// it indexes the compile-time Cayley table.
//
// "The operands have the same shape" is what every bilinear op here requires,
// and with a dynamic extent the types cannot establish it -- so the check is
// emitted rather than assumed. Without it a mismatched pair would read past the
// shorter operand and return a plausible answer.

// CHECK-LABEL: func @ragged_product
// CHECK-NOT:     tessera_clifford
// CHECK-DAG:     tensor.dim
// CHECK-DAG:     cf.assert
// CHECK-DAG:     scf.for
func.func @ragged_product(%a : tensor<?x8xf32>, %b : tensor<?x8xf32>) -> tensor<?x8xf32> {
  %r = "tessera_clifford.geo_product"(%a, %b) { algebra = [3, 0, 0], dtype = "fp32" }
      : (tensor<?x8xf32>, tensor<?x8xf32>) -> tensor<?x8xf32>
  return %r : tensor<?x8xf32>
}

// A unary op has nothing to compare, so it needs the dim for its bound and no
// assertion.
// CHECK-LABEL: func @ragged_reverse
// CHECK-NOT:     tessera_clifford
// CHECK-NOT:     cf.assert
// CHECK-DAG:     tensor.dim
func.func @ragged_reverse(%a : tensor<?x?x8xf32>) -> tensor<?x?x8xf32> {
  %r = "tessera_clifford.reverse"(%a) { algebra = [3, 0, 0], dtype = "fp32" }
      : (tensor<?x?x8xf32>) -> tensor<?x?x8xf32>
  return %r : tensor<?x?x8xf32>
}
