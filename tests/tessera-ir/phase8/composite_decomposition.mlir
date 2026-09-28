// ODS triage WIRE slice 1 (GOV-ODS-CONSUMER-1): the Graph composites the @jit
// frontend emits are rewritten onto the canonical ops their consumers lower.
//   tessera.target_verify(tokens, logits)  -> tessera.softmax(logits){axis = rank-1}
//   tessera.ntk_rope(x, theta){scale = s}  -> tessera.rope(x, theta / s)
// One pattern source (CompositeDecomposition.h), run standalone and by
// tessera-canonicalize (the x86 / NVIDIA pre-lowering stage).
//
// RUN: %tessera_strict_opt %s --tessera-decompose-composite-ops | FileCheck %s
// RUN: %tessera_strict_opt %s --tessera-canonicalize | FileCheck %s

// CHECK-LABEL: func.func @target_verify_to_softmax
// CHECK-SAME: (%{{.*}}: tensor<3xi32>, %[[L:.*]]: tensor<3x8xf32>)
// CHECK-NOT: tessera.target_verify
// CHECK: tessera.softmax %[[L]] {axis = 1 : i64} : (tensor<3x8xf32>) -> tensor<3x8xf32>
func.func @target_verify_to_softmax(%t: tensor<3xi32>, %l: tensor<3x8xf32>) -> tensor<3x8xf32> {
  %0 = tessera.target_verify %t, %l : (tensor<3xi32>, tensor<3x8xf32>) -> tensor<3x8xf32>
  return %0 : tensor<3x8xf32>
}

// A discardable attribute rides forward onto the softmax (Decision #32).
// CHECK-LABEL: func.func @target_verify_keeps_attrs
// CHECK: tessera.softmax %{{.*}} {axis = 1 : i64, tessera.provenance = "spec_decode"}
func.func @target_verify_keeps_attrs(%t: tensor<2xi32>, %l: tensor<2x5xf32>) -> tensor<2x5xf32> {
  %0 = tessera.target_verify %t, %l {tessera.provenance = "spec_decode"} : (tensor<2xi32>, tensor<2x5xf32>) -> tensor<2x5xf32>
  return %0 : tensor<2x5xf32>
}

// scale = 2: theta is divided by a splat of the scale before the rope.
// CHECK-LABEL: func.func @ntk_rope_scaled
// CHECK-SAME: (%[[X:.*]]: tensor<4x8xf32>, %[[TH:.*]]: tensor<4x8xf32>)
// CHECK-NOT: tessera.ntk_rope
// CHECK: %[[S:.*]] = arith.constant dense<2.000000e+00> : tensor<4x8xf32>
// CHECK: %[[D:.*]] = tessera.div %[[TH]], %[[S]]
// CHECK: tessera.rope %[[X]], %[[D]] : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
func.func @ntk_rope_scaled(%x: tensor<4x8xf32>, %th: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = tessera.ntk_rope %x, %th {scale = 2.0 : f64} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// scale = 1 (explicit or the ODS default) is exactly rope: no division, and a
// dynamic theta is fine because no splat has to be materialized.
// CHECK-LABEL: func.func @ntk_rope_unit_scale
// CHECK-SAME: (%[[X:.*]]: tensor<?x8xf32>, %[[TH:.*]]: tensor<?x8xf32>)
// CHECK-NOT: tessera.div
// CHECK-NOT: arith.constant
// CHECK: tessera.rope %[[X]], %[[TH]] : (tensor<?x8xf32>, tensor<?x8xf32>) -> tensor<?x8xf32>
func.func @ntk_rope_unit_scale(%x: tensor<?x8xf32>, %th: tensor<?x8xf32>) -> tensor<?x8xf32> {
  %0 = tessera.ntk_rope %x, %th : (tensor<?x8xf32>, tensor<?x8xf32>) -> tensor<?x8xf32>
  return %0 : tensor<?x8xf32>
}

// Negative (Decision #10a): the canonical ops themselves are not touched.
// CHECK-LABEL: func.func @canonical_ops_untouched
// CHECK-NOT: tessera.div
// CHECK-NOT: arith.constant
// CHECK: tessera.softmax %{{.*}} {axis = 1 : i64}
// CHECK: tessera.rope
// CHECK: return
func.func @canonical_ops_untouched(%x: tensor<4x8xf32>, %th: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = tessera.softmax %x {axis = 1 : i64} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %1 = tessera.rope %0, %th : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}
