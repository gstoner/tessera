// CF2 — lower tessera.control_for to scf.for (carry in iter_args; body kept as
// a func.call). The portable hardware-free step of the CUDA/ROCm control-flow
// path: the loop becomes ONE scf.for wrapper, not one launch per iteration.
//
// RUN: tessera-opt %s -split-input-file --tessera-control-flow-to-scf | FileCheck %s

// ─── Single-carry, no captures (legacy 1-operand form) ──────────────────────
func.func private @body1(%c: tensor<1x8xf32>) -> tensor<1x8xf32>

// CHECK-LABEL: func.func @for_single
// CHECK-DAG:   %[[LB:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[UB:.*]] = arith.constant 8 : index
// CHECK-DAG:   %[[ST:.*]] = arith.constant 1 : index
// CHECK:       %[[R:.*]] = scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[ST]]
// CHECK-SAME:    iter_args(%[[C:.*]] = %{{.*}}) -> (tensor<1x8xf32>)
// CHECK:         %[[N:.*]] = func.call @body1(%[[C]])
// CHECK:         scf.yield %[[N]]
// CHECK-NOT:   tessera.control_for
// CHECK:       return %[[R]]
func.func @for_single(%init: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %r = "tessera.control_for"(%init) {
    body = @body1, start = 0 : i64, stop = 8 : i64, step = 1 : i64
  } : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}

// -----
// ─── carry_arg_index form: one carried value + a loop-invariant capture ─────
func.func private @body_mm(%c: tensor<1x8xf32>, %w: tensor<8x8xf32>)
    -> tensor<1x8xf32>

// The weight %w is NOT an iter_arg — it's passed into the body call as an
// invariant capture, in original operand order.
// CHECK-LABEL: func.func @for_capture
// CHECK:       %[[R:.*]] = scf.for %{{.*}} iter_args(%[[C:.*]] = %{{.*}}) -> (tensor<1x8xf32>)
// CHECK:         func.call @body_mm(%[[C]], %{{.*}}) : (tensor<1x8xf32>, tensor<8x8xf32>) -> tensor<1x8xf32>
// CHECK:         scf.yield
// CHECK-NOT:   tessera.control_for
func.func @for_capture(%init: tensor<1x8xf32>, %w: tensor<8x8xf32>)
    -> tensor<1x8xf32> {
  %r = "tessera.control_for"(%init, %w) {
    body = @body_mm, start = 0 : i64, stop = 4 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<1x8xf32>, tensor<8x8xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}

// -----
// ─── legacy all-carried form → MULTI-iter_args scf.for (pytree carry) ───────
func.func private @body2(%a: tensor<1x8xf32>, %b: tensor<1x8xf32>)
    -> (tensor<1x8xf32>, tensor<1x8xf32>)

// Two carried tensors → scf.for with two iter_args and two results: the shape a
// pytree carry lowers to.
// CHECK-LABEL: func.func @for_pytree
// CHECK:       %[[R:.*]]:2 = scf.for %{{.*}} iter_args(%[[A:.*]] = %{{.*}}, %[[B:.*]] = %{{.*}}) -> (tensor<1x8xf32>, tensor<1x8xf32>)
// CHECK:         %[[N:.*]]:2 = func.call @body2(%[[A]], %[[B]])
// CHECK:         scf.yield %[[N]]#0, %[[N]]#1
// CHECK-NOT:   tessera.control_for
// CHECK:       return %[[R]]#0, %[[R]]#1
func.func @for_pytree(%a: tensor<1x8xf32>, %b: tensor<1x8xf32>)
    -> (tensor<1x8xf32>, tensor<1x8xf32>) {
  %r:2 = "tessera.control_for"(%a, %b) {
    body = @body2, start = 0 : i64, stop = 8 : i64, step = 1 : i64
  } : (tensor<1x8xf32>, tensor<1x8xf32>) -> (tensor<1x8xf32>, tensor<1x8xf32>)
  return %r#0, %r#1 : tensor<1x8xf32>, tensor<1x8xf32>
}
