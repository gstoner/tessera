// CF2b/CF2c — lower tessera.control_if → scf.if and tessera.control_while →
// bounded scf.while (branch/loop bodies kept as func.calls). Same payload-skip
// discipline as control_for: the executable-payload form (then_opcodes /
// body_opcodes, with stub @symbols) is left for the CF0 guard / CF3/CF4 decoder.
//
// RUN: tessera-opt %s -split-input-file --tessera-control-flow-to-scf | FileCheck %s

// ─── control_if → scf.if ────────────────────────────────────────────────────
func.func private @tb(%x: tensor<1x8xf32>) -> tensor<1x8xf32>
func.func private @eb(%x: tensor<1x8xf32>) -> tensor<1x8xf32>

// CHECK-LABEL: func.func @if_lower
// CHECK:       %[[F:.*]] = tensor.extract %{{.*}}[%{{.*}}] : tensor<1xf32>
// CHECK:       %[[C:.*]] = arith.cmpf ogt, %[[F]], %{{.*}}
// CHECK:       %[[R:.*]] = scf.if %[[C]] -> (tensor<1x8xf32>) {
// CHECK:         %[[T:.*]] = func.call @tb(%{{.*}})
// CHECK:         scf.yield %[[T]]
// CHECK:       } else {
// CHECK:         %[[E:.*]] = func.call @eb(%{{.*}})
// CHECK:         scf.yield %[[E]]
// CHECK-NOT:   tessera.control_if
func.func @if_lower(%flag: tensor<1xf32>, %x: tensor<1x8xf32>)
    -> tensor<1x8xf32> {
  %r = "tessera.control_if"(%flag, %x) {
    then_branch = @tb, else_branch = @eb, flag_arg_index = 0 : i64
  } : (tensor<1xf32>, tensor<1x8xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}

// -----
// ─── control_while → bounded scf.while ──────────────────────────────────────
func.func private @wb(%c: tensor<1x4xf32>) -> tensor<1x4xf32>
func.func private @wc(%c: tensor<1x4xf32>) -> tensor<1xf32>

// State is (counter : index, carry); the before region SHORT-CIRCUITS the
// bound — it checks `i < max_iters` first and only calls @cond inside the
// then-branch of an scf.if, so @cond is never evaluated at i == max_iters. The
// after region runs the body and increments. Op result is the carry (#1).
// CHECK-LABEL: func.func @while_lower
// CHECK:       %[[W:.*]]:2 = scf.while (%[[I:.*]] = %{{.*}}, %[[C:.*]] = %{{.*}}) : (index, tensor<1x4xf32>) -> (index, tensor<1x4xf32>)
// CHECK:         %[[LT:.*]] = arith.cmpi ult, %[[I]], %{{.*}}
// CHECK:         %[[CONT:.*]] = scf.if %[[LT]] -> (i1) {
// CHECK:           func.call @wc(%[[C]])
// CHECK:           arith.cmpf ogt
// CHECK:           scf.yield %{{.*}} : i1
// CHECK:         } else {
// CHECK:           %[[F:.*]] = arith.constant false
// CHECK:           scf.yield %[[F]] : i1
// CHECK:         }
// CHECK:         scf.condition(%[[CONT]]) %[[I]], %[[C]]
// CHECK:       } do {
// CHECK:         func.call @wb(%{{.*}})
// CHECK:         arith.addi
// CHECK:         scf.yield
// CHECK:       return %[[W]]#1
// CHECK-NOT:   tessera.control_while
func.func @while_lower(%init: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %r = "tessera.control_while"(%init) {
    body = @wb, cond = @wc, carry_arg_index = 0 : i64, max_iters = 8 : i64
  } : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %r : tensor<1x4xf32>
}

// -----
// ─── payload-form control_if is SKIPPED (stays for the guard) ───────────────
func.func private @tb2()
func.func private @eb2()
// CHECK-LABEL: func.func @if_payload_skipped
// CHECK:       tessera.control_if
func.func @if_payload_skipped(%flag: tensor<1xf32>) -> tensor<1x8xf32> {
  %r = "tessera.control_if"(%flag) {
    then_branch = @tb2, else_branch = @eb2, flag_arg_index = 0 : i64,
    then_opcodes = array<i32: 1>, then_out_id = 1 : i64,
    else_opcodes = array<i32: 1>, else_out_id = 1 : i64
  } : (tensor<1xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}

// -----
// ─── payload-form control_while is SKIPPED ──────────────────────────────────
func.func private @wb2()
func.func private @wc2()
// CHECK-LABEL: func.func @while_payload_skipped
// CHECK:       tessera.control_while
func.func @while_payload_skipped(%init: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %r = "tessera.control_while"(%init) {
    body = @wb2, cond = @wc2, carry_arg_index = 0 : i64, max_iters = 4 : i64,
    body_opcodes = array<i32: 0>, body_out_id = 1 : i64,
    cond_opcodes = array<i32: 0>, cond_out_id = 1 : i64
  } : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %r : tensor<1x4xf32>
}
