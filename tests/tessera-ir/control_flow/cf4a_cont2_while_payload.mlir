// CF4a-cont-2 — decode the tessera.control_while op-list payload (body_opcodes /
// cond_opcodes / ...) into real @body / @cond func.funcs of tessera.* ops, so
// CF2's LowerControlFlowToSCF can lower the loop to a bounded scf.while.
//
// Op-list ABI for control_while: ids 0..n-1 = operands, id n = live carry, body
// & cond op j = id n+1+j. CF2 calls @body / @cond with only the carry, so both
// materialize as single-arg funcs (id n → arg 0): @body is (carry)->carry, @cond
// is (carry)->pred (predicate type inferred from the cond op-list). @body and
// @cond must be distinct symbols.
//
// RUN: tessera-opt %s -split-input-file --tessera-materialize-control-payload \
// RUN:   --allow-unregistered-dialect | FileCheck %s

// ─── body = silu(c), cond = relu(c) ─────────────────────────────────────────
// 1 operand (carry = id 0); live carry = id 1; body/cond op0 → id 2.
func.func private @wb(%c: tensor<4xf32>) -> tensor<4xf32>
func.func private @wc(%c: tensor<4xf32>) -> tensor<4xf32>

// CHECK-LABEL: func.func private @wb(%arg0: tensor<4xf32>) -> tensor<4xf32>
// CHECK:       tessera.silu %arg0
// CHECK-LABEL: func.func private @wc(%arg0: tensor<4xf32>) -> tensor<4xf32>
// CHECK:       tessera.relu %arg0
// CHECK-LABEL: func.func @f
// CHECK:       tessera.control_while %{{.*}} {body = @wb, carry_arg_index = 0 : i64, cond = @wc, max_iters = 4 : i64}
// CHECK-NOT:   body_opcodes
func.func @f(%init: tensor<4xf32>) -> tensor<4xf32> {
  %r = "tessera.control_while"(%init) {
    body = @wb, cond = @wc, carry_arg_index = 0 : i64, max_iters = 4 : i64,
    body_opcodes = array<i32: 23>, body_in0 = array<i32: 1>,
    body_in1 = array<i32: -1>, body_iattr = array<i32: 0>,
    body_fattr = array<f32: 1.0e-05>, body_out_id = 2 : i64,
    cond_opcodes = array<i32: 20>, cond_in0 = array<i32: 1>,
    cond_in1 = array<i32: -1>, cond_iattr = array<i32: 0>,
    cond_fattr = array<f32: 1.0e-05>, cond_out_id = 2 : i64
  } : (tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----
// ─── body_branch == cond_branch (SHARED stub) is left untouched ─────────────
// Materializing @cond would overwrite the @body it shares, so leave it for the
// guard. (@body is (carry)->carry, @cond is (carry)->pred — they cannot be the
// same func.)
func.func private @shared(%c: tensor<4xf32>) -> tensor<4xf32>

// CHECK-LABEL: func.func @g
// CHECK:       tessera.control_while
// CHECK-SAME:  body_opcodes
func.func @g(%init: tensor<4xf32>) -> tensor<4xf32> {
  %r = "tessera.control_while"(%init) {
    body = @shared, cond = @shared, carry_arg_index = 0 : i64, max_iters = 4 : i64,
    body_opcodes = array<i32: 23>, body_in0 = array<i32: 1>,
    body_in1 = array<i32: -1>, body_iattr = array<i32: 0>,
    body_fattr = array<f32: 1.0e-05>, body_out_id = 2 : i64,
    cond_opcodes = array<i32: 20>, cond_in0 = array<i32: 1>,
    cond_in1 = array<i32: -1>, cond_iattr = array<i32: 0>,
    cond_fattr = array<f32: 1.0e-05>, cond_out_id = 2 : i64
  } : (tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}
