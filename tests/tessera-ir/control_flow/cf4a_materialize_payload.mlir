// CF4a — decode the tessera.control_for op-list payload (body_opcodes/...) into
// a real @body func.func of tessera.* ops. The frontend emits loops in this
// payload form (carry-only @body stub + serialized op-list); CF2 skips it. This
// pass materializes the body so CF2 can then lower the loop to scf.for — the
// prerequisite for an executable device body (CF4b ROCm).
//
// Op-list ABI: ids 0..n-1 = operands, id n = live carry, id n+1+j = body op j.
// Opcodes: 0 matmul, 1-4 add/sub/mul/div, 10-12 softmax/rmsnorm/layer_norm,
// 20-24 relu/sigmoid/tanh/silu/gelu. iattr = matmul transpose bits, fattr = eps.
//
// RUN: tessera-opt %s -split-input-file --tessera-materialize-control-payload \
// RUN:   --allow-unregistered-dialect | FileCheck %s

// ─── matmul(carry, w) → silu, with a loop-invariant capture %w ──────────────
// 2 operands (carry=id0, w=id1); live carry=id2; op0 matmul→id3, op1 silu→id4.
func.func private @loop_body(tensor<1x8xf32>) -> tensor<1x8xf32>

// The stub @loop_body is rewritten to take all operands and run the real body.
// CHECK-LABEL: func.func private @loop_body(%arg0: tensor<1x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<1x8xf32>
// CHECK:       %[[MM:.*]] = tessera.matmul %arg0, %arg1 {transpose_a = false, transpose_b = false}
// CHECK:       %[[S:.*]] = tessera.silu %[[MM]]
// CHECK:       return %[[S]]
// CHECK-LABEL: func.func @f
// CHECK:       tessera.control_for %arg0, %arg1 {body = @loop_body, carry_arg_index = 0 : i64, start = 0 : i64, step = 1 : i64, stop = 8 : i64}
// CHECK-NOT:   body_opcodes
func.func @f(%init: tensor<1x8xf32>, %w: tensor<8x8xf32>) -> tensor<1x8xf32> {
  %r = "tessera.control_for"(%init, %w) {
    body = @loop_body, start = 0 : i64, stop = 8 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64,
    body_opcodes = array<i32: 0, 23>, body_in0 = array<i32: 2, 3>,
    body_in1 = array<i32: 1, -1>, body_iattr = array<i32: 0, 0>,
    body_fattr = array<f32: 1.0e-05, 1.0e-05>, body_out_id = 4 : i64
  } : (tensor<1x8xf32>, tensor<8x8xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}

// -----
// ─── single-op body, no capture: carry = rmsnorm(carry) with eps from fattr ─
// 1 operand (carry=id0); live carry=id1; op0 rmsnorm→id2.
func.func private @nb(tensor<1x16xf32>) -> tensor<1x16xf32>

// CHECK-LABEL: func.func private @nb(%arg0: tensor<1x16xf32>) -> tensor<1x16xf32>
// CHECK:       %[[N:.*]] = tessera.rmsnorm %arg0 {eps = 9.9999997473787516E-6 : f64}
// CHECK:       return %[[N]]
func.func @g(%init: tensor<1x16xf32>) -> tensor<1x16xf32> {
  %r = "tessera.control_for"(%init) {
    body = @nb, start = 0 : i64, stop = 4 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64,
    body_opcodes = array<i32: 11>, body_in0 = array<i32: 1>,
    body_in1 = array<i32: -1>, body_iattr = array<i32: 0>,
    body_fattr = array<f32: 1.0e-05>, body_out_id = 2 : i64
  } : (tensor<1x16xf32>) -> tensor<1x16xf32>
  return %r : tensor<1x16xf32>
}

// -----
// ─── an unknown opcode leaves the payload UNTOUCHED (for the guard) ──────────
func.func private @ub(tensor<1x8xf32>) -> tensor<1x8xf32>

// CHECK-LABEL: func.func @h
// CHECK:       tessera.control_for
// CHECK-SAME:  body_opcodes
func.func @h(%init: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %r = "tessera.control_for"(%init) {
    body = @ub, start = 0 : i64, stop = 4 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64,
    body_opcodes = array<i32: 999>, body_in0 = array<i32: 1>,
    body_in1 = array<i32: -1>, body_iattr = array<i32: 0>,
    body_fattr = array<f32: 1.0e-05>, body_out_id = 2 : i64
  } : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}
