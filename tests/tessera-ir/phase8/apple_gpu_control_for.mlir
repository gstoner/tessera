// REQUIRES: tessera-apple-backend
//
// RUN: tessera-opt %s --tessera-control-for-to-apple_gpu | FileCheck %s

// Phase-G G-B — lower the Graph-IR bounded loop `tessera.control_for` to the
// Apple Target-IR op `tessera_apple.gpu.control_loop` (value-preserving; records
// the run_graph_loop runtime symbol). IR-only (Decision #19 hardware-free layer).

// The loop body is a symbol-referenced func.func (not a region) — control_for
// stays a value-semantic leaf. @loop_body need not be resolved for this lowering.
func.func private @loop_body(%c: tensor<1x8xf32>) -> tensor<1x8xf32>

// CHECK-LABEL: func.func @bounded_loop
func.func @bounded_loop(%init: tensor<1x8xf32>) -> tensor<1x8xf32> {
  // CHECK: %[[R:.*]] = tessera_apple.gpu.control_loop %{{.*}} {
  // CHECK-SAME: body = @loop_body
  // CHECK-SAME: start = 0 : i64
  // CHECK-SAME: status = "artifact"
  // CHECK-SAME: step = 1 : i64
  // CHECK-SAME: stop = 8 : i64
  // CHECK-SAME: symbol = "tessera_apple_gpu_run_graph_loop_f32"
  // CHECK-NOT: tessera.control_for
  // CHECK: return %[[R]]
  %r = "tessera.control_for"(%init) {
    body = @loop_body, start = 0 : i64, stop = 8 : i64, step = 1 : i64
  } : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}

// Executable-payload form: the loop closes over a loop-invariant const (a weight
// matrix). Operands are [carry, const]; `carry_arg_index` selects the one carried
// value, the const is an invariant capture, and the op yields a single result.
// The verifier must accept #operands (2) != #results (1) here (mirrors
// control_while) — regression guard against requiring one-result-per-operand.
func.func private @loop_body_mm(%c: tensor<1x8xf32>) -> tensor<1x8xf32>

// CHECK-LABEL: func.func @bounded_loop_with_const
func.func @bounded_loop_with_const(%init: tensor<1x8xf32>,
                                   %w: tensor<8x8xf32>) -> tensor<1x8xf32> {
  // CHECK: %[[R:.*]] = tessera_apple.gpu.control_loop %{{.*}}, %{{.*}} {
  // CHECK-SAME: body = @loop_body_mm
  // CHECK-SAME: carry_arg_index = 0 : i64
  // CHECK-SAME: symbol = "tessera_apple_gpu_run_graph_loop_f32"
  // CHECK-NOT: tessera.control_for
  // CHECK: return %[[R]]
  %r = "tessera.control_for"(%init, %w) {
    body = @loop_body_mm, start = 0 : i64, stop = 4 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64,
    body_opcodes = array<i32: 0, 23>, body_in0 = array<i32: 2, 3>,
    body_in1 = array<i32: 1, -1>, body_iattr = array<i32: 0, 0>,
    body_fattr = array<f32: 1.0e-05, 1.0e-05>, body_out_id = 4 : i64
  } : (tensor<1x8xf32>, tensor<8x8xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}
