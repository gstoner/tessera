// CF1 — control-flow op verifier coverage (the cases the Sprint-V9 fixture
// tests/tessera-ir/phase2/sprint_v9_control_stub_misc_verifiers.mlir does not
// reach): the executable-payload carry_arg_index form of ControlForOp /
// ControlWhileOp, and ControlIfOp's then/else payload symmetry. Positive cases
// confirm the payload form verifies; negative cases pin each emitOpError.
//
// RUN: tessera-opt %s -split-input-file -verify-diagnostics | FileCheck %s

func.func private @body()

// ─── control_for, payload (carry_arg_index) form — positive ─────────────────
// CHECK-LABEL: func.func @for_carry_ok
func.func @for_carry_ok(%c: tensor<1x8xf32>, %w: tensor<8x8xf32>)
    -> tensor<1x8xf32> {
  // One loop-carried operand (index 0) + one invariant capture (%w); a single
  // result whose type matches the carried operand.
  %r = "tessera.control_for"(%c, %w) {
    body = @body, start = 0 : i64, stop = 8 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<1x8xf32>, tensor<8x8xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}

// -----
func.func private @body()
func.func @for_carry_idx_oob(%init: tensor<1x8xf32>) -> tensor<1x8xf32> {
  // expected-error @+1 {{carry_arg_index out of range}}
  %r = "tessera.control_for"(%init) {
    body = @body, start = 0 : i64, stop = 8 : i64, step = 1 : i64,
    carry_arg_index = 5 : i64
  } : (tensor<1x8xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}

// -----
func.func private @body()
func.func @for_carry_multi_result(%a: tensor<1x8xf32>, %b: tensor<1x8xf32>)
    -> (tensor<1x8xf32>, tensor<1x8xf32>) {
  // carry_arg_index selects ONE carried value, so the op yields exactly one
  // result (the rest of the operands are invariant captures).
  // expected-error @+1 {{carries one value}}
  %r:2 = "tessera.control_for"(%a, %b) {
    body = @body, start = 0 : i64, stop = 8 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<1x8xf32>, tensor<1x8xf32>) -> (tensor<1x8xf32>, tensor<1x8xf32>)
  return %r#0, %r#1 : tensor<1x8xf32>, tensor<1x8xf32>
}

// -----
func.func private @body()
func.func @for_result_type_mismatch(%a: tensor<1x8xf32>, %w: tensor<8x8xf32>)
    -> tensor<2x8xf32> {
  // expected-error @+1 {{for result type must match the carried iter_arg type}}
  %r = "tessera.control_for"(%a, %w) {
    body = @body, start = 0 : i64, stop = 8 : i64, step = 1 : i64,
    carry_arg_index = 0 : i64
  } : (tensor<1x8xf32>, tensor<8x8xf32>) -> tensor<2x8xf32>
  return %r : tensor<2x8xf32>
}

// -----
// ─── control_while, payload (carry_arg_index) form ──────────────────────────
func.func private @wb()
func.func private @wc()
// CHECK-LABEL: func.func @while_carry_ok
func.func @while_carry_ok(%c: tensor<1x4xf32>, %w: tensor<4x4xf32>)
    -> tensor<1x4xf32> {
  %r = "tessera.control_while"(%c, %w) {
    body = @wb, cond = @wc, carry_arg_index = 0 : i64, max_iters = 8 : i64
  } : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
  return %r : tensor<1x4xf32>
}

// -----
func.func private @wb()
func.func private @wc()
func.func @while_carry_idx_oob(%init: tensor<1x4xf32>) -> tensor<1x4xf32> {
  // expected-error @+1 {{carry_arg_index out of range}}
  %r = "tessera.control_while"(%init) {
    body = @wb, cond = @wc, carry_arg_index = 5 : i64, max_iters = 8 : i64
  } : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %r : tensor<1x4xf32>
}

// -----
func.func private @wb()
func.func private @wc()
func.func @while_result_type_mismatch(%init: tensor<1x4xf32>)
    -> tensor<2x4xf32> {
  // expected-error @+1 {{while result type must match the carried iter_arg type}}
  %r = "tessera.control_while"(%init) {
    body = @wb, cond = @wc, carry_arg_index = 0 : i64, max_iters = 8 : i64
  } : (tensor<1x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// -----
// ─── control_if — then/else payload symmetry ────────────────────────────────
func.func private @tb()
func.func private @eb()
func.func @if_payload_asymmetric(%flag: tensor<1xf32>, %a: tensor<1x8xf32>)
    -> tensor<1x8xf32> {
  // A then payload without a matching else payload (or vice-versa) is illegal:
  // both branches are serialized to the run_graph_cond ABI together.
  // expected-error @+1 {{then/else payloads must both be present or both absent}}
  %r = "tessera.control_if"(%flag, %a) {
    then_branch = @tb, else_branch = @eb, flag_arg_index = 0 : i64,
    then_opcodes = array<i32: 1>
  } : (tensor<1xf32>, tensor<1x8xf32>) -> tensor<1x8xf32>
  return %r : tensor<1x8xf32>
}

// -----
// ─── control_scan (CF4e) — positive: (init, xs) -> (carry, ys) ──────────────
func.func private @sbody()
// CHECK-LABEL: func.func @scan_ok
func.func @scan_ok(%init: tensor<4xf32>, %xs: tensor<3x4xf32>)
    -> (tensor<4xf32>, tensor<3x4xf32>) {
  %c, %ys = "tessera.control_scan"(%init, %xs) {
    body = @sbody, trip = 3 : i64, carry_arg_index = 0 : i64
  } : (tensor<4xf32>, tensor<3x4xf32>) -> (tensor<4xf32>, tensor<3x4xf32>)
  return %c, %ys : tensor<4xf32>, tensor<3x4xf32>
}

// -----
func.func private @sbody()
func.func @scan_trip_nonpos(%init: tensor<4xf32>, %xs: tensor<3x4xf32>)
    -> (tensor<4xf32>, tensor<3x4xf32>) {
  // expected-error @+1 {{trip must be positive}}
  %c, %ys = "tessera.control_scan"(%init, %xs) {
    body = @sbody, trip = 0 : i64, carry_arg_index = 0 : i64
  } : (tensor<4xf32>, tensor<3x4xf32>) -> (tensor<4xf32>, tensor<3x4xf32>)
  return %c, %ys : tensor<4xf32>, tensor<3x4xf32>
}

// -----
func.func private @sbody()
func.func @scan_carry_mismatch(%init: tensor<4xf32>, %xs: tensor<3x4xf32>)
    -> (tensor<8xf32>, tensor<3x4xf32>) {
  // expected-error @+1 {{carry_out type must match the init carry type}}
  %c, %ys = "tessera.control_scan"(%init, %xs) {
    body = @sbody, trip = 3 : i64, carry_arg_index = 0 : i64
  } : (tensor<4xf32>, tensor<3x4xf32>) -> (tensor<8xf32>, tensor<3x4xf32>)
  return %c, %ys : tensor<8xf32>, tensor<3x4xf32>
}

// -----
func.func private @sbody()
func.func @scan_ys_trip_mismatch(%init: tensor<4xf32>, %xs: tensor<3x4xf32>)
    -> (tensor<4xf32>, tensor<5x4xf32>) {
  // expected-error @+1 {{ys leading dim must equal trip}}
  %c, %ys = "tessera.control_scan"(%init, %xs) {
    body = @sbody, trip = 3 : i64, carry_arg_index = 0 : i64
  } : (tensor<4xf32>, tensor<3x4xf32>) -> (tensor<4xf32>, tensor<5x4xf32>)
  return %c, %ys : tensor<4xf32>, tensor<5x4xf32>
}

// -----
// ─── control_scan with a loop-invariant capture (W) — positive (CF4e-2) ─────
func.func private @sbody()
// CHECK-LABEL: func.func @scan_capture_ok
func.func @scan_capture_ok(%init: tensor<1x4xf32>, %xs: tensor<3x4xf32>,
    %W: tensor<4x4xf32>) -> (tensor<1x4xf32>, tensor<3x4xf32>) {
  %c, %ys = "tessera.control_scan"(%init, %xs, %W) {
    body = @sbody, trip = 3 : i64, carry_arg_index = 0 : i64
  } : (tensor<1x4xf32>, tensor<3x4xf32>, tensor<4x4xf32>)
      -> (tensor<1x4xf32>, tensor<3x4xf32>)
  return %c, %ys : tensor<1x4xf32>, tensor<3x4xf32>
}
