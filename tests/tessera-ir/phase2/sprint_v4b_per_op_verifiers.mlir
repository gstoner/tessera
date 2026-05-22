// RUN: tessera-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// Sprint V4b (2026-05-22) — long-tail per-op verifiers.
//
// Adds shape-preservation + attribute-bounds checks to four ops that
// previously had either no verifier or a trivial `return success();`
// stub: `tessera.cast`, `tessera.softmax`, `tessera.rope`,
// `tessera.dropout`.
//
// Each negative case is paired with a positive twin to lock the
// non-degenerate path.

// ─────────────────────────────────────────────────────────────────────────
// CAST — positive: rank + dims preserved, element type may differ.
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @cast_ok
// CHECK:       tessera.cast
func.func @cast_ok(%x: tensor<4x8xf32>) -> tensor<4x8xbf16> {
  %y = "tessera.cast"(%x) : (tensor<4x8xf32>) -> tensor<4x8xbf16>
  return %y : tensor<4x8xbf16>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// CAST — negative: rank changes.
// ─────────────────────────────────────────────────────────────────────────

func.func @cast_rank_mismatch(%x: tensor<4x8xf32>) -> tensor<32xbf16> {
  // expected-error @+1 {{cast must preserve rank: 2 -> 1}}
  %y = "tessera.cast"(%x) : (tensor<4x8xf32>) -> tensor<32xbf16>
  return %y : tensor<32xbf16>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// CAST — negative: static dim differs.
// ─────────────────────────────────────────────────────────────────────────

func.func @cast_dim_mismatch(%x: tensor<4x8xf32>) -> tensor<4x16xbf16> {
  // expected-error @+1 {{cast must preserve dim 1: 8 vs 16}}
  %y = "tessera.cast"(%x) : (tensor<4x8xf32>) -> tensor<4x16xbf16>
  return %y : tensor<4x16xbf16>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// SOFTMAX — positive: axis = -1 with rank-3 input is in range.
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @softmax_ok
// CHECK:       tessera.softmax
func.func @softmax_ok(%x: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %y = "tessera.softmax"(%x) {axis = -1 : i64}
      : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %y : tensor<2x4x8xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// SOFTMAX — negative: axis out of range (axis=5 for rank-3 input).
// ─────────────────────────────────────────────────────────────────────────

func.func @softmax_axis_out_of_range(%x: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  // expected-error @+1 {{axis out of range: got 5 for rank-3 input (expected -3 <= axis < 3)}}
  %y = "tessera.softmax"(%x) {axis = 5 : i64}
      : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %y : tensor<2x4x8xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// SOFTMAX — negative: shape changes.
// ─────────────────────────────────────────────────────────────────────────

func.func @softmax_shape_mismatch(%x: tensor<2x4x8xf32>) -> tensor<2x4x16xf32> {
  // expected-error @+1 {{softmax must preserve dim 2}}
  %y = "tessera.softmax"(%x) : (tensor<2x4x8xf32>) -> tensor<2x4x16xf32>
  return %y : tensor<2x4x16xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// ROPE — positive: rank + dims preserved.
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @rope_ok
// CHECK:       tessera.rope
func.func @rope_ok(%x: tensor<2x4x8xf32>, %theta: tensor<4x8xf32>) -> tensor<2x4x8xf32> {
  %y = "tessera.rope"(%x, %theta)
      : (tensor<2x4x8xf32>, tensor<4x8xf32>) -> tensor<2x4x8xf32>
  return %y : tensor<2x4x8xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// ROPE — negative: output rank differs from input.
// ─────────────────────────────────────────────────────────────────────────

func.func @rope_rank_mismatch(%x: tensor<2x4x8xf32>, %theta: tensor<4x8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{rope must preserve rank: 3 -> 1}}
  %y = "tessera.rope"(%x, %theta)
      : (tensor<2x4x8xf32>, tensor<4x8xf32>) -> tensor<8xf32>
  return %y : tensor<8xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// DROPOUT — positive: p = 0.1 is in [0, 1).
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @dropout_ok
// CHECK:       tessera.dropout
func.func @dropout_ok(%x: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %y = "tessera.dropout"(%x) {p = 1.000000e-01 : f64}
      : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %y : tensor<2x4x8xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// DROPOUT — negative: p = 1.0 is out of bounds (would zero every elt).
// ─────────────────────────────────────────────────────────────────────────

func.func @dropout_p_ge_one(%x: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  // expected-error @+1 {{dropout probability must satisfy 0.0 <= p < 1.0; got 1}}
  %y = "tessera.dropout"(%x) {p = 1.000000e+00 : f64}
      : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %y : tensor<2x4x8xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// DROPOUT — negative: p = -0.1 is out of bounds.
// ─────────────────────────────────────────────────────────────────────────

func.func @dropout_p_negative(%x: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  // expected-error @+1 {{dropout probability must satisfy 0.0 <= p < 1.0; got -1.000000e-01}}
  %y = "tessera.dropout"(%x) {p = -1.000000e-01 : f64}
      : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %y : tensor<2x4x8xf32>
}
