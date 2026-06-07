// RUN: tessera-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// Sprint V8 (2026-06-07) — norm/softmax-family per-op verifiers.
//
// Adds shape+dtype-preservation (and eps>0 / axis-bounds) checks to four ops
// that previously had no verifier: tessera.rmsnorm, tessera.rmsnorm_safe,
// tessera.softmax_safe, tessera.log_softmax. Mirrors the SoftmaxOp/LayerNormOp
// contracts. Each negative case is paired with a positive twin.

// ─── rmsnorm — positive ─────────────────────────────────────────────────────
// CHECK-LABEL: func.func @rmsnorm_ok
// CHECK:       tessera.rmsnorm
func.func @rmsnorm_ok(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %y = "tessera.rmsnorm"(%x) {eps = 1.0e-05 : f64} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %y : tensor<4x8xf32>
}

// -----
// rmsnorm — negative: element type changes.
func.func @rmsnorm_dtype(%x: tensor<4x8xf32>) -> tensor<4x8xbf16> {
  // expected-error @+1 {{rmsnorm must preserve element type}}
  %y = "tessera.rmsnorm"(%x) : (tensor<4x8xf32>) -> tensor<4x8xbf16>
  return %y : tensor<4x8xbf16>
}

// -----
// rmsnorm — negative: non-positive eps.
func.func @rmsnorm_eps(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error @+1 {{eps must be positive}}
  %y = "tessera.rmsnorm"(%x) {eps = -1.0 : f64} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %y : tensor<4x8xf32>
}

// -----
// ─── rmsnorm_safe — positive ────────────────────────────────────────────────
// CHECK-LABEL: func.func @rmsnorm_safe_ok
// CHECK:       tessera.rmsnorm_safe
func.func @rmsnorm_safe_ok(%x: tensor<2x16xf32>) -> tensor<2x16xf32> {
  %y = "tessera.rmsnorm_safe"(%x) : (tensor<2x16xf32>) -> tensor<2x16xf32>
  return %y : tensor<2x16xf32>
}

// -----
// rmsnorm_safe — negative: rank changes.
func.func @rmsnorm_safe_rank(%x: tensor<2x16xf32>) -> tensor<32xf32> {
  // expected-error @+1 {{rmsnorm_safe must preserve rank}}
  %y = "tessera.rmsnorm_safe"(%x) : (tensor<2x16xf32>) -> tensor<32xf32>
  return %y : tensor<32xf32>
}

// -----
// ─── softmax_safe — positive ────────────────────────────────────────────────
// CHECK-LABEL: func.func @softmax_safe_ok
// CHECK:       tessera.softmax_safe
func.func @softmax_safe_ok(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %y = "tessera.softmax_safe"(%x) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %y : tensor<4x8xf32>
}

// -----
// softmax_safe — negative: dim changes.
func.func @softmax_safe_dim(%x: tensor<4x8xf32>) -> tensor<4x16xf32> {
  // expected-error @+1 {{softmax_safe must preserve dim 1}}
  %y = "tessera.softmax_safe"(%x) : (tensor<4x8xf32>) -> tensor<4x16xf32>
  return %y : tensor<4x16xf32>
}

// -----
// ─── log_softmax — positive ─────────────────────────────────────────────────
// CHECK-LABEL: func.func @log_softmax_ok
// CHECK:       tessera.log_softmax
func.func @log_softmax_ok(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %y = "tessera.log_softmax"(%x) {axis = -1 : i64} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %y : tensor<4x8xf32>
}

// -----
// log_softmax — negative: axis out of range.
func.func @log_softmax_axis(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error @+1 {{axis out of range}}
  %y = "tessera.log_softmax"(%x) {axis = 5 : i64} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %y : tensor<4x8xf32>
}
