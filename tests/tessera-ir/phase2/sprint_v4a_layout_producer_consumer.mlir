// RUN: tessera-opt --tessera-layout-legality %s -split-input-file -verify-diagnostics | FileCheck %s

// Sprint V4a (2026-05-22) — LayoutLegalityPass second rule:
// producer/consumer accept-set mismatch detection.
//
// matmul's accept-set is {row_major, col_major}.  When an operand's
// producer carries a `tessera.layout` attribute outside this set
// (e.g., bsr or packed) without an intervening `tessera.cast`, the
// pass emits LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH.

// ─────────────────────────────────────────────────────────────────────────
// POSITIVE: producer emits row_major; matmul accepts. ✓
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @matmul_row_major_ok
// CHECK:       tessera.matmul
func.func @matmul_row_major_ok(%x: tensor<4x8xbf16>, %w: tensor<8x16xbf16>) -> tensor<4x16xf32> {
  %xc = "tessera.cast"(%x) {tessera.layout = "row_major"}
      : (tensor<4x8xbf16>) -> tensor<4x8xbf16>
  %wc = "tessera.cast"(%w) {tessera.layout = "row_major"}
      : (tensor<8x16xbf16>) -> tensor<8x16xbf16>
  %out = "tessera.matmul"(%xc, %wc) {transposeA = false, transposeB = false}
      : (tensor<4x8xbf16>, tensor<8x16xbf16>) -> tensor<4x16xf32>
  return %out : tensor<4x16xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// NEGATIVE: producer emits bsr layout; matmul rejects.
// ─────────────────────────────────────────────────────────────────────────

func.func @matmul_bsr_lhs_rejected(%x: tensor<4x8xbf16>, %w: tensor<8x16xbf16>) -> tensor<4x16xf32> {
  %xc = "tessera.cast"(%x) {tessera.layout = "bsr"}
      : (tensor<4x8xbf16>) -> tensor<4x8xbf16>
  %wc = "tessera.cast"(%w) {tessera.layout = "row_major"}
      : (tensor<8x16xbf16>) -> tensor<8x16xbf16>
  // expected-error @+1 {{LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH: tessera.matmul operand 'lhs' has layout "bsr" but matmul's accept-set is {row_major, col_major}}}
  %out = "tessera.matmul"(%xc, %wc) {transposeA = false, transposeB = false}
      : (tensor<4x8xbf16>, tensor<8x16xbf16>) -> tensor<4x16xf32>
  return %out : tensor<4x16xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// NEGATIVE: producer emits packed layout on rhs; matmul rejects.
// ─────────────────────────────────────────────────────────────────────────

func.func @matmul_packed_rhs_rejected(%x: tensor<4x8xbf16>, %w: tensor<8x16xbf16>) -> tensor<4x16xf32> {
  %xc = "tessera.cast"(%x) {tessera.layout = "row_major"}
      : (tensor<4x8xbf16>) -> tensor<4x8xbf16>
  %wc = "tessera.cast"(%w) {tessera.layout = "packed"}
      : (tensor<8x16xbf16>) -> tensor<8x16xbf16>
  // expected-error @+1 {{LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH: tessera.matmul operand 'rhs' has layout "packed" but matmul's accept-set is {row_major, col_major}}}
  %out = "tessera.matmul"(%xc, %wc) {transposeA = false, transposeB = false}
      : (tensor<4x8xbf16>, tensor<8x16xbf16>) -> tensor<4x16xf32>
  return %out : tensor<4x16xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// POSITIVE: no producer layout attribute → no enforcement (matmul
// works with the canonical default).
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @matmul_no_producer_layout_attr
// CHECK:       tessera.matmul
func.func @matmul_no_producer_layout_attr(%x: tensor<4x8xbf16>, %w: tensor<8x16xbf16>) -> tensor<4x16xf32> {
  %out = "tessera.matmul"(%x, %w) {transposeA = false, transposeB = false}
      : (tensor<4x8xbf16>, tensor<8x16xbf16>) -> tensor<4x16xf32>
  return %out : tensor<4x16xf32>
}
