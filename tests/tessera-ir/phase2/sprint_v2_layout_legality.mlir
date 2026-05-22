// RUN: tessera-opt --tessera-layout-legality %s -split-input-file -verify-diagnostics | FileCheck %s

// Sprint V2 (2026-05-22) — LayoutLegalityPass first rule:
// tessera.cast with `tessera.layout` attribute outside the canonical
// accept-set emits LAYOUT_LEGALITY_UNKNOWN_LAYOUT and fails the pass.

// ─────────────────────────────────────────────────────────────────────────
// Positive: cast with canonical layout name (row_major) — pass
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @cast_row_major
// CHECK:       tessera.cast
func.func @cast_row_major(%x: tensor<4x16xf32>) -> tensor<4x16xf16> {
  %y = "tessera.cast"(%x) {tessera.layout = "row_major"}
       : (tensor<4x16xf32>) -> tensor<4x16xf16>
  return %y : tensor<4x16xf16>
}

// -----

// Positive: cast with canonical bhsd (attention 4D) — pass
//
// CHECK-LABEL: func.func @cast_bhsd
// CHECK:       tessera.cast
func.func @cast_bhsd(%x: tensor<2x4x16x32xf32>) -> tensor<2x4x16x32xf16> {
  %y = "tessera.cast"(%x) {tessera.layout = "bhsd"}
       : (tensor<2x4x16x32xf32>) -> tensor<2x4x16x32xf16>
  return %y : tensor<2x4x16x32xf16>
}

// -----

// Positive: cast with no layout attribute at all — pass-through
//
// CHECK-LABEL: func.func @cast_no_layout
// CHECK:       tessera.cast
func.func @cast_no_layout(%x: tensor<4x16xf32>) -> tensor<4x16xf16> {
  %y = "tessera.cast"(%x) : (tensor<4x16xf32>) -> tensor<4x16xf16>
  return %y : tensor<4x16xf16>
}

// -----

// Negative: cast with non-canonical layout — emits diagnostic
func.func @cast_unknown_layout(%x: tensor<4x16xf32>) -> tensor<4x16xf16> {
  // expected-error @+1 {{LAYOUT_LEGALITY_UNKNOWN_LAYOUT}}
  %y = "tessera.cast"(%x) {tessera.layout = "exotic_block_format"}
       : (tensor<4x16xf32>) -> tensor<4x16xf16>
  return %y : tensor<4x16xf16>
}
