// RUN: tessera-opt --tessera-layout-legality %s -split-input-file -verify-diagnostics | FileCheck %s

// LayoutLegalityPass producer/consumer accept-set rule extended (2026-06-11)
// from matmul to more consumer ops, each with the operands that actually carry
// the layout contract:
//   tessera.conv2d_nhwc — data operand (#0) must be nhwc (the filter is a
//                         separate weight layout, not checked here);
//   tessera.flash_attn  — Q/K/V (#0..2) must be bhsd.

// ─────────────────────────────────────────────────────────────────────────
// POSITIVE: conv2d_nhwc data operand is nhwc. ✓
// ─────────────────────────────────────────────────────────────────────────
// CHECK-LABEL: func.func @conv_nhwc_ok
// CHECK:       tessera.conv2d_nhwc
func.func @conv_nhwc_ok(%x: tensor<1x8x8x3xf32>, %w: tensor<3x3x3x16xf32>) -> tensor<1x8x8x16xf32> {
  %xc = "tessera.cast"(%x) {tessera.layout = "nhwc"} : (tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
  %o = "tessera.conv2d_nhwc"(%xc, %w) {strides = [1, 1], dilations = [1, 1]}
      : (tensor<1x8x8x3xf32>, tensor<3x3x3x16xf32>) -> tensor<1x8x8x16xf32>
  return %o : tensor<1x8x8x16xf32>
}

// -----

// NEGATIVE: conv2d_nhwc data operand carries nchw — rejected.
func.func @conv_nchw_rejected(%x: tensor<1x8x8x3xf32>, %w: tensor<3x3x3x16xf32>) -> tensor<1x8x8x16xf32> {
  %xc = "tessera.cast"(%x) {tessera.layout = "nchw"} : (tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
  // expected-error @+1 {{LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH: tessera.conv2d_nhwc operand #0 has layout "nchw" but its accept-set is {nhwc}}}
  %o = "tessera.conv2d_nhwc"(%xc, %w) {strides = [1, 1], dilations = [1, 1]}
      : (tensor<1x8x8x3xf32>, tensor<3x3x3x16xf32>) -> tensor<1x8x8x16xf32>
  return %o : tensor<1x8x8x16xf32>
}

// -----

// POSITIVE: flash_attn Q is bhsd. ✓
// CHECK-LABEL: func.func @attn_bhsd_ok
// CHECK:       tessera.flash_attn
func.func @attn_bhsd_ok(%q: tensor<1x2x8x4xf32>, %k: tensor<1x2x8x4xf32>, %v: tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32> {
  %qc = "tessera.cast"(%q) {tessera.layout = "bhsd"} : (tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>
  %o = "tessera.flash_attn"(%qc, %k, %v) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {head_dim = 4 : i64}
      : (tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>
  return %o : tensor<1x2x8x4xf32>
}

// -----

// NEGATIVE: flash_attn Q carries row_major — rejected.
func.func @attn_row_major_q_rejected(%q: tensor<1x2x8x4xf32>, %k: tensor<1x2x8x4xf32>, %v: tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32> {
  %qc = "tessera.cast"(%q) {tessera.layout = "row_major"} : (tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>
  // expected-error @+1 {{LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH: tessera.flash_attn operand #0 has layout "row_major" but its accept-set is {bhsd}}}
  %o = "tessera.flash_attn"(%qc, %k, %v) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {head_dim = 4 : i64}
      : (tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>
  return %o : tensor<1x2x8x4xf32>
}
