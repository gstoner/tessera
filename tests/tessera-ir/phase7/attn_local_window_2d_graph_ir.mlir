// RUN: tessera-opt %s | FileCheck %s
// RUN: tessera-opt %s | tessera-opt | FileCheck %s

// ============================================================================
// Sub-1 — Graph IR roundtrip for tessera.attn_local_window_2d.
//
// The new ODS op parses, prints, and re-parses cleanly with the
// rank-5 operand types and the window=[rh, rw] attribute.
// ============================================================================

// CHECK-LABEL: func @local_window_2d_static
// CHECK:       tessera.attn_local_window_2d
// CHECK-SAME:  window = [1, 1]
// CHECK-SAME:  (tensor<2x4x8x8x16xf32>, tensor<2x4x8x8x16xf32>, tensor<2x4x8x8x16xf32>) -> tensor<2x4x8x8x16xf32>
func.func @local_window_2d_static(
    %q: tensor<2x4x8x8x16xf32>,
    %k: tensor<2x4x8x8x16xf32>,
    %v: tensor<2x4x8x8x16xf32>
) -> tensor<2x4x8x8x16xf32> {
  %o = tessera.attn_local_window_2d %q, %k, %v {window = [1, 1]} :
      (tensor<2x4x8x8x16xf32>, tensor<2x4x8x8x16xf32>, tensor<2x4x8x8x16xf32>)
      -> tensor<2x4x8x8x16xf32>
  return %o : tensor<2x4x8x8x16xf32>
}

// CHECK-LABEL: func @local_window_2d_asymmetric_window
// CHECK:       tessera.attn_local_window_2d
// CHECK-SAME:  window = [2, 3]
func.func @local_window_2d_asymmetric_window(
    %q: tensor<1x2x5x7x4xf32>,
    %k: tensor<1x2x5x7x4xf32>,
    %v: tensor<1x2x5x7x4xf32>
) -> tensor<1x2x5x7x4xf32> {
  %o = tessera.attn_local_window_2d %q, %k, %v {window = [2, 3]} :
      (tensor<1x2x5x7x4xf32>, tensor<1x2x5x7x4xf32>, tensor<1x2x5x7x4xf32>)
      -> tensor<1x2x5x7x4xf32>
  return %o : tensor<1x2x5x7x4xf32>
}

// CHECK-LABEL: func @local_window_2d_bf16
// CHECK:       tessera.attn_local_window_2d
// CHECK-SAME:  tensor<{{[0-9x]+}}xbf16>
func.func @local_window_2d_bf16(
    %q: tensor<1x1x4x4x8xbf16>,
    %k: tensor<1x1x4x4x8xbf16>,
    %v: tensor<1x1x4x4x8xbf16>
) -> tensor<1x1x4x4x8xbf16> {
  %o = tessera.attn_local_window_2d %q, %k, %v {window = [1, 1]} :
      (tensor<1x1x4x4x8xbf16>, tensor<1x1x4x4x8xbf16>, tensor<1x1x4x4x8xbf16>)
      -> tensor<1x1x4x4x8xbf16>
  return %o : tensor<1x1x4x4x8xbf16>
}
