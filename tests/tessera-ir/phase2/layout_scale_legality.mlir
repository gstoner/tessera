// RUN: tessera-opt --tessera-layout-legality %s -split-input-file -verify-diagnostics | FileCheck %s

// LayoutLegalityPass scale-layout rule (2026-06 — DeepGEMM keystone).
// A low-precision scale *operand* on tessera.grouped_gemm / moe_swiglu_block is
// only legal when the op also declares a `scale_layout` attribute — an untyped
// scale tensor has no layout contract for the target-lowering layer to act on.
// (Repack/transpose insertion to a target's wanted layout is the Tile->Target
// lowering's job; this pass enforces the invariant.)

// ─────────────────────────────────────────────────────────────────────────
// POSITIVE: scales(...) + scale_layout declared. ✓
// ─────────────────────────────────────────────────────────────────────────
// CHECK-LABEL: func.func @gg_scaled_ok
// CHECK:       tessera.grouped_gemm
// CHECK-SAME:  scales(%arg3, %arg4)
func.func @gg_scaled_ok(%x: tensor<8x16xf32>, %w: tensor<2x16x32xf32>,
    %gs: tensor<2xi64>, %xs: tensor<8x1xf32>, %ws: tensor<2x1xf32>)
    -> tensor<8x32xf32> {
  %o = tessera.grouped_gemm %x, %w, %gs scales(%xs, %ws)
        {scale_layout = {granularity = "block", block = [1, 128], packing = "ue8m0"}}
       : (tensor<8x16xf32>, tensor<2x16x32xf32>, tensor<2xi64>,
          tensor<8x1xf32>, tensor<2x1xf32>) -> tensor<8x32xf32>
  return %o : tensor<8x32xf32>
}

// -----

// POSITIVE: bare (unscaled) form needs no scale_layout. ✓
// CHECK-LABEL: func.func @gg_bare_ok
// CHECK:       tessera.grouped_gemm
func.func @gg_bare_ok(%x: tensor<12x8xbf16>, %w: tensor<3x8x6xbf16>,
                      %gs: tensor<3xi64>) -> tensor<12x6xf32> {
  %o = tessera.grouped_gemm %x, %w, %gs
       : (tensor<12x8xbf16>, tensor<3x8x6xbf16>, tensor<3xi64>) -> tensor<12x6xf32>
  return %o : tensor<12x6xf32>
}

// -----

// NEGATIVE: grouped_gemm scale operands with no scale_layout. ✗
func.func @gg_scale_no_layout(%x: tensor<8x16xf32>, %w: tensor<2x16x32xf32>,
    %gs: tensor<2xi64>, %xs: tensor<8x1xf32>, %ws: tensor<2x1xf32>)
    -> tensor<8x32xf32> {
  // expected-error @+1 {{LAYOUT_LEGALITY_SCALE_WITHOUT_LAYOUT}}
  %o = tessera.grouped_gemm %x, %w, %gs scales(%xs, %ws)
       : (tensor<8x16xf32>, tensor<2x16x32xf32>, tensor<2xi64>,
          tensor<8x1xf32>, tensor<2x1xf32>) -> tensor<8x32xf32>
  return %o : tensor<8x32xf32>
}

// -----

// NEGATIVE: moe_swiglu_block scale operands with no scale_layout. ✗
func.func @moe_scale_no_layout(%x: tensor<8x16xf32>, %wg: tensor<2x16x32xf32>,
    %wu: tensor<2x16x32xf32>, %wd: tensor<2x32x16xf32>, %gs: tensor<2xi64>,
    %xs: tensor<8x1xf32>, %wgs: tensor<2x1xf32>, %wus: tensor<2x1xf32>,
    %wds: tensor<2x1xf32>) -> tensor<8x16xf32> {
  // expected-error @+1 {{LAYOUT_LEGALITY_SCALE_WITHOUT_LAYOUT}}
  %o = tessera.moe_swiglu_block %x, %wg, %wu, %wd, %gs
         scales(%xs, %wgs, %wus, %wds)
       : (tensor<8x16xf32>, tensor<2x16x32xf32>, tensor<2x16x32xf32>,
          tensor<2x32x16xf32>, tensor<2xi64>, tensor<8x1xf32>, tensor<2x1xf32>,
          tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<8x16xf32>
  return %o : tensor<8x16xf32>
}
