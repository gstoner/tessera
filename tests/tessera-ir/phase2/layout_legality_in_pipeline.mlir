// RUN: tessera-opt --pass-pipeline='builtin.module(tessera-lower-to-x86)' %s -split-input-file -verify-diagnostics | FileCheck %s
//
// 2026-06-17: LayoutLegalityPass is now wired into the named lowering pipelines
// (was standalone --tessera-layout-legality). This fixture proves it actually
// FIRES inside tessera-lower-to-x86: an unknown layout on a tessera.cast is
// caught early, before tiling/codegen, with the stable diagnostic. The gpu /
// cuda13 pipelines wire the same pass at the same point (asserted by
// tests/unit/test_layout_legality_pipeline_wiring.py).

// POSITIVE: a clean row_major cast + matmul lowers through the pipeline. ✓
// CHECK-LABEL: func.func @clean_layout
func.func @clean_layout(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %ac = "tessera.cast"(%a) {tessera.layout = "row_major"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %0 = "tessera.matmul"(%ac, %b) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// -----

// NEGATIVE: an unknown layout name is rejected by the wired legality pass.
// NB: a dtype-changing cast (f32 -> f16) so the earlier tessera-canonicalize
// EraseIdentityCast pattern doesn't erase it before layout-legality runs — the
// cast must survive to carry the (bogus) layout to the check.
func.func @unknown_layout_rejected(%a: tensor<4x8xf32>) -> tensor<4x8xf16> {
  // expected-error @+1 {{LAYOUT_LEGALITY_UNKNOWN_LAYOUT}}
  %ac = "tessera.cast"(%a) {tessera.layout = "bogus_layout"} : (tensor<4x8xf32>) -> tensor<4x8xf16>
  return %ac : tensor<4x8xf16>
}
