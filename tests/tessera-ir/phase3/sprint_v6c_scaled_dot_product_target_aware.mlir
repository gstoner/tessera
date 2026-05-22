// RUN: tessera-opt %s -split-input-file -verify-diagnostics | FileCheck %s
//
// Sprint V6c (2026-05-22) — target-aware tile size verifier on
// FA-4 Tile IR `tessera.attn.scaled_dot_product`.  Generalizes the
// Sprint V3 FlashAttnOp head_dim pattern to the canonical attention
// kernel of the FA-4 Tile IR layer.
//
// Sprint V7b (2026-05-22) unblocked this fixture by registering an
// eager-load extension on the `tessera.attn` dialect: when the
// `tessera` Graph IR dialect loads, `tessera.attn` is automatically
// `getOrLoadDialect`'d so the parser finds the op without needing
// a pass to trigger the lookup.
//
// Per-SM tile_q × tile_kv ceilings (Sprint V6c FA-4 tile size table):
//   sm_70 / sm_75 / sm_80 / sm_86 / sm_89  → 64 × 128 (FA-2 baseline)
//   sm_90 / sm_90a / sm_100 / sm_120       → 128 × 256 (FA-3/FA-4)
//   no SM tag                              → no limit (CPU reference)
//
// Three cases: 1 positive + 2 negative + 1 no-target-sm positive.

// ─────────────────────────────────────────────────────────────────────────
// POSITIVE: SM_90 with tile_q=64, tile_kv=128 (well within 128×256).
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @sdp_sm90_within_limits
// CHECK:       tessera.attn.scaled_dot_product
func.func @sdp_sm90_within_limits(
    %q: tensor<64x64xf16>, %k: tensor<128x64xf16>
) -> tensor<64x128xf32>
    attributes { tessera.target_sm = "sm_90" } {
  %s = "tessera.attn.scaled_dot_product"(%q, %k) {scale = 0.125 : f32}
       : (tensor<64x64xf16>, tensor<128x64xf16>) -> tensor<64x128xf32>
  return %s : tensor<64x128xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// NEGATIVE 1: SM_80 with tile_q=128 (limit is 64) — tile_q exceeded.
// ─────────────────────────────────────────────────────────────────────────

func.func @sdp_sm80_tile_q_overflow(
    %q: tensor<128x64xf16>, %k: tensor<128x64xf16>
) -> tensor<128x128xf32>
    attributes { tessera.target_sm = "sm_80" } {
  // expected-error @+1 {{tile_q=128 exceeds the SM sm_80 ScaledDotProduct kernel limit of 64}}
  %s = "tessera.attn.scaled_dot_product"(%q, %k) {scale = 0.125 : f32}
       : (tensor<128x64xf16>, tensor<128x64xf16>) -> tensor<128x128xf32>
  return %s : tensor<128x128xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// NEGATIVE 2: SM_90 with tile_kv=512 (limit is 256) — tile_kv exceeded.
// ─────────────────────────────────────────────────────────────────────────

func.func @sdp_sm90_tile_kv_overflow(
    %q: tensor<64x64xf16>, %k: tensor<512x64xf16>
) -> tensor<64x512xf32>
    attributes { tessera.target_sm = "sm_90" } {
  // expected-error @+1 {{tile_kv=512 exceeds the SM sm_90 ScaledDotProduct kernel limit of 256}}
  %s = "tessera.attn.scaled_dot_product"(%q, %k) {scale = 0.125 : f32}
       : (tensor<64x64xf16>, tensor<512x64xf16>) -> tensor<64x512xf32>
  return %s : tensor<64x512xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// POSITIVE: no tessera.target_sm — CPU reference path, no limit applied.
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @sdp_no_target_sm
// CHECK:       tessera.attn.scaled_dot_product
func.func @sdp_no_target_sm(
    %q: tensor<1024x128xf32>, %k: tensor<2048x128xf32>
) -> tensor<1024x2048xf32> {
  %s = "tessera.attn.scaled_dot_product"(%q, %k) {scale = 0.0883 : f32}
       : (tensor<1024x128xf32>, tensor<2048x128xf32>) -> tensor<1024x2048xf32>
  return %s : tensor<1024x2048xf32>
}
