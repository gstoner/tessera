// RUN: tessera-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// Sprint V3 (2026-05-22) — target-aware head_dim ceiling on FlashAttnOp.
//
// When the parent function carries ``tessera.target_sm = "sm_XX"``,
// the FlashAttnOp verifier enforces the per-SM head_dim limit:
//   sm_70 / sm_75 / sm_80 / sm_86 / sm_89 : ≤ 128
//   sm_90 / sm_90a / sm_100 / sm_100a / sm_120 / sm_120a : ≤ 256
// Functions without the attribute (CPU path) skip this check.

// ─────────────────────────────────────────────────────────────────────────
// Positive: SM_90 with head_dim = 256 (at limit) — OK
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @flash_attn_sm90_at_limit
// CHECK:       tessera.flash_attn
func.func @flash_attn_sm90_at_limit(
    %q: tensor<2x4x16x256xf32>,
    %k: tensor<2x4x16x256xf32>,
    %v: tensor<2x4x16x256xf32>
) -> tensor<2x4x16x256xf32>
    attributes { tessera.target_sm = "sm_90" } {
  %o = "tessera.flash_attn"(%q, %k, %v) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {
    head_dim = 256 : i64,
    causal = false
  } : (tensor<2x4x16x256xf32>, tensor<2x4x16x256xf32>, tensor<2x4x16x256xf32>)
    -> tensor<2x4x16x256xf32>
  return %o : tensor<2x4x16x256xf32>
}

// -----

// Negative: SM_90 with head_dim = 257 — exceeds the 256 ceiling.
func.func @flash_attn_sm90_overflow(
    %q: tensor<2x4x16x257xf32>,
    %k: tensor<2x4x16x257xf32>,
    %v: tensor<2x4x16x257xf32>
) -> tensor<2x4x16x257xf32>
    attributes { tessera.target_sm = "sm_90" } {
  // expected-error @+1 {{head_dim=257 exceeds the SM sm_90 flash-attention kernel limit of 256}}
  %o = "tessera.flash_attn"(%q, %k, %v) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {
    head_dim = 257 : i64,
    causal = false
  } : (tensor<2x4x16x257xf32>, tensor<2x4x16x257xf32>, tensor<2x4x16x257xf32>)
    -> tensor<2x4x16x257xf32>
  return %o : tensor<2x4x16x257xf32>
}

// -----

// Negative: SM_80 with head_dim = 256 — SM_80 ceiling is 128.
func.func @flash_attn_sm80_overflow(
    %q: tensor<2x4x16x256xf32>,
    %k: tensor<2x4x16x256xf32>,
    %v: tensor<2x4x16x256xf32>
) -> tensor<2x4x16x256xf32>
    attributes { tessera.target_sm = "sm_80" } {
  // expected-error @+1 {{head_dim=256 exceeds the SM sm_80 flash-attention kernel limit of 128}}
  %o = "tessera.flash_attn"(%q, %k, %v) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {
    head_dim = 256 : i64,
    causal = false
  } : (tensor<2x4x16x256xf32>, tensor<2x4x16x256xf32>, tensor<2x4x16x256xf32>)
    -> tensor<2x4x16x256xf32>
  return %o : tensor<2x4x16x256xf32>
}

// -----

// Positive: no tessera.target_sm attribute — verifier doesn't apply
// the limit (CPU reference path).
//
// CHECK-LABEL: func.func @flash_attn_no_target_sm
// CHECK:       tessera.flash_attn
func.func @flash_attn_no_target_sm(
    %q: tensor<2x4x16x512xf32>,
    %k: tensor<2x4x16x512xf32>,
    %v: tensor<2x4x16x512xf32>
) -> tensor<2x4x16x512xf32> {
  %o = "tessera.flash_attn"(%q, %k, %v) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {
    head_dim = 512 : i64,
    causal = false
  } : (tensor<2x4x16x512xf32>, tensor<2x4x16x512xf32>, tensor<2x4x16x512xf32>)
    -> tensor<2x4x16x512xf32>
  return %o : tensor<2x4x16x512xf32>
}
