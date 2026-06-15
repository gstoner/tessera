// RUN: tessera-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// VarlenSdpaOp (2026-06-15) — packed-sequence SDPA with cu_seqlens as
// first-class operands (the Cosmos-3 "two-way flat attention" IR contract).
// Verifier pins: positive head_dim; rank-3 packed q/k/v sharing head axis +
// head_dim; k/v sharing total_k; rank-1 integer cu_seqlens; head_dim attr ==
// q innermost dim; target-aware per-SM head_dim ceiling.

// ─────────────────────────────────────────────────────────────────────────
// Positive: square blocks (Reasoner pathway). H=2, total=9 (blocks 4,5),
// head_dim=8, causal. cu_seqlens_q == cu_seqlens_k = [0,4,9].
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @varlen_square_causal
// CHECK:       tessera.varlen_sdpa
func.func @varlen_square_causal(
    %q: tensor<2x9x8xf32>, %k: tensor<2x9x8xf32>, %v: tensor<2x9x8xf32>,
    %cuq: tensor<3xi32>, %cuk: tensor<3xi32>
) -> tensor<2x9x8xf32> {
  %o = tessera.varlen_sdpa %q, %k, %v, %cuq, %cuk {
    head_dim = 8 : i64, causal = true
  } : (tensor<2x9x8xf32>, tensor<2x9x8xf32>, tensor<2x9x8xf32>,
       tensor<3xi32>, tensor<3xi32>) -> tensor<2x9x8xf32>
  return %o : tensor<2x9x8xf32>
}

// -----

// Positive: rectangular blocks (Generator pathway, Cosmos Fig. 14b).
// total_q=6, cu_q=[0,2,5,6]; total_k=15, cu_k=[0,5,10,15]; head_dim=16.
// cu_seqlens_q != cu_seqlens_k — the first-class rectangular case.

// CHECK-LABEL: func.func @varlen_rectangular_bidirectional
// CHECK:       tessera.varlen_sdpa
func.func @varlen_rectangular_bidirectional(
    %q: tensor<2x6x16xf32>, %k: tensor<2x15x16xf32>, %v: tensor<2x15x16xf32>,
    %cuq: tensor<4xi32>, %cuk: tensor<4xi32>
) -> tensor<2x6x16xf32> {
  %o = tessera.varlen_sdpa %q, %k, %v, %cuq, %cuk {
    head_dim = 16 : i64, causal = false
  } : (tensor<2x6x16xf32>, tensor<2x15x16xf32>, tensor<2x15x16xf32>,
       tensor<4xi32>, tensor<4xi32>) -> tensor<2x6x16xf32>
  return %o : tensor<2x6x16xf32>
}

// -----

// Negative: head_dim attribute disagrees with q innermost dim (8 vs 16).
func.func @varlen_head_dim_mismatch(
    %q: tensor<2x9x16xf32>, %k: tensor<2x9x16xf32>, %v: tensor<2x9x16xf32>,
    %cuq: tensor<3xi32>, %cuk: tensor<3xi32>
) -> tensor<2x9x16xf32> {
  // expected-error @+1 {{head_dim attribute (8) must match q innermost dim (16)}}
  %o = tessera.varlen_sdpa %q, %k, %v, %cuq, %cuk {
    head_dim = 8 : i64, causal = false
  } : (tensor<2x9x16xf32>, tensor<2x9x16xf32>, tensor<2x9x16xf32>,
       tensor<3xi32>, tensor<3xi32>) -> tensor<2x9x16xf32>
  return %o : tensor<2x9x16xf32>
}

// -----

// Negative: q and k disagree on the head axis (2 vs 3).
func.func @varlen_head_axis_mismatch(
    %q: tensor<2x9x8xf32>, %k: tensor<3x9x8xf32>, %v: tensor<3x9x8xf32>,
    %cuq: tensor<3xi32>, %cuk: tensor<3xi32>
) -> tensor<2x9x8xf32> {
  // expected-error @+1 {{q and k must share the head axis; got 2 vs 3}}
  %o = tessera.varlen_sdpa %q, %k, %v, %cuq, %cuk {
    head_dim = 8 : i64, causal = false
  } : (tensor<2x9x8xf32>, tensor<3x9x8xf32>, tensor<3x9x8xf32>,
       tensor<3xi32>, tensor<3xi32>) -> tensor<2x9x8xf32>
  return %o : tensor<2x9x8xf32>
}

// -----

// Negative: k and v disagree on total_k (9 vs 10).
func.func @varlen_kv_length_mismatch(
    %q: tensor<2x9x8xf32>, %k: tensor<2x9x8xf32>, %v: tensor<2x10x8xf32>,
    %cuq: tensor<3xi32>, %cuk: tensor<3xi32>
) -> tensor<2x9x8xf32> {
  // expected-error @+1 {{k and v must share the packed key length (total_k); got 9 vs 10}}
  %o = tessera.varlen_sdpa %q, %k, %v, %cuq, %cuk {
    head_dim = 8 : i64, causal = false
  } : (tensor<2x9x8xf32>, tensor<2x9x8xf32>, tensor<2x10x8xf32>,
       tensor<3xi32>, tensor<3xi32>) -> tensor<2x9x8xf32>
  return %o : tensor<2x9x8xf32>
}

// -----

// Negative: cu_seqlens_q is rank-2, not a rank-1 offset vector.
func.func @varlen_cu_seqlens_rank(
    %q: tensor<2x9x8xf32>, %k: tensor<2x9x8xf32>, %v: tensor<2x9x8xf32>,
    %cuq: tensor<3x1xi32>, %cuk: tensor<3xi32>
) -> tensor<2x9x8xf32> {
  // expected-error @+1 {{cu_seqlens_q must be a rank-1 cu_seqlens vector}}
  %o = tessera.varlen_sdpa %q, %k, %v, %cuq, %cuk {
    head_dim = 8 : i64, causal = false
  } : (tensor<2x9x8xf32>, tensor<2x9x8xf32>, tensor<2x9x8xf32>,
       tensor<3x1xi32>, tensor<3xi32>) -> tensor<2x9x8xf32>
  return %o : tensor<2x9x8xf32>
}

// -----

// Negative: cu_seqlens_k has a float element type.
func.func @varlen_cu_seqlens_dtype(
    %q: tensor<2x9x8xf32>, %k: tensor<2x9x8xf32>, %v: tensor<2x9x8xf32>,
    %cuq: tensor<3xi32>, %cuk: tensor<3xf32>
) -> tensor<2x9x8xf32> {
  // expected-error @+1 {{cu_seqlens_k must have an integer element type}}
  %o = tessera.varlen_sdpa %q, %k, %v, %cuq, %cuk {
    head_dim = 8 : i64, causal = false
  } : (tensor<2x9x8xf32>, tensor<2x9x8xf32>, tensor<2x9x8xf32>,
       tensor<3xi32>, tensor<3xf32>) -> tensor<2x9x8xf32>
  return %o : tensor<2x9x8xf32>
}

// -----

// Negative: SM_80 with head_dim = 256 exceeds the 128 ceiling.
func.func @varlen_sm80_overflow(
    %q: tensor<2x9x256xf32>, %k: tensor<2x9x256xf32>, %v: tensor<2x9x256xf32>,
    %cuq: tensor<3xi32>, %cuk: tensor<3xi32>
) -> tensor<2x9x256xf32>
    attributes { tessera.target_sm = "sm_80" } {
  // expected-error @+1 {{head_dim=256 exceeds the SM sm_80 flash-attention kernel limit of 128}}
  %o = tessera.varlen_sdpa %q, %k, %v, %cuq, %cuk {
    head_dim = 256 : i64, causal = false
  } : (tensor<2x9x256xf32>, tensor<2x9x256xf32>, tensor<2x9x256xf32>,
       tensor<3xi32>, tensor<3xi32>) -> tensor<2x9x256xf32>
  return %o : tensor<2x9x256xf32>
}
