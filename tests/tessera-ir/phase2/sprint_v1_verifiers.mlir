// RUN: tessera-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// Sprint V1 (2026-05-22) — hasVerifier = 1 + verify() for TransposeOp,
// LayerNormOp, MoeDispatchOp.  Each section tests one positive case
// (verifier passes; FileCheck the op is preserved) and one negative
// case (verifier rejects; expected-error matches the diagnostic).

// ─────────────────────────────────────────────────────────────────────────
// TransposeOp — positive: rank + permutation + element type match
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @transpose_ok
// CHECK:       tessera.transpose
func.func @transpose_ok(%x: tensor<4x8xf32>) -> tensor<8x4xf32> {
  %y = "tessera.transpose"(%x) : (tensor<4x8xf32>) -> tensor<8x4xf32>
  return %y : tensor<8x4xf32>
}

// -----

// TransposeOp — negative: rank mismatch
func.func @transpose_rank_mismatch(%x: tensor<4x8xf32>) -> tensor<4x8x1xf32> {
  // expected-error @+1 {{transpose must preserve rank: 2 -> 3}}
  %y = "tessera.transpose"(%x) : (tensor<4x8xf32>) -> tensor<4x8x1xf32>
  return %y : tensor<4x8x1xf32>
}

// -----

// TransposeOp — negative: element type mismatch
func.func @transpose_elem_mismatch(%x: tensor<4x8xf32>) -> tensor<8x4xf16> {
  // expected-error @+1 {{transpose must preserve element type}}
  %y = "tessera.transpose"(%x) : (tensor<4x8xf32>) -> tensor<8x4xf16>
  return %y : tensor<8x4xf16>
}

// -----

// TransposeOp — negative: static dims aren't a permutation
func.func @transpose_dims_not_perm(%x: tensor<4x8xf32>) -> tensor<7x5xf32> {
  // expected-error @+1 {{output static dims must be a permutation of input static dims}}
  %y = "tessera.transpose"(%x) : (tensor<4x8xf32>) -> tensor<7x5xf32>
  return %y : tensor<7x5xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// LayerNormOp — positive: shape-preserving, eps positive
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @layer_norm_ok
// CHECK:       tessera.layer_norm
func.func @layer_norm_ok(%x: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %y = "tessera.layer_norm"(%x) {eps = 1.0e-5 : f64}
       : (tensor<4x16xf32>) -> tensor<4x16xf32>
  return %y : tensor<4x16xf32>
}

// -----

// LayerNormOp — negative: rank mismatch
func.func @layer_norm_rank_mismatch(%x: tensor<4x16xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{layer_norm must preserve rank}}
  %y = "tessera.layer_norm"(%x) : (tensor<4x16xf32>) -> tensor<4xf32>
  return %y : tensor<4xf32>
}

// -----

// LayerNormOp — negative: non-positive eps
func.func @layer_norm_zero_eps(%x: tensor<4x16xf32>) -> tensor<4x16xf32> {
  // expected-error @+1 {{eps must be positive for stable rsqrt}}
  %y = "tessera.layer_norm"(%x) {eps = 0.0 : f64}
       : (tensor<4x16xf32>) -> tensor<4x16xf32>
  return %y : tensor<4x16xf32>
}

// -----

// LayerNormOp — negative: dim size mismatch
func.func @layer_norm_dim_mismatch(%x: tensor<4x16xf32>) -> tensor<4x8xf32> {
  // expected-error @+1 {{layer_norm must preserve dim 1}}
  %y = "tessera.layer_norm"(%x) : (tensor<4x16xf32>) -> tensor<4x8xf32>
  return %y : tensor<4x8xf32>
}

// -----

// ─────────────────────────────────────────────────────────────────────────
// MoeDispatchOp — positive: token count match on dim 0
// ─────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @moe_dispatch_ok
// CHECK:       tessera.moe_dispatch
func.func @moe_dispatch_ok(
    %x: tensor<128x256xf32>, %route: tensor<128x4xi32>
) -> tensor<128x256xf32> {
  %p = "tessera.moe_dispatch"(%x, %route)
       : (tensor<128x256xf32>, tensor<128x4xi32>) -> tensor<128x256xf32>
  return %p : tensor<128x256xf32>
}

// -----

// MoeDispatchOp — negative: token count mismatch
func.func @moe_dispatch_token_mismatch(
    %x: tensor<128x256xf32>, %route: tensor<127x4xi32>
) -> tensor<128x256xf32> {
  // expected-error @+1 {{token count mismatch: x[0]=128 route[0]=127}}
  %p = "tessera.moe_dispatch"(%x, %route)
       : (tensor<128x256xf32>, tensor<127x4xi32>) -> tensor<128x256xf32>
  return %p : tensor<128x256xf32>
}
