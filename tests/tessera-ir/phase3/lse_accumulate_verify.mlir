// RUN: tessera-opt %s -split-input-file -verify-diagnostics

// LseAccumulateOp accepts scalar-f32 OR rank-1 [tile_q] per-row statistics
// (running_m, running_l, lse). The cross-shape checks must validate EVERY
// rank-1 statistic against the acc tile_q and against the other rank-1
// statistics — independent of which (if any) of the three are scalar. A scalar
// running_m must not mask a mismatched rank-1 running_l / lse.

// A scalar running_m must not hide a mismatched rank-1 running_l: acc tile_q is
// 64 but running_l is a per-row [32] vector.
func.func @scalar_m_masks_bad_l(%acc: tensor<64x64xf32>, %m: f32, %l: tensor<32xf32>)
    -> (tensor<64x64xf32>, f32) {
  // expected-error @+1 {{running_l length must match acc tile_q}}
  %out, %lse = "tessera_attn.lse_accumulate"(%acc, %m, %l)
      : (tensor<64x64xf32>, f32, tensor<32xf32>) -> (tensor<64x64xf32>, f32)
  return %out, %lse : tensor<64x64xf32>, f32
}

// -----

// Likewise a rank-1 lse result [32] that disagrees with acc tile_q [64] must be
// rejected even when running_m is scalar and running_l already matches.
func.func @scalar_m_masks_bad_lse(%acc: tensor<64x64xf32>, %m: f32, %l: tensor<64xf32>)
    -> (tensor<64x64xf32>, tensor<32xf32>) {
  // expected-error @+1 {{lse length must match acc tile_q}}
  %out, %lse = "tessera_attn.lse_accumulate"(%acc, %m, %l)
      : (tensor<64x64xf32>, f32, tensor<64xf32>) -> (tensor<64x64xf32>, tensor<32xf32>)
  return %out, %lse : tensor<64x64xf32>, tensor<32xf32>
}

// -----

// Two rank-1 statistics that match acc tile_q but disagree with each other in
// element type are rejected via the pairwise same-tensor check.
func.func @rank1_stats_element_type_mismatch(%acc: tensor<64x64xf32>, %m: tensor<64xf32>, %l: tensor<64xf16>)
    -> (tensor<64x64xf32>, f32) {
  // expected-error @+1 {{running_m/running_l element types must match}}
  %out, %lse = "tessera_attn.lse_accumulate"(%acc, %m, %l)
      : (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf16>) -> (tensor<64x64xf32>, f32)
  return %out, %lse : tensor<64x64xf32>, f32
}

// -----

// Valid: all statistics are scalar f32 — this is exactly what TileIRLoweringPass
// emits in the reduced-loop form, and must verify clean.
func.func @all_scalar_stats_ok(%acc: tensor<64x64xf32>, %m: f32, %l: f32)
    -> (tensor<64x64xf32>, f32) {
  %out, %lse = "tessera_attn.lse_accumulate"(%acc, %m, %l)
      : (tensor<64x64xf32>, f32, f32) -> (tensor<64x64xf32>, f32)
  return %out, %lse : tensor<64x64xf32>, f32
}

// -----

// Valid: a scalar running_m mixed with rank-1 running_l / lse that both match
// the acc tile_q [64] verifies clean.
func.func @mixed_scalar_and_rank1_ok(%acc: tensor<64x64xf32>, %m: f32, %l: tensor<64xf32>)
    -> (tensor<64x64xf32>, tensor<64xf32>) {
  %out, %lse = "tessera_attn.lse_accumulate"(%acc, %m, %l)
      : (tensor<64x64xf32>, f32, tensor<64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>)
  return %out, %lse : tensor<64x64xf32>, tensor<64xf32>
}
