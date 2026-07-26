// RUN: tessera-opt --allow-unregistered-dialect \
// RUN:   --tessera-tile-ir-lowering='tile-q=64 tile-kv=64 sm=120' \
// RUN:   -split-input-file -verify-diagnostics %s

// The canonical streaming path requires an explicit seed when dropout is
// active; replaying an implicit block-local seed would repeat the first mask.
func.func @dropout_without_seed(
    %q: tensor<8x16xf16>, %k: tensor<17x16xf16>,
    %v: tensor<17x16xf16>) -> tensor<8x16xf32> {
  // expected-error @+1 {{TILE_IR_LOWERING}}
  %out = "tessera.flash_attn"(%q, %k, %v)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {
        causal = true, dropout_p = 2.5e-1 : f64, head_dim = 16 : i64
      } : (tensor<8x16xf16>, tensor<17x16xf16>, tensor<17x16xf16>)
          -> tensor<8x16xf32>
  return %out : tensor<8x16xf32>
}

// -----

// Batch/head distribution remains a separate tiling step. The rank-4 value
// must not fall back to the former whole-tensor straight-line lowering.
func.func @rank4_requires_distribution(
    %q: tensor<1x8x2x16xf16>, %k: tensor<1x17x2x16xf16>,
    %v: tensor<1x17x2x16xf16>) -> tensor<1x8x2x16xf32> {
  // expected-error @+1 {{TILE_IR_LOWERING}}
  %out = "tessera.flash_attn"(%q, %k, %v)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {
        causal = true, head_dim = 16 : i64
      } : (tensor<1x8x2x16xf16>, tensor<1x17x2x16xf16>,
           tensor<1x17x2x16xf16>) -> tensor<1x8x2x16xf32>
  return %out : tensor<1x8x2x16xf32>
}

// -----

// One max/sum pair is owned by every query row. Scalar state would couple
// independent rows and is not a valid canonical streaming recurrence.
func.func @streaming_update_rejects_scalar_state(
    %scores: tensor<8x16xf32>, %value: tensor<16x32xf16>,
    %m: f32, %l: f32, %acc: tensor<8x32xf32>)
    -> (tensor<8x32xf32>, f32, f32) {
  // expected-error @+2 {{running and updated max/sum must be rank-1 per-query tensors}}
  %new_acc, %new_m, %new_l =
      "tessera_attn.streaming_update"(%scores, %value, %m, %l, %acc)
      : (tensor<8x16xf32>, tensor<16x32xf16>, f32, f32,
         tensor<8x32xf32>) -> (tensor<8x32xf32>, f32, f32)
  return %new_acc, %new_m, %new_l : tensor<8x32xf32>, f32, f32
}

// -----

// The logical source length is part of the ragged-tail correctness contract.
func.func @boundary_rejects_empty_logical_source(
    %scores: tensor<8x16xf32>, %q: index, %kv: index)
    -> tensor<8x16xf32> {
  // expected-error @+1 {{logical_sk must be positive}}
  %masked = "tessera_attn.boundary_mask"(%scores, %q, %kv) {
      causal = true, window_left = -1 : i64, window_right = -1 : i64,
      logical_sk = 0 : i64
    } : (tensor<8x16xf32>, index, index) -> tensor<8x16xf32>
  return %masked : tensor<8x16xf32>
}
