// RUN: tessera-opt %s --split-input-file --verify-diagnostics
//
// Negative coverage for `SelectiveSsmOp::verify` (Track-R Phase 0b). The
// previous verifier was too loose (rank-2 `a`'s trailing dim was never checked
// against the state dim N; `state` was only rank-checked, not shape-checked)
// and too strict (direct ranked-shape compares rejected dynamic-compatible
// shapes). These cases lock the tightened checks; `selective_ssm.mlir` covers
// the positive paths, including dynamic dims.

// rank-2 `a` trailing dim must equal the state dim N (here N=4, a is (8, 5)).
func.func @a_rank2_bad_n(%x: tensor<2x16x8xf32>, %a: tensor<8x5xf32>,
                         %b: tensor<2x16x4xf32>, %c: tensor<2x16x4xf32>,
                         %delta: tensor<2x16x8xf32>) -> tensor<2x16x8xf32> {
  // expected-error @+1 {{a trailing dim must equal state dim N}}
  %y = tessera.selective_ssm %x, %a, %b, %c, %delta {chunk_size = 64 : i64}
      : (tensor<2x16x8xf32>, tensor<8x5xf32>, tensor<2x16x4xf32>,
         tensor<2x16x4xf32>, tensor<2x16x8xf32>) -> tensor<2x16x8xf32>
  return %y : tensor<2x16x8xf32>
}

// -----

// `state` must be (B, D, N) — here D=16, N=8 expected but state is (1, 16, 4).
func.func @bad_state_shape(%x: tensor<1x32x16xf32>, %a: tensor<16x8xf32>,
                           %b: tensor<1x32x8xf32>, %c: tensor<1x32x8xf32>,
                           %delta: tensor<1x32x16xf32>,
                           %s: tensor<1x16x4xf32>) -> tensor<1x32x16xf32> {
  // expected-error @+1 {{state must have shape (B, D, N) compatible with x and b}}
  %y = tessera.selective_ssm %x, %a, %b, %c, %delta init(%s)
      : (tensor<1x32x16xf32>, tensor<16x8xf32>, tensor<1x32x8xf32>,
         tensor<1x32x8xf32>, tensor<1x32x16xf32>,
         tensor<1x16x4xf32>) -> tensor<1x32x16xf32>
  return %y : tensor<1x32x16xf32>
}

// -----

// `a`'s leading dim is the channel dim D (here D=8, a is (7,)).
func.func @a_bad_leading(%x: tensor<2x16x8xf32>, %a: tensor<7xf32>,
                         %b: tensor<2x16x4xf32>, %c: tensor<2x16x4xf32>,
                         %delta: tensor<2x16x8xf32>) -> tensor<2x16x8xf32> {
  // expected-error @+1 {{a leading dim must equal x channel dim D}}
  %y = tessera.selective_ssm %x, %a, %b, %c, %delta {chunk_size = 64 : i64}
      : (tensor<2x16x8xf32>, tensor<7xf32>, tensor<2x16x4xf32>,
         tensor<2x16x4xf32>, tensor<2x16x8xf32>) -> tensor<2x16x8xf32>
  return %y : tensor<2x16x8xf32>
}
