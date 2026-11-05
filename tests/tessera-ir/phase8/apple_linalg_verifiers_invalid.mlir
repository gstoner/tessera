// L-series linalg verifier negatives (review R4 + RV-P2a, 2026-06-03).
//
// RUN: tessera-opt %s --split-input-file --verify-diagnostics -o /dev/null

func.func @svd_full_bad_s(%a: tensor<6x4xf32>)
    -> (tensor<6x6xf32>, tensor<9xf32>, tensor<4x4xf32>) {
  // |S| must equal min(M, N) = 4, not 9 (full_matrices).
  // expected-error @+1 {{number of singular values S must equal min(M, N)}}
  %u, %s, %v = tessera.svd %a {full_matrices = true}
      : (tensor<6x4xf32>) -> (tensor<6x6xf32>, tensor<9xf32>, tensor<4x4xf32>)
  return %u, %s, %v : tensor<6x6xf32>, tensor<9xf32>, tensor<4x4xf32>
}

// -----

func.func @svd_reduced_bad_s(%a: tensor<6x4xf32>)
    -> (tensor<6x4xf32>, tensor<9xf32>, tensor<4x4xf32>) {
  // |S| must equal min(M, N) = 4, not 9 (reduced).
  // expected-error @+1 {{number of singular values S must equal min(M, N)}}
  %u, %s, %v = tessera.svd %a
      : (tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<9xf32>, tensor<4x4xf32>)
  return %u, %s, %v : tensor<6x4xf32>, tensor<9xf32>, tensor<4x4xf32>
}

// -----

func.func @svd_full_bad_u(%a: tensor<6x4xf32>)
    -> (tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>) {
  // full_matrices ⇒ U must be square M×M (6×6), not 6×4.
  // expected-error @+1 {{U must be square (M×M) for full_matrices SVD}}
  %u, %s, %v = tessera.svd %a {full_matrices = true}
      : (tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>)
  return %u, %s, %v : tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>
}

// -----

func.func @qr_nonsquare_r(%a: tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4x3xf32>) {
  // expected-error @+1 {{R must be square}}
  %q, %r = tessera.qr %a : (tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4x3xf32>)
  return %q, %r : tensor<6x4xf32>, tensor<4x3xf32>
}

// -----

func.func @qr_bad_r_order(%a: tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<5x5xf32>) {
  // R order must equal A columns (4), not 5.
  // expected-error @+1 {{R order must equal the number of columns of A}}
  %q, %r = tessera.qr %a : (tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<5x5xf32>)
  return %q, %r : tensor<6x4xf32>, tensor<5x5xf32>
}

// -----

// E2E-REAL-6 (2026-11-04): svd and cholesky_solve admit a rank-3 batch, so the
// verifier checks the batch dimension and the trailing matrix relations.
func.func @svd_batch_mismatch(%a: tensor<2x6x4xf32>)
    -> (tensor<3x6x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32>) {
  // expected-error @+1 {{A, U, S and V batch dimensions must match}}
  %u, %s, %v = tessera.svd %a
      : (tensor<2x6x4xf32>) -> (tensor<3x6x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32>)
  return %u, %s, %v : tensor<3x6x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32>
}

// -----

func.func @svd_batched_bad_s(%a: tensor<2x6x4xf32>)
    -> (tensor<2x6x4xf32>, tensor<2x5xf32>, tensor<2x4x4xf32>) {
  // expected-error @+1 {{number of singular values S must equal min(M, N)}}
  %u, %s, %v = tessera.svd %a
      : (tensor<2x6x4xf32>) -> (tensor<2x6x4xf32>, tensor<2x5xf32>, tensor<2x4x4xf32>)
  return %u, %s, %v : tensor<2x6x4xf32>, tensor<2x5xf32>, tensor<2x4x4xf32>
}

// -----

func.func @svd_rank_mismatch(%a: tensor<2x6x4xf32>)
    -> (tensor<2x6x4xf32>, tensor<4xf32>, tensor<2x4x4xf32>) {
  // expected-error @+1 {{singular values S must have rank one less than A}}
  %u, %s, %v = tessera.svd %a
      : (tensor<2x6x4xf32>) -> (tensor<2x6x4xf32>, tensor<4xf32>, tensor<2x4x4xf32>)
  return %u, %s, %v : tensor<2x6x4xf32>, tensor<4xf32>, tensor<2x4x4xf32>
}

// -----

func.func @cholesky_solve_batch_mismatch(%a: tensor<2x3x3xf32>, %b: tensor<3x3x2xf32>)
    -> tensor<3x3x2xf32> {
  // expected-error @+1 {{A and B batch dimensions must match}}
  %x = tessera.cholesky_solve %a, %b : (tensor<2x3x3xf32>, tensor<3x3x2xf32>) -> tensor<3x3x2xf32>
  return %x : tensor<3x3x2xf32>
}

// -----

func.func @cholesky_solve_rank4(%a: tensor<1x2x3x3xf32>, %b: tensor<1x2x3x2xf32>)
    -> tensor<1x2x3x2xf32> {
  // expected-error @+1 {{expects matching rank-2 or rank-3 A, B, and result tensors}}
  %x = tessera.cholesky_solve %a, %b : (tensor<1x2x3x3xf32>, tensor<1x2x3x2xf32>) -> tensor<1x2x3x2xf32>
  return %x : tensor<1x2x3x2xf32>
}
