// L-series linalg pilot — L1: tessera.cholesky verifier negative cases.
//
// RUN: tessera-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect -o /dev/null

func.func @chol_nonsquare(%a: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error @+1 {{'tessera.cholesky' op input matrix must be square}}
  %0 = tessera.cholesky %a : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

func.func @chol_shape_mismatch(%a: tensor<4x4xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{'tessera.cholesky' op result must have the same shape as the input}}
  %0 = tessera.cholesky %a : (tensor<4x4xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

func.func @chol_rank3(%a: tensor<2x4x4xf32>) -> tensor<2x4x4xf32> {
  // expected-error @+1 {{'tessera.cholesky' op expects rank-2 input and result tensors}}
  %0 = tessera.cholesky %a : (tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %0 : tensor<2x4x4xf32>
}
