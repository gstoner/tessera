// RUN: tessera-opt %s --split-input-file --verify-diagnostics -o /dev/null
func.func @batch(%a: tensor<2x4x4xf32>, %b: tensor<2x4x3xf32>) -> tensor<2x4x3xf32> {
  %l = tessera.cholesky %a : (tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  %x = tessera.tri_solve %l, %b : (tensor<2x4x4xf32>, tensor<2x4x3xf32>) -> tensor<2x4x3xf32>
  return %x : tensor<2x4x3xf32>
}
// -----
func.func @dynamic(%a: tensor<?x4x4xf32>, %b: tensor<?x4x?xf32>) -> tensor<?x4x?xf32> {
  %x = tessera.tri_solve %a, %b : (tensor<?x4x4xf32>, tensor<?x4x?xf32>) -> tensor<?x4x?xf32>
  return %x : tensor<?x4x?xf32>
}
// -----
func.func @chol_batch_mismatch(%a: tensor<2x4x4xf32>) -> tensor<3x4x4xf32> {
  // expected-error @+1 {{result must have the same shape as the input}}
  %l = tessera.cholesky %a : (tensor<2x4x4xf32>) -> tensor<3x4x4xf32>
  return %l : tensor<3x4x4xf32>
}
// -----
func.func @solve_batch_mismatch(%a: tensor<2x4x4xf32>, %b: tensor<3x4x1xf32>) -> tensor<3x4x1xf32> {
  // expected-error @+1 {{A and B batch dimensions must match}}
  %x = tessera.tri_solve %a, %b : (tensor<2x4x4xf32>, tensor<3x4x1xf32>) -> tensor<3x4x1xf32>
  return %x : tensor<3x4x1xf32>
}
// -----
func.func @solve_result_mismatch(%a: tensor<2x4x4xf32>, %b: tensor<2x4x1xf32>) -> tensor<3x4x1xf32> {
  // expected-error @+1 {{result must have the same shape as B}}
  %x = tessera.tri_solve %a, %b : (tensor<2x4x4xf32>, tensor<2x4x1xf32>) -> tensor<3x4x1xf32>
  return %x : tensor<3x4x1xf32>
}
// -----
func.func @solve_rank_mismatch(%a: tensor<2x4x4xf32>, %b: tensor<4x1xf32>) -> tensor<4x1xf32> {
  // expected-error @+1 {{expects matching rank-2 or rank-3 A, B, and result tensors}}
  %x = tessera.tri_solve %a, %b : (tensor<2x4x4xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
  return %x : tensor<4x1xf32>
}
