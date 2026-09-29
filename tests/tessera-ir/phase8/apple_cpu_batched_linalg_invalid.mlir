// REQUIRES: tessera-apple-backend
// RUN: tessera-opt --tessera-lower-to-apple_cpu-full --split-input-file --verify-diagnostics %s
// Rank-3 is valid target-neutral Graph IR, but the Apple CPU ABI is rank-2.
func.func @batched_cholesky(%a: tensor<2x4x4xf32>) -> tensor<2x4x4xf32> {
  // expected-error@+1 {{apple_cpu value lowering: tessera.cholesky requires rank-2 tensor operands}}
  %r = tessera.cholesky %a : (tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %r : tensor<2x4x4xf32>
}
// -----
func.func @batched_tri_solve(%a: tensor<2x4x4xf32>, %b: tensor<2x4x3xf32>) -> tensor<2x4x3xf32> {
  // expected-error@+1 {{apple_cpu value lowering: tessera.tri_solve requires rank-2 tensor operands}}
  %r = tessera.tri_solve %a, %b : (tensor<2x4x4xf32>, tensor<2x4x3xf32>) -> tensor<2x4x3xf32>
  return %r : tensor<2x4x3xf32>
}
