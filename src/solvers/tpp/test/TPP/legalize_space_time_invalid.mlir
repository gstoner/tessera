// RUN: tessera-opt %s -tpp-legalize-space-time -verify-diagnostics
//
// Unknown schemes are illegal (spec section 6): the pass emits a diagnostic
// and fails rather than silently passing bad metadata downstream.

func.func @bad_scheme(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // expected-error @+1 {{unknown spatial scheme 'quadratic'}}
  %y = "tpp.grad"(%x) { scheme = "quadratic" } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  return %y : tensor<32x32xf32>
}
