// RUN: tessera-opt %s -tpp-legalize-space-time -verify-diagnostics

func.func @missing_scheme(%x: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{requires explicit 'scheme'}}
  %y = "tpp.grad"(%x) {order = 2 : i64, spacing = [1.0 : f64, 1.0 : f64]} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %y : tensor<8x8xf32>
}

func.func @missing_spacing(%x: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{requires explicit per-axis 'spacing'}}
  %y = "tpp.grad"(%x) {scheme = "central", order = 2 : i64} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %y : tensor<8x8xf32>
}

func.func @bad_spacing(%x: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{every spacing value must be finite and > 0}}
  %y = "tpp.grad"(%x) {scheme = "central", order = 2 : i64, spacing = [1.0 : f64, 0.0 : f64]} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %y : tensor<8x8xf32>
}
