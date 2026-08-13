// RUN: tessera-opt %s -lower-tpp-to-target-ir -verify-diagnostics

module attributes {tessera.target = "cpu"} {
  func.func @unlegalized(%x: tensor<8x8xf32>) -> tensor<8x8xf32> {
    // expected-error @+1 {{target lowering requires validated scheme, order, and spacing}}
    %y = "tpp.grad"(%x) {scheme = "central", order = 2 : i64, spacing = [1.0 : f64, 1.0 : f64]} : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %y : tensor<8x8xf32>
  }
}
