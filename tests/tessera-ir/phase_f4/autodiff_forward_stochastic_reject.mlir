// RUN: not tessera-opt %s --tessera-autodiff-forward 2>&1 | FileCheck %s

module {
  func.func @unreplayable(%x: tensor<8xf32>) -> tensor<8xf32>
      attributes {tessera.autodiff = "forward"} {
    %y = "tessera.dropout"(%x) {
      p = 2.500000e-01 : f64, estimator = "constant_noise",
      tessera.effect_kind = "random", training = true
    } : (tensor<8xf32>) -> tensor<8xf32>
    return %y : tensor<8xf32>
  }
}

// CHECK: active stochastic operation requires explicit constant_noise replay
