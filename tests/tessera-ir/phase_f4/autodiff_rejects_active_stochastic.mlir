// RUN: not tessera-opt --tessera-autodiff-paired %s 2>&1 | FileCheck %s

// Active stochastic programs require an explicit estimator; deterministic
// seeds make replay reproducible but do not define a mathematical adjoint.

module {
  func.func @stochastic(%x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.dropout"(%x) {
      p = 2.500000e-01 : f64,
      seed = 7 : i64,
      tessera.effect_kind = "random",
      training = true
    } : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK: AUTODIFF_STOCHASTIC_EFFECT:
    return %y : tensor<4xf32>
  }
}
