// RUN: tessera-opt --tessera-autodiff-paired %s -split-input-file | FileCheck %s

// W4-EFFECTS-1 slice E2b — the pathwise adjoint of a KEYED dropout.
//
// Under the declared `constant_noise` estimator the forward is
//   y = x * m / (1 - p),  m in {0,1} drawn from the op's key,
// so the Jacobian is diag(m / (1 - p)). A diagonal operator equals its own
// transpose, hence the adjoint is the SAME operation applied to the
// cotangent: dx = dropout(dout, same key). That is only sound because the
// draw REPLAYS from the key — verified numerically in
// tests/unit/test_recorded_product.py (J v == diag(m) v bitwise).

module {
  func.func @keyed_dropout(%x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.dropout"(%x) {
      p = 2.500000e-01 : f64, seed = 7 : i64,
      tessera.effect_kind = "random", training = true,
      tessera.recorded_product.effect_class = "keyed_rng",
      tessera.recorded_product.digest = "1111111111111111111111111111111111111111111111111111111111111111"
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %y : tensor<4xf32>
  }
}

// The region is admitted — neither refusal fires — and a paired backward exists.
// CHECK-LABEL: func.func @keyed_dropout(
// CHECK-SAME: tessera.autodiff.paired = @keyed_dropout__bwd
// CHECK-NOT: AUTODIFF_STOCHASTIC
// CHECK-NOT: AUTODIFF_OP_NOT_DIFFERENTIABLE

// The backward applies the SAME keyed draw to the cotangent: same seed, same
// probability. A different key here would silently produce a wrong gradient.
// CHECK-LABEL: func.func @keyed_dropout__bwd(
// CHECK-SAME: tessera.autodiff.role = "backward"
// CHECK: tessera.dropout %arg1
// CHECK-SAME: p = 2.500000e-01 : f64
// CHECK-SAME: seed = 7 : i64
// CHECK: return
