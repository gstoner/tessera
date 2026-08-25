// RUN: tessera-opt --tessera-autodiff-paired %s | FileCheck %s

// W4-EFFECTS-1 slice E2b — the pathwise adjoint of a KEYED dropout.
//
// Under the declared `constant_noise` estimator the forward is
//   y = x * m / (1 - p),  m in {0,1} drawn from the op's key,
// so the Jacobian is diag(m / (1 - p)). A diagonal operator equals its own
// transpose, hence the adjoint is the SAME operation applied to the
// cotangent: dx = dropout(dout, same key). That is only sound because the
// draw REPLAYS from the key — verified numerically in
// tests/unit/test_recorded_product.py (J v == diag(m) v bitwise).
//
// The recorded product here is a REAL one, emitted by the E1 carrier: its
// digest is the sha256 of the payload beside it, and the payload names this
// operation. The pass recomputes both, so a fabricated digest cannot admit
// an op (PR #630 review).

module {
  func.func @keyed_dropout(%x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.dropout"(%x) {
      p = 2.500000e-01 : f64, seed = 7 : i64,
      tessera.effect_kind = "random", training = true,
      tessera.recorded_product.schema = "tessera.recorded_product.v1", tessera.recorded_product.effect_class = "keyed_rng", tessera.recorded_product.digest = "d12f47730f725eeabca93e9fc56c0650b0e518c5597dc8b7334d05728bb71f11", tessera.recorded_product.payload = "{\"effect_class\":\"keyed_rng\",\"op\":\"tessera.dropout\",\"op_occurrence\":\"bb0.op0\",\"product\":{\"dtype\":\"f32\",\"key\":{\"counter\":0,\"seed\":7},\"shape\":[4]},\"schema\":\"tessera.recorded_product.v1\",\"write_set\":[]}"
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %y : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @keyed_dropout(
// CHECK-SAME: tessera.autodiff.paired = @keyed_dropout__bwd
// CHECK-NOT: AUTODIFF_STOCHASTIC
// CHECK-NOT: AUTODIFF_OP_NOT_DIFFERENTIABLE

// CHECK-LABEL: func.func @keyed_dropout__bwd(
// CHECK-SAME: tessera.autodiff.role = "backward"
// CHECK: tessera.dropout %arg1
// CHECK-SAME: p = 2.500000e-01 : f64
// CHECK-SAME: seed = 7 : i64
// CHECK: return
