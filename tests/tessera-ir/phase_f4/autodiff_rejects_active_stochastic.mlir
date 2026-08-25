// RUN: not tessera-opt --tessera-autodiff-paired %s 2>&1 | FileCheck %s

// W4-EFFECTS-1 slice E2. A stochastic op raises TWO independent questions —
// can the draw be REPLAYED, and can it be DIFFERENTIATED — which the old
// blanket AUTODIFF_STOCHASTIC_EFFECT diagnostic conflated. This fixture's
// original prose already drew the distinction ("deterministic seeds make
// replay reproducible but do not define a mathematical adjoint"); now the
// compiler answers the two separately.
//
// Here: a seeded dropout carrying NO recorded product. Replayability fails
// first, and says so — a seed in the op's attributes is not a recorded
// product, and absence is not permission.

module {
  func.func @stochastic(%x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.dropout"(%x) {
      p = 2.500000e-01 : f64,
      seed = 7 : i64,
      tessera.effect_kind = "random",
      training = true
    } : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK: AUTODIFF_STOCHASTIC_NO_PRODUCT:
    // CHECK-SAME: carries no recorded product
    return %y : tensor<4xf32>
  }
}
