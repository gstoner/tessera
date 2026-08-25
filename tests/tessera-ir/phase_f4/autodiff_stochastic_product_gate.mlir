// RUN: not tessera-opt --tessera-autodiff-paired %s -split-input-file 2>&1 | FileCheck %s

// W4-EFFECTS-1 slice E2 — the REPLAYABILITY gate, refusal by refusal. Each
// module isolates one reason, so the diagnostic names why the draw cannot be
// replayed rather than lumping it in with "stochastic".
//
// The admitted path — a keyed draw that passes this gate AND has an adjoint —
// is autodiff_dropout_pathwise_adjoint.mlir; keeping it separate is what
// makes each file state exactly one thing.

// ── a product of the wrong class does not establish reproducibility ───────
module {
  func.func @wrong_class(%x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.dropout"(%x) {
      p = 2.500000e-01 : f64, seed = 7 : i64,
      tessera.effect_kind = "random", training = true,
      tessera.recorded_product.effect_class = "observational",
      tessera.recorded_product.digest = "1111111111111111111111111111111111111111111111111111111111111111"
    } : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK: AUTODIFF_STOCHASTIC_UNKEYED:
    // CHECK-SAME: 'observational' product
    return %y : tensor<4xf32>
  }
}

// -----
// ── a keyed_rng claim without an addressable digest is not verifiable ─────
module {
  func.func @unaddressable(%x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.dropout"(%x) {
      p = 2.500000e-01 : f64, seed = 7 : i64,
      tessera.effect_kind = "random", training = true,
      tessera.recorded_product.effect_class = "keyed_rng",
      tessera.recorded_product.digest = "short"
    } : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK: AUTODIFF_STOCHASTIC_NO_PRODUCT:
    // CHECK-SAME: without a 64-character content digest
    return %y : tensor<4xf32>
  }
}
