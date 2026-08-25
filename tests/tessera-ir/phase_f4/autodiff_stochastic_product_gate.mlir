// RUN: not tessera-opt %s -split-input-file --tessera-autodiff-paired 2>&1 | FileCheck %s

// W4-EFFECTS-1 slice E2 — the REPLAYABILITY gate, refusal by refusal. Each
// module isolates one reason, so the diagnostic names why the draw cannot be
// replayed rather than lumping it in with "stochastic".
//
// The carrier is verified as a CHAIN — sha256(payload) == digest, and the
// payload names this op and this class — so a fabricated digest, a missing
// payload, or a product copied from another operation are all refused
// (PR #630 review). The admitted path lives in
// autodiff_dropout_pathwise_adjoint.mlir, so each file states one thing.

// ── a product of the wrong class does not establish reproducibility ──
module {
  func.func @wrong_class(%x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.dropout"(%x) {
      p = 2.500000e-01 : f64, seed = 7 : i64,
      tessera.effect_kind = "random", training = true,
      tessera.recorded_product.schema = "tessera.recorded_product.v1",
      tessera.recorded_product.effect_class = "observational",
      tessera.recorded_product.digest = "d12f47730f725eeabca93e9fc56c0650b0e518c5597dc8b7334d05728bb71f11",
      tessera.recorded_product.payload = "{\"effect_class\":\"keyed_rng\",\"op\":\"tessera.dropout\",\"op_occurrence\":\"bb0.op0\",\"product\":{\"dtype\":\"f32\",\"key\":{\"counter\":0,\"seed\":7},\"shape\":[4]},\"schema\":\"tessera.recorded_product.v1\",\"write_set\":[]}"
    } : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK: AUTODIFF_STOCHASTIC_UNKEYED:
    // CHECK-SAME: 'observational' product
    return %y : tensor<4xf32>
  }
}

// -----
// ── a digest with no payload addresses nothing and cannot be verified ──
module {
  func.func @no_payload(%x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.dropout"(%x) {
      p = 2.500000e-01 : f64, seed = 7 : i64,
      tessera.effect_kind = "random", training = true,
      tessera.recorded_product.schema = "tessera.recorded_product.v1",
      tessera.recorded_product.effect_class = "keyed_rng",
      tessera.recorded_product.digest = "d12f47730f725eeabca93e9fc56c0650b0e518c5597dc8b7334d05728bb71f11"
    } : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK: AUTODIFF_STOCHASTIC_NO_PRODUCT:
    // CHECK-SAME: not the payload it addresses
    return %y : tensor<4xf32>
  }
}

// -----
// ── a fabricated digest over a real payload is caught by recomputation ──
module {
  func.func @fabricated_digest(%x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.dropout"(%x) {
      p = 2.500000e-01 : f64, seed = 7 : i64,
      tessera.effect_kind = "random", training = true,
      tessera.recorded_product.schema = "tessera.recorded_product.v1",
      tessera.recorded_product.effect_class = "keyed_rng",
      tessera.recorded_product.digest = "1111111111111111111111111111111111111111111111111111111111111111",
      tessera.recorded_product.payload = "{\"effect_class\":\"keyed_rng\",\"op\":\"tessera.dropout\",\"op_occurrence\":\"bb0.op0\",\"product\":{\"dtype\":\"f32\",\"key\":{\"counter\":0,\"seed\":7},\"shape\":[4]},\"schema\":\"tessera.recorded_product.v1\",\"write_set\":[]}"
    } : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK: AUTODIFF_STOCHASTIC_NO_PRODUCT:
    // CHECK-SAME: does not hash to its declared digest
    return %y : tensor<4xf32>
  }
}

// -----
// ── a VERIFIABLE product recorded for another operation cannot be pasted on ──
module {
  func.func @foreign_product(%x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.dropout"(%x) {
      p = 2.500000e-01 : f64, seed = 7 : i64,
      tessera.effect_kind = "random", training = true,
      tessera.recorded_product.schema = "tessera.recorded_product.v1",
      tessera.recorded_product.effect_class = "keyed_rng",
      tessera.recorded_product.digest = "174cf5097d1af26063371e0e30886fb2c7faf5bdad858049b3b54dcf523dc668",
      tessera.recorded_product.payload = "{\"effect_class\":\"keyed_rng\",\"op\":\"tessera.other_op\",\"op_occurrence\":\"bb0.op0\",\"product\":{\"dtype\":\"f32\",\"key\":{\"seed\":7},\"shape\":[4]},\"schema\":\"tessera.recorded_product.v1\",\"write_set\":[]}"
    } : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK: AUTODIFF_STOCHASTIC_NO_PRODUCT:
    // CHECK-SAME: does not name tessera.dropout
    return %y : tensor<4xf32>
  }
}

// -----
// ── a carrier without its schema is not a carrier this compiler understands ──
module {
  func.func @no_schema(%x: tensor<4xf32>) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = "tessera.dropout"(%x) {
      p = 2.500000e-01 : f64, seed = 7 : i64,
      tessera.effect_kind = "random", training = true,
      tessera.recorded_product.effect_class = "keyed_rng",
      tessera.recorded_product.digest = "d12f47730f725eeabca93e9fc56c0650b0e518c5597dc8b7334d05728bb71f11",
      tessera.recorded_product.payload = "{\"effect_class\":\"keyed_rng\",\"op\":\"tessera.dropout\",\"op_occurrence\":\"bb0.op0\",\"product\":{\"dtype\":\"f32\",\"key\":{\"counter\":0,\"seed\":7},\"shape\":[4]},\"schema\":\"tessera.recorded_product.v1\",\"write_set\":[]}"
    } : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK: AUTODIFF_STOCHASTIC_NO_PRODUCT:
    // CHECK-SAME: no supported recorded-product schema
    return %y : tensor<4xf32>
  }
}
