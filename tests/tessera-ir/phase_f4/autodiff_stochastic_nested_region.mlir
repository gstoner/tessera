// RUN: not tessera-opt %s -split-input-file --tessera-autodiff-paired 2>&1 | FileCheck %s

// W4-EFFECTS-1, PR #630 review (P2) — admission is the SAME inside a region.
//
// E2 admitted a keyed draw at the top level of a differentiated function. That
// left a hole with teeth: `RegionAdjointInterface::isReplayable` rejected any
// non-pure op, so the moment the very same admitted dropout appeared inside an
// `scf.if` — the control flow W4 exists to differentiate through — the
// function failed with AUTODIFF_REGION_ADJOINT. The admitted family was
// admissible only in straight-line code, which is not a useful family.
//
// The region walk now calls the SAME `recordedProductFailure` verifier the
// paired pass calls (#31: one implementation per boundary), so a top-level
// draw and a nested draw cannot diverge in what they accept — including the
// hash-chain check, which is what the second module proves.

// ── a verified keyed draw nested in scf.if differentiates ──
module {
  func.func @nested_if(%x: tensor<4xf32>, %c: i1) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = scf.if %c -> tensor<4xf32> {
      %d = "tessera.dropout"(%x) {
        p = 2.500000e-01 : f64, seed = 7 : i64,
        tessera.effect_kind = "random", training = true,
        tessera.recorded_product.schema = "tessera.recorded_product.v1",
        tessera.recorded_product.effect_class = "keyed_rng",
        tessera.recorded_product.digest = "709428c2a8817b2aa57b4c2f8cc59842b16128a86db27b470192ffbf3a098e9d",
        tessera.recorded_product.payload = "{\"effect_class\":\"keyed_rng\",\"op\":\"tessera.dropout\",\"op_occurrence\":\"bb0.op0\",\"product\":{\"dtype\":\"f32\",\"key\":{\"counter\":0,\"seed\":7},\"p\":0.25,\"shape\":[4]},\"schema\":\"tessera.recorded_product.v1\",\"write_set\":[]}"
      } : (tensor<4xf32>) -> tensor<4xf32>
      scf.yield %d : tensor<4xf32>
    } else {
      scf.yield %x : tensor<4xf32>
    }
    return %y : tensor<4xf32>
  }
}

// The second module is refused, so the run fails overall and stderr carries
// its diagnostic; the first module's IR still prints.
// CHECK: error: {{.*}}[AUTODIFF_REGION_ADJOINT]
// CHECK: func.func @nested_if(
// The region is replayable, so the pairing happened AND the predicate is
// saved rather than recomputed — a second draw would not reproduce the first.
// CHECK-SAME: tessera.autodiff.paired = @nested_if__bwd
// CHECK-SAME: tessera.autodiff.residual_sources = ["scf.if:predicate"]
// CHECK: func.func @nested_if__bwd(

// -----

// ── the SAME nesting with a product naming another op is still refused ──
// Teeth for the sharing claim: had the region walk merely checked that SOME
// product attribute was present, this would differentiate too.
module {
  func.func @nested_if_foreign_product(%x: tensor<4xf32>, %c: i1) -> tensor<4xf32>
      attributes {tessera.autodiff = "reverse"} {
    %y = scf.if %c -> tensor<4xf32> {
      %d = "tessera.dropout"(%x) {
        p = 2.500000e-01 : f64, seed = 7 : i64,
        tessera.effect_kind = "random", training = true,
        tessera.recorded_product.schema = "tessera.recorded_product.v1",
        tessera.recorded_product.effect_class = "keyed_rng",
        tessera.recorded_product.digest = "a1ba98b80fa05272ab5315a1dfe79d8377f80949b54202ff407c69fe7841adc1",
        tessera.recorded_product.payload = "{\"effect_class\":\"keyed_rng\",\"op\":\"tessera.rng_philox_uniform\",\"op_occurrence\":\"bb0.op0\",\"product\":{\"dtype\":\"f32\",\"key\":{\"counter\":0,\"seed\":7},\"shape\":[4]},\"schema\":\"tessera.recorded_product.v1\",\"write_set\":[]}"
      } : (tensor<4xf32>) -> tensor<4xf32>
      scf.yield %d : tensor<4xf32>
    } else {
      scf.yield %x : tensor<4xf32>
    }
    return %y : tensor<4xf32>
  }
}

// CHECK-NOT: func.func @nested_if_foreign_product__bwd
