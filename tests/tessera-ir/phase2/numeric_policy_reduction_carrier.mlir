// RUN: tessera-opt %s --tessera-to-linalg --split-input-file | FileCheck %s
// RUN: tessera-opt %s --tessera-record-metadata --tessera-to-linalg \
// RUN:   --tessera-verify-metadata-obligation --split-input-file \
// RUN:   | FileCheck %s --check-prefix=OBLIGATION

// NUMPOL-CARRIER-1 (integrated-plan queue row 3b) step 2 — the reduction
// family CARRIES its declared accumulator into the emitted arithmetic.
//
// Measured before this change: `{storage="bf16", accum="fp32"}` on rmsnorm and
// softmax lowered to `arith.addf ... : bf16`. The accumulator contract was not
// merely dropped as metadata — the emitted code CONTRADICTED it, accumulating
// in the storage dtype on the very op that performs the accumulation. bf16
// carries 8 significand bits, so the running sum stagnates once it exceeds
// ~256x the increment and the error grows with the reduced extent:
//
//   D=4096: rmsnorm 2.94e-01 emitted vs 1.72e-06 declared
//           softmax denominator 5.95e-01 emitted vs 1.03e-06 declared
//
// The cast placement was decided by measurement, not taste: truncating the
// REDUCED value back to storage before the sqrt/divide leaves 5.6e-04, while
// truncating only the RESULT reaches 1.7e-06 — 326x apart. The latter is also
// the faithful reading of Decision #15a, where storage is the dtype of the
// tensor and the tensor is the result.

// ── rmsnorm: the mean-square reduction runs in the declared accumulator ──
module {
  func.func @rmsnorm_accum(%x: tensor<8x128xbf16>, %g: tensor<128xbf16>)
      -> tensor<8x128xbf16> {
    %r = "tessera.rmsnorm"(%x, %g) {
      epsilon = 1.000000e-05 : f32,
      numeric_policy = {storage = "bf16", accum = "fp32"}
    } : (tensor<8x128xbf16>, tensor<128xbf16>) -> tensor<8x128xbf16>
    return %r : tensor<8x128xbf16>
  }
}
// CHECK-LABEL: func.func @rmsnorm_accum
// the input is widened once...
// CHECK: arith.extf {{.*}} : bf16 to f32
// ...the reduction accumulates in f32, not bf16...
// CHECK: linalg.reduce
// CHECK: arith.addf {{.*}} : f32
// ...and only the RESULT returns to storage.
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: return {{.*}} : tensor<8x128xbf16>
// CHECK-NOT: arith.addf {{.*}} : bf16

// -----

// ── softmax: max-reduce, exp, and sum-of-exp all in the accumulator ──
module {
  func.func @softmax_accum(%x: tensor<8x128xbf16>) -> tensor<8x128xbf16> {
    %s = "tessera.softmax"(%x) {
      axis = 1 : i64, numeric_policy = {storage = "bf16", accum = "fp32"}
    } : (tensor<8x128xbf16>) -> tensor<8x128xbf16>
    return %s : tensor<8x128xbf16>
  }
}
// CHECK-LABEL: func.func @softmax_accum
// CHECK: arith.maximumf {{.*}} : f32
// CHECK: math.exp {{.*}} : f32
// CHECK: arith.addf {{.*}} : f32
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK-NOT: arith.addf {{.*}} : bf16

// -----

// ── layer_norm reduces TWICE, so it compounds what rmsnorm shows once ──
module {
  func.func @layer_norm_accum(%x: tensor<8x128xbf16>, %g: tensor<128xbf16>,
                              %b: tensor<128xbf16>) -> tensor<8x128xbf16> {
    %r = "tessera.layer_norm"(%x, %g, %b) {
      epsilon = 1.000000e-05 : f32,
      numeric_policy = {storage = "bf16", accum = "fp32"}
    } : (tensor<8x128xbf16>, tensor<128xbf16>, tensor<128xbf16>)
        -> tensor<8x128xbf16>
    return %r : tensor<8x128xbf16>
  }
}
// CHECK-LABEL: func.func @layer_norm_accum
// CHECK: arith.addf {{.*}} : f32
// CHECK: arith.addf {{.*}} : f32
// CHECK-NOT: arith.addf {{.*}} : bf16

// -----

// ── NEGATIVE, and the one that keeps this honest: with NO policy the emitted
// IR is unchanged. A carrier that rewrites every program is not a carrier,
// it is a global dtype promotion — and it would silently make bf16 training
// cost fp32 bandwidth. Nothing here may widen without being asked. ──
module {
  func.func @no_policy_is_unchanged(%x: tensor<8x128xbf16>) -> tensor<8x128xbf16> {
    %s = "tessera.softmax"(%x) {axis = 1 : i64}
      : (tensor<8x128xbf16>) -> tensor<8x128xbf16>
    return %s : tensor<8x128xbf16>
  }
}
// CHECK-LABEL: func.func @no_policy_is_unchanged
// CHECK: arith.addf {{.*}} : bf16
// CHECK-NOT: arith.extf
// CHECK-NOT: f32

// -----

// ── NEGATIVE: a policy whose accum equals its storage widens nothing ──
module {
  func.func @equal_accum_widens_nothing(%x: tensor<8x128xf32>) -> tensor<8x128xf32> {
    %s = "tessera.softmax"(%x) {
      axis = 1 : i64, numeric_policy = {storage = "fp32", accum = "fp32"}
    } : (tensor<8x128xf32>) -> tensor<8x128xf32>
    return %s : tensor<8x128xf32>
  }
}
// CHECK-LABEL: func.func @equal_accum_widens_nothing
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.truncf
// A trailing CHECK-NOT scans to end-of-input, so it is bounded here by the
// next module's label — without this it would also claim the obligation
// modules below emit no casts, which they do and should.
// CHECK-LABEL: func.func @all_policies_consumed

// -----

// ── Decision #32 at this boundary ──
//
// The policy dictionary does not survive into linalg, and should not: the
// accumulator is now the ELEMENT TYPE of the emitted reduction, which is a
// stronger carrier than an attribute nobody has to read. #32 requires that be
// SAID, not assumed, so the pass declares it and the obligation verifier
// checks it — where before this boundary dropped the policy silently and
// nothing was looking (the record/verify pair brackets only Graph→Tile).
//
// The two reasons are not interchangeable. The verifier counts a MULTISET of
// values, so a function that lowers one of two policy-carrying ops loses a
// value while the NAME survives; `re_expressed` is the reason for that case
// and is accepted only while the name is still present. The first version of
// this pass declared `represented_in_type` unconditionally and the verifier
// rejected it — which is the mechanism working.
module {
  func.func @all_policies_consumed(%x: tensor<8x128xbf16>) -> tensor<8x128xbf16> {
    %s = "tessera.softmax"(%x) {
      axis = 1 : i64, numeric_policy = {storage = "bf16", accum = "fp32"}
    } : (tensor<8x128xbf16>) -> tensor<8x128xbf16>
    return %s : tensor<8x128xbf16>
  }
}
// OBLIGATION-LABEL: func.func @all_policies_consumed
// OBLIGATION-SAME: tessera.lowering.dropped = {numeric_policy = "represented_in_type"}

// -----

module {
  func.func @one_policy_survives(%a: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
    %r = "tessera.rmsnorm"(%a) {
      epsilon = 1.000000e-05 : f32,
      numeric_policy = {storage = "bf16", accum = "fp32"}
    } : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
    // cholesky is not lowered by this pass, so its policy is still there.
    %s = "tessera.cholesky"(%r) {
      numeric_policy = {storage = "bf16", accum = "fp32"}
    } : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %s : tensor<128x128xbf16>
  }
}
// OBLIGATION-LABEL: func.func @one_policy_survives
// OBLIGATION-SAME: tessera.lowering.dropped = {numeric_policy = "re_expressed"}

// -----

// ── and with NO policy there is nothing to declare: a record that explains
// nothing is an error in its own right, because it reads as a considered
// exception while silently licensing a real future drop. ──
module {
  func.func @nothing_to_declare(%x: tensor<8x128xbf16>) -> tensor<8x128xbf16> {
    %s = "tessera.softmax"(%x) {axis = 1 : i64}
      : (tensor<8x128xbf16>) -> tensor<8x128xbf16>
    return %s : tensor<8x128xbf16>
  }
}
// OBLIGATION-LABEL: func.func @nothing_to_declare
// OBLIGATION-NOT: tessera.lowering.dropped
