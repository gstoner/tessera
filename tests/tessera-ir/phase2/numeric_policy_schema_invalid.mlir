// RUN: not tessera-opt %s --tessera-ir-contracts --split-input-file 2>&1 | FileCheck %s

// NUMPOL-CARRIER-1 step 1 — the refusals, one reason per module.
// Accept-set: numeric_policy_schema.mlir.

// ── a typo is not an ignorable key ──
// The real `accum` is ABSENT here. Because `getAs<StringAttr>` returns null
// identically for a missing key and a misspelled one, this op used to carry a
// policy that looked like it stated an accumulator contract and stated none —
// a semantic key defaulting silently, which #21a exists to forbid.
module {
  func.func @typo(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xf32> {
    // expected-error @+1 {{NUMERIC_POLICY_UNKNOWN_KEY}}
    %0 = "tessera.matmul"(%a, %b) {
      numeric_policy = {storage = "bf16", accumulator = "fp32"}
    } : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
// CHECK: NUMERIC_POLICY_UNKNOWN_KEY
// CHECK-SAME: "accumulator"

// -----

// ── a non-string value reads back as ABSENT through StringAttr lookup ──
module {
  func.func @not_a_string(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xf32> {
    // expected-error @+1 {{NUMERIC_POLICY_NON_STRING_VALUE}}
    %0 = "tessera.matmul"(%a, %b) {
      numeric_policy = {storage = "bf16", accum = 32 : i64}
    } : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
// CHECK: NUMERIC_POLICY_NON_STRING_VALUE

// -----

// ── a dtype nobody defines ──
module {
  func.func @unknown_accum(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xf32> {
    // expected-error @+1 {{NUMERIC_POLICY_UNKNOWN_ACCUM}}
    %0 = "tessera.matmul"(%a, %b) {
      numeric_policy = {storage = "bf16", accum = "float128"}
    } : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
// CHECK: NUMERIC_POLICY_UNKNOWN_ACCUM

// -----

// ── the accumulator may not be NARROWER than the storage ──
// Refused rather than warned, on measurement rather than taste: at the same
// 48 dtype bits, fp16-storage/fp32-accum is 25.8x more accurate than this
// (K=4096 dot product, median relative error vs an fp64 reference, 48 trials),
// and — decisively — the result here is BIT-IDENTICAL to also narrowing the
// storage, so the wider storage is unobservable and buys only memory traffic.
// There is no program for which this policy is the right answer.
module {
  func.func @narrowing(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // expected-error @+1 {{NUMERIC_POLICY_NARROWING_ACCUM}}
    %0 = "tessera.matmul"(%a, %b) {
      numeric_policy = {storage = "fp32", accum = "fp16"}
    } : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
// CHECK: NUMERIC_POLICY_NARROWING_ACCUM
// CHECK-SAME: 24 significand bits
// CHECK-SAME: 11 bits

// -----

// ── same WIDTH, fewer significand bits, is still narrowing ──
// fp16 and bf16 are both 16 bits and carry 11 and 8 significand bits. A
// width-based rule would call this pair legal; the accumulator's contract is
// precision, so the comparison is on significand bits.
module {
  func.func @same_width_narrower(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    // expected-error @+1 {{NUMERIC_POLICY_NARROWING_ACCUM}}
    %0 = "tessera.matmul"(%a, %b) {
      numeric_policy = {storage = "fp16", accum = "bf16"}
    } : (tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf16>
    return %0 : tensor<64x64xf16>
  }
}
// CHECK: NUMERIC_POLICY_NARROWING_ACCUM

// -----

// ── an integer accumulator cannot hold a floating-point product ──
module {
  func.func @float_into_int(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xf32> {
    // expected-error @+1 {{NUMERIC_POLICY_NARROWING_ACCUM}}
    %0 = "tessera.matmul"(%a, %b) {
      numeric_policy = {storage = "bf16", accum = "int32"}
    } : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
// CHECK: NUMERIC_POLICY_NARROWING_ACCUM
// CHECK-SAME: cannot accumulate into integer

// -----

// ── a math mode that does not reduce its storage ──
// TF32 carries 11 significand bits; bf16 storage carries 8. Declaring tf32
// here rounds nothing, so it is either a no-op or a false statement about the
// arithmetic. Decision #15a states TF32 as an fp32 math mode.
module {
  func.func @mode_not_reducing(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xf32> {
    // expected-error @+1 {{NUMERIC_POLICY_MATH_MODE_NOT_REDUCING}}
    %0 = "tessera.matmul"(%a, %b) {
      numeric_policy = {storage = "bf16", accum = "fp32", math_mode = "tf32"}
    } : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
// CHECK: NUMERIC_POLICY_MATH_MODE_NOT_REDUCING

// -----

// ── an unknown mode name ──
module {
  func.func @unknown_mode(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // expected-error @+1 {{NUMERIC_POLICY_UNKNOWN_MATH_MODE}}
    %0 = "tessera.matmul"(%a, %b) {
      numeric_policy = {storage = "fp32", accum = "fp32", math_mode = "fast"}
    } : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
// CHECK: NUMERIC_POLICY_UNKNOWN_MATH_MODE

// -----

// ── a numeric_policy that is not a dictionary at all ──
// `getAttrOfType<DictionaryAttr>` returns null for a WRONGLY TYPED attribute
// exactly as it does for an absent one, so this used to be skipped in silence
// by the checker AND by every consumer of the attribute. Measured: the
// spectral scheduler emitted `numeric_policy = "f32;ortho"`, a private
// semicolon-delimited string, and it was invisible to all of them — while
// carrying something that is not a Decision #15a policy at all (its value
// could be "deterministic_f32_ascending_frames", a reduction-ORDER contract,
// not a dtype). That contract is now named tessera.spectral_accumulation /
// tessera.spectral_normalization, and this refusal is what stops the
// collision recurring: one attribute name means one thing, or its consumers
// read a different contract than its producers wrote (#31).
//
// Ops that DECLARE numeric_policy in ODS are already covered by the attribute
// constraint; `tessera.relu` does not declare it, so the attribute is
// discardable here — which is precisely where the spectral case lived.
module {
  func.func @policy_is_a_string(%a: tensor<4xf32>) -> tensor<4xf32> {
    // expected-error @+1 {{NUMERIC_POLICY_NOT_A_DICTIONARY}}
    %0 = "tessera.relu"(%a) {numeric_policy = "f32;ortho"}
      : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
// CHECK: NUMERIC_POLICY_NOT_A_DICTIONARY
// CHECK-SAME: got "f32;ortho"
