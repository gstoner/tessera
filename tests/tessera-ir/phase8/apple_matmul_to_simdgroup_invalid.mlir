// REQUIRES: tessera-apple-backend
// RUN: tessera-opt --split-input-file --verify-diagnostics --tessera-matmul-to-apple-simdgroup %s
//
// APPLE-ACCUM-1 negative fixture: the simdgroup lowering reads the program's
// numeric_policy.accum and refuses, with a stable diagnostic naming storage and
// accumulator, anything it cannot faithfully execute. Each case must produce
// the error and NO lowering (verify-diagnostics fails on any unexpected
// diagnostic, and the pass signals failure).

// The accumulator selects semantics (Decision #21a). An f16 GEMM with no
// numeric_policy is refused rather than given an fp32 accumulator it never
// asked for.
func.func @missing_accumulator(%a: tensor<16x16xf16>, %b: tensor<16x8xf16>)
    -> tensor<16x8xf32> {
  // expected-error @+1 {{APPLE_SIMDGROUP_ACCUM_MISSING}}
  %c = "tessera.matmul"(%a, %b)
      : (tensor<16x16xf16>, tensor<16x8xf16>) -> tensor<16x8xf32>
  return %c : tensor<16x8xf32>
}

// -----

// Metal compiles a bfloat accumulator, but measured on the M1 Max it runs fp32
// accumulation truncated to bf16 at the store -- not the bf16 accumulation this
// program declares.
func.func @bf16_accumulator_is_refused(%a: tensor<16x16xbf16>, %b: tensor<16x8xbf16>)
    -> tensor<16x8xbf16> {
  // expected-error @+1 {{storage 'bf16' in accum="bf16": a bf16 simdgroup accumulator is not bf16 accumulation on Apple7}}
  %c = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "bf16", accum = "bf16"}}
      : (tensor<16x16xbf16>, tensor<16x8xbf16>) -> tensor<16x8xbf16>
  return %c : tensor<16x8xbf16>
}

// -----

// simdgroup_matrix has no integer element type.
func.func @int32_accumulator_is_refused(%a: tensor<16x16xf16>, %b: tensor<16x8xf16>)
    -> tensor<16x8xf32> {
  // expected-error @+1 {{APPLE_SIMDGROUP_ACCUM_UNSUPPORTED}}
  %c = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "fp16", accum = "int32"}}
      : (tensor<16x16xf16>, tensor<16x8xf16>) -> tensor<16x8xf32>
  return %c : tensor<16x8xf32>
}

// -----

// An f16 accumulator into a bf16 result would round the already-rounded
// accumulator a second time; there is no single-rounding epilogue for it.
func.func @double_rounding_result_is_refused(%a: tensor<16x16xf16>, %b: tensor<16x8xf16>)
    -> tensor<16x8xbf16> {
  // expected-error @+1 {{no single-rounding conversion from an f16 accumulator to an bf16 result}}
  %c = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "fp16", accum = "fp16"}}
      : (tensor<16x16xf16>, tensor<16x8xf16>) -> tensor<16x8xbf16>
  return %c : tensor<16x8xbf16>
}

// -----

// A policy storage that contradicts the operands is refused, not believed.
func.func @policy_storage_contradicts_operands(%a: tensor<16x16xf16>, %b: tensor<16x8xf16>)
    -> tensor<16x8xf32> {
  // expected-error @+1 {{APPLE_SIMDGROUP_STORAGE_MISMATCH: numeric_policy.storage="bf16" does not name the operands' element type f16}}
  %c = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "bf16", accum = "fp32"}}
      : (tensor<16x16xf16>, tensor<16x8xf16>) -> tensor<16x8xf32>
  return %c : tensor<16x8xf32>
}
