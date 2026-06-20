// RUN: tessera-opt --tessera-ir-contracts %s -split-input-file -verify-diagnostics --allow-unregistered-dialect | FileCheck %s

// IRContractLegalityPass (2026-06-19) — dtype / aliasing / buffer-binding
// contracts. LayoutLegalityPass's sibling for the three remaining contract
// families in COMPILER_AUDIT's "Layout and binding contracts are uneven" item.

// ─────────────────────────────────────────────────────────────────────────
// DTYPE — POSITIVE: low-precision storage with a wider accumulator. ✓
// ─────────────────────────────────────────────────────────────────────────
// CHECK-LABEL: func.func @dtype_fp8_ok
func.func @dtype_fp8_ok(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "fp8_e4m3", accum = "fp32"}}
       : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// DTYPE — POSITIVE: bf16 storage + fp32 accum (the canonical matmul policy). ✓
// CHECK-LABEL: func.func @dtype_bf16_ok
func.func @dtype_bf16_ok(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "bf16", accum = "fp32"}}
       : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// DTYPE — NEGATIVE: TF32 as a storage dtype is illegal (Decision #15a). ✗
func.func @dtype_tf32_storage(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{DTYPE_LEGALITY_TF32_AS_STORAGE}}
  %0 = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "tf32", accum = "fp32"}}
       : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// DTYPE — NEGATIVE: low-precision storage with no accumulator declared. ✗
func.func @dtype_lowp_no_accum(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{DTYPE_LEGALITY_LOWP_WITHOUT_WIDE_ACCUM}}
  %0 = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "fp8_e4m3"}}
       : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// DTYPE — NEGATIVE: low-precision storage with a non-wider accumulator. ✗
func.func @dtype_lowp_narrow_accum(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{DTYPE_LEGALITY_LOWP_WITHOUT_WIDE_ACCUM}}
  %0 = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "fp8_e4m3", accum = "fp8_e5m2"}}
       : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// DTYPE — NEGATIVE: storage names an unknown dtype. ✗
func.func @dtype_unknown_storage(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // expected-error @+1 {{DTYPE_LEGALITY_UNKNOWN_STORAGE}}
  %0 = "tessera.matmul"(%a, %b) {numeric_policy = {storage = "float7", accum = "fp32"}}
       : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// ALIASING — POSITIVE: in-place with a valid aliases index. ✓
// CHECK-LABEL: func.func @alias_ok
func.func @alias_ok(%a: tensor<8xf32>) -> tensor<8xf32> {
  %0 = "tessera.relu"(%a) {tessera.inplace = true, tessera.aliases = 0 : i64}
       : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// ALIASING — NEGATIVE: in-place op without an aliases declaration. ✗
func.func @alias_missing(%a: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{ALIAS_LEGALITY_MISSING_ALIASES}}
  %0 = "tessera.relu"(%a) {tessera.inplace = true} : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// ALIASING — NEGATIVE: aliases index out of operand range. ✗
func.func @alias_oob(%a: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{ALIAS_LEGALITY_OPERAND_OOB}}
  %0 = "tessera.relu"(%a) {tessera.inplace = true, tessera.aliases = 5 : i64}
       : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// BUFFER — POSITIVE: a known buffer role. ✓
// CHECK-LABEL: func.func @buffer_ok
func.func @buffer_ok(%a: tensor<8xf32>) -> tensor<8xf32> {
  %0 = "tessera.relu"(%a) {tessera.binding = "buf0", tessera.buffer_role = "input"}
       : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// BUFFER — NEGATIVE: an unknown buffer role. ✗
func.func @buffer_bad_role(%a: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{BUFFER_BINDING_UNKNOWN_ROLE}}
  %0 = "tessera.relu"(%a) {tessera.buffer_role = "bogus"} : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// BUFFER — NEGATIVE: one binding id bound to two different roles. ✗
func.func @buffer_conflict(%a: tensor<8xf32>) -> tensor<8xf32> {
  %0 = "tessera.relu"(%a) {tessera.binding = "buf0", tessera.buffer_role = "input"}
       : (tensor<8xf32>) -> tensor<8xf32>
  // expected-error @+1 {{BUFFER_BINDING_CONFLICT}}
  %1 = "tessera.relu"(%0) {tessera.binding = "buf0", tessera.buffer_role = "scratch"}
       : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}
