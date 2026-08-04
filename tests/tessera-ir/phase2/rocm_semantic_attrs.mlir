// RUN: tessera-opt %s | FileCheck %s
//
// W1.1b — the ACCEPTING side of the split dtype constraint.
//
// `rocm_semantic_attrs_invalid.mlir` proves what is rejected. This file proves
// the split is a split and not simply a narrowing: the same literal `"int32"`
// is legal on `compare` and illegal on every other op carrying `$dtype`. A
// pair of fixtures where one op accepts a value and another rejects it is the
// only evidence a per-operation constraint is doing per-operation work; two
// rejections would be equally satisfied by one shared narrow set.

// CHECK-LABEL: func.func @compare_accepts_integers
func.func @compare_accepts_integers() {
  // `compare` maps either kind to a bool, so it is the one op of the 21
  // carrying `$dtype` whose set genuinely includes integers.
  // CHECK: tessera_rocm.compare
  // CHECK-SAME: dtype = "int32"
  "tessera_rocm.compare"() { name = "c_i32", dtype = "int32" } : () -> ()
  // CHECK: tessera_rocm.compare
  // CHECK-SAME: dtype = "bf16"
  "tessera_rocm.compare"() { name = "c_bf16", dtype = "bf16" } : () -> ()
  return
}

// CHECK-LABEL: func.func @float_ops_keep_their_three_widths
func.func @float_ops_keep_their_three_widths() {
  // The unchanged textual form matters: 288 existing fixtures pass without
  // edit because this work added a constraint, not a syntax.
  // CHECK: tessera_rocm.softmax
  "tessera_rocm.softmax"() { name = "s_f32", dtype = "f32" } : () -> ()
  // CHECK: tessera_rocm.softmax
  "tessera_rocm.softmax"() { name = "s_f16", dtype = "f16" } : () -> ()
  // CHECK: tessera_rocm.softmax
  "tessera_rocm.softmax"() { name = "s_bf16", dtype = "bf16" } : () -> ()
  return
}

// CHECK-LABEL: func.func @fpquant_dtype_is_the_input_width
func.func @fpquant_dtype_is_the_input_width() {
  // Worth pinning explicitly, because it is the op that made the original
  // union look justified: `fpquant` IS a quantisation op, so `fp8_e4m3` reads
  // like a plausible `$dtype`. It is not — `$dtype` is the width of the value
  // being quantised, and the target grid is `mantissa_bits`/`max_normal`/
  // `min_exp`. Admitting fp8 here would have accepted a program that means
  // nothing.
  // CHECK: tessera_rocm.fpquant
  // CHECK-SAME: mantissa_bits = 3
  "tessera_rocm.fpquant"() {
    name = "q", dtype = "bf16",
    max_normal = 448.0 : f32, mantissa_bits = 3, min_exp = -6
  } : () -> ()
  return
}
