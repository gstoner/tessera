// RUN: tessera-opt %s --tessera-ir-contracts --split-input-file | FileCheck %s

// NUMPOL-CARRIER-1 (integrated-plan queue row 3b) step 1 — `numeric_policy`
// gets a SCHEMA.
//
// The attribute is a bare `DictionaryAttrBase` whose ODS predicate checks only
// "is a dictionary". Measured against this tree before these checks existed,
// five malformed policies were all ACCEPTED (exit 0) while the documented
// TF32-as-storage violation correctly failed — so the pass was running and
// simply had nothing to say. A carrier cannot be built on a payload with no
// schema, which is why this step comes before carrying anything.
//
// The unknown-key case is the sharpest: `getAs<StringAttr>("accum")` returns
// null for a MISSPELLED key exactly as it does for an absent one, so the op
// carried a policy that looked like it stated an accumulator contract and
// stated none.
//
// Negative cases live in numeric_policy_schema_invalid.mlir; this file is the
// accept-set, because a checker that only ever refuses proves nothing.

// ── the everyday mixed policy ──
module {
  func.func @mixed(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xf32> {
    %0 = "tessera.matmul"(%a, %b) {
      numeric_policy = {storage = "bf16", accum = "fp32"}
    } : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
// CHECK-LABEL: func.func @mixed
// CHECK: numeric_policy = {accum = "fp32", storage = "bf16"}

// -----

// ── integer storage with a FLOAT accumulator: the dequantized-weight path.
// Cross-family int→float is legal and must stay legal; only float→int is not,
// since an integer accumulator cannot hold a floating-point product. ──
module {
  func.func @dequant(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xf32> {
    %0 = "tessera.matmul"(%a, %b) {
      numeric_policy = {storage = "int4", accum = "fp32"}
    } : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xf32>
    %1 = "tessera.matmul"(%a, %b) {
      numeric_policy = {storage = "int4", accum = "int32"}
    } : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
// CHECK-LABEL: func.func @dequant
// CHECK: accum = "fp32", storage = "int4"
// CHECK: accum = "int32", storage = "int4"

// -----

// ── math_mode names a NARROWER arithmetic on wider storage. TF32 (11
// significand bits) on fp32 (24) reduces; that is what makes it a mode rather
// than a storage dtype (Decision #15a). The `softmax` key is the attention
// family's separately-accumulated statistic and is part of the schema. ──
module {
  func.func @modes(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %0 = "tessera.matmul"(%a, %b) {
      numeric_policy = {storage = "fp32", accum = "fp32", math_mode = "tf32",
                        rounding = "round_to_nearest_even",
                        softmax = "fp32"}
    } : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
// CHECK-LABEL: func.func @modes
// CHECK: math_mode = "tf32"
// CHECK-SAME: rounding = "round_to_nearest_even"
// CHECK-SAME: softmax = "fp32"

// -----

// ── equal precision is not narrowing: fp32 into fp32, bf16 into bf16 ──
module {
  func.func @equal(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %0 = "tessera.matmul"(%a, %b) {
      numeric_policy = {storage = "bf16", accum = "bf16"}
    } : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %0 : tensor<64x64xbf16>
  }
}
// CHECK-LABEL: func.func @equal

// -----

// ── the FULL canonical policy: quantization and determinism ──
// PR #631 review. The first version of this schema derived its key set from
// the policies that appear in fixtures rather than from the normative
// definition — `NumericPolicy(storage, accum, rounding, scale, quant_axis,
// deterministic[, math_mode])` in
// docs/reference/tessera_tensor_attributes.md and
// python/tessera/compiler/primitive_coverage.py. It therefore invented
// `rounding_mode` for the canonical `rounding` and omitted `scale`,
// `quant_axis`, `deterministic` and `scale_layout` outright. Every in-tree
// fixture passed, because none carries a quantization or determinism policy —
// and the production legality pipeline would have rejected the first one that
// did.
//
// The value KIND matters for the same reason the key set does: `quant_axis` is
// an integer, `deterministic` a boolean and `scale_layout` a nested
// dictionary, so a blanket "every value is a string" rule refuses three
// canonical fields. Checking the declared kind per key is what makes the
// wrongly-typed-value rule correct rather than merely strict.
module {
  func.func @canonical_quant(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>)
      -> tensor<64x64xf32> {
    %0 = "tessera.matmul"(%a, %b) {numeric_policy = {
      storage = "int8", accum = "int32",
      rounding = "round_to_nearest_even",
      scale = "per_channel", quant_axis = 1 : i64,
      deterministic = true, math_mode = "ieee",
      scale_layout = {granularity = "per_channel"}
    }} : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
// CHECK-LABEL: func.func @canonical_quant
// MLIR prints dictionary entries sorted by name, so the checks follow that
// order rather than the order written above.
// CHECK: deterministic = true
// CHECK-SAME: quant_axis = 1 : i64
// CHECK-SAME: scale_layout = {granularity = "per_channel"}
