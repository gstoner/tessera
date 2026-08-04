// RUN: not tessera-opt %s --split-input-file \
// RUN:   --tessera-verify-metadata-obligation 2>&1 | FileCheck %s
//
// W1.3 / Decision #32 — the five rejections.
//
// Decision #10a: a pass that marks or permits ships a fixture whose correct
// output is a refusal. Four of these five are cases where the permissive answer
// is the tempting one, and all four fail CLOSED.

// 1. The defect the whole item exists for: an attribute present before the
// boundary, absent after, with nothing recorded.
module attributes {tessera.metadata_snapshot = {silent = {numeric_policy = ["{accum = \22fp32\22}", 1]}}} {
  // CHECK: METADATA_OBLIGATION_SILENT_DROP
  // CHECK-SAME: `numeric_policy`
  func.func @silent(%a: tensor<8xf32>) -> tensor<8xf32> {
    return %a : tensor<8xf32>
  }
}

// -----

// 2. A reason outside the closed set. Decision #21a: the drop reason SELECTS
// SEMANTICS -- it is the difference between "this moved into the type" and
// "nobody has done this yet" -- so an unrecognised value is an error, never a
// permissive default.
module attributes {tessera.metadata_snapshot = {bad_reason = {layout = ["\22row_major\22", 1]}}} {
  // CHECK: METADATA_OBLIGATION_UNKNOWN_REASON
  // CHECK-SAME: not_needed
  func.func @bad_reason(%a: tensor<8xf32>) -> tensor<8xf32>
      attributes {tessera.lowering.dropped = {layout = "not_needed"}} {
    return %a : tensor<8xf32>
  }
}

// -----

// 3. Debt with no owner. A bare `not_yet_carried` is a silent drop with extra
// syntax: it satisfies the verifier's letter, records nothing actionable, and
// would sit in the tree indefinitely because no item is on the hook for it.
module attributes {tessera.metadata_snapshot = {unowned_debt = {distribution = ["\22block\22", 1]}}} {
  // CHECK: METADATA_OBLIGATION_DEBT_UNATTRIBUTED
  func.func @unowned_debt(%a: tensor<8xf32>) -> tensor<8xf32>
      attributes {
        tessera.lowering.dropped = {distribution = "not_yet_carried"}
      } {
    return %a : tensor<8xf32>
  }
}

// -----

// 4. A declared drop that did not happen -- Decision #29 applied to this
// mechanism itself. The attribute is still present, so the exception carries
// nothing; but it reads in review as a considered decision, and it would
// silently license a REAL future drop of `layout` that nobody looked at.
module attributes {tessera.metadata_snapshot = {stale = {layout = ["\22row_major\22", 1]}}} {
  // CHECK: METADATA_OBLIGATION_STALE_DECLARATION
  func.func @stale(%a: tensor<8xf32>) -> tensor<8xf32>
      attributes {tessera.lowering.dropped = {layout = "consumed_by_pass"}} {
    %0 = "tessera.cast"(%a) {layout = "row_major"} : (tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

// -----

// 5. Verify with no snapshot. The dangerous default is success: the pass would
// be green on every pipeline that forgot to record, which is indistinguishable
// from a pipeline that has no losses. Fail closed instead -- an unrun check
// must not look like a passed one.
// CHECK: METADATA_OBLIGATION_NO_SNAPSHOT
module {
  func.func @no_snapshot(%a: tensor<8xf32>) -> tensor<8xf32> {
    return %a : tensor<8xf32>
  }
}

// -----

// 6. A value REPLACED while the name survives. PR #500 review: the first
// version snapshotted only NAMES, so `accum = "fp32"` becoming `accum = "fp16"`
// moved no name and fired nothing — precisely the instruction-selection
// corruption this verifier exists to prevent, invisible to it. The same
// name-only blindness also covered a lost occurrence when a sibling op kept
// the attribute, which is why the snapshot counts values.
module attributes {
  tessera.metadata_snapshot = {
    replaced = {numeric_policy = ["{accum = \22fp32\22}", 1]}}
} {
  // CHECK: METADATA_OBLIGATION_VALUE_DROP
  func.func @replaced(%a: tensor<8xf32>) -> tensor<8xf32> {
    %0 = "tessera.cast"(%a) {numeric_policy = {accum = "fp16"}}
      : (tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

// -----

// 7. A declaration for an attribute that was NEVER present. Also PR #500
// review, and the more dangerous stale shape: it looks harmless right up until
// the function acquires that attribute, at which point it silently licenses a
// real drop nobody reviewed.
module attributes {
  tessera.metadata_snapshot = {
    never_had_it = {numeric_policy = ["{accum = \22fp32\22}", 1]}}
} {
  // CHECK: METADATA_OBLIGATION_STALE_DECLARATION
  // CHECK-SAME: never present before the boundary either
  func.func @never_had_it(%a: tensor<8xf32>) -> tensor<8xf32>
      attributes {tessera.lowering.dropped = {layout = "consumed_by_pass"}} {
    %0 = "tessera.cast"(%a) {numeric_policy = {accum = "fp32"}}
      : (tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

// -----

// 8. `re_expressed` claimed when the attribute is gone entirely. The reason
// asserts the VALUE was re-encoded; if the name is gone too, nothing was
// re-expressed and it is a silent drop wearing the most permissive label.
module attributes {
  tessera.metadata_snapshot = {
    nothing_to_re_express = {layout = ["\22row_major\22", 1]}}
} {
  // CHECK: METADATA_OBLIGATION_UNKNOWN_REASON
  // CHECK-SAME: nothing was re-expressed
  func.func @nothing_to_re_express(%a: tensor<8xf32>) -> tensor<8xf32>
      attributes {tessera.lowering.dropped = {layout = "re_expressed"}} {
    return %a : tensor<8xf32>
  }
}
