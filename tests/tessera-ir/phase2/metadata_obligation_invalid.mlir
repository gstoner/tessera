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
module attributes {tessera.metadata_snapshot = {silent = ["numeric_policy"]}} {
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
module attributes {tessera.metadata_snapshot = {bad_reason = ["layout"]}} {
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
module attributes {tessera.metadata_snapshot = {unowned_debt = ["distribution"]}} {
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
module attributes {tessera.metadata_snapshot = {stale = ["layout"]}} {
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
