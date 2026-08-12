// REQUIRES: tessera-rocm-backend
// RUN: not tessera-opt --split-input-file %s 2>&1 | FileCheck %s
//
// W1.1b — semantic string attributes state their legal set and fail closed.
//
// Every ROCm kernel op took a bare `StrAttr` for `dtype`, `reduction` and
// `mode`, and the passes read them as free strings: an unrecognised value fell
// through a chain of `==` comparisons to whatever the last `else` happened to
// do. Decision #21a requires a semantic key to be rejected, not defaulted.
//
// This is the NEGATIVE fixture the positive ones cannot replace (Decision
// #10a): a constraint that only ever accepts proves nothing about what it
// rejects, and all 288 existing fixtures pass unchanged precisely because the
// textual form did not change.

func.func @bad_dtype() {
  // CHECK: attribute 'dtype' failed to satisfy constraint: ROCm float kernel dtype
  "tessera_rocm.binary_loss"() {
    name = "bce", dtype = "float128", reduction = "mean"
  } : () -> ()
  return
}

// -----

// A REAL dtype that is wrong for THIS op — the case a single shared constraint
// could not catch, raised on PR #499. The first cut declared one
// `ROCM_DTypeAttr` over the union of every dtype any ROCm op accepts, so this
// passed ODS and was rejected later inside `GenerateROCMSoftmaxKernel.cpp`,
// which is the layering W1.1b exists to remove. `float128` above does not test
// this: no op accepts it, so a union rejects it too.
func.func @real_dtype_wrong_op() {
  // CHECK: attribute 'dtype' failed to satisfy constraint: ROCm float kernel dtype
  "tessera_rocm.softmax"() { name = "s", dtype = "int8" } : () -> ()
  return
}

// -----

// The other half of the split, and the half that makes it a split rather than
// a narrowing: `compare` maps either kind to a bool, so integers ARE legal
// there (see rocm_semantic_attrs.mlir for the accepting case) — but int8 is
// not among the widths its generator implements.
func.func @compare_takes_ints_but_not_this_one() {
  // CHECK: attribute 'dtype' failed to satisfy constraint: ROCm numeric kernel dtype
  "tessera_rocm.compare"() { name = "c", dtype = "int8" } : () -> ()
  return
}
