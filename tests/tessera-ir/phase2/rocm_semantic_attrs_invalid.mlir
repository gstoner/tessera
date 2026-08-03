// RUN: not tessera-opt %s 2>&1 | FileCheck %s
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
  // CHECK: attribute 'dtype' failed to satisfy constraint: ROCm kernel storage dtype
  "tessera_rocm.binary_loss"() {
    name = "bce", dtype = "float128", reduction = "mean"
  } : () -> ()
  return
}
