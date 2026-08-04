// RUN: not tessera-opt --split-input-file %s 2>&1 | FileCheck %s
//
// W1.1b — the three `$kind` ops now fail CLOSED (Decision #21a).
//
// Before this, each of these generated a complete GPU kernel and returned 0.
// The bug was not a missing check but a trailing `else` doing double duty as a
// real branch AND an unnamed default, so nothing looked wrong at the call site.

func.func @predicate_bogus_kind() {
  // Silently tested `isfinite`.
  // CHECK: attribute 'kind' failed to satisfy constraint: ROCm predicate kind
  "tessera_rocm.predicate"() { name = "k", kind = "isnormal", dtype = "f32" } : () -> ()
  return
}

// -----

func.func @clifford_bogus_kind() {
  // Silently computed the geometric product -- a different algebra operation,
  // returning plausible numbers for a product nobody asked for.
  // CHECK: attribute 'kind' failed to satisfy constraint: ROCm Cl(3,0)
  "tessera_rocm.clifford"() { name = "k", kind = "inner", dtype = "f32" } : () -> ()
  return
}

// -----

func.func @optimizer_bogus_kind() {
  // Silently trained with Adam.
  //
  // This case originally used `adafactor` as its "plausible but unimplemented"
  // example. It is neither: the ROCm optimizer pass implements it and
  // `test_adafactor_full_backward_executes_on_gfx1151` runs it on hardware. The
  // unit suite caught that the enum had omitted it. An invented token is used
  // here instead, because for THIS op every plausible optimizer name turned out
  // to be real.
  // CHECK: attribute 'kind' failed to satisfy constraint: ROCm optimizer
  "tessera_rocm.optimizer"() { name = "k", kind = "not_an_optimizer", dtype = "f32" } : () -> ()
  return
}
