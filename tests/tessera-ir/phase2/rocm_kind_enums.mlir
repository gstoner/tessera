// RUN: tessera-opt %s | FileCheck %s
//
// W1.1b — `$kind` on the three ops that failed OPEN.
//
// 17 ops carry `$kind` (the plan says 4; three of them are `I64Attr`, not
// strings). Each was probed with a bogus kind, inspecting the OUTPUT rather
// than the exit code: 14 already reject it with a named error. These three
// returned 0 AND generated a complete GPU kernel.
//
// This is the accepting half, and it is the half that matters most here: each
// set includes the ELSE BRANCH'S OWN NAME, and those are exactly the values a
// `kind ==` scan of the consumer does not see. `isfinite`, `geometric_product`
// and `adam` are all real values producers emit, and a constraint derived from
// the obvious grep would have rejected them.

// CHECK-LABEL: func.func @predicate_kinds
func.func @predicate_kinds() {
  // CHECK: tessera_rocm.predicate
  "tessera_rocm.predicate"() { name = "a", kind = "isnan", dtype = "f32" } : () -> ()
  "tessera_rocm.predicate"() { name = "b", kind = "isinf", dtype = "f32" } : () -> ()
  // `isfinite` is the trailing `else` in emitPredBody -- a real predicate that
  // never appears in a `kind ==` comparison.
  // CHECK: kind = "isfinite"
  "tessera_rocm.predicate"() { name = "c", kind = "isfinite", dtype = "f32" } : () -> ()
  return
}

// -----

// CHECK-LABEL: func.func @clifford_kinds
func.func @clifford_kinds() {
  // `geometric_product` is `tableFor`'s trailing `else { *t = GP; }`. It is the
  // most consequential of the three unnamed defaults: an unrecognised kind
  // silently computed a DIFFERENT ALGEBRA OPERATION.
  // CHECK: kind = "geometric_product"
  "tessera_rocm.clifford"() { name = "a", kind = "geometric_product", dtype = "f32" } : () -> ()
  "tessera_rocm.clifford"() { name = "b", kind = "wedge", dtype = "f32" } : () -> ()
  "tessera_rocm.clifford"() { name = "c", kind = "left_contraction", dtype = "f32" } : () -> ()
  return
}

// -----

// CHECK-LABEL: func.func @optimizer_kinds
func.func @optimizer_kinds() {
  "tessera_rocm.optimizer"() { name = "a", kind = "sgd", dtype = "f32" } : () -> ()
  "tessera_rocm.optimizer"() { name = "b", kind = "momentum", dtype = "f32" } : () -> ()
  "tessera_rocm.optimizer"() { name = "c", kind = "nesterov", dtype = "f32" } : () -> ()
  "tessera_rocm.optimizer"() { name = "d", kind = "lion", dtype = "f32" } : () -> ()
  // `adam` is the fallthrough: `if (kind == "adamw")` only adds decoupled
  // weight decay, so plain Adam is the path an unmatched kind reached. It is
  // emitted by `runtime.py` in four places -- omitting it would have broken a
  // real optimizer, the same way an earlier `mode` set omitting "set" did.
  // CHECK: kind = "adam"
  "tessera_rocm.optimizer"() { name = "e", kind = "adam", dtype = "f32" } : () -> ()
  "tessera_rocm.optimizer"() { name = "f", kind = "adamw", dtype = "f32" } : () -> ()
  // CHECK: kind = "adafactor"
  "tessera_rocm.optimizer"() { name = "g", kind = "adafactor", dtype = "f32" } : () -> ()
  return
}
