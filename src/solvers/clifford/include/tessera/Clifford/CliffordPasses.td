//===- CliffordPasses.td ----------------------------------*- tablegen -*-===//
// Pass declarations for Tessera Clifford / Geometric Algebra dialect.
//
// GA7 ships a single annotation pass — `tessera-clifford-annotate-algebra` —
// that resolves the algebra signature on every clifford op and attaches
// derived metadata (`tessera.clifford.dim`, per-op `tessera.clifford.grades`,
// signature validation against the v1 allow-list).
//
// GA8 (lowering passes — declared here as stubs, implementations come with
// the GA8 sprint):
//   - tessera-clifford-expand-product-table
//   - tessera-clifford-grade-fusion
//   - tessera-clifford-rotor-sandwich-fold
//
// GA8 stubs land in this file so the pass pipeline is discoverable from
// `ts-clifford-opt --help` even before the bodies are filled in. Each stub
// pass is a no-op walk that emits a diagnostic note explaining where the
// real implementation will go.
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def CliffordAnnotateAlgebra : Pass<"tessera-clifford-annotate-algebra", "mlir::ModuleOp"> {
  let summary =
    "Resolve the algebra signature on every tessera_clifford.* op and "
    "attach derived metadata (dim, allow-list flag, default grade set).";
  let description = [{
    Walks every clifford op, reads its `algebra` attribute, validates it
    against the v1 allow-list (Cl(3,0) and Cl(1,3)), and attaches:

      tessera.clifford.dim          : 2^(p+q+r)  — algebra dimension
      tessera.clifford.allow_listed : bool — true iff in v1 allow-list
      tessera.clifford.canonical    : bool — true iff signature is well-formed

    NOTE (W0.7): the GA8 passes below do NOT yet gate on `canonical` -- they
    are annotation-only and reference the attribute nowhere. An earlier
    version of this text asserted that they "gate on `canonical=true` and
    refuse to proceed on out-of-allow-list signatures"; that was never true.
    `tessera.clifford.canonical` is therefore a declaration without a
    consumer (Decision #29). Either the gate lands with the real GA8
    lowering, or the attribute is deleted -- it must not read as an enforced
    contract in the meantime.
  }];
}

def CliffordExpandProductTable : Pass<"tessera-clifford-expand-product-table", "mlir::ModuleOp"> {
  let summary =
    "[annotation-only] Emit a remark per clifford.geo_product describing the "
    "planned Cayley-table lowering. Rewrites no IR.";
  let description = [{
    W0.7 -- "stub" was ambiguous between "does nothing" and "does something
    partial". This pass is ANNOTATION-ONLY: it emits a remark on each
    geo_product op describing the intended lowering and performs no IR
    rewriting. It does not read `tessera.clifford.canonical`.

    The real lowering (compile-time-known sparse contraction from the
    algebra's Cayley table) is GA8, and batched operands are W3.6.
  }];
}

def CliffordGradeFusion : Pass<"tessera-clifford-grade-fusion", "mlir::ModuleOp"> {
  let summary =
    "[annotation-only] Recognize `grade(k, geo_product(a, b))` chains. "
    "Rewrites no IR; the restricted-contraction lowering is GA8.";
}

def CliffordRotorSandwichFold : Pass<"tessera-clifford-rotor-sandwich-fold", "mlir::ModuleOp"> {
  let summary =
    "[annotation-only] Mark R · x · R† chains. Rewrites no IR -- the marker "
    "it writes has no consumer yet (W3.6); the direct rotor-conjugation "
    "kernel is GA8.";
}
