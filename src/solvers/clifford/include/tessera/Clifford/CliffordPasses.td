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

    Downstream lowering passes (GA8) gate on `canonical=true` and refuse to
    proceed on out-of-allow-list signatures, emitting a precise diagnostic
    naming the op and the unsupported signature.
  }];
}

def CliffordExpandProductTable : Pass<"tessera-clifford-expand-product-table", "mlir::ModuleOp"> {
  let summary =
    "[GA8 stub] Lower clifford.geo_product to a compile-time-known sparse "
    "contraction using the algebra's Cayley table.";
  let description = [{
    Implementation pending GA8. The GA7 stub emits a remark on each
    geo_product op describing the lowering plan; no IR rewriting yet.
  }];
}

def CliffordGradeFusion : Pass<"tessera-clifford-grade-fusion", "mlir::ModuleOp"> {
  let summary =
    "[GA8 stub] Fuse chains `grade(k, geo_product(a, b))` into a single "
    "restricted contraction that only emits the grade-k slice.";
}

def CliffordRotorSandwichFold : Pass<"tessera-clifford-rotor-sandwich-fold", "mlir::ModuleOp"> {
  let summary =
    "[GA8 stub] Recognize R · x · R† chains and emit a direct rotor-"
    "conjugation kernel instead of two geometric products.";
}
