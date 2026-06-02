//===- CliffordOps.td --------------------------------------*- tablegen -*-===//
// Tessera Clifford / Geometric Algebra Dialect.
//
// 17 ops mirroring the GA4 registered surface (`category="geometric_algebra"`
// in `python/tessera/compiler/primitive_coverage.py`).  Each op carries an
// `algebra` attribute (`#clifford.algebra<p, q, r>`) and an optional
// `grades` attribute restricting the input/output grades.  Multivector
// operands are typed as generic `AnyType` for v1 — the GA8 lowering passes
// interpret the layout (linearized coefficient vector, last-axis blade
// order) before emitting backend kernels.
//
// Naming maps 1:1 to the Python surface in `tessera.ga.ops`:
//
//   clifford.geo_product     ↔ geometric_product
//   clifford.grade           ↔ grade_projection
//   clifford.wedge           ↔ wedge
//   clifford.left_contract   ↔ left_contraction
//   clifford.inner           ↔ inner
//   clifford.reverse         ↔ reverse
//   clifford.grade_involute  ↔ grade_involution
//   clifford.conjugate       ↔ conjugate
//   clifford.hodge_star      ↔ hodge_star  (GA5)
//   clifford.norm            ↔ norm
//   clifford.exp             ↔ exp_mv
//   clifford.log             ↔ log_mv
//   clifford.rotor_sandwich  ↔ rotor_sandwich
//   clifford.ext_deriv       ↔ ext_deriv   (GA5)
//   clifford.codiff          ↔ codiff      (GA5)
//   clifford.vec_deriv       ↔ vec_deriv   (GA5)
//   clifford.integral        ↔ integral    (GA5)
//
// See `docs/audit/domain/DOMAIN_AUDIT.md` § GA7.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Clifford_Dialect : Dialect {
  let name = "tessera_clifford";
  let cppNamespace = "::tessera::clifford";
  let summary =
    "Tessera Geometric Algebra (Clifford) dialect — signature-parameterized "
    "multivector operations.";
  let description = [{
    Operations operate on multivector values typed as opaque `AnyType` and
    carry their `Cl(p, q, r)` signature as an op-level attribute. The v1
    allow-list per `docs/audit/domain/DOMAIN_AUDIT.md` § Q1 is Cl(3,0) and
    Cl(1,3); the dialect parses any non-negative (p, q, r) but downstream
    passes restrict to the allowed signatures via the annotation pass.
  }];
}

//===----------------------------------------------------------------------===//
// Common attribute conventions
//
// `algebra` is an `I64ArrayAttr` `[p, q, r]` carrying the signature; v1
// allow-list (per ga_scope_lock.md § Q1) is {(3,0,0), (1,3,0)} enforced by
// `AnnotateAlgebraPass`. `dtype` is a `StrAttr` from the canonical set
// {"fp32", "fp16", "bf16", "fp64"}.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Core multivector ops (GA3)
//===----------------------------------------------------------------------===//

class CliffordOp<string mnemonic, list<Trait> traits = []>
    : Op<Clifford_Dialect, mnemonic, traits>;

def CL_GeoProductOp : CliffordOp<"geo_product", [Pure]> {
  let summary = "Geometric (Clifford) product a · b.";
  let description = [{
    Bilinear. For grade-pure inputs, the result has components at every
    grade reachable by the Cayley table of the algebra. Lowered by
    GA8's `-clifford-expand-product-table` pass to a compile-time-known
    sparse contraction.
  }];
  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    I64ArrayAttr:$algebra,
    StrAttr:$dtype
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs `algebra` $algebra `dtype` $dtype
    attr-dict `:` functional-type(operands, results)
  }];
}

def CL_GradeProjectionOp : CliffordOp<"grade", [Pure]> {
  let summary = "Project a multivector onto a grade subset.";
  let arguments = (ins
    AnyType:$value,
    I64ArrayAttr:$grades,
    I64ArrayAttr:$algebra,
    StrAttr:$dtype
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $value `grades` $grades `algebra` $algebra `dtype` $dtype
    attr-dict `:` functional-type(operands, results)
  }];
}

def CL_WedgeOp : CliffordOp<"wedge", [Pure]> {
  let summary = "Outer (exterior) product a ∧ b.";
  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    I64ArrayAttr:$algebra,
    StrAttr:$dtype
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs `algebra` $algebra `dtype` $dtype
    attr-dict `:` functional-type(operands, results)
  }];
}

def CL_LeftContractOp : CliffordOp<"left_contract", [Pure]> {
  let summary = "Left contraction a ⌋ b.";
  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    I64ArrayAttr:$algebra,
    StrAttr:$dtype
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs `algebra` $algebra `dtype` $dtype
    attr-dict `:` functional-type(operands, results)
  }];
}

def CL_InnerOp : CliffordOp<"inner", [Pure]> {
  let summary = "Scalar inner product ⟨a, b⟩ = scalar_part(a · reverse(b)).";
  let description = [{
    Result is scalar (rank-0 tensor or scalar value), not a multivector.
  }];
  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    I64ArrayAttr:$algebra,
    StrAttr:$dtype
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs `algebra` $algebra `dtype` $dtype
    attr-dict `:` functional-type(operands, results)
  }];
}

def CL_NormOp : CliffordOp<"norm", [Pure]> {
  let summary = "Multivector norm |a| = sqrt(⟨a, a⟩).";
  let arguments = (ins
    AnyType:$value,
    I64ArrayAttr:$algebra,
    StrAttr:$dtype
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $value `algebra` $algebra `dtype` $dtype
    attr-dict `:` functional-type(operands, results)
  }];
}

def CL_ReverseOp : CliffordOp<"reverse", [Pure]> {
  let summary = "Reversion anti-automorphism a†.";
  let arguments = (ins
    AnyType:$value,
    I64ArrayAttr:$algebra,
    StrAttr:$dtype
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $value `algebra` $algebra `dtype` $dtype
    attr-dict `:` functional-type(operands, results)
  }];
}

def CL_GradeInvoluteOp : CliffordOp<"grade_involute", [Pure]> {
  let summary = "Grade-involution â (sign flip on odd grades).";
  let arguments = (ins
    AnyType:$value,
    I64ArrayAttr:$algebra,
    StrAttr:$dtype
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $value `algebra` $algebra `dtype` $dtype
    attr-dict `:` functional-type(operands, results)
  }];
}

def CL_ConjugateOp : CliffordOp<"conjugate", [Pure]> {
  let summary = "Clifford conjugation: reverse ∘ grade_involute.";
  let arguments = (ins
    AnyType:$value,
    I64ArrayAttr:$algebra,
    StrAttr:$dtype
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $value `algebra` $algebra `dtype` $dtype
    attr-dict `:` functional-type(operands, results)
  }];
}

def CL_HodgeStarOp : CliffordOp<"hodge_star", [Pure]> {
  let summary = "Hodge star ⋆ω = reverse(ω) · I.";
  let arguments = (ins
    AnyType:$value,
    I64ArrayAttr:$algebra,
    StrAttr:$dtype
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $value `algebra` $algebra `dtype` $dtype
    attr-dict `:` functional-type(operands, results)
  }];
}

def CL_ExpOp : CliffordOp<"exp", [Pure]> {
  let summary =
    "Power-series exponential exp(a); closed-form for Cl(3,0) bivectors.";
  let arguments = (ins
    AnyType:$value,
    I64ArrayAttr:$algebra,
    StrAttr:$dtype,
    DefaultValuedAttr<I64Attr, "24">:$terms
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $value `algebra` $algebra `dtype` $dtype `terms` $terms
    attr-dict `:` functional-type(operands, results)
  }];
}

def CL_LogOp : CliffordOp<"log", [Pure]> {
  let summary = "Logarithm log(a); closed-form for Cl(3,0) rotors.";
  let arguments = (ins
    AnyType:$value,
    I64ArrayAttr:$algebra,
    StrAttr:$dtype,
    DefaultValuedAttr<I64Attr, "64">:$terms
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $value `algebra` $algebra `dtype` $dtype `terms` $terms
    attr-dict `:` functional-type(operands, results)
  }];
}

def CL_RotorSandwichOp : CliffordOp<"rotor_sandwich", [Pure]> {
  let summary = "Apply R x R† — rotor conjugation rotation of x.";
  let description = [{
    The headline equivariance-from-algebra primitive (Decision GA-L4).
    Recognized by GA8's `-clifford-rotor-sandwich-fold` pass and
    lowered to a fused kernel that bypasses the two intermediate
    geometric products.
  }];
  let arguments = (ins
    AnyType:$rotor,
    AnyType:$value,
    I64ArrayAttr:$algebra,
    StrAttr:$dtype
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $rotor `,` $value `algebra` $algebra `dtype` $dtype
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Differential-form ops (GA5)
//===----------------------------------------------------------------------===//

def CL_ExtDerivOp : CliffordOp<"ext_deriv", [Pure]> {
  let summary = "Exterior derivative dω of a sampled multivector field.";
  let description = [{
    Operates on a `MultivectorField` (sampled field on a uniform Euclidean
    grid). `spacing` is the per-spatial-axis step size; the lowering pass
    generates central-difference kernels.
  }];
  let arguments = (ins
    AnyType:$field,
    I64ArrayAttr:$algebra,
    StrAttr:$dtype,
    F64ArrayAttr:$spacing
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $field `algebra` $algebra `dtype` $dtype `spacing` $spacing
    attr-dict `:` functional-type(operands, results)
  }];
}

def CL_CodiffOp : CliffordOp<"codiff", [Pure]> {
  let summary = "Codifferential d*ω = ±⋆ d ⋆ ω.";
  let arguments = (ins
    AnyType:$field,
    I64ArrayAttr:$algebra,
    StrAttr:$dtype,
    F64ArrayAttr:$spacing
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $field `algebra` $algebra `dtype` $dtype `spacing` $spacing
    attr-dict `:` functional-type(operands, results)
  }];
}

def CL_VecDerivOp : CliffordOp<"vec_deriv", [Pure]> {
  let summary = "Geometric gradient ∂F = Σ_i e_i ∂_i F on a sampled field.";
  let arguments = (ins
    AnyType:$field,
    I64ArrayAttr:$algebra,
    StrAttr:$dtype,
    F64ArrayAttr:$spacing
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $field `algebra` $algebra `dtype` $dtype `spacing` $spacing
    attr-dict `:` functional-type(operands, results)
  }];
}

def CL_IntegralOp : CliffordOp<"integral", [Pure]> {
  let summary = "Riemann-sum integration over a Manifold.";
  let description = [{
    `manifold_kind` identifies the manifold type (`"euclidean"`, `"sphere"`,
    `"son"`). `weights` is an attribute pointer (string) the runtime
    resolves to the manifold's discretization weights.
  }];
  let arguments = (ins
    AnyType:$integrand,
    I64ArrayAttr:$algebra,
    StrAttr:$dtype,
    StrAttr:$manifold_kind,
    StrAttr:$weights_ref
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $integrand `algebra` $algebra `dtype` $dtype `manifold` $manifold_kind
    `weights` $weights_ref attr-dict `:` functional-type(operands, results)
  }];
}
