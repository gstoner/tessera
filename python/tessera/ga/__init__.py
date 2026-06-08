"""Tessera geometric-algebra namespace (Clifford algebras).

This module is the entry point for the GA-series primitive surface
sequenced in `docs/audit/domain/DOMAIN_AUDIT.md`. GA0 (scope lock) ships the
namespace; GA1 onwards populates it with the algebra signature object,
multivector type, grade-aware constraints, and the Clifford primitive
library.

Scope-locked at GA0:
    - First-class signatures: Cl(3,0) and Cl(1,3) only for v1.
    - Multivector is a sibling tensor kind, not a 7th tensor attribute.
    - Backend order: x86 -> Apple CPU -> Apple GPU -> NVIDIA after Phase G.

See `docs/audit/domain/DOMAIN_AUDIT.md` for the locked decisions and
`docs/audit/domain/DOMAIN_AUDIT.md` for the full sprint sequence.
"""

from tessera.ga.calculus import (
    MultivectorField,
    codiff,
    ext_deriv,
    hodge_star,
    hodge_star_field,
    integral,
    vec_deriv,
)
from tessera.ga.constraints import (
    Even,
    GradeIn,
    IsForm,
    IsRotor,
    Odd,
)
from tessera.ga.manifold import (
    Euclidean,
    Manifold,
    SOn,
    Sphere,
)
from tessera.ga.multivector import (
    Multivector,
    MultivectorSpec,
)
from tessera.ga.ops import (
    conjugate,
    exp_mv,
    geometric_product,
    grade_involution,
    grade_projection,
    inner,
    left_contraction,
    log_mv,
    norm,
    norm_squared,
    reverse,
    rotor_from_axis,
    rotor_sandwich,
    rotor_sandwich_norm,
    wedge,
)
from tessera.ga.signature import (
    Basis,
    Cl,
    TesseraAlgebraError,
    V1_ALLOWED_SIGNATURES,
)
from tessera.ga.types import (
    DiffForm,
    Morphism,
    MorphismSpec,
    Rotor,
    VectorField,
)

__version__ = "0.0.0-ga5"

__all__ = [
    # Signature (GA1)
    "Basis",
    "Cl",
    "TesseraAlgebraError",
    "V1_ALLOWED_SIGNATURES",
    # Multivector value + annotation (GA2 + GA3)
    "Multivector",
    "MultivectorSpec",
    # Annotation markers (GA2)
    "DiffForm",
    "Morphism",
    "MorphismSpec",
    "Rotor",
    "VectorField",
    # Constraint predicates (GA2)
    "Even",
    "GradeIn",
    "IsForm",
    "IsRotor",
    "Odd",
    # Operations (GA3)
    "conjugate",
    "exp_mv",
    "geometric_product",
    "grade_involution",
    "grade_projection",
    "inner",
    "left_contraction",
    "log_mv",
    "norm",
    "norm_squared",
    "reverse",
    "rotor_from_axis",
    "rotor_sandwich",
    "rotor_sandwich_norm",
    "wedge",
    # Differential calculus + manifolds (GA5)
    "MultivectorField",
    "codiff",
    "ext_deriv",
    "hodge_star",
    "hodge_star_field",
    "integral",
    "vec_deriv",
    "Euclidean",
    "Manifold",
    "SOn",
    "Sphere",
    "__version__",
]
