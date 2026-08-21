"""AD-OPERATOR-1 — derivatives as first-class matrix-free operators.

`OperatorTangent` promotes the matvec pattern `implicit.py` grew locally
(closures for `J @ v` and `Jᵀ @ r`, threaded by hand into CG/GMRES) into a
type: a linear map accessed only through its action, with **composition as
the operator product** and **`.T` as an involution** (AUTODIFF_NEXTGEN_PLAN
§3.5). The adjoint identity ``⟨A v, u⟩ = ⟨v, Aᵀ u⟩`` is a checkable law of
the type itself — the operator-level statement of Law 3, swept by
`tests/unit/test_operator_tangent.py`.

Nothing is materialized: an `OperatorTangent` is callable (`A(v)` is the
matvec), so the existing matrix-free solvers (`implicit.cg_solve`,
`implicit.gmres_solve`) consume it unchanged — solve-consumption is the
third leg of the §3.5 design (IFT, iHVP, Newton–Krylov as compositions fed
to matrix-free solves).

**Decision #31 declaration — relationship to the materialized `@f__bwd`
ABI.** The compiler's paired-function autodiff (W6.1) materializes the
transpose as a sibling function `@f__bwd` at IR level; that is the one
production lowering. `OperatorTangent.T` is the *unmaterialized twin of the
same mathematical object*: the adjoint characterized by ``⟨Jv,u⟩=⟨v,Jᵀu⟩``,
applied through actions instead of a lowered body. Neither replaces the
other: the materialized ABI is where codegen and packaging happen; the
operator form is what matrix-free solves consume (this module, and the
landed AD-SOLVER-IFT-1/W3.5 parent/child packages, which already execute
residual + transposed solve + adjoint on AVX-512/gfx1151 without a dense
Jacobian). This paragraph is the recorded #31 relationship — one production
lowering, one declared operator twin — kept here, next to the type, so it
cannot drift silently.

The H3 well-posedness certificate (`RootConditionCertificate`,
`certify_root`) lives here too: `CORE_SUBSTRATE_VIEW.md` S8's "one fix,
three consumers" item (game theory, Riemannian-OT, S-series). For a root
``F(x*, θ) = 0`` the implicit function theorem needs ``∂ₓF`` nonsingular at
the solution; for KKT-form residuals that is exactly what *strict
complementarity* guarantees — a weakly-active constraint (zero multiplier
on an active constraint) makes the KKT Jacobian singular and the implicit
derivative ill-posed. The certificate measures the hypothesis instead of
assuming it, in the S4 certificate discipline alongside `LinearSolveInfo`:
measured numbers plus a verdict, and **fail-closed** when the check cannot
be evaluated (non-finite residual or Jacobian is an error, never a pass).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .errors import TesseraAutodiffError

__all__ = [
    "OperatorTangent",
    "RootConditionCertificate",
    "certify_root",
]


class TesseraOperatorError(TesseraAutodiffError):
    """Raised on a structurally invalid operator use (shape mismatch,
    transpose of a forward-only operator, composition of incompatible
    maps)."""


@dataclass(frozen=True)
class OperatorTangent:
    """A matrix-free linear map ``v ↦ A v`` with a declared adjoint.

    ``fwd`` maps a vector reshapeable to ``in_shape`` to the flattened
    output; ``adj`` is the transposed action (``None`` for a forward-only
    operator — taking ``.T`` then **fails closed** rather than silently
    finite-differencing an adjoint, unless the operator declares itself
    self-adjoint). Instances are callable: ``A(v)`` is the matvec, so
    `cg_solve`/`gmres_solve` consume an operator anywhere they accepted a
    bare closure.
    """

    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    fwd: Callable[[np.ndarray], np.ndarray]
    adj: Optional[Callable[[np.ndarray], np.ndarray]] = None
    self_adjoint: bool = False
    provenance: str = "numerical_oracle"
    _label: str = field(default="operator", repr=False)

    # ── shape vocabulary ────────────────────────────────────────────────────
    @property
    def shape(self) -> tuple[int, int]:
        m = int(np.prod(self.out_shape)) if self.out_shape else 1
        n = int(np.prod(self.in_shape)) if self.in_shape else 1
        return (m, n)

    # ── application ─────────────────────────────────────────────────────────
    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Apply the map; returns the FLATTENED output (the convention the
        matrix-free solvers already use)."""
        m, n = self.shape
        arr = np.asarray(v, dtype=np.float64)
        if arr.size != n:
            raise TesseraOperatorError(
                f"{self._label}: input has {arr.size} values, expected {n} "
                f"(in_shape {self.in_shape})"
            )
        out = np.asarray(self.fwd(arr.reshape(self.in_shape)),
                         dtype=np.float64).reshape(-1)
        if out.size != m:
            raise TesseraOperatorError(
                f"{self._label}: action returned {out.size} values, "
                f"expected {m} (out_shape {self.out_shape})"
            )
        return out

    __call__ = matvec

    # ── transpose as involution ─────────────────────────────────────────────
    @property
    def T(self) -> "OperatorTangent":
        if self.self_adjoint:
            if self.in_shape != self.out_shape:
                raise TesseraOperatorError(
                    f"{self._label}: declared self-adjoint but maps "
                    f"{self.in_shape} → {self.out_shape}"
                )
            return self
        if self.adj is None:
            raise TesseraOperatorError(
                f"{self._label}: no adjoint action was supplied — refusing "
                f"to invent one (a finite-difference fallback here would be "
                f"a silently different operator; construct the adjoint "
                f"explicitly or declare self_adjoint)"
            )
        return OperatorTangent(
            in_shape=self.out_shape,
            out_shape=self.in_shape,
            fwd=self.adj,
            adj=self.fwd,
            self_adjoint=False,
            provenance=self.provenance,
            _label=f"{self._label}.T",
        )

    # ── algebra ─────────────────────────────────────────────────────────────
    def __matmul__(self, other: Any) -> Any:
        """Operator ∘ operator (composition as product) or operator @ vector
        (application, returning the logical `out_shape`)."""
        if isinstance(other, OperatorTangent):
            if self.shape[1] != other.shape[0]:
                raise TesseraOperatorError(
                    f"cannot compose {self._label}{self.shape} with "
                    f"{other._label}{other.shape}: inner dimensions differ"
                )
            outer, inner = self, other

            def composed_fwd(v: np.ndarray) -> np.ndarray:
                return outer.matvec(inner.matvec(v))

            composed_adj: Optional[Callable[[np.ndarray], np.ndarray]] = None
            if ((inner.adj is not None or inner.self_adjoint)
                    and (outer.adj is not None or outer.self_adjoint)):
                def _adj(u: np.ndarray) -> np.ndarray:
                    return inner.T.matvec(outer.T.matvec(u))

                composed_adj = _adj
            return OperatorTangent(
                in_shape=inner.in_shape,
                out_shape=outer.out_shape,
                fwd=composed_fwd,
                adj=composed_adj,
                self_adjoint=False,
                provenance=(self.provenance
                            if self.provenance == other.provenance
                            else "composed"),
                _label=f"({self._label} @ {other._label})",
            )
        return self.matvec(other).reshape(self.out_shape)

    def __neg__(self) -> "OperatorTangent":
        adj_fn = self.adj
        negated_adj: Optional[Callable[[np.ndarray], np.ndarray]] = None
        if adj_fn is not None:
            def _neg_adj(u: np.ndarray) -> np.ndarray:
                return -np.asarray(adj_fn(u), dtype=np.float64).reshape(-1)

            negated_adj = _neg_adj
        return OperatorTangent(
            in_shape=self.in_shape,
            out_shape=self.out_shape,
            fwd=lambda v: -self.matvec(v),
            adj=negated_adj,
            self_adjoint=self.self_adjoint,
            provenance=self.provenance,
            _label=f"(-{self._label})",
        )

    # ── constructors ────────────────────────────────────────────────────────
    @classmethod
    def from_matvec_pair(
        cls,
        matvec: Callable[[np.ndarray], np.ndarray],
        rmatvec: Optional[Callable[[np.ndarray], np.ndarray]],
        *,
        in_shape: tuple[int, ...],
        out_shape: tuple[int, ...],
        provenance: str = "numerical_oracle",
        label: str = "operator",
    ) -> "OperatorTangent":
        return cls(in_shape=tuple(in_shape), out_shape=tuple(out_shape),
                   fwd=matvec, adj=rmatvec, provenance=provenance,
                   _label=label)

    @classmethod
    def self_adjoint_from(
        cls,
        matvec: Callable[[np.ndarray], np.ndarray],
        *,
        shape: tuple[int, ...],
        provenance: str = "numerical_oracle",
        label: str = "operator",
    ) -> "OperatorTangent":
        """A declared-symmetric map (an HVP, a Gauss–Newton normal
        operator). The declaration is a claim the adjoint law can check."""
        return cls(in_shape=tuple(shape), out_shape=tuple(shape),
                   fwd=matvec, adj=None, self_adjoint=True,
                   provenance=provenance, _label=label)

    def materialize(self) -> np.ndarray:
        """Dense (m, n) matrix via n basis matvecs — the explicit
        small-system oracle (used by the dense solver path and the H3
        certificate), never the default execution path."""
        m, n = self.shape
        basis = np.eye(n, dtype=np.float64)
        return np.column_stack([self.matvec(basis[j]) for j in range(n)])


# ── H3: well-posedness certificate at implicit solutions ─────────────────────


@dataclass(frozen=True)
class RootConditionCertificate:
    """Measured IFT well-posedness at a claimed root ``F(x*, θ) = 0``.

    S4 discipline: numbers plus a verdict, never a bare boolean. ``strict``
    holds iff the point is actually a root (``residual_norm`` within
    ``residual_tol`` of zero, scaled) AND ``∂ₓF`` is non-degenerate there
    (``sigma_min > degeneracy_tol · sigma_max`` — scale-invariant). For a
    KKT residual the second condition is what strict complementarity
    guarantees; its failure is exactly the degenerate case where the
    implicit derivative is ill-posed.
    """

    residual_norm: float
    solution_scale: float
    sigma_min: float
    sigma_max: float
    condition_number: float
    residual_tol: float
    degeneracy_tol: float
    residual_ok: bool
    nondegenerate: bool
    strict: bool
    method: str = "dense_svd_oracle"


def certify_root(
    optimality: Callable[..., np.ndarray],
    solution: np.ndarray,
    params: Sequence[np.ndarray],
    *,
    eps: float = 1e-6,
    residual_tol: float = 1e-6,
    degeneracy_tol: float = 1e-8,
) -> RootConditionCertificate:
    """Measure the IFT hypothesis at ``solution`` instead of assuming it.

    Fail-closed: a residual or Jacobian that cannot be evaluated (raises,
    or produces non-finite values) is an error — the certificate never
    converts "could not check" into a pass (#21a's discipline applied to a
    numeric hypothesis). The Jacobian is materialized through the operator
    oracle (`OperatorTangent.materialize`) — reference-lane cost, matching
    the dense solver path this module already ships.
    """
    from .implicit import TesseraImplicitDiffError, _partial_jacobian_matvecs

    x = np.asarray(solution, dtype=np.float64)
    full_args = [x, *(np.asarray(p, dtype=np.float64) for p in params)]
    try:
        residual = np.asarray(optimality(*full_args), dtype=np.float64)
    except Exception as exc:
        raise TesseraImplicitDiffError(
            f"root certificate cannot be evaluated: the optimality residual "
            f"raised {type(exc).__name__} at the claimed solution"
        ) from exc
    if not np.all(np.isfinite(residual)):
        raise TesseraImplicitDiffError(
            "root certificate cannot be evaluated: the optimality residual "
            "is non-finite at the claimed solution"
        )
    residual_norm = float(np.linalg.norm(residual.reshape(-1)))
    solution_scale = max(1.0, float(np.linalg.norm(x.reshape(-1))))

    matvec, _rmatvec, out_shape, in_shape = _partial_jacobian_matvecs(
        optimality, full_args, 0, eps=eps
    )
    A_op = OperatorTangent.from_matvec_pair(
        matvec, None, in_shape=in_shape, out_shape=out_shape,
        label="∂ₓF",
    )
    dense = A_op.materialize()
    if not np.all(np.isfinite(dense)):
        raise TesseraImplicitDiffError(
            "root certificate cannot be evaluated: ∂ₓF is non-finite at "
            "the claimed solution"
        )
    svals = np.linalg.svd(dense, compute_uv=False)
    sigma_max = float(svals[0]) if svals.size else 0.0
    sigma_min = float(svals[-1]) if svals.size else 0.0
    tiny = float(np.finfo(np.float64).tiny)
    condition = float(sigma_max / max(sigma_min, tiny))

    residual_ok = residual_norm <= residual_tol * solution_scale
    nondegenerate = sigma_min > degeneracy_tol * max(sigma_max, tiny)
    return RootConditionCertificate(
        residual_norm=residual_norm,
        solution_scale=solution_scale,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        condition_number=condition,
        residual_tol=residual_tol,
        degeneracy_tol=degeneracy_tol,
        residual_ok=residual_ok,
        nondegenerate=nondegenerate,
        strict=residual_ok and nondegenerate,
    )
