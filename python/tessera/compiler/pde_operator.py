"""Typed, exact-rational PDE operator and stability contracts.

This module is deliberately independent of NumPy and backend runtimes.  It is
the compiler authority for constant-coefficient scalar second-order operators;
architecture packages may consume its verdicts but may not recreate them with
floating-point tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction


class PDEClassification(str, Enum):
    ELLIPTIC = "elliptic"
    PARABOLIC = "parabolic"
    HYPERBOLIC = "hyperbolic"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PDETerm:
    """One ``a_alpha D^alpha`` term with an exact rational coefficient."""

    derivative_orders: tuple[int, ...]
    coefficient: Fraction

    def __post_init__(self) -> None:
        if not self.derivative_orders or any(v < 0 for v in self.derivative_orders):
            raise ValueError("PDE derivative orders must be a non-empty nonnegative tuple")
        if self.coefficient.denominator <= 0:
            raise ValueError("PDE coefficient denominator must be positive")

    @property
    def order(self) -> int:
        return sum(self.derivative_orders)


@dataclass(frozen=True)
class PDEOperator:
    """Typed scalar constant-coefficient differential operator.

    Classification is derived from ``terms`` and is intentionally absent from
    the type identity.  A variable coefficient must use the future field-valued
    ABI and therefore cannot accidentally inherit this constant-coefficient
    proof.
    """

    domain_rank: int
    max_order: int
    element_dtype: str
    terms: tuple[PDETerm, ...]

    def __post_init__(self) -> None:
        if self.domain_rank <= 0 or self.max_order <= 0:
            raise ValueError("PDE rank and maximum order must be positive")
        if self.element_dtype not in {"f32", "f64"}:
            raise ValueError("exact PDE analysis currently admits only f32/f64 storage")
        if not self.terms:
            raise ValueError("PDE operator requires at least one term")
        for term in self.terms:
            if len(term.derivative_orders) != self.domain_rank:
                raise ValueError("PDE term rank does not match operator domain rank")
            if term.order > self.max_order:
                raise ValueError("PDE term exceeds the declared maximum order")

    def principal_matrix(self) -> tuple[tuple[Fraction, ...], ...]:
        """Return the exact symmetric matrix of the second-order symbol.

        For a mixed term ``c D_i D_j``, the symmetric quadratic-form entries
        are ``c/2`` at ``(i,j)`` and ``(j,i)``.
        """
        if self.max_order != 2:
            raise ValueError("principal-matrix classification requires order two")
        matrix = [[Fraction(0) for _ in range(self.domain_rank)]
                  for _ in range(self.domain_rank)]
        for term in self.terms:
            if term.order != 2:
                continue
            active = [i for i, order in enumerate(term.derivative_orders) for _ in range(order)]
            if len(active) != 2:
                raise AssertionError("validated second-order multi-index is malformed")
            i, j = active
            if i == j:
                matrix[i][i] += term.coefficient
            else:
                matrix[i][j] += term.coefficient / 2
                matrix[j][i] += term.coefficient / 2
        return tuple(tuple(row) for row in matrix)

    def classify(self) -> PDEClassification:
        pos, neg, zero = exact_symmetric_inertia(self.principal_matrix())
        rank = self.domain_rank
        if zero == 0 and (pos == rank or neg == rank):
            return PDEClassification.ELLIPTIC
        if zero > 0 and (pos == rank - zero or neg == rank - zero):
            return PDEClassification.PARABOLIC
        if zero == 0 and min(pos, neg) == 1:
            return PDEClassification.HYPERBOLIC
        return PDEClassification.MIXED


def exact_symmetric_inertia(
    values: tuple[tuple[Fraction, ...], ...]
) -> tuple[int, int, int]:
    """Compute positive/negative/zero inertia by exact congruence elimination."""
    n = len(values)
    if any(len(row) != n for row in values):
        raise ValueError("principal matrix must be square")
    matrix = [list(row) for row in values]
    indices = list(range(n))
    pos = neg = zero = 0
    while indices:
        pivot = next((i for i in indices if matrix[i][i] != 0), None)
        if pivot is None:
            pair = next(((i, j) for i in indices for j in indices
                         if i < j and matrix[i][j] != 0), None)
            if pair is None:
                zero += len(indices)
                break
            # Use a genuine symmetric 2x2 pivot.  Counting the block and merely
            # deleting it would be wrong when it is coupled to the remaining
            # rows; its exact Schur complement is D - C^T B^-1 C with
            # B^-1=[[0,1/b],[1/b,0]].
            pos += 1
            neg += 1
            i, j = pair
            b = matrix[i][j]
            rest = [k for k in indices if k not in pair]
            for k in rest:
                for ell in rest:
                    matrix[k][ell] -= (
                        matrix[k][i] * matrix[j][ell]
                        + matrix[k][j] * matrix[i][ell]
                    ) / b
            indices.remove(i)
            indices.remove(j)
            continue
        diagonal = matrix[pivot][pivot]
        pos += diagonal > 0
        neg += diagonal < 0
        rest = [i for i in indices if i != pivot]
        for i in rest:
            for j in rest:
                matrix[i][j] -= matrix[i][pivot] * matrix[pivot][j] / diagonal
        indices = rest
    return int(pos), int(neg), zero


@dataclass(frozen=True)
class ExplicitDiffusionCertificate:
    stable: bool
    dt: Fraction
    dt_max: Fraction
    classification: PDEClassification
    proof: str = "exact_diagonal_diffusion_ftcs_v1"


def certify_explicit_diagonal_diffusion(
    operator: PDEOperator, *, spacing: tuple[Fraction, ...], dt: Fraction
) -> ExplicitDiffusionCertificate:
    """Prove the centered FTCS bound ``dt <= 1/(2 sum_i a_i/h_i^2)``.

    Anything outside the diagonal positive-diffusion envelope fails closed.
    Negative-definite operators are normalized by a global sign; mixed terms,
    missing directions, and non-elliptic symbols are rejected rather than
    sampled numerically.
    """
    if len(spacing) != operator.domain_rank or any(h <= 0 for h in spacing):
        raise ValueError("PDE spacing must be positive and match the domain rank")
    if dt <= 0:
        raise ValueError("PDE timestep must be positive")
    classification = operator.classify()
    if classification is not PDEClassification.ELLIPTIC:
        raise ValueError("explicit diffusion certificate requires an elliptic operator")
    matrix = operator.principal_matrix()
    if any(matrix[i][j] != 0 for i in range(operator.domain_rank)
           for j in range(operator.domain_rank) if i != j):
        raise ValueError("mixed derivatives require a separate stability certificate")
    diagonal = [matrix[i][i] for i in range(operator.domain_rank)]
    sign = 1 if all(v > 0 for v in diagonal) else -1
    rates = [sign * value / (h * h) for value, h in zip(diagonal, spacing)]
    if any(rate <= 0 for rate in rates):
        raise ValueError("diffusion principal coefficients must share one nonzero sign")
    dt_max = Fraction(1, 2) / sum(rates, Fraction(0))
    return ExplicitDiffusionCertificate(
        stable=dt <= dt_max,
        dt=dt,
        dt_max=dt_max,
        classification=classification,
    )


__all__ = [
    "ExplicitDiffusionCertificate",
    "PDEClassification",
    "PDEOperator",
    "PDETerm",
    "certify_explicit_diagonal_diffusion",
    "exact_symmetric_inertia",
]
