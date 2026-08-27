"""Metrics and tangent spaces — the gradient is `metric^-1 . differential` (MC3).

The differential `df = f'(x)[dx]` is a **linear form**, and a gradient is
whatever you take an inner product with to recover it (Bright/Edelman/Johnson
§5.1, via the Riesz representation theorem). So the differential is invariant
and the *gradient is not*: change the inner product and you change the vector.
Under a weighted inner product `<x, y>_W = x^T W y`,

    grad_W f = W^-1 grad f

and on a constraint surface the differential is restricted to the tangent
space, so the gradient is the projection of the ambient one — on the sphere
`x^T x = 1` the constraint linearizes to `x^T dx = 0`, giving
`grad_S f = (I - x x^T) grad f` (§13.1.2); on the orthogonal group `Q^T dQ` is
antisymmetric (§13.2), which is the whole `n(n-1)/2` parameter count.

**Why this module exists (Decision #29).** `manifold` was a declared semantic
key with no consumer, and three places in-tree had independently paid for the
same missing abstraction: `ebm/geo_sampling.py` hand-rolls sphere projection
plus a normalize retraction, `ga/manifold.py` defines `Euclidean`/`Sphere`/
`SOn` as *quadrature domains* (`sample_points()` + `weights()`, no tangent
operations at all), and `optim.muon` is steepest descent under a spectral-norm
geometry without anything saying so. One protocol, named consumers.

The consumer is :func:`tessera.autodiff.grad`, which accepts ``metric=`` and
returns the gradient *in that geometry*. Preconditioning, natural gradient and
Muon are all this one idea: a different metric, not a different derivative.

Scope note: this is the Python reference lane. The MLIR-side `manifold`
attribute still reaches no backend; closing that is a lowering concern, not
this module's.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Metric",
    "Euclidean",
    "Weighted",
    "Sphere",
    "Orthogonal",
    "riemannian_gradient",
]

_METRIC_DIAGNOSTIC_CODE = "E_METRIC_CONTRACT"


@runtime_checkable
class Metric(Protocol):
    """An inner product, plus the tangent structure of its domain.

    Three operations, each with a distinct job:

    ``inner(u, v)``            the inner product itself — what "gradient" is
                               defined relative to.
    ``sharp(covector)``        raise an index: map the differential's
                               representation in the *ambient* basis to the
                               vector whose ``inner`` with ``dx`` reproduces
                               ``df``. This is the ``W^-1`` of §5.1.
    ``project_tangent(x, v)``  restrict to the tangent space at ``x``. The
                               identity for an open domain; a real projection
                               on a constraint surface.

    ``retract(x, v)`` is the step operation an optimizer needs — move from
    ``x`` along ``v`` and land back on the manifold.
    """

    def inner(self, u: np.ndarray, v: np.ndarray) -> float: ...
    def sharp(self, covector: np.ndarray) -> np.ndarray: ...
    def project_tangent(self, x: np.ndarray, v: np.ndarray) -> np.ndarray: ...
    def retract(self, x: np.ndarray, v: np.ndarray) -> np.ndarray: ...


class Euclidean:
    """The default: `<u, v> = sum(u * v)` on an open domain.

    Every operation is the identity, which is exactly why the gradient people
    usually mean is "the" gradient — not because it is canonical, but because
    this metric is assumed.
    """

    def inner(self, u, v) -> float:
        return float(np.sum(np.asarray(u) * np.asarray(v)))

    def sharp(self, covector):
        return np.asarray(covector, dtype=np.float64)

    def project_tangent(self, x, v):
        return np.asarray(v, dtype=np.float64)

    def retract(self, x, v):
        return np.asarray(x, dtype=np.float64) + np.asarray(v, dtype=np.float64)

    def __repr__(self) -> str:
        return "Euclidean()"


class Weighted:
    """`<u, v>_W = u^T W v` for symmetric positive-definite `W` (§5.1).

    ``W`` may be given as a full matrix or as a 1-D array of positive weights
    (the diagonal case, which is what "per-parameter learning rate" means).
    Positive-definiteness is checked, not assumed: an indefinite ``W`` is not
    an inner product, and the "gradient" it produces can point uphill.
    """

    _diagonal: np.ndarray | None
    _matrix: np.ndarray | None

    def __init__(self, W) -> None:
        arr = np.asarray(W, dtype=np.float64)
        if arr.ndim == 1:
            if np.any(arr <= 0.0):
                raise ValueError(
                    f"{_METRIC_DIAGNOSTIC_CODE}: diagonal metric weights must "
                    f"all be positive; got min {arr.min():.3e}. A non-positive "
                    f"weight makes this not an inner product."
                )
            self._diagonal = arr
            self._matrix = None
        elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            if not np.allclose(arr, arr.T):
                raise ValueError(
                    f"{_METRIC_DIAGNOSTIC_CODE}: a metric matrix must be "
                    f"symmetric; got max asymmetry "
                    f"{np.max(np.abs(arr - arr.T)):.3e}."
                )
            eigenvalues = np.linalg.eigvalsh(arr)
            if eigenvalues.min() <= 0.0:
                raise ValueError(
                    f"{_METRIC_DIAGNOSTIC_CODE}: a metric matrix must be "
                    f"positive definite; smallest eigenvalue "
                    f"{eigenvalues.min():.3e}. An indefinite W is not an inner "
                    f"product, and the vector it produces can point uphill."
                )
            self._diagonal = None
            self._matrix = arr
        else:
            raise ValueError(
                f"{_METRIC_DIAGNOSTIC_CODE}: metric weights must be a 1-D "
                f"positive vector or a square SPD matrix; got shape {arr.shape}."
            )

    def inner(self, u, v) -> float:
        u = np.asarray(u, dtype=np.float64).ravel()
        v = np.asarray(v, dtype=np.float64).ravel()
        if self._diagonal is not None:
            return float(np.sum(self._diagonal * u * v))
        matrix = self._matrix
        assert matrix is not None
        return float(u @ matrix @ v)

    def sharp(self, covector):
        g = np.asarray(covector, dtype=np.float64)
        if self._diagonal is not None:
            return g / self._diagonal.reshape(g.shape)
        matrix = self._matrix
        assert matrix is not None  # exactly one of the two is set by __init__
        return np.linalg.solve(matrix, g.ravel()).reshape(g.shape)

    def project_tangent(self, x, v):
        return np.asarray(v, dtype=np.float64)

    def retract(self, x, v):
        return np.asarray(x, dtype=np.float64) + np.asarray(v, dtype=np.float64)

    def __repr__(self) -> str:
        kind = "diagonal" if self._diagonal is not None else "matrix"
        return f"Weighted({kind})"


class Sphere:
    """The unit sphere `x^T x = 1` with the induced Euclidean metric (§13.1).

    Differentiating the constraint gives `2 x^T dx = 0`, so tangents are
    orthogonal to the radius and the gradient is the ambient one with its
    radial component removed: `(I - x x^T) g`. The retraction is
    renormalization — the projection back onto the sphere.
    """

    def inner(self, u, v) -> float:
        return float(np.sum(np.asarray(u) * np.asarray(v)))

    def sharp(self, covector):
        return np.asarray(covector, dtype=np.float64)

    @staticmethod
    def _point(x, *, operation: str) -> np.ndarray:
        point = np.asarray(x, dtype=np.float64)
        norm = float(np.linalg.norm(point))
        if not np.all(np.isfinite(point)) or not np.isfinite(norm):
            raise ValueError(
                f"{_METRIC_DIAGNOSTIC_CODE}: Sphere.{operation} requires a "
                "finite point on the unit sphere."
            )
        if not np.isclose(norm, 1.0, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError(
                f"{_METRIC_DIAGNOSTIC_CODE}: Sphere.{operation} requires "
                f"||x||=1; got ||x||={norm:.17g}."
            )
        return point

    def project_tangent(self, x, v):
        x = self._point(x, operation="project_tangent")
        v = np.asarray(v, dtype=np.float64)
        if v.shape != x.shape or not np.all(np.isfinite(v)):
            raise ValueError(
                f"{_METRIC_DIAGNOSTIC_CODE}: Sphere.project_tangent requires "
                "a finite tangent candidate with the same shape as x."
            )
        return v - x * float(np.sum(x * v))

    def retract(self, x, v):
        point = self._point(x, operation="retract")
        tangent = np.asarray(v, dtype=np.float64)
        if tangent.shape != point.shape or not np.all(np.isfinite(tangent)):
            raise ValueError(
                f"{_METRIC_DIAGNOSTIC_CODE}: Sphere.retract requires a finite "
                "step with the same shape as x."
            )
        moved = point + tangent
        nrm = float(np.linalg.norm(moved))
        if nrm == 0.0:
            raise ValueError(
                f"{_METRIC_DIAGNOSTIC_CODE}: retraction landed at the origin, "
                f"which is not on the unit sphere; the step was exactly -x."
            )
        return moved / nrm

    def __repr__(self) -> str:
        return "Sphere()"


class Orthogonal:
    """The orthogonal group `Q^T Q = I` (§13.2).

    Differentiating the constraint gives `Q^T dQ + dQ^T Q = 0`, i.e. `Q^T dQ`
    is **antisymmetric** — which is the whole reason `O(n)` has `n(n-1)/2`
    parameters rather than `n^2`. The tangent projection is therefore
    `Q * skew(Q^T V)`, and the retraction is the `Q` factor of a QR
    factorization with the sign convention fixed so it is a true retraction.
    """

    def inner(self, u, v) -> float:
        return float(np.sum(np.asarray(u) * np.asarray(v)))

    def sharp(self, covector):
        return np.asarray(covector, dtype=np.float64)

    @staticmethod
    def _point(x, *, operation: str) -> np.ndarray:
        Q = np.asarray(x, dtype=np.float64)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError(
                f"{_METRIC_DIAGNOSTIC_CODE}: Orthogonal.{operation} requires "
                f"a square rank-2 point; got shape {Q.shape}."
            )
        identity = np.eye(Q.shape[0], dtype=np.float64)
        if not np.all(np.isfinite(Q)) or not np.allclose(
            Q.T @ Q, identity, rtol=1.0e-10, atol=1.0e-12
        ):
            raise ValueError(
                f"{_METRIC_DIAGNOSTIC_CODE}: Orthogonal.{operation} requires "
                "Q.T @ Q == I at the supplied point."
            )
        return Q

    def project_tangent(self, x, v):
        Q = self._point(x, operation="project_tangent")
        V = np.asarray(v, dtype=np.float64)
        if V.shape != Q.shape or not np.all(np.isfinite(V)):
            raise ValueError(
                f"{_METRIC_DIAGNOSTIC_CODE}: Orthogonal.project_tangent "
                "requires a finite tangent candidate with the same shape as Q."
            )
        M = Q.T @ V
        return Q @ (0.5 * (M - M.T))

    def retract(self, x, v):
        Q0 = self._point(x, operation="retract")
        tangent = np.asarray(v, dtype=np.float64)
        if tangent.shape != Q0.shape or not np.all(np.isfinite(tangent)):
            raise ValueError(
                f"{_METRIC_DIAGNOSTIC_CODE}: Orthogonal.retract requires a "
                "finite step with the same shape as Q."
            )
        moved = Q0 + tangent
        Q, R = np.linalg.qr(moved)
        # numpy's QR leaves the sign of each column free; pinning it to a
        # positive diagonal makes `retract(x, 0) == x` , which is what makes
        # this a retraction rather than an arbitrary nearby point.
        return Q * np.sign(np.sign(np.diag(R)) + 0.5)

    def __repr__(self) -> str:
        return "Orthogonal()"


def riemannian_gradient(metric, x, euclidean_gradient) -> np.ndarray:
    """The gradient of the same differential, read in ``metric``'s geometry.

    ``project_tangent(x, sharp(g))`` — raise the index, then restrict to the
    tangent space. For :class:`Euclidean` both steps are the identity, which is
    why the distinction is usually invisible.
    """
    if not isinstance(metric, Metric):
        raise TypeError(
            f"{_METRIC_DIAGNOSTIC_CODE}: {type(metric).__name__} is not a "
            f"Metric — it must provide inner/sharp/project_tangent/retract."
        )
    raised = metric.sharp(np.asarray(euclidean_gradient, dtype=np.float64))
    return metric.project_tangent(np.asarray(x, dtype=np.float64), raised)
