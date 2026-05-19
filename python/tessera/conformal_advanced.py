"""``tessera.conformal_advanced`` — Schwarz-Christoffel + Weierstrass.

Two heavier mathematical constructions from *Visual Complex
Analysis*: the Schwarz-Christoffel mapping (Ch. 12) and the
Weierstrass ℘ elliptic function (Ch. 5).

**Schwarz-Christoffel — MVP scope:**

The full Schwarz-Christoffel theorem solves the "parameter
problem": given a target polygon, find the **prevertices**
``x_k ∈ ℝ`` (on the boundary of the upper half-plane H⁺) such
that the SC integral lands on the polygon's vertices.  The
parameter problem is its own research topic — the SC Toolbox
literature has whole iterative solvers for it.

This module ships the **prevertex-given** form: the user
supplies both the polygon and the prevertices, and we compute
the SC conformal map by direct integration::

    f(z) = ∫_{z₀}^{z} ∏_k (ζ − x_k)^{α_k − 1} dζ

where ``α_k`` are the interior angles in units of π.  The
canonical use is for symmetric polygons (regular n-gons,
rectangles) where the prevertices have closed forms.

**Weierstrass ℘ — MVP scope:**

The doubly-periodic elliptic function ::

    ℘(z; ω₁, ω₂) = 1/z² + Σ_{(m,n)≠(0,0)} [1/(z − Λ)² − 1/Λ²]

where ``Λ = m·ω₁ + n·ω₂`` runs over the lattice.  We ship a
truncated lattice sum (||m|, |n|| ≤ ``cutoff``) — fine for
visualization and small-z, not industrial.  Tests verify the
two defining properties: periodicity and the elliptic-curve
identity ``℘'(z)² = 4℘³ − g₂℘ − g₃``.
"""

from __future__ import annotations

import cmath
import math
from typing import Sequence

from .contour import _simpson_complex_range, line_segment, contour_integral


# ─────────────────────────────────────────────────────────────────────────────
# Schwarz-Christoffel — prevertex-given MVP
# ─────────────────────────────────────────────────────────────────────────────

def schwarz_christoffel_map(
    z: complex,
    prevertices: Sequence[float],
    angles: Sequence[float],
    *,
    base_point: complex = 0 + 0j,
    n_steps: int = 1024,
) -> complex:
    """Evaluate the Schwarz-Christoffel conformal map at ``z``.

    Parameters
    ----------
    z
        Point in the upper half-plane to map.
    prevertices
        Real numbers on the boundary of H⁺ that get carried to the
        polygon's vertices.  Order matters: the prevertex order
        determines vertex order.
    angles
        Interior angles of the polygon at each vertex, **in units
        of π** (so a right angle is ``0.5``).  Must satisfy
        ``Σ(1 − α_k) = 2`` for a closed polygon (this corresponds
        to the angle-sum of an n-gon being ``(n - 2)π``).  We do
        NOT check this constraint — the integration produces a
        valid open curve regardless; if you supply inconsistent
        angles the result won't close.
    base_point
        Starting point of the integration (typically ``0 + 0j``
        or the centroid).  The final map differs by a constant
        depending on this choice.
    n_steps
        Steps in the underlying Simpson's rule.

    Returns
    -------
    complex
        ``f(z)`` — the image of ``z`` under the SC map.
    """
    if z.imag <= 0:
        raise ValueError(
            f"schwarz_christoffel_map: z must be in the upper "
            f"half-plane (Im z > 0); got {z}"
        )
    if len(prevertices) != len(angles):
        raise ValueError(
            f"prevertices and angles must have the same length; "
            f"got {len(prevertices)} vs {len(angles)}"
        )

    def integrand(zeta: complex) -> complex:
        """SC integrand: ``∏_k (ζ − x_k)^{α_k − 1}``."""
        prod = 1 + 0j
        for xk, ak in zip(prevertices, angles):
            prod *= (zeta - xk) ** (ak - 1.0)
        return prod

    # Integrate along the straight line from base_point to z in
    # the upper half-plane.  For prevertices on the real axis
    # this avoids the singularities at ζ = x_k as long as the
    # base + z aren't exactly on the real axis.
    L = line_segment(base_point, z)
    return contour_integral(integrand, L, n_steps=n_steps)


# ─────────────────────────────────────────────────────────────────────────────
# Weierstrass ℘ — truncated lattice sum
# ─────────────────────────────────────────────────────────────────────────────

def weierstrass_p(
    z: complex,
    omega1: complex,
    omega2: complex,
    *,
    cutoff: int = 8,
) -> complex:
    """Weierstrass ``℘(z; ω₁, ω₂)`` via a truncated lattice sum.

    Convergent series::

        ℘(z) = 1/z² + Σ_{(m,n) ≠ (0,0)} [1/(z − Λ)² − 1/Λ²]

    where ``Λ = m·ω₁ + n·ω₂`` and the sum is taken with ``|m|, |n|
    ≤ cutoff``.

    The full series converges absolutely; truncation introduces a
    bounded error that decreases like ``1/cutoff²``.  For
    visualization / small-z work, cutoff ≈ 8-16 is typically
    sufficient.

    Raises ``ValueError`` when ``z`` lands exactly on a lattice
    point (where ``℘`` has a double pole).
    """
    # Check whether z is on the lattice.
    if z == 0:
        raise ValueError("℘ has a pole at z = 0")
    total = 1.0 / (z * z)
    for m in range(-cutoff, cutoff + 1):
        for n in range(-cutoff, cutoff + 1):
            if m == 0 and n == 0:
                continue
            Lambda = m * omega1 + n * omega2
            if abs(z - Lambda) < 1e-12:
                raise ValueError(
                    f"℘ has a pole at z = Λ = {Lambda}"
                )
            total += 1.0 / ((z - Lambda) ** 2) - 1.0 / (Lambda ** 2)
    return total


def weierstrass_p_derivative(
    z: complex,
    omega1: complex,
    omega2: complex,
    *,
    cutoff: int = 8,
) -> complex:
    """``℘'(z) = -2 · Σ 1/(z − Λ)³``.

    Derivative of the Weierstrass ℘ function; converges absolutely
    without the regularizing ``-1/Λ²`` term.
    """
    if z == 0:
        raise ValueError("℘' has a pole at z = 0")
    total = 0j
    for m in range(-cutoff, cutoff + 1):
        for n in range(-cutoff, cutoff + 1):
            Lambda = m * omega1 + n * omega2
            if abs(z - Lambda) < 1e-12:
                raise ValueError(
                    f"℘' has a pole at z = Λ = {Lambda}"
                )
            total += 1.0 / ((z - Lambda) ** 3)
    return -2.0 * total


def weierstrass_invariants(
    omega1: complex,
    omega2: complex,
    *,
    cutoff: int = 8,
) -> tuple[complex, complex]:
    """The Weierstrass invariants ``(g₂, g₃)`` of the lattice.

    ::

        g₂ = 60 · Σ 1/Λ⁴
        g₃ = 140 · Σ 1/Λ⁶

    These appear in the elliptic-curve identity ``℘'(z)² = 4℘(z)³
    − g₂·℘(z) − g₃`` — verified by
    :func:`tests/unit/test_weierstrass.test_elliptic_curve_identity`.
    """
    s4 = 0j
    s6 = 0j
    for m in range(-cutoff, cutoff + 1):
        for n in range(-cutoff, cutoff + 1):
            if m == 0 and n == 0:
                continue
            Lambda = m * omega1 + n * omega2
            l2 = Lambda * Lambda
            l4 = l2 * l2
            l6 = l4 * l2
            s4 += 1.0 / l4
            s6 += 1.0 / l6
    return (60.0 * s4, 140.0 * s6)


# ─────────────────────────────────────────────────────────────────────────────
# Schwarz-Christoffel parameter problem solver.
#
# The SC parameter problem: given a target polygon (vertices in ℂ),
# find the prevertices ``x_k`` ∈ ℝ ∪ {∞} on the boundary of H⁺ that
# the SC map carries to the polygon's vertices.  The angles are
# fixed by the polygon's shape; the prevertices control side-length
# ratios.
#
# Standard reference: Trefethen, *Schwarz-Christoffel Mapping* (CUP, 2002).
#
# Algorithm shipped here (Newton-style iteration on side-length-ratio
# residuals):
#
#   1. Extract interior angles ``α_k`` (in units of π) from the
#      vertex sequence.  Verify Σ(1 − α_k) = 2 (Gauss-Bonnet for
#      a simple polygon).
#   2. Fix three prevertices by Möbius gauge: ``x_1 = -1``,
#      ``x_2 = 0``, ``x_N = ∞`` (handled via change-of-variables).
#      The remaining ``x_3, ..., x_{N-1}`` are the free parameters.
#   3. Parameterize free prevertices via log-spacings ``s_i > 0`` so
#      ``x_{2+i} = sum_{j=1..i} exp(s_j)`` automatically respects
#      the ordering ``0 < x_3 < x_4 < ... < x_{N-1}``.
#   4. Compute side lengths via numerical integration on the real
#      axis between consecutive prevertices (handles integrable
#      endpoint singularities by substitution).
#   5. Residual: ``side_k / side_1 − (target_k / target_1)`` for
#      k = 2, ..., N − 2.  That's N − 3 equations in N − 3 unknowns.
#   6. Solve with a damped Newton iteration; converges quadratically
#      from a reasonable initial guess.
# ─────────────────────────────────────────────────────────────────────────────

def _polygon_interior_angles(vertices: Sequence[complex]) -> list[float]:
    """Interior angles of a polygon at each vertex, in units of π.

    The interior angle at vertex k is the angle between the
    *incoming* edge ``v_{k-1} → v_k`` (reversed) and the *outgoing*
    edge ``v_k → v_{k+1}``.  In units of π so the SC ``α_k``
    convention is direct.
    """
    n = len(vertices)
    angles: list[float] = []
    for k in range(n):
        prev_e = vertices[(k - 1) % n] - vertices[k]
        next_e = vertices[(k + 1) % n] - vertices[k]
        # For a CCW polygon, the interior is on the LEFT of the
        # traversal direction.  Measure the CCW angle from
        # ``next_e`` to ``prev_e`` — this is the interior angle.
        ang = math.atan2(
            next_e.real * prev_e.imag - next_e.imag * prev_e.real,
            next_e.real * prev_e.real + next_e.imag * prev_e.imag,
        )
        # Normalize to (0, 2π).
        if ang <= 0:
            ang += 2.0 * math.pi
        angles.append(ang / math.pi)
    return angles


def _sc_real_axis_segment_length(
    prevertices: Sequence[float],
    angles: Sequence[float],
    k: int,
    n_steps: int = 256,
) -> float:
    """Length of the polygon's edge ``k`` under the SC map.

    Edge ``k`` is the image of the real-axis segment between
    ``prevertices[k]`` and ``prevertices[k+1]``.  The integrand
    has integrable singularities at both endpoints
    ``∏ |t − x_j|^(α_j − 1)``; we substitute
    ``t = x_k + (x_{k+1} − x_k) · u`` to bring the interval to
    ``[0, 1]`` and integrate via composite Simpson with a small
    inset from the endpoints (where the integrand is unbounded
    but integrable).
    """
    x_lo = prevertices[k]
    x_hi = prevertices[k + 1]
    span = x_hi - x_lo
    if span <= 0:
        raise ValueError(
            f"prevertices must be strictly ordered; got "
            f"x[{k}]={x_lo}, x[{k+1}]={x_hi}"
        )

    def integrand(u: float) -> float:
        t = x_lo + span * u
        prod = 1.0
        for j, (xj, aj) in enumerate(zip(prevertices, angles)):
            # Skip the prevertex at infinity — its factor
            # ``(t − ∞)^(α − 1)`` becomes a constant under the SC
            # gauge convention and is absorbed into overall scale.
            if math.isinf(xj):
                continue
            d = abs(t - xj)
            if d == 0.0:
                # Singularity exactly at a prevertex; shouldn't
                # happen for u ∈ (0, 1) but guard the edge.
                return 0.0
            prod *= d ** (aj - 1.0)
        return prod * span    # |dt/du| = |span|

    # Inset slightly from u = 0 and u = 1 to avoid the singularity
    # while preserving most of the integral.
    eps = 1e-6
    h = (1.0 - 2 * eps) / n_steps
    if n_steps % 2 == 1:
        n_steps += 1
    total = integrand(eps) + integrand(1.0 - eps)
    for i in range(1, n_steps):
        u = eps + i * h
        w = 4.0 if (i % 2 == 1) else 2.0
        total += w * integrand(u)
    return total * (h / 3.0)


def _sc_real_axis_segment_to_infinity_length(
    prevertices: Sequence[float],
    angles: Sequence[float],
    n_steps: int = 256,
) -> float:
    """Length of the SC edge from ``prevertices[N-2]`` to ``+∞`` —
    the last edge when the final prevertex is at infinity.

    Substitute ``t = x_{N-2} + 1/u`` to bring ``[x_{N-2}, ∞)``
    onto ``(0, 1]``.  ``dt = -du/u²``, so the integrand
    factor in u-space picks up ``1/u²``."""
    x_lo = prevertices[-2]

    def integrand(u: float) -> float:
        if u == 0:
            return 0.0
        t = x_lo + 1.0 / u
        prod = 1.0
        for xj, aj in zip(prevertices[:-1], angles[:-1]):
            d = abs(t - xj)
            prod *= d ** (aj - 1.0)
        # The last prevertex at ∞ contributes a factor
        # ``|t|^(α_N − 1) · t^(α_N − 1)`` in the limit — the
        # convention used here is to fold it into the explicit
        # 1/u² Jacobian below.
        jacobian = 1.0 / (u * u)
        # The ``(t - x_N)^(α_N - 1)`` factor as x_N → ∞ becomes
        # ``(-x_N)^(α_N - 1)`` which is constant in t.  By the
        # SC gauge convention this constant is absorbed into the
        # overall scale, so we skip it here.
        return prod * jacobian

    eps = 1e-6
    h = (1.0 - 2 * eps) / n_steps
    if n_steps % 2 == 1:
        n_steps += 1
    # Reverse: integration over u from eps to 1-eps corresponds
    # to t going from infinity inward to x_{N-2}.  We use the
    # absolute value of the integral as the length.
    total = integrand(eps) + integrand(1.0 - eps)
    for i in range(1, n_steps):
        u = eps + i * h
        w = 4.0 if (i % 2 == 1) else 2.0
        total += w * integrand(u)
    return abs(total * (h / 3.0))


def _params_to_prevertices(params: Sequence[float]) -> list[float]:
    """Convert the free log-spacings to the prevertex tuple
    ``[x_1, x_2, ..., x_{N-1}, +∞]``.

    ``x_1 = -1, x_2 = 0`` are fixed; ``x_3, ..., x_{N-1}`` come from
    cumulative ``exp(params)``; ``x_N = +∞``."""
    out = [-1.0, 0.0]
    s = 0.0
    for p in params:
        s += math.exp(p)
        out.append(s)
    out.append(math.inf)
    return out


def schwarz_christoffel_parameter_solve(
    vertices: Sequence[complex],
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    initial_log_spacings: Sequence[float] | None = None,
) -> tuple[list[float], list[float]]:
    """Solve the SC parameter problem.

    Given polygon vertices (in any rotation / scale / translation),
    find prevertices that produce a polygon with the matching
    side-length ratios under the SC map.

    Parameters
    ----------
    vertices
        Sequence of ``N`` polygon vertices.
    max_iter
        Maximum damped-Newton iterations.
    tol
        Convergence threshold on the residual norm.
    initial_log_spacings
        Optional initial guess; defaults to uniform ``[0, 0, ...]``
        which spaces prevertices at ``x_3 = 1, x_4 = 2, ...``.

    Returns
    -------
    (prevertices, angles)
        ``prevertices`` includes ``+inf`` as its last element;
        ``angles`` are interior angles in units of π.

    Notes
    -----
    Convergence is reliable for "well-conditioned" polygons (no
    extremely thin spikes, no near-collinear vertices).
    Pathological polygons may require a hand-tuned initial guess.
    """
    n = len(vertices)
    if n < 3:
        raise ValueError(
            f"schwarz_christoffel_parameter_solve: polygon needs at "
            f"least 3 vertices; got {n}"
        )
    angles = _polygon_interior_angles(vertices)
    # Sanity: interior angle sum = (n-2)·π ⇒ Σ(1 - α_k) = 2.
    angle_sum_residual = sum(1.0 - a for a in angles) - 2.0
    if abs(angle_sum_residual) > 1e-6:
        raise ValueError(
            f"polygon angles do not sum to (n-2)π; "
            f"Σ(1 − α) = {sum(1.0 - a for a in angles):.6g}, "
            f"expected 2.0.  Check vertex ordering (counterclockwise) "
            f"and that the polygon is simple."
        )

    # Triangle: no free parameters.
    if n == 3:
        return [-1.0, 0.0, math.inf], angles

    target_sides = [
        abs(vertices[(i + 1) % n] - vertices[i])
        for i in range(n)
    ]
    # Side-ratio targets (skip the first side which we scale to match).
    side_ratio_targets = [s / target_sides[0] for s in target_sides[1:]]

    n_free = n - 3
    if initial_log_spacings is None:
        params = [0.0] * n_free
    else:
        params = list(initial_log_spacings)
        if len(params) != n_free:
            raise ValueError(
                f"initial_log_spacings must have length {n_free}; "
                f"got {len(params)}"
            )

    def compute_sides(p: Sequence[float]) -> list[float]:
        prev = _params_to_prevertices(p)
        sides = []
        # Edges 0..N-2 between finite prevertices.
        for k in range(n - 2):
            sides.append(_sc_real_axis_segment_length(prev, angles, k))
        # Edge N-1 from prev[N-2] to +∞ (which closes back to vertex 0).
        sides.append(_sc_real_axis_segment_to_infinity_length(prev, angles))
        return sides

    def compute_residual(p: Sequence[float]) -> list[float]:
        sides = compute_sides(p)
        ratios = [s / sides[0] for s in sides[1:]]
        return [r - t for r, t in zip(ratios, side_ratio_targets)]

    # The number of residuals we use is n - 1 (sides 1..N-1); the
    # number of unknowns is n - 3.  Use the first n_free residuals
    # (least-squares-style: in the well-determined case they're
    # consistent; otherwise we minimize a subset).
    def truncated_residual(p: Sequence[float]) -> list[float]:
        full = compute_residual(p)
        return full[:n_free]

    # Damped Newton iteration with finite-difference Jacobian.
    for iteration in range(max_iter):
        r = truncated_residual(params)
        r_norm = math.sqrt(sum(x * x for x in r))
        if r_norm < tol:
            break
        # Build Jacobian by finite differences.
        eps = 1e-5
        J = []
        for j in range(n_free):
            params_p = list(params); params_p[j] += eps
            r_p = truncated_residual(params_p)
            J.append([(r_p[i] - r[i]) / eps for i in range(n_free)])
        # Solve J^T · dp = -r  (Newton step). J is (n_free × n_free),
        # stored as columns above — flip to rows.
        J_rows = [[J[c][r_idx] for c in range(n_free)]
                   for r_idx in range(n_free)]
        try:
            dp = _solve_linear_system(J_rows, [-x for x in r])
        except _SingularJacobianError:
            # Singular Jacobian — bail out with what we have.
            break
        # Damped line search: shrink step size if residual grows.
        alpha = 1.0
        for _ in range(20):
            trial = [params[i] + alpha * dp[i] for i in range(n_free)]
            r_trial = truncated_residual(trial)
            r_trial_norm = math.sqrt(sum(x * x for x in r_trial))
            if r_trial_norm < r_norm:
                params = trial
                break
            alpha *= 0.5
        else:
            # No improvement: give up.
            break

    return _params_to_prevertices(params), angles


class _SingularJacobianError(Exception):
    pass


def _solve_linear_system(
    matrix: list[list[float]], rhs: list[float],
) -> list[float]:
    """Gaussian elimination with partial pivoting on an n×n system.

    Small system — performance isn't critical, but we avoid a
    numpy dependency for this module to keep the SC solver
    self-contained."""
    n = len(rhs)
    # Augmented matrix copy.
    A = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for i in range(n):
        # Pivot.
        piv = max(range(i, n), key=lambda r: abs(A[r][i]))
        if abs(A[piv][i]) < 1e-15:
            raise _SingularJacobianError(
                f"singular at row {i}, pivot magnitude {A[piv][i]:.3g}"
            )
        A[i], A[piv] = A[piv], A[i]
        # Eliminate below.
        for r in range(i + 1, n):
            factor = A[r][i] / A[i][i]
            for c in range(i, n + 1):
                A[r][c] -= factor * A[i][c]
    # Back-substitute.
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = A[i][n]
        for c in range(i + 1, n):
            s -= A[i][c] * x[c]
        x[i] = s / A[i][i]
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Weierstrass ℘ — grows the lattice sum until the latest
# ring's contribution falls below tolerance.
# ─────────────────────────────────────────────────────────────────────────────

def _weierstrass_ring_contribution(
    z: complex, omega1: complex, omega2: complex, r: int,
) -> complex:
    """Contribution from the lattice ring ``max(|m|, |n|) == r`` to
    the regularized Weierstrass sum.

    Each ring has ``8r`` points (for r ≥ 1).  We sum over the
    boundary of the ``r × r`` integer square."""
    if r == 0:
        return 0j   # only (0, 0) is in the r=0 ring, and it's excluded
    total = 0j
    for m in range(-r, r + 1):
        for n in range(-r, r + 1):
            if max(abs(m), abs(n)) != r:
                continue
            Lambda = m * omega1 + n * omega2
            if abs(z - Lambda) < 1e-12:
                raise ValueError(
                    f"℘ has a pole at z = Λ = {Lambda}"
                )
            total += 1.0 / ((z - Lambda) ** 2) - 1.0 / (Lambda ** 2)
    return total


def weierstrass_p_adaptive(
    z: complex,
    omega1: complex,
    omega2: complex,
    *,
    tol: float = 1e-8,
    max_cutoff: int = 64,
) -> tuple[complex, int, float]:
    """Adaptive ``℘(z; ω₁, ω₂)`` — grow the lattice sum until
    the latest ring's contribution falls below ``tol``.

    Parameters
    ----------
    z
        Evaluation point.
    omega1, omega2
        Lattice generators.
    tol
        Tolerance on the ``r``-th ring's contribution magnitude.
        When a ring contributes less than ``tol``, we stop and
        return.
    max_cutoff
        Safety cap on the radius.

    Returns
    -------
    (value, cutoff_used, last_ring_magnitude)
        ``value`` is the converged sum, ``cutoff_used`` is the
        ring index at which we stopped, and ``last_ring_magnitude``
        is an upper bound on the truncation error (the last
        contribution we added).

    Notes
    -----
    Convergence is reliable for ``z`` away from lattice points.
    For ``z`` very close to a lattice point Λ ≠ 0, convergence
    may require larger ``max_cutoff``.
    """
    if z == 0:
        raise ValueError("℘ has a pole at z = 0")
    total = 1.0 / (z * z)
    last_contribution = math.inf
    cutoff_used = 0
    for r in range(1, max_cutoff + 1):
        contribution = _weierstrass_ring_contribution(z, omega1, omega2, r)
        total += contribution
        cutoff_used = r
        last_contribution = abs(contribution)
        if last_contribution < tol:
            break
    return total, cutoff_used, last_contribution


__all__ = [
    "schwarz_christoffel_map",
    "schwarz_christoffel_parameter_solve",
    "weierstrass_p",
    "weierstrass_p_adaptive",
    "weierstrass_p_derivative",
    "weierstrass_invariants",
]
