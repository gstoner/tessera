"""``tessera.contour`` — Bundle C of the Needham completion (Ch. 7-9).

Contour integration, winding number, and residue extraction.  This
is the second half of *Visual Complex Analysis*: where Bundle A/B
made differentiation visible, Bundle C makes integration visible.

Public surface:

  - :class:`Contour` — a parametric curve ``γ: [0, 1] → ℂ`` plus
    an optional ``γ': [0, 1] → ℂ`` derivative (auto-computed via
    central differences when absent).
  - :func:`circle(center, radius, ccw=True)` — ``γ(t) = center +
    radius · exp(±2πi t)``.
  - :func:`line_segment(z0, z1)` — straight-line interpolation.
  - :func:`polygon(vertices)` — closed piecewise-linear contour.
  - :func:`contour_integral(f, contour, n_steps=1024)` —
    composite Simpson's rule over the t-parameterization.
  - :func:`winding_number(contour, z0)` — argument-variation
    integral, rounded to the nearest integer.
  - :func:`residue(f, z0, radius=0.1)` — Cauchy's formula
    ``(1/2πi) ∮ f(z) dz`` around a small circle at ``z₀``.

The acceptance proofs in :mod:`tests/unit/test_contour.py`
include:

  - ``∮ dz`` around a closed loop = 0 (Cauchy's theorem).
  - ``∮ 1/(z − a) dz = 2πi`` for ``a`` inside (Cauchy's formula).
  - Winding number is ±1 / 0 as expected.
  - Argument principle: ``∮ f'/f dz / (2πi)`` counts zeros minus
    poles inside.
  - Residue theorem: ``∮ f dz = 2πi · Σ Res(f, z_k)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence


# ─────────────────────────────────────────────────────────────────────────────
# Contour — a parametric curve with optional analytic derivative.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Contour:
    """A parametric curve ``γ: [0, 1] → ℂ`` with an optional
    derivative.  Used by :func:`contour_integral` and friends.

    The contour is **closed** when ``γ(0) == γ(1)`` — the
    standard convention for Cauchy-style integrals.  Constructors
    in this module produce closed contours (``circle``,
    ``polygon``) or open ones (``line_segment``); callers can
    inspect ``is_closed`` to check.

    ``segments`` is an optional sequence of ``t``-breakpoints
    where ``γ'`` is discontinuous (e.g., polygon vertices).  The
    integrator splits at these so Simpson's rule sees smooth
    sub-integrands.  Defaults to a single segment ``(0.0, 1.0)``.
    """
    gamma: Callable[[float], complex]
    gamma_prime: Optional[Callable[[float], complex]] = None
    # Optional label for debug / report formatting.
    name: str = ""
    # t-breakpoints between smooth pieces (e.g., polygon vertices).
    segments: tuple[float, ...] = (0.0, 1.0)

    def __call__(self, t: float) -> complex:
        return self.gamma(float(t))

    def tangent(self, t: float) -> complex:
        """``γ'(t)``.  Uses the analytic derivative when supplied;
        otherwise falls back to central differences."""
        if self.gamma_prime is not None:
            return self.gamma_prime(float(t))
        h = 1e-6
        return (self.gamma(t + h) - self.gamma(t - h)) / (2.0 * h)

    @property
    def is_closed(self, *, tol: float = 1e-9) -> bool:
        return abs(self.gamma(0.0) - self.gamma(1.0)) <= tol


# ─────────────────────────────────────────────────────────────────────────────
# Constructors
# ─────────────────────────────────────────────────────────────────────────────

def circle(center: complex, radius: float, *, ccw: bool = True) -> Contour:
    """A circle of radius ``radius`` around ``center``.  ``ccw``
    controls orientation (counterclockwise is the standard
    positive direction in Cauchy's theorem)."""
    sign = 1.0 if ccw else -1.0
    c = complex(center)
    r = float(radius)

    def g(t: float) -> complex:
        return c + r * complex(math.cos(sign * 2.0 * math.pi * t),
                                math.sin(sign * 2.0 * math.pi * t))

    def gp(t: float) -> complex:
        return sign * 2.0 * math.pi * r * complex(
            -math.sin(sign * 2.0 * math.pi * t),
             math.cos(sign * 2.0 * math.pi * t),
        )

    return Contour(gamma=g, gamma_prime=gp, name=f"circle({c}, {r})")


def line_segment(z0: complex, z1: complex) -> Contour:
    """The straight line from ``z0`` to ``z1``.  ``γ(t) = z0 +
    t·(z1 − z0)``.  Not closed."""
    a = complex(z0)
    b = complex(z1)
    dz = b - a

    def g(t: float) -> complex:
        return a + t * dz

    def gp(_t: float) -> complex:
        return dz

    return Contour(gamma=g, gamma_prime=gp, name=f"line({a}→{b})")


def polygon(vertices: Sequence[complex]) -> Contour:
    """A closed polygon with the given vertices.  The contour
    visits ``v0 → v1 → ... → vN → v0`` over ``t ∈ [0, 1]``,
    with each edge given equal parameter length ``1/N``."""
    pts = [complex(v) for v in vertices]
    if len(pts) < 3:
        raise ValueError(
            f"polygon requires at least 3 vertices; got {len(pts)}"
        )
    n_edges = len(pts)
    # Closed: append v0 at the end.
    closed = pts + [pts[0]]

    def g(t: float) -> complex:
        t = max(0.0, min(1.0, t))
        scaled = t * n_edges
        k = int(scaled)
        if k >= n_edges:
            k = n_edges - 1
        local = scaled - k
        return closed[k] * (1.0 - local) + closed[k + 1] * local

    def gp(t: float) -> complex:
        # Left-bias at edge boundaries so γ'(k/n) belongs to the
        # edge ENDING at that vertex (the one we're closing),
        # not the next edge starting from it.  This is what the
        # per-segment integrator wants — the right endpoint of
        # segment k should carry segment k's tangent.
        t = max(0.0, min(1.0, t))
        scaled = t * n_edges
        # Subtract a tiny ε so a value at an exact boundary
        # rounds DOWN to the previous edge.
        k = int(scaled - 1e-12)
        k = max(0, min(n_edges - 1, k))
        return n_edges * (closed[k + 1] - closed[k])

    # Break t into segments at each vertex boundary so the
    # integrator can use a smooth Simpson's per edge instead of
    # straddling the kinks at vertices.
    segs = tuple(k / n_edges for k in range(n_edges + 1))
    return Contour(
        gamma=g, gamma_prime=gp,
        name=f"polygon({n_edges})", segments=segs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Composite Simpson's rule for complex contour integrals.
# ─────────────────────────────────────────────────────────────────────────────

def _simpson_complex_range(
    integrand: Callable[[float], complex],
    t_lo: float, t_hi: float,
    n_steps: int,
) -> complex:
    """``∫_{t_lo}^{t_hi} integrand(t) dt`` via composite Simpson's
    rule.  ``n_steps`` is the number of subintervals — rounded
    up to even.

    Boundary samples are nudged ε-inward so contours with a
    discontinuous γ' (polygons) have their tangents evaluated
    on the correct segment.  For smooth contours the ε-nudge
    is far below numerical precision.
    """
    if n_steps < 2:
        n_steps = 2
    if n_steps % 2 == 1:
        n_steps += 1
    span = t_hi - t_lo
    h = span / n_steps
    eps = span * 1e-9
    total = integrand(t_lo + eps) + integrand(t_hi - eps)
    for i in range(1, n_steps):
        t = t_lo + i * h
        w = 4.0 if (i % 2 == 1) else 2.0
        total = total + w * integrand(t)
    return total * (h / 3.0)


def contour_integral(
    f: Callable[[complex], complex],
    contour: Contour,
    *,
    n_steps: int = 1024,
) -> complex:
    """Compute ``∮ f(z) dz`` over the parameterized contour.

    Pull-back to the t-parameterization::

        ∫_γ f(z) dz = ∫_0^1 f(γ(t)) · γ'(t) dt

    Implementation: composite Simpson's rule over each smooth
    sub-segment of the contour (e.g., each edge of a polygon),
    then sum.  ``n_steps`` is divided proportionally among the
    sub-segments.
    """
    def integrand(t: float) -> complex:
        return f(contour(t)) * contour.tangent(t)
    segs = contour.segments
    if len(segs) < 2:
        return _simpson_complex_range(integrand, 0.0, 1.0, n_steps)
    n_pieces = len(segs) - 1
    per_piece = max(2, n_steps // n_pieces)
    total = 0j
    for i in range(n_pieces):
        total = total + _simpson_complex_range(
            integrand, segs[i], segs[i + 1], per_piece,
        )
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Winding number — via argument variation.
# ─────────────────────────────────────────────────────────────────────────────

def winding_number(
    contour: Contour, z0: complex, *,
    n_steps: int = 2048,
) -> int:
    """Winding number of the contour around ``z₀``.

    Uses the argument-variation method::

        n(γ, z₀) = (1 / 2π) ∮ d(arg(γ(t) - z₀))

    Numerically: sample the contour at ``n_steps`` points, sum
    the unwrapped differences of ``arg(γ(t) - z₀)``, divide by
    ``2π``, round to the nearest integer.

    Returns 0 when ``z₀`` is outside the contour's image, +1 for
    a single counterclockwise encirclement, -1 for clockwise,
    etc.

    Raises :class:`ValueError` when the contour passes within
    ``1e-9`` of ``z₀`` (the winding number is undefined there).
    """
    samples = n_steps + 1
    total_delta = 0.0
    prev_arg = None
    for i in range(samples):
        t = i / n_steps
        diff = contour(t) - z0
        if abs(diff) < 1e-9:
            raise ValueError(
                f"contour passes through z₀={z0!r} at t≈{t}; winding "
                "number is undefined"
            )
        arg = math.atan2(diff.imag, diff.real)
        if prev_arg is not None:
            d = arg - prev_arg
            # Unwrap into (-π, π].
            while d > math.pi:
                d -= 2.0 * math.pi
            while d < -math.pi:
                d += 2.0 * math.pi
            total_delta += d
        prev_arg = arg
    return int(round(total_delta / (2.0 * math.pi)))


# ─────────────────────────────────────────────────────────────────────────────
# Residue — Cauchy's formula.
# ─────────────────────────────────────────────────────────────────────────────

def residue(
    f: Callable[[complex], complex],
    z0: complex,
    *,
    radius: float = 0.1,
    n_steps: int = 1024,
) -> complex:
    """``Res(f, z₀)`` via a circle integral.

    ``Res(f, z₀) = (1 / 2πi) ∮_{|z-z₀|=r} f(z) dz``

    Provided ``radius`` is small enough that ``z₀`` is the only
    pole inside the circle, the result is independent of
    ``radius``.  Callers near multiple poles should narrow
    ``radius`` accordingly.
    """
    c = circle(z0, radius, ccw=True)
    return contour_integral(f, c, n_steps=n_steps) / (2.0j * math.pi)


# ─────────────────────────────────────────────────────────────────────────────
# Worked-example helpers — argument principle + residue theorem.
# These aren't new primitives, just named compositions over the
# above that make the textbook theorems testable in one call.
# ─────────────────────────────────────────────────────────────────────────────

def argument_principle_count(
    f: Callable[[complex], complex],
    f_prime: Callable[[complex], complex],
    contour: Contour,
    *,
    n_steps: int = 1024,
) -> int:
    """``Z − P`` over the contour: zeros minus poles of ``f``
    enclosed.

    Uses ``∮ (f' / f) dz / (2πi) = Z − P`` (the argument
    principle).  ``f_prime`` is supplied separately because we
    don't want to numerical-differentiate inside an integrand —
    it amplifies finite-difference noise.

    Returns the rounded integer count.
    """
    def g(z: complex) -> complex:
        return f_prime(z) / f(z)
    value = contour_integral(g, contour, n_steps=n_steps) / (2.0j * math.pi)
    return int(round(value.real))


def residue_theorem_sum(
    f: Callable[[complex], complex],
    contour: Contour,
    poles_inside: Sequence[complex],
    *,
    radius: float = 0.05,
    n_steps: int = 1024,
) -> complex:
    """``2πi · Σ Res(f, p_k)`` for poles ``p_k`` inside the
    contour.  By the residue theorem, this equals ``∮_γ f(z) dz``.

    Tests use this to verify the equality on closed forms with
    known residues.
    """
    total = 0j
    for p in poles_inside:
        total = total + residue(f, p, radius=radius, n_steps=n_steps)
    return 2.0j * math.pi * total


__all__ = [
    "Contour",
    "circle",
    "line_segment",
    "polygon",
    "contour_integral",
    "winding_number",
    "residue",
    "argument_principle_count",
    "residue_theorem_sum",
]
