"""``tessera.complex`` — Visual Complex / conformal pilot (M7).

Step 1 of the M7 sequence pins the **complex-scalar representation
decision** that the milestone plan left open:

- ``Cl(0,1)`` (single basis ``e₁`` with ``e₁² = -1``) was the plan's
  first recommendation but is **not** in the v1 GA allow-list
  (``ga_scope_lock.md`` Q1 lists only ``Cl(1,3)`` and ``Cl(3,0)``).
- The even subalgebra of ``Cl(2,0)`` is isomorphic to ℂ but is
  also outside the allow-list.
- That leaves the plan's third option: a **non-GA
  :class:`ComplexScalar` sibling kind**.

Per Decision #15a (``Multivector`` is a sibling tensor kind, not a
seventh tensor attribute), ``ComplexScalar`` is the obvious second
sibling — complex numbers are not tensors and they're not
multivectors.  The representation is a thin numpy-backed
``(real, imag)`` pair so it composes naturally with the rest of
the Tessera surface: tensors, multivectors, energy programs.

If we later need full GA interop, ``ComplexScalar.to_ga_even_cl20()``
ships a 4-component multivector encoding (`1 + i⟼1·e + (re)·e + …`)
once Cl(2,0) is unlocked.  Until then this module stays GA-free
and the GA scope lock is untouched.

This file is the namespace skeleton.  Concrete primitives
(``complex_mul``, ``complex_exp``, ``mobius``, ``stereographic``,
``conformal_jacobian``, ``laplacian_2d``) land in subsequent M7
steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


__all__ = [
    "ComplexScalar",
    "from_pair",
    "to_pair",
    "from_numpy",
    "to_numpy",
    "is_complex",
    "complex_mul",
    "complex_exp",
    "complex_conjugate",
    "complex_abs",
    "complex_div",
    "mobius",
    "stereographic",
    "stereographic_inverse",
    "conformal_jacobian",
    "laplacian_2d",
    "check_cauchy_riemann",
    "NotHolomorphicError",
    "analytic",
    "conformal_energy_on_sphere",
]


# ─────────────────────────────────────────────────────────────────────────────
# Sibling value kind
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ComplexScalar:
    """A (batched) complex scalar — sibling value kind to
    :class:`tessera.ga.Multivector`.

    The dataclass is frozen so the value is a true sibling
    (immutable identity), but the ``re`` / ``im`` numpy arrays
    are mutable — Tessera's broader convention is that the
    *handle* is value-typed and the buffers it points at are
    explicit.

    Both fields must have identical shape.  Element dtype is
    preserved; M7 v1 focuses on ``float32`` / ``float64`` paths.
    """
    re: np.ndarray
    im: np.ndarray

    def __post_init__(self) -> None:
        re = np.asarray(self.re)
        im = np.asarray(self.im)
        if re.shape != im.shape:
            raise ValueError(
                f"ComplexScalar: re.shape={re.shape} != im.shape={im.shape}"
            )
        # ``object.__setattr__`` because the dataclass is frozen.
        object.__setattr__(self, "re", re)
        object.__setattr__(self, "im", im)

    # ── shape / dtype ──────────────────────────────────────────

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.re.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.re.dtype

    @property
    def ndim(self) -> int:
        return self.re.ndim

    # ── construction helpers ───────────────────────────────────

    @classmethod
    def from_numpy(cls, z: np.ndarray | complex) -> "ComplexScalar":
        """Build from a numpy complex array or a python ``complex``."""
        arr = np.asarray(z)
        if np.issubdtype(arr.dtype, np.complexfloating):
            return cls(arr.real.copy(), arr.imag.copy())
        # Real input — treat as ``re + 0i``.
        return cls(arr.astype(np.float64, copy=False),
                   np.zeros_like(arr, dtype=np.float64))

    def to_numpy(self, *, dtype: np.dtype | str = np.complex128) -> np.ndarray:
        """Materialize as a numpy complex array."""
        return (self.re.astype(dtype) + 1j * self.im.astype(dtype))

    # ── repr ───────────────────────────────────────────────────

    def __repr__(self) -> str:
        if self.re.ndim == 0:
            return f"ComplexScalar({float(self.re)} + {float(self.im)}i)"
        return f"ComplexScalar(shape={self.shape}, dtype={self.dtype})"


def from_pair(re: Any, im: Any) -> ComplexScalar:
    """Build a :class:`ComplexScalar` from explicit ``(re, im)``
    numpy arrays.  Mirrors :class:`ComplexScalar`'s constructor
    but stays usable as a public namespace helper."""
    return ComplexScalar(np.asarray(re), np.asarray(im))


def to_pair(z: ComplexScalar) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a :class:`ComplexScalar` into ``(re, im)``."""
    return z.re, z.im


def from_numpy(z: np.ndarray | complex) -> ComplexScalar:
    """Top-level alias of :meth:`ComplexScalar.from_numpy`."""
    return ComplexScalar.from_numpy(z)


def to_numpy(z: ComplexScalar, *, dtype: np.dtype | str = np.complex128) -> np.ndarray:
    """Top-level alias of :meth:`ComplexScalar.to_numpy`."""
    return z.to_numpy(dtype=dtype)


def is_complex(x: Any) -> bool:
    """Predicate the M7 verifier uses to gate operations on
    complex inputs.  Returns ``True`` for :class:`ComplexScalar`
    or any numpy complex-dtype array."""
    if isinstance(x, ComplexScalar):
        return True
    if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.complexfloating):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 primitives — complex_mul, complex_exp, complex_conjugate, complex_abs.
#
# Each primitive operates on :class:`ComplexScalar` (or anything
# :func:`from_numpy` accepts) and returns a :class:`ComplexScalar`.
# Implementations are pure numpy; the future ``@energy_jit``-style
# decorator that produces fused MSL kernels can read the operand
# pair representation directly.
# ─────────────────────────────────────────────────────────────────────────────

def _as_pair(z: Any) -> tuple[np.ndarray, np.ndarray]:
    """Coerce ``z`` to a ``(re, im)`` pair of numpy arrays.

    Accepts ``ComplexScalar``, complex numpy arrays, real numpy
    arrays (zero imaginary), and python scalars.  This is the
    single ingest gate for the M7 primitives — every op should go
    through it so the rules for "what counts as complex" stay
    centralized.
    """
    if isinstance(z, ComplexScalar):
        return z.re, z.im
    arr = np.asarray(z)
    if np.issubdtype(arr.dtype, np.complexfloating):
        return arr.real, arr.imag
    return arr.astype(np.float64, copy=False), np.zeros_like(arr, dtype=np.float64)


def complex_mul(z: Any, w: Any) -> ComplexScalar:
    """``(a + b·i)·(c + d·i) = (ac − bd) + (ad + bc)·i``.

    Supports batched inputs — every operation broadcasts via
    numpy's normal rules.
    """
    a, b = _as_pair(z)
    c, d = _as_pair(w)
    re = a * c - b * d
    im = a * d + b * c
    return ComplexScalar(re, im)


def complex_exp(z: Any) -> ComplexScalar:
    """Euler form: ``e^(a + b·i) = e^a · (cos b, sin b)``."""
    a, b = _as_pair(z)
    ea = np.exp(a)
    return ComplexScalar(ea * np.cos(b), ea * np.sin(b))


def complex_conjugate(z: Any) -> ComplexScalar:
    """``conj(a + b·i) = a − b·i``.  This is NOT holomorphic — the
    M7 Cauchy-Riemann verifier (Step 6) must catch any analyitic
    claim about ``complex_conjugate``."""
    a, b = _as_pair(z)
    return ComplexScalar(a.copy(), -b)


def complex_abs(z: Any) -> np.ndarray:
    """``|a + b·i| = sqrt(a² + b²)`` — real-valued, returns a
    plain numpy array."""
    a, b = _as_pair(z)
    return np.sqrt(a * a + b * b)


def complex_div(z: Any, w: Any, *, eps: float = 1e-12) -> ComplexScalar:
    """``z / w = z · conj(w) / |w|²``.

    Returns ``ComplexScalar(+inf, +inf)`` where ``|w| < eps`` so
    Möbius transformations can pass the point-at-infinity through
    a downstream stereographic projection without raising.
    """
    a, b = _as_pair(z)
    c, d = _as_pair(w)
    denom = c * c + d * d
    safe = np.where(denom > eps, denom, 1.0)
    re = np.where(denom > eps, (a * c + b * d) / safe, np.inf)
    im = np.where(denom > eps, (b * c - a * d) / safe, np.inf)
    return ComplexScalar(re, im)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Möbius transformation
#
# f(z; a, b, c, d) = (a z + b) / (c z + d)
#
# Mobius transformations are the conformal automorphisms of the
# Riemann sphere.  They map the family of "generalized circles"
# (true circles + straight lines) to itself.  Composition is
# represented by 2x2 matrix multiplication of the coefficients,
# which the test suite verifies.
# ─────────────────────────────────────────────────────────────────────────────

def mobius(
    z: Any, a: Any, b: Any, c: Any, d: Any, *, eps: float = 1e-12,
) -> ComplexScalar:
    """``f(z) = (a·z + b) / (c·z + d)``.

    All inputs are coerced through :func:`_as_pair`.  At the pole
    ``z = -d/c`` the result is :class:`ComplexScalar` with both
    components set to ``+inf`` (the Riemann sphere's point at
    infinity).

    Raises :class:`ValueError` when ``a·d − b·c`` (the determinant
    of the Mobius matrix) is zero — that would collapse the whole
    plane to a single point and is the only genuinely degenerate
    case worth catching at construction.
    """
    a_pair = from_pair(*_as_pair(a))
    b_pair = from_pair(*_as_pair(b))
    c_pair = from_pair(*_as_pair(c))
    d_pair = from_pair(*_as_pair(d))
    # det = a·d − b·c
    ad = complex_mul(a_pair, d_pair)
    bc = complex_mul(b_pair, c_pair)
    det_re = ad.re - bc.re
    det_im = ad.im - bc.im
    if (np.abs(det_re) < eps).all() and (np.abs(det_im) < eps).all():
        raise ValueError(
            "Mobius matrix is singular (a·d − b·c ≈ 0): the map collapses"
        )
    az = complex_mul(a_pair, z)
    numerator = ComplexScalar(az.re + b_pair.re, az.im + b_pair.im)
    cz = complex_mul(c_pair, z)
    denominator = ComplexScalar(cz.re + d_pair.re, cz.im + d_pair.im)
    return complex_div(numerator, denominator, eps=eps)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Stereographic projection (S² ↔ ℂ).
#
# Forward: project from the north pole of the unit sphere onto the
# complex plane tangent at the south pole (== the equatorial plane
# through the origin, by convention).
#
#   f(x, y, z) = (x + i·y) / (1 − z)   for z ≠ 1
#   f(0, 0, 1) = ∞  (the point at infinity)
#
# Inverse: from ζ ∈ ℂ back to the sphere
#
#   g(u + i·v) = ( 2u / (1+|ζ|²), 2v / (1+|ζ|²), (|ζ|² − 1)/(|ζ|² + 1) )
#
# Together these are the canonical conformal bijection between
# the Riemann sphere and the extended complex plane.
# ─────────────────────────────────────────────────────────────────────────────

def stereographic(
    p: Any, *, eps: float = 1e-12,
) -> ComplexScalar:
    """Stereographic projection from the north pole of S² to ℂ.

    Parameters
    ----------
    p
        A 3-vector ``(x, y, z)`` on the unit sphere — either a
        numpy array with last-axis length 3, or a tuple of three
        scalars/arrays.  The function does NOT re-normalize; the
        caller is responsible for keeping ``x² + y² + z² = 1``.
    eps
        Threshold for detecting the north pole — when ``|1 − z|
        < eps`` the output is ``(+inf, +inf)``.

    Returns
    -------
    ComplexScalar
        The projected point in ℂ.
    """
    if isinstance(p, tuple) and len(p) == 3:
        x, y, z = (np.asarray(c, dtype=np.float64) for c in p)
    else:
        arr = np.asarray(p, dtype=np.float64)
        if arr.shape[-1] != 3:
            raise ValueError(
                f"stereographic: expected 3-vector or last-axis 3; got {arr.shape}"
            )
        x = arr[..., 0]
        y = arr[..., 1]
        z = arr[..., 2]
    denom = 1.0 - z
    safe = np.where(np.abs(denom) > eps, denom, 1.0)
    near_north = np.abs(denom) <= eps
    re = np.where(near_north, np.inf, x / safe)
    im = np.where(near_north, np.inf, y / safe)
    return ComplexScalar(re, im)


def stereographic_inverse(zeta: Any) -> np.ndarray:
    """Inverse stereographic projection ℂ → S² ⊂ ℝ³.

    Returns a numpy array with the same batch shape as the input
    and a trailing axis of size 3.  ``ζ`` representing the point
    at infinity (``inf + inf·i``) maps to the north pole ``(0, 0, 1)``.
    """
    u, v = _as_pair(zeta)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    infinite = np.isinf(u) | np.isinf(v)
    mag_sq = np.where(infinite, 0.0, u * u + v * v)
    denom = 1.0 + mag_sq
    x = np.where(infinite, 0.0, 2.0 * u / denom)
    y = np.where(infinite, 0.0, 2.0 * v / denom)
    z = np.where(infinite, 1.0, (mag_sq - 1.0) / denom)
    return np.stack([x, y, z], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Conformal Jacobian + 2-D Laplacian.
#
# For a complex-analytic ``f = u + i·v`` the Jacobian at ``z₀`` is
#
#     J = [[u_x, u_y], [v_x, v_y]]
#
# and the Cauchy-Riemann equations (``u_x = v_y``, ``u_y = -v_x``)
# turn it into a uniform scale + rotation:
#
#     J = ρ · R(θ)    where  ρ = |f'(z₀)|,  θ = arg(f'(z₀)).
#
# :func:`conformal_jacobian` returns ``(scale, rotation_angle)``
# via central differences.  For a non-analytic ``f`` the
# decomposition is approximate; the M7 Cauchy-Riemann verifier
# (Step 6) is what catches the non-analytic case structurally.
#
# :func:`laplacian_2d` runs a 5-point stencil over a uniform grid.
# Harmonic functions (the real/imaginary parts of an analytic
# function) yield ≈ 0; non-harmonic input surfaces as a non-zero
# residual.
# ─────────────────────────────────────────────────────────────────────────────

def conformal_jacobian(
    f: Any, z0: Any, *, h: float = 1e-5,
) -> tuple[float, float]:
    """Estimate ``(scale, rotation_angle)`` of ``f`` at ``z₀`` via
    central differences.

    Parameters
    ----------
    f
        Callable ``ℂ → ℂ`` (or :class:`ComplexScalar`-valued).
        Anything :func:`from_numpy` accepts works.
    z0
        Center of the linearization — a python complex, a numpy
        complex scalar, or a :class:`ComplexScalar`.
    h
        Central-difference step.

    Returns
    -------
    (scale, rotation_angle)
        ``scale = |df/dz|``, ``rotation_angle = arg(df/dz)``.
    """
    if isinstance(z0, ComplexScalar):
        z0_c = complex(float(z0.re), float(z0.im))
    else:
        z0_c = complex(np.asarray(z0).item())

    def _eval(z: complex) -> complex:
        out = f(z)
        if isinstance(out, ComplexScalar):
            return complex(float(out.re), float(out.im))
        return complex(out)

    # The Cauchy-Riemann derivative is df/dz = (f(z + h) - f(z - h)) / (2h).
    deriv = (_eval(z0_c + h) - _eval(z0_c - h)) / (2.0 * h)
    return float(abs(deriv)), float(np.angle(deriv))


def laplacian_2d(
    field: np.ndarray, *, dx: float = 1.0, dy: float | None = None,
) -> np.ndarray:
    """5-point stencil estimate of ``Δu = u_xx + u_yy``.

    The input is a 2-D numpy array on a uniform grid.  Boundary
    cells are filled with zeros (Dirichlet); the test surface
    avoids the boundary by construction.
    """
    if field.ndim != 2:
        raise ValueError(
            f"laplacian_2d: expected 2-D array, got shape {field.shape}"
        )
    if dy is None:
        dy = dx
    out = np.zeros_like(field, dtype=np.float64)
    inner = (slice(1, -1), slice(1, -1))
    out[inner] = (
        (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / (dy * dy)
        + (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]) / (dx * dx)
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Cauchy-Riemann verifier.
#
# A complex function ``f(z) = u(x, y) + i·v(x, y)`` is holomorphic
# iff it satisfies the Cauchy-Riemann equations:
#
#     ∂u/∂x = ∂v/∂y          (CR-1)
#     ∂u/∂y = -∂v/∂x         (CR-2)
#
# Equivalently, ``∂f/∂x = -i · ∂f/∂y`` as complex numbers.  We
# verify this numerically via central differences in the real and
# imaginary directions and compare the two estimates.
#
# Future work (the symbolic path): an AST-lowered complex function
# whose ops live in a closed whitelist can be checked symbolically
# by composing the M6 ``energy_vjp``-style closed-form derivatives.
# This is the dependency the original plan called out; M7 v1 ships
# the numerical path so it's testable today.
# ─────────────────────────────────────────────────────────────────────────────

class NotHolomorphicError(Exception):
    """Raised by :func:`check_cauchy_riemann` /
    :func:`analytic` when the Cauchy-Riemann residual exceeds the
    caller's tolerance."""


def check_cauchy_riemann(
    f: Any, z0: Any, *, h: float = 1e-5, atol: float = 1e-3,
) -> tuple[bool, float]:
    """Numerical CR check via central differences.

    Returns ``(passes, residual)`` where ``residual`` is
    ``|∂f/∂x + i · ∂f/∂y|`` (Wirtinger ``∂f/∂z̄``).  An analytic
    ``f`` has residual ``≈ 0``; the test passes when
    ``residual <= atol``.
    """
    if isinstance(z0, ComplexScalar):
        z0_c = complex(float(z0.re), float(z0.im))
    else:
        z0_c = complex(np.asarray(z0).item())

    def _eval(z: complex) -> complex:
        out = f(z)
        if isinstance(out, ComplexScalar):
            return complex(float(out.re), float(out.im))
        return complex(out)

    f_x = (_eval(z0_c + h) - _eval(z0_c - h)) / (2.0 * h)
    f_y = (_eval(z0_c + h * 1j) - _eval(z0_c - h * 1j)) / (2.0 * h)
    # Wirtinger ∂f/∂z̄ = (∂f/∂x + i·∂f/∂y) / 2.  For holomorphic f
    # this vanishes.  Our residual is the magnitude of that.
    residual = float(abs(0.5 * (f_x + 1j * f_y)))
    return (residual <= atol, residual)


def analytic(
    fn: Any = None, *, probes: int = 5, h: float = 1e-5, atol: float = 1e-3,
):
    """Decorator that marks a complex function as analytic.

    At first call, samples ``probes`` random points in the unit
    disk and runs :func:`check_cauchy_riemann` at each; if any
    residual exceeds ``atol``, raises :class:`NotHolomorphicError`
    with the offending point and residual.  The check then runs
    once per probe set per function; subsequent calls are
    overhead-free.

    Usage::

        @complex.analytic
        def f(z): return complex_mul(z, z)
    """
    def _decorate(fn):
        verified = [False]

        def _wrap(z):
            if not verified[0]:
                rng = np.random.RandomState(0)
                probe_points = rng.randn(probes) + 1j * rng.randn(probes)
                for p in probe_points:
                    passes, residual = check_cauchy_riemann(
                        fn, p, h=h, atol=atol,
                    )
                    if not passes:
                        raise NotHolomorphicError(
                            f"@analytic({fn.__qualname__}) failed CR at "
                            f"z = {p:g}; ∂f/∂z̄ residual = {residual:.3g} "
                            f"> tol {atol}"
                        )
                verified[0] = True
            return fn(z)

        _wrap.__wrapped__ = fn
        _wrap.__name__ = getattr(fn, "__name__", "analytic_wrapped")
        _wrap.__qualname__ = getattr(fn, "__qualname__", _wrap.__name__)
        return _wrap

    # Support both `@analytic` and `@analytic(probes=...)` usage.
    if fn is not None and callable(fn):
        return _decorate(fn)
    return _decorate


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Conformal energy example.
#
# Concrete demonstration of the M7 thesis: a quadratic energy
# defined in ℂ stays geometrically meaningful when lifted to S²
# via stereographic projection, because the projection is conformal.
#
# Given a target point ``p* ∈ S²``, the energy of any other
# ``p ∈ S²`` is the squared distance between their stereographic
# projections::
#
#     E(p) = ‖stereographic(p) − stereographic(p*)‖²
#
# This is exactly :func:`tessera.energy.norm_sq` applied to the
# 2-D difference vector — the connection M7 promised in the plan.
# Minimum is at ``p = p*``, energy = 0.
# ─────────────────────────────────────────────────────────────────────────────

def conformal_energy_on_sphere(
    p: Any, p_target: Any,
) -> np.ndarray:
    """Stereographic-lifted quadratic energy on S².

    Both inputs are 3-vectors on the unit sphere.  Returns a
    scalar (or batched scalar) energy via
    :func:`tessera.energy.norm_sq` applied to the 2-D difference
    of stereographic projections.

    Routes through the existing :mod:`tessera.energy` namespace
    so the M7 surface composes with the M6 energy IR rather than
    inventing parallel machinery.
    """
    from . import energy as _tensor_energy
    zeta = stereographic(p)
    zeta_star = stereographic(p_target)
    diff = np.stack(
        [zeta.re - zeta_star.re, zeta.im - zeta_star.im], axis=-1,
    )
    return _tensor_energy.norm_sq(diff)
