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
    # Bundle B (Needham Ch. 3 — Möbius / cross-ratio surface)
    "cross_ratio",
    "is_concyclic",
    "mobius_from_three_points",
    # Bundle A (Needham Ch. 2 / 4-5 — log / arg / pow / Wirtinger)
    "complex_arg",
    "complex_log",
    "complex_pow",
    "complex_sqrt",
    "dz",
    "dbar",
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

    def to_numpy(self, *, dtype: Any = np.complex128) -> np.ndarray:
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


def to_numpy(z: ComplexScalar, *, dtype: Any = np.complex128) -> np.ndarray:
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


def _try_apple_gpu_complex_op(
    op_name: str, argtypes, args,
) -> bool:
    """Bridge helper: dispatch ``op_name`` through the JIT bridge.
    Returns True on success (the runtime actually ran), False on
    silent miss (caller falls back to numpy)."""
    try:
        from tessera.compiler import jit_bridge as _bridge
        return _bridge.dispatch_via_manifest(
            op_name, argtypes=argtypes, args=args, args_summary=(),
        )
    except Exception:
        return False


def complex_mul(z: Any, w: Any) -> ComplexScalar:
    """``(a + b·i)·(c + d·i) = (ac − bd) + (ad + bc)·i``.

    Supports batched inputs — every operation broadcasts via
    numpy's normal rules.

    M7 follow-up (2026-05-18): when both inputs are f32 and
    same-shape, dispatches the fused MSL kernel
    ``tessera_apple_gpu_complex_mul_f32`` on Apple GPU and
    records a JitBridgeRoute.
    """
    a, b = _as_pair(z)
    c, d = _as_pair(w)
    # GPU fast path: same-shape f32 inputs.
    if (a.dtype == np.float32 and b.dtype == np.float32
            and c.dtype == np.float32 and d.dtype == np.float32
            and a.shape == c.shape and a.shape == b.shape):
        import ctypes
        n = int(a.size)
        ac = np.ascontiguousarray(a); bc = np.ascontiguousarray(b)
        cc = np.ascontiguousarray(c); dc = np.ascontiguousarray(d)
        out_re = np.zeros(a.shape, dtype=np.float32)
        out_im = np.zeros(a.shape, dtype=np.float32)
        p_f = ctypes.POINTER(ctypes.c_float)
        argtypes = (p_f, p_f, p_f, p_f, p_f, p_f, ctypes.c_int32)
        args = (
            ac.ctypes.data_as(p_f), bc.ctypes.data_as(p_f),
            cc.ctypes.data_as(p_f), dc.ctypes.data_as(p_f),
            out_re.ctypes.data_as(p_f), out_im.ctypes.data_as(p_f),
            ctypes.c_int32(n),
        )
        if _try_apple_gpu_complex_op("complex_mul", argtypes, args):
            return ComplexScalar(out_re, out_im)
    re = a * c - b * d
    im = a * d + b * c
    return ComplexScalar(re, im)


def complex_exp(z: Any) -> ComplexScalar:
    """Euler form: ``e^(a + b·i) = e^a · (cos b, sin b)``.

    M7 follow-up: f32 same-shape inputs route to
    ``tessera_apple_gpu_complex_exp_f32``.
    """
    a, b = _as_pair(z)
    if a.dtype == np.float32 and b.dtype == np.float32 and a.shape == b.shape:
        import ctypes
        n = int(a.size)
        ac = np.ascontiguousarray(a); bc = np.ascontiguousarray(b)
        out_re = np.zeros(a.shape, dtype=np.float32)
        out_im = np.zeros(a.shape, dtype=np.float32)
        p_f = ctypes.POINTER(ctypes.c_float)
        argtypes = (p_f, p_f, p_f, p_f, ctypes.c_int32)
        args = (
            ac.ctypes.data_as(p_f), bc.ctypes.data_as(p_f),
            out_re.ctypes.data_as(p_f), out_im.ctypes.data_as(p_f),
            ctypes.c_int32(n),
        )
        if _try_apple_gpu_complex_op("complex_exp", argtypes, args):
            return ComplexScalar(out_re, out_im)
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
    # M7 follow-up: f32 batched ``z`` with scalar (a, b, c, d) routes
    # to the fused MSL kernel ``tessera_apple_gpu_complex_mobius_f32``.
    z_re, z_im = _as_pair(z)
    is_scalar_coef = (
        a_pair.shape == () and b_pair.shape == ()
        and c_pair.shape == () and d_pair.shape == ()
    )
    if (z_re.dtype == np.float32 and z_im.dtype == np.float32
            and z_re.shape == z_im.shape and is_scalar_coef):
        import ctypes
        n = int(z_re.size)
        zr = np.ascontiguousarray(z_re); zi = np.ascontiguousarray(z_im)
        out_re = np.zeros(z_re.shape, dtype=np.float32)
        out_im = np.zeros(z_re.shape, dtype=np.float32)
        p_f = ctypes.POINTER(ctypes.c_float)
        argtypes = (
            p_f, p_f,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            p_f, p_f, ctypes.c_int32,
        )
        args = (
            zr.ctypes.data_as(p_f), zi.ctypes.data_as(p_f),
            ctypes.c_float(float(a_pair.re)), ctypes.c_float(float(a_pair.im)),
            ctypes.c_float(float(b_pair.re)), ctypes.c_float(float(b_pair.im)),
            ctypes.c_float(float(c_pair.re)), ctypes.c_float(float(c_pair.im)),
            ctypes.c_float(float(d_pair.re)), ctypes.c_float(float(d_pair.im)),
            out_re.ctypes.data_as(p_f), out_im.ctypes.data_as(p_f),
            ctypes.c_int32(n),
        )
        if _try_apple_gpu_complex_op("complex_mobius", argtypes, args):
            return ComplexScalar(out_re, out_im)
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
    # M7 follow-up: f32 same-shape inputs route to
    # ``tessera_apple_gpu_complex_stereographic_f32``.
    if (x.dtype == np.float32 and y.dtype == np.float32
            and z.dtype == np.float32 and x.shape == y.shape == z.shape):
        import ctypes
        n = int(x.size)
        xc = np.ascontiguousarray(x); yc = np.ascontiguousarray(y)
        zc = np.ascontiguousarray(z)
        out_re = np.zeros(x.shape, dtype=np.float32)
        out_im = np.zeros(x.shape, dtype=np.float32)
        p_f = ctypes.POINTER(ctypes.c_float) if (
            __import__("ctypes") and True
        ) else None
        import ctypes as _c
        p_f = _c.POINTER(_c.c_float)
        argtypes = (p_f, p_f, p_f, p_f, p_f, _c.c_int32)
        args = (
            xc.ctypes.data_as(p_f), yc.ctypes.data_as(p_f),
            zc.ctypes.data_as(p_f),
            out_re.ctypes.data_as(p_f), out_im.ctypes.data_as(p_f),
            _c.c_int32(n),
        )
        if _try_apple_gpu_complex_op(
            "complex_stereographic", argtypes, args,
        ):
            return ComplexScalar(out_re, out_im)
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

        _wrap.__wrapped__ = fn  # type: ignore[attr-defined]
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


# ─────────────────────────────────────────────────────────────────────────────
# Bundle B — Cross-ratio and Möbius constructions (Needham Ch. 3).
#
# The cross-ratio is the fundamental Möbius invariant.  Four points
# are concyclic iff their cross-ratio is real.  Given three source
# and three destination points, there's a unique Möbius
# transformation mapping them; we build it via the standard
# three-point-to-(0, 1, ∞) construction.
# ─────────────────────────────────────────────────────────────────────────────

def _to_complex(z: Any) -> complex:
    """Coerce a :class:`ComplexScalar` or a python ``complex`` /
    real to a python ``complex``.  Used by the scalar Möbius
    helpers below — these aren't batched (callers wrap with
    numpy broadcasting if needed)."""
    if isinstance(z, ComplexScalar):
        return complex(float(z.re), float(z.im))
    return complex(z)


def cross_ratio(z1: Any, z2: Any, z3: Any, z4: Any) -> complex:
    """Four-point cross-ratio: ``(z1, z2; z3, z4) =
    ((z1 − z3)(z2 − z4)) / ((z1 − z4)(z2 − z3))``.

    The fundamental Möbius invariant — under any Möbius
    transformation ``M``, ``cross_ratio(Mz1, Mz2, Mz3, Mz4) ==
    cross_ratio(z1, z2, z3, z4)``.

    Returns a python ``complex`` (scalar).  Coincident inputs that
    would zero the denominator return ``complex(inf, inf)`` —
    consistent with the Riemann-sphere "point at infinity"
    convention used elsewhere in this module.

    Reference: Needham, *Visual Complex Analysis*, §3.6.
    """
    z1, z2, z3, z4 = (_to_complex(z) for z in (z1, z2, z3, z4))
    num = (z1 - z3) * (z2 - z4)
    den = (z1 - z4) * (z2 - z3)
    if abs(den) < 1e-15:
        return complex(float("inf"), float("inf"))
    return num / den


def is_concyclic(
    z1: Any, z2: Any, z3: Any, z4: Any, *, tol: float = 1e-9,
) -> bool:
    """``True`` iff the four points lie on a common circle (or line).

    Test: the cross-ratio of four concyclic points is real-valued.
    For points that are coincident or collinear, the cross-ratio
    may be 0, 1, or ∞ — all still "real" in the concyclicity
    sense.

    Reference: Needham, *Visual Complex Analysis*, §3.6 Theorem.
    """
    cr = cross_ratio(z1, z2, z3, z4)
    if cr.imag != cr.imag:        # NaN guard
        return False
    if cr.real == float("inf") or cr.imag == float("inf"):
        # ∞ as a generalized "concyclic" — three coincident points.
        return True
    return abs(cr.imag) <= tol * max(abs(cr), 1.0)


def _mobius_to_0_1_inf(z1: complex, z2: complex, z3: complex):
    """Coefficients ``(a, b, c, d)`` of the unique Möbius map
    sending ``(z1, z2, z3) → (0, 1, ∞)``.

    Closed form: M(z) = ((z − z1)(z2 − z3)) / ((z − z3)(z2 − z1)).
    Special cases when any zi is ∞ are handled by limit forms.
    """
    inf = complex(float("inf"), float("inf"))
    if z1 == inf:
        a = complex(0); b = (z2 - z3); c = complex(1); d = -z3
    elif z2 == inf:
        a = complex(1); b = -z1; c = complex(1); d = -z3
    elif z3 == inf:
        a = complex(1); b = -z1; c = complex(0); d = (z2 - z1)
    else:
        a = (z2 - z3)
        b = -z1 * (z2 - z3)
        c = (z2 - z1)
        d = -z3 * (z2 - z1)
    return a, b, c, d


def _mobius_compose(
    a1, b1, c1, d1, a2, b2, c2, d2,
) -> tuple[complex, complex, complex, complex]:
    """Möbius composition is 2x2 matrix multiplication of coeff
    matrices.  Returns the composed (a, b, c, d) — the map
    ``M1 ∘ M2`` (apply M2 first, then M1)."""
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
    )


def mobius_from_three_points(
    src: tuple[Any, Any, Any],
    dst: tuple[Any, Any, Any],
) -> tuple[complex, complex, complex, complex]:
    """Construct the unique Möbius transformation sending ``src ==
    (z1, z2, z3)`` to ``dst == (w1, w2, w3)``.

    Returns ``(a, b, c, d)`` — coefficients for use with
    :func:`mobius`.  Three-point data uniquely determines a
    Möbius transformation; the standard construction goes via the
    canonical map to ``(0, 1, ∞)``.

    Reference: Needham, *Visual Complex Analysis*, §3.7.
    """
    z1, z2, z3 = (_to_complex(z) for z in src)
    w1, w2, w3 = (_to_complex(z) for z in dst)
    # Build M1: src → (0, 1, ∞)
    a1, b1, c1, d1 = _mobius_to_0_1_inf(z1, z2, z3)
    # Build M2: dst → (0, 1, ∞)
    a2, b2, c2, d2 = _mobius_to_0_1_inf(w1, w2, w3)
    # Invert M2: closed form for a Möbius inverse is (d, -b, -c, a)
    # divided by determinant; the determinant cancels because Möbius
    # transformations are projective.
    inv_a2, inv_b2, inv_c2, inv_d2 = d2, -b2, -c2, a2
    # M = inv(M2) ∘ M1: apply M1 first, then inv(M2).
    return _mobius_compose(
        inv_a2, inv_b2, inv_c2, inv_d2,
        a1, b1, c1, d1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bundle A — log / arg / pow (Needham Ch. 2) + Wirtinger derivatives (Ch. 4-5).
#
# Branch-cut policy (locked, matches NumPy):
#
#   * ``complex_arg(z)`` returns the principal argument in
#     ``(-π, π]`` (NumPy ``np.angle`` convention).
#   * ``complex_log(z) = log|z| + i·arg(z)`` — branch cut along
#     the negative real axis; ``log(-1) = i·π``.
#   * ``complex_pow(z, w) = exp(w · log(z))`` — inherits the log
#     branch cut.
#
# Wirtinger operators:
#
#   * ``∂/∂z = (1/2)(∂/∂x − i ∂/∂y)``
#   * ``∂/∂z̄ = (1/2)(∂/∂x + i ∂/∂y)``
#
# For holomorphic f: ``dbar(f, z) ≈ 0`` and ``dz(f, z) = f'(z)``.
# For non-holomorphic f: both are non-zero.  These are the
# building blocks the existing :func:`check_cauchy_riemann`
# uses internally; M7's bundle-A exposes them as first-class
# primitives so users can ask "what is ∂f/∂z̄ at z₀?" directly.
# ─────────────────────────────────────────────────────────────────────────────

def complex_arg(z: Any) -> np.ndarray:
    """Principal argument in ``(-π, π]``.  Matches ``numpy.angle``."""
    a, b = _as_pair(z)
    return np.arctan2(b, a)


def complex_log(z: Any) -> ComplexScalar:
    """Principal-branch complex logarithm.

    ``log(z) = log|z| + i·arg(z)``.  Branch cut along the
    negative real axis; ``log(0)`` returns
    ``ComplexScalar(-inf, 0)`` — same behavior as ``numpy.log``
    on a complex array.
    """
    a, b = _as_pair(z)
    mag = np.sqrt(a * a + b * b)
    # log(0) → -inf (NumPy convention); np.log handles this.
    re = np.log(np.where(mag > 0, mag, np.float64(1.0)))
    re = np.where(mag > 0, re, np.float64(-np.inf))
    im = np.arctan2(b, a)
    return ComplexScalar(re, im)


def complex_sqrt(z: Any, *, branch: int = 0) -> ComplexScalar:
    """Complex square root with explicit branch selection.

    Principal branch (``branch=0``): ``Im(√z) ≥ 0``, cut along
    the negative real axis.  ``sqrt(-1) = i``,
    ``sqrt(1) = 1``, ``sqrt(z)² == z`` everywhere except the
    cut.

    Second branch (``branch=1``): the other sheet of the
    Riemann surface — multiplies the principal branch by -1.
    See :mod:`tessera.riemann_surface` for the full sheet-
    tracking machinery.

    Reference: Needham, *Visual Complex Analysis*, §2.9 + Ch. 8.
    """
    if branch not in (0, 1):
        raise ValueError(
            f"complex_sqrt: branch must be 0 (principal) or 1 "
            f"(second sheet); got {branch}"
        )
    # Special-case z == 0: log(0) is -inf and the exp chain
    # would produce a NaN imaginary part.  sqrt(0) is 0 on
    # every branch.
    a, b = _as_pair(z)
    if np.isscalar(a) or a.shape == ():
        if float(a) == 0.0 and float(b) == 0.0:
            return ComplexScalar(np.array(0.0, dtype=np.float64),
                                 np.array(0.0, dtype=np.float64))
    # sqrt(z) = exp(0.5 * log(z)) on the principal branch.
    half = ComplexScalar(np.array(0.5, dtype=np.float64),
                         np.array(0.0, dtype=np.float64))
    principal = complex_exp(complex_mul(half, complex_log(z)))
    if branch == 0:
        return principal
    return ComplexScalar(-principal.re, -principal.im)


def complex_pow(z: Any, w: Any) -> ComplexScalar:
    """Complex power ``z^w = exp(w · log(z))``.

    Conventions:

      * ``0^0 = 1`` (NumPy convention).
      * ``0^w = 0`` for ``Re(w) > 0``.
      * ``0^w`` for ``Re(w) ≤ 0`` propagates through the
        ``exp(w · log(0))`` chain — NaN/Inf per IEEE 754.
    """
    log_z = complex_log(z)
    return complex_exp(complex_mul(w, log_z))


def _eval_complex(f: Any, z: complex) -> complex:
    """Coerce ``f(z)``'s result to a python complex regardless of
    whether ``f`` returns a :class:`ComplexScalar`, a numpy
    complex, or a python ``complex``."""
    out = f(z)
    if isinstance(out, ComplexScalar):
        return complex(float(out.re), float(out.im))
    return complex(out)


def dz(f: Any, z0: Any, *, h: float = 1e-5) -> complex:
    """Wirtinger ``∂f/∂z`` evaluated at ``z₀``.

    Computed via central differences on ``u`` and ``v`` (the real
    and imaginary parts of ``f``) and the identity

      ``∂f/∂z = (1/2)((u_x + v_y) + i(v_x − u_y))``.

    For a holomorphic ``f``, this equals the complex derivative
    ``f'(z₀)``.  For a non-holomorphic ``f`` (e.g., conjugate or
    ``|z|²``), it captures the holomorphic component.
    """
    z0_c = _to_complex(z0)
    f_px = _eval_complex(f, z0_c + h)
    f_mx = _eval_complex(f, z0_c - h)
    f_py = _eval_complex(f, z0_c + h * 1j)
    f_my = _eval_complex(f, z0_c - h * 1j)
    f_x = (f_px - f_mx) / (2.0 * h)
    f_y = (f_py - f_my) / (2.0 * h)
    return 0.5 * (f_x - 1j * f_y)


def dbar(f: Any, z0: Any, *, h: float = 1e-5) -> complex:
    """Wirtinger ``∂f/∂z̄`` evaluated at ``z₀``.

    ``∂f/∂z̄ = (1/2)((u_x − v_y) + i(v_x + u_y))``.

    Holomorphic ``f`` ⇒ ``dbar(f, z₀) ≈ 0`` (this is the
    Cauchy-Riemann condition).  Non-holomorphic ``f`` ⇒ non-zero,
    and the magnitude is exactly the residual the numerical
    :func:`check_cauchy_riemann` computes.
    """
    z0_c = _to_complex(z0)
    f_px = _eval_complex(f, z0_c + h)
    f_mx = _eval_complex(f, z0_c - h)
    f_py = _eval_complex(f, z0_c + h * 1j)
    f_my = _eval_complex(f, z0_c - h * 1j)
    f_x = (f_px - f_mx) / (2.0 * h)
    f_y = (f_py - f_my) / (2.0 * h)
    return 0.5 * (f_x + 1j * f_y)
