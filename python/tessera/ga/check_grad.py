"""GA6 prep — multivector autodiff verification harness.

This module is **GA6 preparation infrastructure**, not the GA6
implementation itself.  Per the sprint plan + the milestone status
doc, GA6 (Clifford reverse-mode autodiff) is the highest-risk
sprint on the GA roadmap — multivector VJPs require correctly
threading the **reverse anti-automorphism** through every chain-rule
application.  Front-loading the test infrastructure lets us catch
sign / order errors at the harness level before they get burned
into the VJP table.

What this module provides:

  - :func:`multivector_check_grad` — finite-difference a
    Multivector-valued function and compare against a candidate
    VJP closure.  Returns the worst-case relative error so callers
    can ``assert err < tol`` in tests.
  - :func:`multivector_check_grad_scalar` — analogue for functions
    that return a numpy scalar (the common case for
    ``norm`` / ``inner`` / ``norm_squared``).

Both helpers are pure numpy (no MSL); they exist so GA6 can ship
its VJP table against a verifier that's already proven on the
handful of analytic VJPs Tessera already has.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from tessera.ga.multivector import Multivector


__all__ = [
    "multivector_check_grad",
    "multivector_check_grad_scalar",
]


def _finite_diff_mv(
    fn: Callable[[Multivector], Multivector],
    mv: Multivector,
    out_cotangent: np.ndarray,
    *,
    eps: float = 1e-3,
) -> np.ndarray:
    """Central finite-difference VJP for a Multivector-valued ``fn``.

    Returns the cotangent on ``mv.coefficients`` as a plain numpy
    array of the same shape.  ``out_cotangent`` is the seed cotangent
    on the output (same shape as ``fn(mv).coefficients``).
    """
    algebra = mv.algebra
    coeffs = np.asarray(mv.coefficients, dtype=np.float64)
    grad = np.zeros_like(coeffs)
    flat_in = coeffs.reshape(-1)
    flat_grad = grad.reshape(-1)
    for i in range(flat_in.size):
        plus = flat_in.copy()
        plus[i] += eps
        minus = flat_in.copy()
        minus[i] -= eps
        out_plus = np.asarray(
            fn(Multivector(plus.reshape(coeffs.shape), algebra)).coefficients,
            dtype=np.float64,
        )
        out_minus = np.asarray(
            fn(Multivector(minus.reshape(coeffs.shape), algebra)).coefficients,
            dtype=np.float64,
        )
        dout = (out_plus - out_minus) / (2.0 * eps)
        flat_grad[i] = float(np.sum(out_cotangent * dout))
    return grad


def _finite_diff_mv_scalar(
    fn: Callable[[Multivector], np.ndarray],
    mv: Multivector,
    *,
    eps: float = 1e-3,
) -> np.ndarray:
    """Central finite-difference VJP for a scalar-valued ``fn``.

    Used when ``fn`` returns a numpy scalar (or array of scalars),
    e.g., ``ga.norm`` / ``ga.inner`` / ``ga.norm_squared``.
    """
    algebra = mv.algebra
    coeffs = np.asarray(mv.coefficients, dtype=np.float64)
    grad = np.zeros_like(coeffs)
    flat_in = coeffs.reshape(-1)
    flat_grad = grad.reshape(-1)
    for i in range(flat_in.size):
        plus = flat_in.copy()
        plus[i] += eps
        minus = flat_in.copy()
        minus[i] -= eps
        out_plus = float(
            np.asarray(fn(Multivector(plus.reshape(coeffs.shape), algebra)),
                       dtype=np.float64)
        )
        out_minus = float(
            np.asarray(fn(Multivector(minus.reshape(coeffs.shape), algebra)),
                       dtype=np.float64)
        )
        flat_grad[i] = (out_plus - out_minus) / (2.0 * eps)
    return grad


def multivector_check_grad(
    fn: Callable[[Multivector], Multivector],
    vjp: Callable[[Multivector, np.ndarray], np.ndarray],
    mv: Multivector,
    out_cotangent: np.ndarray,
    *,
    eps: float = 1e-3,
    rtol: float = 5e-4,
    atol: float = 1e-6,
) -> tuple[bool, float]:
    """Verify a candidate VJP against finite differences.

    Parameters
    ----------
    fn : Callable[[Multivector], Multivector]
        The forward op under test.
    vjp : Callable[[Multivector, np.ndarray], np.ndarray]
        Candidate VJP. Takes ``(input_mv, out_cotangent)`` and
        returns the cotangent on ``input_mv.coefficients`` as a
        numpy array of the same shape.
    mv : Multivector
        Point of linearization.
    out_cotangent : np.ndarray
        Seed cotangent on the output (must match
        ``fn(mv).coefficients.shape``).

    Returns
    -------
    (ok, max_rel_err) : tuple[bool, float]
        ``ok`` is ``True`` iff ``np.allclose`` holds at ``(rtol, atol)``;
        ``max_rel_err`` is the largest pointwise relative error.
    """
    fd = _finite_diff_mv(fn, mv, out_cotangent, eps=eps)
    analytic = np.asarray(vjp(mv, out_cotangent), dtype=np.float64)
    diff = np.abs(fd - analytic)
    scale = np.maximum(np.abs(fd), np.abs(analytic)) + atol
    rel = (diff / scale).max() if diff.size else 0.0
    ok = bool(np.allclose(fd, analytic, rtol=rtol, atol=atol))
    return ok, float(rel)


def multivector_check_grad_scalar(
    fn: Callable[[Multivector], np.ndarray],
    vjp: Callable[[Multivector], np.ndarray],
    mv: Multivector,
    *,
    eps: float = 1e-3,
    rtol: float = 5e-4,
    atol: float = 1e-6,
) -> tuple[bool, float]:
    """Verify a candidate scalar-VJP against finite differences.

    Like :func:`multivector_check_grad` but for ops that return a
    numpy scalar (``norm`` / ``inner`` / ``norm_squared``).  The
    ``vjp`` callable here takes only ``(input_mv,)`` and returns
    the input cotangent (the output cotangent is implicit = 1).
    """
    fd = _finite_diff_mv_scalar(fn, mv, eps=eps)
    analytic = np.asarray(vjp(mv), dtype=np.float64)
    diff = np.abs(fd - analytic)
    scale = np.maximum(np.abs(fd), np.abs(analytic)) + atol
    rel = (diff / scale).max() if diff.size else 0.0
    ok = bool(np.allclose(fd, analytic, rtol=rtol, atol=atol))
    return ok, float(rel)
