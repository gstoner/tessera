"""Host-level solver primitives that the capability plans share (P2 tranche).

First entry: ``tridiagonal_solve`` — the Thomas algorithm, the op the PDE plan
calls "required for Crank–Nicolson; missing everywhere today"
(PDE_STENCIL_CAPABILITY_PLAN.md §III.1) and the S6 tranche's remaining genuine
gap after the scan/segment_sum premise was corrected by measurement.

Reference-tier discipline matches the game family: fp64 accumulation with the
right-hand side's storage dtype preserved (Decision #15a), fail-closed shape
and diagonal-dominance-free zero-pivot contracts (a singular tridiagonal system
is an error, never a NaN that flows), and a full VJP through the adjoint
system — the transpose solve — so reverse-mode AD needs no unrolling.
"""

from __future__ import annotations

import numpy as np

from .custom import custom_primitive


def _storage_dtype(x: np.ndarray) -> np.dtype:
    dt = np.asarray(x).dtype
    if dt.kind in ("i", "u", "b"):
        return np.dtype(np.float64)
    return dt


def _thomas(dl: np.ndarray, d: np.ndarray, du: np.ndarray,
            b: np.ndarray) -> np.ndarray:
    """Solve the tridiagonal system in fp64 (Thomas / no pivoting).

    ``dl``/``d``/``du`` are the sub/main/super diagonals of length n (dl[0]
    and du[n-1] are ignored by convention); ``b`` is [..., n] (batched over
    leading axes). Zero forward-elimination pivots fail closed.
    """
    dl = np.asarray(dl, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    du = np.asarray(du, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    n = d.shape[-1]
    if dl.shape[-1] != n or du.shape[-1] != n:
        raise ValueError(
            f"tridiagonal_solve diagonals must share length n = {n}; got "
            f"dl={dl.shape[-1]}, du={du.shape[-1]}")
    if bb.shape[-1] != n:
        raise ValueError(
            f"tridiagonal_solve rhs trailing extent must be n = {n}; got "
            f"{bb.shape[-1]}")
    # Forward elimination (vectorized over the batch axes of b).
    cp = np.empty(n, dtype=np.float64)
    dp = np.empty(bb.shape, dtype=np.float64)
    piv = d[0]
    if piv == 0.0:
        raise ValueError(
            "tridiagonal_solve pivot is exactly zero at row 0 — the system "
            "is singular for the Thomas recurrence; a singular solve fails "
            "closed rather than emitting NaNs")
    cp[0] = du[0] / piv
    dp[..., 0] = bb[..., 0] / piv
    for i in range(1, n):
        piv = d[i] - dl[i] * cp[i - 1]
        if piv == 0.0:
            raise ValueError(
                f"tridiagonal_solve pivot is exactly zero at row {i} — the "
                "system is singular for the Thomas recurrence; a singular "
                "solve fails closed rather than emitting NaNs")
        cp[i] = du[i] / piv
        dp[..., i] = (bb[..., i] - dl[i] * dp[..., i - 1]) / piv
    x = np.empty_like(dp)
    x[..., n - 1] = dp[..., n - 1]
    for i in range(n - 2, -1, -1):
        x[..., i] = dp[..., i] - cp[i] * x[..., i + 1]
    return x


def _tridiagonal_solve_impl(dl: np.ndarray, d: np.ndarray, du: np.ndarray,
                            b: np.ndarray) -> np.ndarray:
    return _thomas(dl, d, du, b).astype(_storage_dtype(b), copy=False)


def _tridiagonal_solve_vjp(dout: np.ndarray, dl: np.ndarray, d: np.ndarray,
                           du: np.ndarray, b: np.ndarray, **_kw: object):
    """Adjoint of ``x = A⁻¹ b``: solve the TRANSPOSE system.

    ``λ = A⁻ᵀ dout``; then ``∂b = λ`` and, from ``∂A = −λ xᵀ`` restricted to
    the three diagonals: ``∂d[i] = −λ[i]·x[i]``, ``∂dl[i] = −λ[i]·x[i−1]``,
    ``∂du[i] = −λ[i]·x[i+1]``. Aᵀ is tridiagonal with the sub/super diagonals
    exchanged (shifted by one under this storage convention), so the adjoint
    reuses the same Thomas kernel — the transpose IS another tridiagonal
    solve, which is what makes the VJP O(n).
    """
    x = _thomas(dl, d, du, b)
    n = np.asarray(d).shape[-1]
    dlf = np.asarray(dl, dtype=np.float64)
    duf = np.asarray(du, dtype=np.float64)
    # Aᵀ in the same (dl, d, du) storage: (Aᵀ)[i, i-1] = A[i-1, i] = du[i-1].
    dl_t = np.zeros(n)
    dl_t[1:] = duf[:-1]
    du_t = np.zeros(n)
    du_t[:-1] = dlf[1:]
    lam = _thomas(dl_t, np.asarray(d, dtype=np.float64), du_t,
                  np.asarray(dout, dtype=np.float64))
    grad_b = lam
    # Batched systems share the diagonals: sum diagonal cotangents over the
    # leading axes.
    batch_axes = tuple(range(lam.ndim - 1))
    grad_d = -(lam * x).sum(axis=batch_axes) if batch_axes else -(lam * x)
    lx = np.zeros_like(lam)
    lx[..., 1:] = lam[..., 1:] * x[..., :-1]
    grad_dl = lx.sum(axis=batch_axes) if batch_axes else lx
    ux = np.zeros_like(lam)
    ux[..., :-1] = lam[..., :-1] * x[..., 1:]
    grad_du = ux.sum(axis=batch_axes) if batch_axes else ux
    return -grad_dl, grad_d, -grad_du, grad_b


tridiagonal_solve = custom_primitive(
    "tridiagonal_solve", vjp=_tridiagonal_solve_vjp,
)(_tridiagonal_solve_impl)
