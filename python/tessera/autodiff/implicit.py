"""Implicit differentiation: custom_root, adjoint state, and IHVP (T3).

Motivation (from the *Elements of Differentiable Programming* review). Many
useful functions are defined implicitly — as the root of ``F(x, θ) = 0`` or the
solution of an optimization problem — and do not decompose into elementary ops,
so ordinary autodiff cannot reach through them (Blondel & Roulet Ch. 10). The
implicit function theorem gives their derivatives directly. With
``A ≔ ∂₁F(x*, θ)`` and ``B ≔ ∂₂F(x*, θ)``:

    JVP:  solve  A t = -B v            (tangent of x* along θ-direction v)
    VJP:  solve  Aᵀ r =  u  →  ∂L/∂θ = -Bᵀ r

so the whole thing is built from the residual's own JVP/VJP and a linear solve —
no unrolling of the solver, and constant memory regardless of solver iterations
(§10.4). The adjoint-state method (§10.5) is the same identity specialized to a
constrained objective ``L(x*(w), w)`` with ``c(x*, w) = 0``.

Both need a linear solve accessed only through matvecs. This module ships a
matrix-free conjugate-gradient solver (Algorithm 8.1) for the SPD case and a
dense fallback for the general case, matching the CG/GMRES vocabulary the
compiler already names in ``compiler/solver_config.py`` (Decision #23: reference
vocabulary, reimplemented here — no SciPy dependency).

This is the numpy reference lane, consistent with the rest of ``autodiff/``. It
unblocks second-order optimizers (IHVP → Newton / natural gradient), energy /
Langevin samplers, and the manifold / OT work, none of which had a differentiable
fixed point before.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "TesseraImplicitDiffError",
    "cg_solve",
    "ihvp",
    "root_vjp",
    "root_jvp",
    "custom_root",
    "adjoint_state_grad",
]


class TesseraImplicitDiffError(RuntimeError):
    """Raised on a non-convergent solve or a malformed implicit-diff request."""


# ── Matrix-free conjugate gradient (Blondel & Roulet Algorithm 8.1) ──────────
def cg_solve(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    tol: float = 1e-8,
    maxiter: int | None = None,
) -> np.ndarray:
    """Solve ``A x = b`` for symmetric positive-definite ``A`` via CG.

    ``A`` is accessed only through ``matvec(v) = A @ v`` (matrix-free). Raises
    ``TesseraImplicitDiffError`` if it fails to reach ``tol`` within ``maxiter``
    (default ``10 * n``) — never returns a silently-unconverged result.
    """
    b = np.asarray(b, dtype=np.float64)
    n = b.size
    shape = b.shape
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    r = b - np.asarray(matvec(x), dtype=np.float64)
    p = r.copy()
    rs_old = float(np.vdot(r, r).real)
    if rs_old <= tol * tol:
        return x
    limit = maxiter if maxiter is not None else 10 * max(n, 1)
    for _ in range(limit):
        Ap = np.asarray(matvec(p), dtype=np.float64)
        pAp = float(np.vdot(p, Ap).real)
        if pAp == 0.0:
            raise TesseraImplicitDiffError(
                "CG breakdown: pᵀAp = 0 (A may not be positive-definite)"
            )
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(np.vdot(r, r).real)
        if rs_new <= tol * tol:
            return x.reshape(shape)
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    raise TesseraImplicitDiffError(
        f"CG did not converge to tol={tol} in {limit} iterations "
        f"(residual {np.sqrt(rs_old):.3e})"
    )


# ── Inverse-Hessian vector product (Blondel & Roulet §8.4) ───────────────────
def ihvp(
    fn: Callable[[np.ndarray], Any],
    x: np.ndarray,
    u: np.ndarray,
    *,
    tol: float = 1e-8,
    maxiter: int | None = None,
    eps: float = 1e-4,
) -> np.ndarray:
    """Inverse-Hessian vector product ``∇²fn(x)⁻¹ u`` via matrix-free CG.

    The Hessian is applied only through ``hvp`` (never materialized), so this is
    O(n) memory. ``fn`` must be scalar-valued and locally strictly convex at
    ``x`` (SPD Hessian) for CG to apply — otherwise use a regularized Hessian or
    a general solver. This is the linear map Newton's method needs
    (``x - ∇²L⁻¹ ∇L``) without forming or factorizing the Hessian.
    """
    from .grad import hvp as _hvp

    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)

    def _matvec(v: np.ndarray) -> np.ndarray:
        return np.asarray(_hvp(fn, x, v.reshape(x.shape), eps=eps), dtype=np.float64)

    return cg_solve(_matvec, u, tol=tol, maxiter=maxiter)


# ── Jacobians of the residual (finite-difference reference matvecs) ──────────
def _partial_jacobian_matvecs(
    F: Callable[..., np.ndarray],
    args: Sequence[np.ndarray],
    argnum: int,
    *,
    eps: float = 1e-6,
):
    """Return ``(matvec, rmatvec)`` for ``J = ∂_argnum F(*args)``.

    ``matvec(v) = J @ v`` and ``rmatvec(r) = Jᵀ @ r``, both by central finite
    differences on ``F`` — the residual's JVP and VJP without requiring the user
    to supply them. ``J`` is treated as a linear map from the flattened
    ``args[argnum]`` space to the flattened output space.
    """
    args = [np.asarray(a, dtype=np.float64) for a in args]
    a = args[argnum]
    f0 = np.asarray(F(*args), dtype=np.float64)
    out_shape = f0.shape

    def _perturbed(delta: np.ndarray) -> np.ndarray:
        new = list(args)
        new[argnum] = a + delta
        return np.asarray(F(*new), dtype=np.float64)

    def matvec(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(a.shape)
        return ((_perturbed(eps * v) - _perturbed(-eps * v)) / (2 * eps)).reshape(-1)

    # Jᵀ via the definition ⟨J v, r⟩ = ⟨v, Jᵀ r⟩: build columns only when needed.
    # For the modest sizes of the reference lane, materialize J once for rmatvec.
    n_in = a.size
    n_out = int(np.prod(out_shape)) if out_shape else 1

    def rmatvec(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        cols = np.empty((n_out, n_in), dtype=np.float64)
        basis = np.zeros(n_in, dtype=np.float64)
        for j in range(n_in):
            basis[j] = 1.0
            cols[:, j] = matvec(basis)
            basis[j] = 0.0
        return cols.T @ r  # Jᵀ r

    return matvec, rmatvec, out_shape, a.shape


# ── custom_root: differentiate x*(θ) defined by F(x*, θ) = 0 ─────────────────
def root_vjp(
    F: Callable[..., np.ndarray],
    solution: np.ndarray,
    params: Sequence[np.ndarray],
    cotangent: np.ndarray,
    *,
    argnums: int | Sequence[int] = 0,
    eps: float = 1e-6,
) -> Any:
    """VJP of an implicit root ``x*(params)`` with ``F(x*, params) = 0``.

    Given the solution ``x*`` and an output cotangent ``u = ∂L/∂x*``, returns
    ``∂L/∂params[argnum]`` for each requested ``argnum`` (a single array if
    ``argnums`` is an int). Implements ``Aᵀ r = u`` then ``-Bᵀ r`` where
    ``A = ∂_x F``, ``B = ∂_param F`` (§10.4). The solve is dense here (robust for
    the reference lane); ``A`` need not be symmetric.
    """
    x = np.asarray(solution, dtype=np.float64)
    full_args = [x, *(np.asarray(p, dtype=np.float64) for p in params)]
    u = np.asarray(cotangent, dtype=np.float64).reshape(-1)

    # A = ∂₁F(x*, params): dense Jacobian in x (arg 0 of F).
    A_matvec, _A_rmatvec, out_shape, _ = _partial_jacobian_matvecs(
        F, full_args, 0, eps=eps
    )
    n = x.size
    A = np.empty((int(np.prod(out_shape)) if out_shape else 1, n), dtype=np.float64)
    basis = np.zeros(n, dtype=np.float64)
    for j in range(n):
        basis[j] = 1.0
        A[:, j] = A_matvec(basis)
        basis[j] = 0.0
    if A.shape[0] != A.shape[1]:
        raise TesseraImplicitDiffError(
            f"root_vjp expects square ∂_xF (got {A.shape}); F must map to the "
            f"solution space"
        )
    try:
        r = np.linalg.solve(A.T, u)
    except np.linalg.LinAlgError as exc:
        raise TesseraImplicitDiffError(
            "∂_xF is singular at the solution; the implicit function theorem "
            "does not apply here"
        ) from exc

    if isinstance(argnums, int):
        single = True
        idxs: tuple[int, ...] = (argnums,)
    else:
        single = False
        idxs = tuple(argnums)
    grads = []
    for an in idxs:
        # B = ∂_{param an} F ; param an is F-argument (an + 1).
        _B_matvec, B_rmatvec, _os, param_shape = _partial_jacobian_matvecs(
            F, full_args, an + 1, eps=eps
        )
        grad = -B_rmatvec(r).reshape(param_shape)
        grads.append(grad)
    return grads[0] if single else tuple(grads)


def root_jvp(
    F: Callable[..., np.ndarray],
    solution: np.ndarray,
    params: Sequence[np.ndarray],
    tangents: Sequence[np.ndarray | None],
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """JVP of an implicit root: solve ``A t = -Σ Bᵢ vᵢ`` (§10.4).

    ``tangents`` are the input directions for each param (``None`` to skip).
    Returns the tangent of ``x*`` — the directional derivative of the solution.
    """
    x = np.asarray(solution, dtype=np.float64)
    full_args = [x, *(np.asarray(p, dtype=np.float64) for p in params)]
    A_matvec, _A_rmatvec, out_shape, _ = _partial_jacobian_matvecs(
        F, full_args, 0, eps=eps
    )
    n = x.size
    A = np.empty((int(np.prod(out_shape)) if out_shape else 1, n), dtype=np.float64)
    basis = np.zeros(n, dtype=np.float64)
    for j in range(n):
        basis[j] = 1.0
        A[:, j] = A_matvec(basis)
        basis[j] = 0.0

    rhs = np.zeros(n, dtype=np.float64)
    for i, v in enumerate(tangents):
        if v is None:
            continue
        Bi_matvec, _rm, _os, _ps = _partial_jacobian_matvecs(
            F, full_args, i + 1, eps=eps
        )
        rhs = rhs - Bi_matvec(np.asarray(v, dtype=np.float64))
    try:
        t = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError as exc:
        raise TesseraImplicitDiffError(
            "∂_xF is singular at the solution; IFT does not apply"
        ) from exc
    return t.reshape(x.shape)


def custom_root(
    optimality: Callable[..., np.ndarray],
    *,
    argnums: int | Sequence[int] = 0,
    eps: float = 1e-6,
) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """Wrap a solver so its output is differentiable through the root condition.

    ``optimality(x, *params) -> residual`` is the function whose zero defines the
    solution (a stationarity condition ``∇₁f = 0``, a fixed-point residual, or a
    nonlinear system). The decorated ``solver(*params) -> x*`` gains
    ``.vjp(x*, params, u)`` and ``.jvp(x*, params, tangents)`` methods computing
    the implicit derivatives without unrolling the solver.

    Example::

        @custom_root(lambda x, theta: x * x - theta)   # x*(θ) = √θ
        def sqrt_solver(theta):
            return np.sqrt(theta)

        xs = sqrt_solver(theta)
        dtheta = sqrt_solver.vjp(xs, (theta,), np.ones_like(xs))  # 1/(2√θ)
    """

    def wrap(solver: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        def vjp(solution, params, cotangent):
            return root_vjp(
                optimality, solution, params, cotangent, argnums=argnums, eps=eps
            )

        def jvp(solution, params, tangents):
            return root_jvp(optimality, solution, params, tangents, eps=eps)

        solver.vjp = vjp  # type: ignore[attr-defined]
        solver.jvp = jvp  # type: ignore[attr-defined]
        solver.optimality = optimality  # type: ignore[attr-defined]
        return solver

    return wrap


# ── Adjoint-state method (Blondel & Roulet §10.5) ────────────────────────────
def adjoint_state_grad(
    L: Callable[[np.ndarray, np.ndarray], float],
    c: Callable[[np.ndarray, np.ndarray], np.ndarray],
    state: np.ndarray,
    w: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Gradient of ``L(s*(w), w)`` s.t. ``c(s*(w), w) = 0`` (adjoint state).

    Solves the adjoint system ``∂₁cᵀ r = -∇₁L`` then returns
    ``∇₂L + ∂₂cᵀ r`` (§10.5). ``state`` must be the converged ``s*(w)``. This is
    the constant-memory way to differentiate an equality-constrained objective —
    the workhorse behind optimal-control and PDE-constrained gradients.
    """
    s = np.asarray(state, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    def _grad_fd(f, arr, other, which):
        g = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            hi = arr.copy(); hi[idx] += eps
            lo = arr.copy(); lo[idx] -= eps
            if which == 0:
                g[idx] = (f(hi, other) - f(lo, other)) / (2 * eps)
            else:
                g[idx] = (f(other, hi) - f(other, lo)) / (2 * eps)
            it.iternext()
        return g

    grad1_L = _grad_fd(L, s, w, 0)
    grad2_L = _grad_fd(L, w, s, 1)

    # ∂₁c and ∂₂c as dense Jacobians via finite differences.
    _m1, rmat1, _os1, _ps1 = _partial_jacobian_matvecs(c, [s, w], 0, eps=eps)
    _m2, rmat2, _os2, _ps2 = _partial_jacobian_matvecs(c, [s, w], 1, eps=eps)

    # Build ∂₁c to solve the (small) adjoint system directly.
    n = s.size
    c0 = np.asarray(c(s, w), dtype=np.float64)
    m = c0.size
    dcds = np.empty((m, n), dtype=np.float64)
    basis = np.zeros(n, dtype=np.float64)
    for j in range(n):
        basis[j] = 1.0
        dcds[:, j] = _m1(basis)
        basis[j] = 0.0
    try:
        r = np.linalg.solve(dcds.T, -grad1_L.reshape(-1))
    except np.linalg.LinAlgError as exc:
        raise TesseraImplicitDiffError(
            "∂₁c is singular at the state; adjoint system unsolvable"
        ) from exc
    return (grad2_L.reshape(-1) + rmat2(r)).reshape(w.shape)
