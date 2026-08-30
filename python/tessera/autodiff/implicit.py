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

Both need a linear solve accessed only through matvecs. This module ships
restarted GMRES for general square systems and conjugate gradient (Algorithm
8.1) for the SPD case. A dense solve remains an explicit small-system oracle,
not the default execution path. The implementation matches the CG/GMRES
vocabulary the compiler names in ``compiler/solver_config.py`` without a SciPy
dependency.

This is the numpy reference lane, consistent with the rest of ``autodiff/``. It
unblocks second-order optimizers (IHVP → Newton / natural gradient), energy /
Langevin samplers, and the manifold / OT work, none of which had a differentiable
fixed point before.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .operator import OperatorTangent, certify_root

#: How many Jacobian entries the finite-difference adjoint may memoize.
#: 8M float64 is 64 MB — a cap against a pathological allocation, not a real
#: working limit: at n ≈ 2800 a single un-memoized adjoint application already
#: costs 5600 residual evaluations, so the streaming path is unusable long
#: before the memo becomes expensive.
_FD_JACOBIAN_ELEMENT_BUDGET = 8 * 1024 * 1024

__all__ = [
    "TesseraImplicitDiffError",
    "cg_solve",
    "gmres_solve",
    "LinearSolveInfo",
    "ResidualLinearization",
    "ihvp",
    "root_vjp",
    "root_jvp",
    "custom_root",
    "adjoint_state_grad",
]


class TesseraImplicitDiffError(RuntimeError):
    """Raised on a non-convergent solve or a malformed implicit-diff request."""


@dataclass(frozen=True)
class LinearSolveInfo:
    """Measured convergence record for a matrix-free linear solve."""

    algorithm: str
    iterations: int
    restarts: int
    matvecs: int
    residual_norm: float
    rhs_norm: float
    converged: bool
    product_source: str = "numerical_oracle"


@dataclass(frozen=True)
class ResidualLinearization:
    """Exact matrix-free residual products supplied by compiler-owned AD.

    The four actions are deliberately separate: an implementation may lower
    forward and transposed products to different target packages, and no dense
    Jacobian is implied.  ``root_jvp``/``root_vjp`` retain finite differences
    only as the explicit NumPy oracle when this carrier is absent.
    """

    residual_shape: tuple[int, ...]
    solution_jvp: Callable[[np.ndarray], np.ndarray]
    solution_vjp: Callable[[np.ndarray], np.ndarray]
    parameter_jvp: Callable[[int, np.ndarray], np.ndarray]
    parameter_vjp: Callable[[int, np.ndarray], np.ndarray]
    provenance: str = "compiler_graph_products"


def gmres_solve(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    tol: float = 1e-8,
    maxiter: int | None = None,
    restart: int | None = None,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, LinearSolveInfo]:
    """Solve a general square linear system with restarted matrix-free GMRES.

    Arnoldi uses modified Gram--Schmidt plus one selective reorthogonalization
    pass. Convergence is decided from the *true* residual after each candidate,
    not only the Hessenberg least-squares estimate. This keeps the reference
    reliable for non-normal residual Jacobians and exposes the work counters
    needed by AD-RESIDUAL-EVAL-1.
    """

    rhs = np.asarray(b, dtype=np.float64)
    shape = rhs.shape
    flat_rhs = rhs.reshape(-1)
    n = flat_rhs.size
    if n == 0:
        raise TesseraImplicitDiffError("GMRES requires a non-empty right-hand side")
    if tol <= 0.0:
        raise ValueError("GMRES tolerance must be positive")
    limit = int(maxiter) if maxiter is not None else 10 * n
    width = int(restart) if restart is not None else min(32, n)
    if limit <= 0 or width <= 0:
        raise ValueError("GMRES maxiter and restart must be positive")
    width = min(width, n, limit)
    x = np.zeros(n, dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64).reshape(-1).copy()
    if x.size != n:
        raise TesseraImplicitDiffError("GMRES x0 shape does not match b")

    matvecs = 0

    def apply(v: np.ndarray) -> np.ndarray:
        nonlocal matvecs
        out = np.asarray(matvec(v.reshape(shape)), dtype=np.float64).reshape(-1)
        matvecs += 1
        if out.size != n:
            raise TesseraImplicitDiffError(f"GMRES matvec returned {out.size} values, expected {n}")
        if not np.all(np.isfinite(out)):
            raise TesseraImplicitDiffError("GMRES matvec produced non-finite values")
        return out

    rhs_norm = float(np.linalg.norm(flat_rhs))
    # Relative, not `tol * max(1, ||b||)`: the solution scales linearly with
    # the right-hand side, so an absolute floor makes x = 0 "converged" for
    # every ||b|| <= tol — a 100% error on a cotangent that is merely small
    # (a late-training gradient), reported as a success.
    threshold = tol * rhs_norm
    iterations = 0
    restarts = 0
    residual = flat_rhs - apply(x)
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm <= threshold:
        info = LinearSolveInfo("gmres", 0, 0, matvecs, residual_norm, rhs_norm, True)
        result = x.reshape(shape)
        return (result, info) if return_info else result

    while iterations < limit:
        beta = float(np.linalg.norm(residual))
        basis = np.zeros((width + 1, n), dtype=np.float64)
        hessenberg = np.zeros((width + 1, width), dtype=np.float64)
        basis[0] = residual / beta
        cycle_x = x.copy()
        cycle_width = min(width, limit - iterations)
        for column in range(cycle_width):
            w = apply(basis[column])
            # Modified Gram--Schmidt.
            for row in range(column + 1):
                coeff = float(np.dot(basis[row], w))
                hessenberg[row, column] += coeff
                w -= coeff * basis[row]
            # Daniel--Gragg--Kaufman--Stewart-style reorthogonalization:
            # a second MGS pass is cheap at these reference sizes and prevents
            # loss of orthogonality from producing optimistic residuals.
            for row in range(column + 1):
                correction = float(np.dot(basis[row], w))
                hessenberg[row, column] += correction
                w -= correction * basis[row]
            h_next = float(np.linalg.norm(w))
            hessenberg[column + 1, column] = h_next
            if h_next > np.finfo(np.float64).eps:
                basis[column + 1] = w / h_next

            e1 = np.zeros(column + 2, dtype=np.float64)
            e1[0] = beta
            least_squares, *_ = np.linalg.lstsq(hessenberg[: column + 2, : column + 1], e1, rcond=None)
            candidate = cycle_x + basis[: column + 1].T @ least_squares
            true_residual = flat_rhs - apply(candidate)
            iterations += 1
            residual_norm = float(np.linalg.norm(true_residual))
            if residual_norm <= threshold:
                info = LinearSolveInfo(
                    "gmres",
                    iterations,
                    restarts,
                    matvecs,
                    residual_norm,
                    rhs_norm,
                    True,
                )
                result = candidate.reshape(shape)
                return (result, info) if return_info else result
            if h_next <= np.finfo(np.float64).eps:
                raise TesseraImplicitDiffError("GMRES Arnoldi breakdown before reaching the true-residual tolerance")
            x = candidate
            residual = true_residual
        restarts += 1

    info = LinearSolveInfo("gmres", iterations, restarts, matvecs, residual_norm, rhs_norm, False)
    raise TesseraImplicitDiffError(
        f"GMRES did not converge to tol={tol} in {limit} iterations "
        f"(true residual {info.residual_norm:.3e}, {matvecs} matvecs)"
    )


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
    r: np.ndarray = np.asarray(b - np.asarray(matvec(x), dtype=np.float64), dtype=np.float64)
    p: np.ndarray = np.array(r, dtype=np.float64, copy=True)
    rs_old: float = float(np.sum(r * r))
    # Relative to ||b||, for the same reason as `gmres_solve`: an absolute
    # residual floor calls x = 0 converged whenever ||b|| <= tol.
    threshold_sq = (tol * float(np.linalg.norm(b.reshape(-1)))) ** 2
    if rs_old <= threshold_sq:
        return x
    limit = maxiter if maxiter is not None else 10 * max(n, 1)
    for _ in range(limit):
        Ap = np.asarray(matvec(p), dtype=np.float64)
        pAp = float(np.vdot(p, Ap).real)
        if pAp == 0.0:
            raise TesseraImplicitDiffError("CG breakdown: pᵀAp = 0 (A may not be positive-definite)")
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(np.vdot(r, r).real)
        if rs_new <= threshold_sq:
            return x.reshape(shape)
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    raise TesseraImplicitDiffError(
        f"CG did not converge to tol={tol} in {limit} iterations (residual {np.sqrt(rs_old):.3e})"
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

    # The Hessian as a declared-self-adjoint OperatorTangent: `H.T is H`
    # by symmetry of second derivatives, and CG consumes the operator
    # directly (solve-consumption, §3.5). The declaration is checkable —
    # the operator-level adjoint law holds up to the HVP's FD noise.
    H = OperatorTangent.self_adjoint_from(
        lambda v: np.asarray(_hvp(fn, x, v.reshape(x.shape), eps=eps),
                             dtype=np.float64).reshape(-1),
        shape=x.shape,
        label="∇²fn",
    )
    return cg_solve(H, u.reshape(-1), tol=tol, maxiter=maxiter).reshape(u.shape)


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
        # Central differencing of g(t) = F(a + t v) has truncation error
        # (h^2/6)·D^3F[v,v,v], i.e. relative error O((h·||v||)^2) — so a fixed
        # h degrades quadratically in ||v||, and the operator is applied to
        # un-normalized vectors (a user tangent in `root_jvp`, the GMRES
        # iterate). Scaling the step by ||v||_inf keeps the perturbation
        # O(eps) whatever the direction's magnitude, which is what makes the
        # matvec a scale-invariant linear map rather than a secant.
        h = eps / max(1.0, float(np.max(np.abs(v))))
        return ((_perturbed(h * v) - _perturbed(-h * v)) / (2 * h)).reshape(-1)

    # Jᵀ via the definition ⟨J v, r⟩ = ⟨v, Jᵀ r⟩, which needs EVERY basis
    # column of J — there is no matrix-free form of this adjoint, only a
    # question of whether the columns are kept. They do not depend on `r`, so
    # they are built once and reused. Rebuilding them per call made the
    # "matrix-free" default cost 2·n residual evaluations on every GMRES
    # application (and GMRES applies the adjoint 2·iters + 1 times), strictly
    # dominating the dense oracle it ships alongside, which pays 2·n once.
    # The O(n)-memory rationale this replaced was not being honoured on the
    # default path anyway: `custom_root` defaults to `certify=True` and
    # `certify_root` materializes the same dense Jacobian before the solve.
    n_in = a.size
    n_out = int(np.prod(out_shape)) if out_shape else 1
    cache: list[Optional[np.ndarray]] = [None]
    warned = [False]

    def _cached_jacobian() -> Optional[np.ndarray]:
        """The ``(n_out, n_in)`` FD Jacobian, or None above the memo budget."""
        if cache[0] is None:
            if n_out * n_in > _FD_JACOBIAN_ELEMENT_BUDGET:
                return None
            basis = np.zeros(n_in, dtype=np.float64)
            columns = np.empty((n_out, n_in), dtype=np.float64)
            for j in range(n_in):
                basis[j] = 1.0
                columns[:, j] = matvec(basis)
                basis[j] = 0.0
            cache[0] = columns
        return cache[0]

    def rmatvec(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        if r.size != n_out:
            raise TesseraImplicitDiffError(f"residual cotangent has {r.size} values, expected {n_out}")
        columns = _cached_jacobian()
        if columns is not None:
            return columns.T @ r
        # Above the budget, stream the columns as before. That is a
        # performance fallback, so it announces itself (Decision #21a) — at
        # this size every adjoint application costs 2·n residual evaluations.
        if not warned[0]:
            warned[0] = True
            warnings.warn(
                f"finite-difference adjoint is not memoized: caching the "
                f"{n_out}x{n_in} Jacobian would exceed the "
                f"{_FD_JACOBIAN_ELEMENT_BUDGET}-element budget, so each "
                f"application re-evaluates the residual {2 * n_in} times. "
                f"Supply an exact linearization to avoid the sweep entirely.",
                RuntimeWarning,
                stacklevel=3,
            )
        result = np.empty(n_in, dtype=np.float64)
        basis = np.zeros(n_in, dtype=np.float64)
        for j in range(n_in):
            basis[j] = 1.0
            result[j] = float(np.dot(matvec(basis), r))
            basis[j] = 0.0
        return result

    return matvec, rmatvec, out_shape, a.shape


# ── operator constructors: the local matvec pattern, promoted (§3.5) ─────────
def _partial_operator(
    F: Callable[..., np.ndarray],
    args: Sequence[np.ndarray],
    argnum: int,
    *,
    eps: float = 1e-6,
    label: str = "∂F",
) -> OperatorTangent:
    """``∂_argnum F(*args)`` as an `OperatorTangent` (FD numerical oracle)."""
    matvec, rmatvec, out_shape, in_shape = _partial_jacobian_matvecs(
        F, args, argnum, eps=eps
    )
    return OperatorTangent.from_matvec_pair(
        matvec, rmatvec, in_shape=in_shape, out_shape=out_shape,
        provenance="numerical_oracle", label=label,
    )


def _linearization_solution_operator(
    linearization: ResidualLinearization,
    solution_shape: tuple[int, ...],
) -> OperatorTangent:
    """``A = ∂ₓF`` from compiler-supplied exact products."""
    return OperatorTangent.from_matvec_pair(
        linearization.solution_jvp,
        linearization.solution_vjp,
        in_shape=solution_shape,
        out_shape=linearization.residual_shape,
        provenance=linearization.provenance,
        label="∂ₓF",
    )


def _linearization_parameter_operator(
    linearization: ResidualLinearization,
    index: int,
    param_shape: tuple[int, ...],
) -> OperatorTangent:
    """``Bᵢ = ∂_{paramᵢ}F`` from compiler-supplied exact products."""
    return OperatorTangent.from_matvec_pair(
        functools.partial(linearization.parameter_jvp, index),
        functools.partial(linearization.parameter_vjp, index),
        in_shape=param_shape,
        out_shape=linearization.residual_shape,
        provenance=linearization.provenance,
        label=f"∂θ{index}F",
    )


# ── custom_root: differentiate x*(θ) defined by F(x*, θ) = 0 ─────────────────
def root_vjp(
    F: Callable[..., np.ndarray],
    solution: np.ndarray,
    params: Sequence[np.ndarray],
    cotangent: np.ndarray,
    *,
    argnums: int | Sequence[int] = 0,
    eps: float = 1e-6,
    linear_solver: str = "gmres",
    tol: float = 1e-8,
    maxiter: int | None = None,
    restart: int | None = None,
    linearization: ResidualLinearization | None = None,
    return_solve_info: bool = False,
) -> Any:
    """VJP of an implicit root ``x*(params)`` with ``F(x*, params) = 0``.

    Given the solution ``x*`` and an output cotangent ``u = ∂L/∂x*``, returns
    ``∂L/∂params[argnum]`` for each requested ``argnum`` (a single array if
    ``argnums`` is an int). Implements ``Aᵀ r = u`` then ``-Bᵀ r`` where
    ``A = ∂_x F``, ``B = ∂_param F`` (§10.4). The default restarted GMRES solve
    is matrix-free and ``A`` need not be symmetric; ``linear_solver="dense"``
    is an explicit small-system oracle.
    """
    x = np.asarray(solution, dtype=np.float64)
    full_args = [x, *(np.asarray(p, dtype=np.float64) for p in params)]
    u = np.asarray(cotangent, dtype=np.float64).reshape(-1)

    # A = ∂₁F(x*, params) as an OperatorTangent; its `.T` is the adjoint
    # the VJP solve consumes (§3.5: composition + transpose + solve).
    if linearization is None:
        A_op = _partial_operator(F, full_args, 0, eps=eps, label="∂ₓF")
    else:
        A_op = _linearization_solution_operator(linearization, x.shape)
    product_source = A_op.provenance
    n = x.size
    n_out = A_op.shape[0]
    if n_out != n:
        raise TesseraImplicitDiffError(
            f"root_vjp expects square ∂_xF (got {(n_out, n)}); F must map to the solution space"
        )
    if linear_solver == "gmres":
        r, solve_info = gmres_solve(
            A_op.T,
            u,
            tol=tol,
            maxiter=maxiter,
            restart=restart,
            return_info=True,
        )
    elif linear_solver == "dense":
        A = A_op.materialize()
        try:
            r = np.linalg.solve(A.T, u)
        except np.linalg.LinAlgError as exc:
            raise TesseraImplicitDiffError(
                "∂_xF is singular at the solution; the implicit function theorem does not apply here"
            ) from exc
        true_residual = A.T @ r - u
        solve_info = LinearSolveInfo(
            "dense",
            1,
            0,
            n,
            float(np.linalg.norm(true_residual)),
            float(np.linalg.norm(u)),
            True,
        )
    else:
        raise ValueError("linear_solver must be 'gmres' or 'dense'")

    if isinstance(argnums, int):
        single = True
        idxs: tuple[int, ...] = (argnums,)
    else:
        single = False
        idxs = tuple(argnums)
    grads = []
    for an in idxs:
        # B = ∂_{param an} F ; param an is F-argument (an + 1). The
        # parameter cotangent is the transposed action Bᵀ r, negated —
        # i.e. the ∂L/∂θ operator is the composition (−Bᵀ) ∘ A⁻ᵀ.
        if linearization is None:
            B_op = _partial_operator(
                F, full_args, an + 1, eps=eps, label=f"∂θ{an}F"
            )
        else:
            B_op = _linearization_parameter_operator(
                linearization, an, full_args[an + 1].shape
            )
        grads.append(-(B_op.T @ np.asarray(r, dtype=np.float64)))
    solve_info = replace(solve_info, product_source=product_source)
    result = grads[0] if single else tuple(grads)
    return (result, solve_info) if return_solve_info else result


def root_jvp(
    F: Callable[..., np.ndarray],
    solution: np.ndarray,
    params: Sequence[np.ndarray],
    tangents: Sequence[np.ndarray | None],
    *,
    eps: float = 1e-6,
    linear_solver: str = "gmres",
    tol: float = 1e-8,
    maxiter: int | None = None,
    restart: int | None = None,
    linearization: ResidualLinearization | None = None,
    return_solve_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, LinearSolveInfo]:
    """JVP of an implicit root: solve ``A t = -Σ Bᵢ vᵢ`` (§10.4).

    ``tangents`` are the input directions for each param (``None`` to skip).
    Returns the tangent of ``x*`` — the directional derivative of the solution.
    """
    x = np.asarray(solution, dtype=np.float64)
    full_args = [x, *(np.asarray(p, dtype=np.float64) for p in params)]
    if linearization is None:
        A_op = _partial_operator(F, full_args, 0, eps=eps, label="∂ₓF")
    else:
        A_op = _linearization_solution_operator(linearization, x.shape)
    product_source = A_op.provenance
    n = x.size
    n_out = A_op.shape[0]
    if n_out != n:
        raise TesseraImplicitDiffError(f"root_jvp expects square ∂_xF (got {(n_out, n)})")

    rhs = np.zeros(n, dtype=np.float64)
    for i, v in enumerate(tangents):
        if v is None:
            continue
        if linearization is None:
            B_op = _partial_operator(
                F, full_args, i + 1, eps=eps, label=f"∂θ{i}F"
            )
        else:
            B_op = _linearization_parameter_operator(
                linearization, i, full_args[i + 1].shape
            )
        rhs = rhs - B_op.matvec(np.asarray(v, dtype=np.float64))
    if linear_solver == "gmres":
        t, solve_info = gmres_solve(
            A_op,
            rhs,
            tol=tol,
            maxiter=maxiter,
            restart=restart,
            return_info=True,
        )
    elif linear_solver == "dense":
        A = A_op.materialize()
        try:
            t = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError as exc:
            raise TesseraImplicitDiffError("∂_xF is singular at the solution; IFT does not apply") from exc
        true_residual = A @ t - rhs
        solve_info = LinearSolveInfo(
            "dense",
            1,
            0,
            n,
            float(np.linalg.norm(true_residual)),
            float(np.linalg.norm(rhs)),
            True,
        )
    else:
        raise ValueError("linear_solver must be 'gmres' or 'dense'")
    solve_info = replace(solve_info, product_source=product_source)
    result = t.reshape(x.shape)
    return (result, solve_info) if return_solve_info else result


def custom_root(
    optimality: Callable[..., np.ndarray],
    *,
    argnums: int | Sequence[int] = 0,
    eps: float = 1e-6,
    certify: bool = True,
    residual_tol: float = 1e-6,
    degeneracy_tol: float = 1e-8,
) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """Wrap a solver so its output is differentiable through the root condition.

    ``optimality(x, *params) -> residual`` is the function whose zero defines the
    solution (a stationarity condition ``∇₁f = 0``, a fixed-point residual, or a
    nonlinear system). The decorated ``solver(*params) -> x*`` gains
    ``.vjp(x*, params, u)`` and ``.jvp(x*, params, tangents)`` methods computing
    the implicit derivatives without unrolling the solver.

    **H3 well-posedness certificate** (AD-OPERATOR-1, `CORE_SUBSTRATE_VIEW.md`
    S8): with ``certify=True`` (the default) every differentiation of a given
    solution first *measures* the implicit-function-theorem hypothesis via
    `operator.certify_root` — the point is actually a root
    (``‖F(x*, θ)‖ ≤ residual_tol``, scaled) and ``∂ₓF`` is non-degenerate
    there (``σ_min > degeneracy_tol · σ_max``; for KKT-form residuals this is
    what strict complementarity guarantees). A degenerate root **rejects with
    the measured numbers** instead of returning an ill-posed gradient, and a
    certificate that cannot be evaluated fails closed. ``certify=False`` is
    an explicit opt-out for callers who have already certified the solution
    elsewhere — the check is skipped, never silently weakened.

    Example::

        @custom_root(lambda x, theta: x * x - theta)   # x*(θ) = √θ
        def sqrt_solver(theta):
            return np.sqrt(theta)

        xs = sqrt_solver(theta)
        dtheta = sqrt_solver.vjp(xs, (theta,), np.ones_like(xs))  # 1/(2√θ)
    """

    def wrap(solver: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        op_name = f"custom_root:{solver.__module__}.{solver.__qualname__}"
        selected_argnums = (argnums,) if isinstance(argnums, int) else tuple(argnums)

        def _certified(solution, params):
            if not certify:
                return None
            certificate = certify_root(
                optimality, solution, params, eps=eps,
                residual_tol=residual_tol, degeneracy_tol=degeneracy_tol,
            )
            if not certificate.strict:
                reasons = []
                if not certificate.residual_ok:
                    reasons.append(
                        f"the point is not a root: ‖F(x*, θ)‖ = "
                        f"{certificate.residual_norm:.3e} exceeds "
                        f"{residual_tol:.1e} × scale "
                        f"{certificate.solution_scale:.3e}"
                    )
                if not certificate.nondegenerate:
                    reasons.append(
                        f"∂ₓF is degenerate at the solution: σ_min = "
                        f"{certificate.sigma_min:.3e} vs σ_max = "
                        f"{certificate.sigma_max:.3e} (condition "
                        f"{certificate.condition_number:.3e}) — for a KKT "
                        f"residual this is a strict-complementarity failure"
                    )
                raise TesseraImplicitDiffError(
                    f"{op_name}: implicit differentiation is ill-posed here — "
                    + "; ".join(reasons)
                    + ". Pass certify=False only if the solution is "
                    "certified elsewhere."
                )
            return certificate

        def vjp(solution, params, cotangent):
            _certified(solution, params)
            return root_vjp(optimality, solution, params, cotangent, argnums=argnums, eps=eps)

        def jvp(solution, params, tangents):
            _certified(solution, params)
            return root_jvp(optimality, solution, params, tangents, eps=eps)

        @functools.wraps(solver)
        def wrapped(*params, **kwargs):
            # The residual is differentiated at this invocation's solution, so
            # bind it in the VJP closure stored on the tape entry.  Each call
            # gets its own closure and therefore composes correctly when the
            # same custom root appears more than once in one tape.
            solution_box: dict[str, np.ndarray] = {}

            def forward(*forward_params):
                solution = np.asarray(solver(*forward_params, **kwargs))
                solution_box["value"] = solution
                return solution

            def implicit_vjp(dout, *forward_params, **_unused):
                # H3: certify the recorded solution once per backward —
                # a degenerate root rejects with measured numbers here,
                # never propagates an ill-posed gradient onto the tape.
                if "certificate" not in solution_box:
                    solution_box["certificate"] = _certified(
                        solution_box["value"], forward_params
                    )
                selected_grads = root_vjp(
                    optimality,
                    solution_box["value"],
                    forward_params,
                    dout,
                    # Always request a tuple, even for one selected argument:
                    # Tape VJPs must return one cotangent for *every* recorded
                    # positional input, with None for intentionally excluded
                    # parameters.
                    argnums=selected_argnums,
                    eps=eps,
                )
                return tuple(
                    selected_grads[selected_argnums.index(i)] if i in selected_argnums else None
                    for i in range(len(forward_params))
                )

            from .tape import record_custom_vjp_call

            return record_custom_vjp_call(op_name, forward, implicit_vjp, *params)

        wrapped.vjp = vjp  # type: ignore[attr-defined]
        wrapped.jvp = jvp  # type: ignore[attr-defined]
        wrapped.optimality = optimality  # type: ignore[attr-defined]
        # The measured certificate itself, for callers who want the numbers
        # (S4 discipline: certificates are data, not just gates).
        wrapped.certificate = (  # type: ignore[attr-defined]
            lambda solution, params: certify_root(
                optimality, solution, params, eps=eps,
                residual_tol=residual_tol, degeneracy_tol=degeneracy_tol,
            )
        )
        return wrapped

    return wrap


# ── Adjoint-state method (Blondel & Roulet §10.5) ────────────────────────────
def adjoint_state_grad(
    L: Callable[[np.ndarray, np.ndarray], float],
    c: Callable[[np.ndarray, np.ndarray], np.ndarray],
    state: np.ndarray,
    w: np.ndarray,
    *,
    eps: float = 1e-6,
    tol: float = 1e-8,
    maxiter: int | None = None,
    restart: int | None = None,
    return_solve_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, LinearSolveInfo]:
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
            hi = arr.copy()
            hi[idx] += eps
            lo = arr.copy()
            lo[idx] -= eps
            if which == 0:
                g[idx] = (f(hi, other) - f(lo, other)) / (2 * eps)
            else:
                g[idx] = (f(other, hi) - f(other, lo)) / (2 * eps)
            it.iternext()
        return g

    grad1_L = _grad_fd(L, s, w, 0)
    grad2_L = _grad_fd(L, w, s, 1)

    # ∂₁c and ∂₂c as OperatorTangents — matrix-free; the adjoint system
    # consumes ∂₁cᵀ through the type's own transpose, and the parameter
    # pullback is ∂₂cᵀ applied to the multiplier. The residual-state
    # Jacobian must be square for the IFT adjoint system.
    C_s = _partial_operator(c, [s, w], 0, eps=eps, label="∂₁c")
    C_w = _partial_operator(c, [s, w], 1, eps=eps, label="∂₂c")
    residual_size = C_s.shape[0]
    if residual_size != s.size:
        raise TesseraImplicitDiffError(f"adjoint_state_grad expects square ∂₁c (got {(residual_size, s.size)})")
    r, solve_info = gmres_solve(
        C_s.T,
        -grad1_L.reshape(-1),
        tol=tol,
        maxiter=maxiter,
        restart=restart,
        return_info=True,
    )
    result = (grad2_L.reshape(-1) + C_w.T.matvec(r)).reshape(w.shape)
    return (result, solve_info) if return_solve_info else result
