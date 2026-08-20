"""Declared degeneracy policy for the matrix-factorization derivative rules.

Motivation (MC2, from the *Matrix Calculus* review —
``docs/audit/compiler/MATRIX_CALCULUS_REVIEW.md``).  The backward rules for
``svd``/``qr``/``cholesky``/``tri_solve`` are derived under a **regularity
hypothesis**: singular values are simple, ``R`` is full rank, ``A`` is
positive definite, the triangular factor is nonsingular.  When that hypothesis
fails the derivative does not merely become inaccurate — it **does not exist**.
Bright/Edelman/Johnson (arXiv:2501.14787) §14: at eigenvalue/singular-value
crossings the factors cease to be differentiable, and for a defective matrix
``dλ`` scales like ``sqrt(||dA||)``, so no finite generalized derivative exists
at all.

Whether the rule refuses, restricts itself to the well-defined part, or returns
a regularized approximation is therefore a **semantic** choice, and under
Architecture Decision #21a a semantic key never defaults silently.

What this module replaces
-------------------------
``vjp_svd`` used to build its coupling matrix as::

    F = 1.0 / (s2[None, :] - s2[:, None] + np.eye(len(s)) * eps)   # eps = 1e-12
    np.fill_diagonal(F, 0.0)

The ``eps`` regularizer was added **only on the diagonal**, and the next line
zeroed the diagonal — so the guard protected exactly the entries it then
erased, and the off-diagonal ``1/(s_j^2 - s_i^2)`` was completely
unregularized.  On a repeated singular value the rule returned an all-``NaN``
gradient behind bare numpy ``RuntimeWarning``s; on a *near*-repeated spectrum
nothing warned at all and the gradient was merely enormous and wrong.  That is
the opposite of the contract the same tree states elsewhere —
``python/tessera/solvers_ops.py``: *"a singular solve fails closed rather than
emitting NaNs"*.

Declared policies
-----------------
``FAIL_CLOSED``  (default)  Refuse.  Raise :class:`TesseraDegeneracyError`
                 naming the op, the offending index pair, and the measured
                 gap.  The honest answer when the derivative does not exist.
``GENERALIZED``  Admit the part of the derivative that survives the
                 degeneracy and refuse the rest.  For ``svd`` this is
                 well-defined and useful: within a degenerate cluster the
                 individual factors ``u_i``/``v_i``/``s_i`` are not
                 differentiable, but any quantity invariant under rotation
                 *within* the cluster is.  The rule therefore admits the
                 cotangent iff (a) its antisymmetric ``U``/``V`` couplings
                 vanish on degenerate pairs and (b) its ``ds`` weights are
                 constant across each degenerate cluster — the exact
                 conditions under which the limit exists.  This is what makes
                 ``grad(sum of singular values)`` (the nuclear norm) work on a
                 degenerate matrix, where it is genuinely differentiable.
                 For ``qr``/``cholesky``/``tri_solve`` a rank-deficient factor
                 leaves *no* well-defined part, so ``GENERALIZED`` is not
                 offered there (Decision #29: do not declare what has no
                 implementation).
``DAMPED``       Spelled ``"damped:<tau>"``.  Replace the exact reciprocal
                 ``1/g`` by the Tikhonov form ``g / (g^2 + tau^2)``, with
                 ``tau`` measured **relative to** ``s_max^2``.  The result is
                 an approximation the caller has explicitly accepted.  ``svd``
                 only, for the same reason as above.
``UNCHECKED``    Run the rule with no guard at all — the caller asserts the
                 input is regular.  This is the pre-2026-08-20 behaviour, kept
                 reachable *by explicit request* so that an opt-out exists,
                 which is precisely what distinguishes it from a silent
                 default.

Selecting a policy
------------------
The declared per-op default lives in :data:`FACTORIZATION_DEGENERACY`.  Override
it for a region of code with the :func:`degeneracy_policy` context manager::

    from tessera.autodiff import degeneracy

    with degeneracy.degeneracy_policy("generalized"):
        g = grad(nuclear_norm)(A)         # A may have repeated singular values

Consumers (Decision #29): ``vjp.py`` and ``jvp.py`` route every factorization
rule through the helpers below, and ``tests/unit/test_factorization_degeneracy.py``
is the enforcer — it probes each op *exactly at* its degeneracy and asserts the
declared policy is what happens, including the negative fixture whose correct
output is the diagnostic (Decision #10a).
"""

from __future__ import annotations

import contextlib
import threading
import warnings
from typing import Iterable, Mapping, Sequence

import numpy as np

from .errors import TesseraDegeneracyError, TesseraDegeneracyWarning

__all__ = [
    "FAIL_CLOSED",
    "GENERALIZED",
    "DAMPED",
    "UNCHECKED",
    "DEGENERACY_DIAGNOSTIC_CODE",
    "FACTORIZATION_DEGENERACY",
    "SUPPORTED_POLICIES",
    "TesseraDegeneracyError",
    "TesseraDegeneracyWarning",
    "existence_tolerance",
    "conditioning_tolerance",
    "admissibility_tolerance",
    "degeneracy_policy",
    "active_policy",
    "declared_policy",
    "parse_policy",
    "singular_value_clusters",
    "svd_coupling",
    "check_factor_rank",
]

# ── Named policies (the `degeneracy_policy` semantic key) ────────────────────
FAIL_CLOSED = "fail_closed"
GENERALIZED = "generalized"
DAMPED = "damped"
UNCHECKED = "unchecked"

#: Stable diagnostic code; registered in
#: ``tessera.compiler.diagnostic_codes`` and drift-gated by
#: ``tests/unit/test_diagnostic_code_registry.py``.
DEGENERACY_DIAGNOSTIC_CODE = "E_DEGENERATE_FACTORIZATION"

# ── Declared policy per op (consumed by vjp.py / jvp.py) ─────────────────────
FACTORIZATION_DEGENERACY: Mapping[str, str] = {
    "svd": FAIL_CLOSED,
    "qr": FAIL_CLOSED,
    "cholesky": FAIL_CLOSED,
    "tri_solve": FAIL_CLOSED,
}

#: Which policies each op actually implements.  Selecting one that is not
#: listed raises rather than degrading to something else (Decision #29 — an
#: undeclared-but-accepted policy would be a declaration with no consumer).
SUPPORTED_POLICIES: Mapping[str, frozenset] = {
    "svd": frozenset({FAIL_CLOSED, GENERALIZED, DAMPED, UNCHECKED}),
    "qr": frozenset({FAIL_CLOSED, UNCHECKED}),
    "cholesky": frozenset({FAIL_CLOSED, UNCHECKED}),
    "tri_solve": frozenset({FAIL_CLOSED, UNCHECKED}),
}


# ── Policy resolution ────────────────────────────────────────────────────────
_LOCAL = threading.local()


def parse_policy(spec: str) -> tuple[str, float | None]:
    """Split ``"damped:1e-6"`` into ``("damped", 1e-6)``.

    Bare policy names return ``(name, None)``.  An unknown name or a malformed
    ``damped`` parameter raises — the key never falls back.
    """
    if not isinstance(spec, str):
        raise TesseraDegeneracyError(
            f"{DEGENERACY_DIAGNOSTIC_CODE}: degeneracy policy must be a string, "
            f"got {type(spec).__name__}"
        )
    name, _, arg = spec.partition(":")
    name = name.strip()
    if name == DAMPED:
        if not arg:
            raise TesseraDegeneracyError(
                f"{DEGENERACY_DIAGNOSTIC_CODE}: policy 'damped' requires a "
                f"relative damping parameter, spelled 'damped:<tau>' "
                f"(e.g. 'damped:1e-6'); got {spec!r}"
            )
        try:
            tau = float(arg)
        except ValueError as exc:
            raise TesseraDegeneracyError(
                f"{DEGENERACY_DIAGNOSTIC_CODE}: could not read the damping "
                f"parameter in {spec!r}; expected 'damped:<float>'"
            ) from exc
        if not (tau > 0.0) or not np.isfinite(tau):
            raise TesseraDegeneracyError(
                f"{DEGENERACY_DIAGNOSTIC_CODE}: damping parameter must be a "
                f"finite positive number, got {tau!r}"
            )
        return DAMPED, tau
    if arg:
        raise TesseraDegeneracyError(
            f"{DEGENERACY_DIAGNOSTIC_CODE}: policy {name!r} takes no parameter, "
            f"got {spec!r}"
        )
    if name not in (FAIL_CLOSED, GENERALIZED, UNCHECKED):
        raise TesseraDegeneracyError(
            f"{DEGENERACY_DIAGNOSTIC_CODE}: unknown degeneracy policy {spec!r}; "
            f"expected one of 'fail_closed', 'generalized', 'damped:<tau>', "
            f"'unchecked'"
        )
    return name, None


def declared_policy(op: str) -> str:
    """Return the declared default policy for ``op``.

    Raises ``KeyError`` (fail closed) rather than defaulting: a factorization
    rule with no declared policy is a bug in this registry, not a silent
    ``fail_closed``.
    """
    try:
        return FACTORIZATION_DEGENERACY[op]
    except KeyError as exc:
        raise KeyError(
            f"factorization op {op!r} has no declared degeneracy policy; add it "
            f"to FACTORIZATION_DEGENERACY (Decision #21a: a semantic key must "
            f"never be chosen silently)"
        ) from exc


@contextlib.contextmanager
def degeneracy_policy(spec: str, *, tol: float | None = None):
    """Override the declared degeneracy policy for the enclosing block.

    Thread-local, so a policy chosen on one thread never leaks into another.
    ``tol`` overrides the *existence* threshold (see
    :func:`existence_tolerance`); the conditioning warning band above it is
    skipped when ``tol`` is raised past :func:`conditioning_tolerance`.
    """
    parse_policy(spec)  # validate eagerly, at the `with`, not at first use
    previous = getattr(_LOCAL, "override", None)
    _LOCAL.override = (spec, tol)
    try:
        yield
    finally:
        _LOCAL.override = previous


def active_policy(op: str, explicit: str | None = None) -> tuple[str, float | None, float | None]:
    """Resolve ``(policy_name, tau, tol_override)`` for ``op``.

    Precedence: an ``explicit`` argument, then the context-manager override,
    then the declared default.  Rejects a policy the op does not implement.
    """
    tol_override: float | None = None
    if explicit is not None:
        spec = explicit
    else:
        override = getattr(_LOCAL, "override", None)
        if override is not None:
            spec, tol_override = override
        else:
            spec = declared_policy(op)
    name, tau = parse_policy(spec)
    supported = SUPPORTED_POLICIES.get(op, frozenset({FAIL_CLOSED, UNCHECKED}))
    if name not in supported:
        raise TesseraDegeneracyError(
            f"{DEGENERACY_DIAGNOSTIC_CODE}: {op} does not implement degeneracy "
            f"policy {name!r}; it supports {sorted(supported)}. "
            f"({name!r} is declared for other factorizations — Decision #29 "
            f"forbids accepting a policy this rule has no implementation for.)"
        )
    return name, tau, tol_override


def existence_tolerance(n: int = 1, dtype=np.float64) -> float:
    """Relative threshold below which two quantities are *numerically equal*.

    ``n * eps`` — the same criterion ``numpy.linalg.matrix_rank`` uses to call a
    singular value zero.  Inside this band the derivative genuinely **does not
    exist** (coincident singular values, a rank-deficient factor), which is a
    semantic condition, so the guard fails closed here (Decision #21a).

    Deliberately *not* ``sqrt(eps)``: refusing merely ill-conditioned inputs
    would reject gradients that exist and still carry eight good digits.
    """
    return float(max(1, int(n)) * np.finfo(np.dtype(dtype)).eps)


def admissibility_tolerance(dtype=np.float64) -> float:
    """Relative threshold below which a cotangent component counts as *zero*.

    Used only by :data:`GENERALIZED`, to decide whether the caller's cotangent
    actually asks for the non-existent part of the derivative.  This is a
    question about **numerical noise in the cotangent**, not about the input
    spectrum, so it deliberately does not reuse :func:`existence_tolerance`: a
    cotangent that arrived through a chain of ops carries rounding noise many
    orders above ``n*eps``, and judging it at that threshold would refuse every
    realistic call and make the policy unusable.

    ``sqrt(eps)`` is the same half-the-digits convention the rest of this module
    uses.  The honest caveat: a *genuine* contribution smaller than this is
    dropped rather than refused.  Since the discarded term scales like
    ``(cotangent) / gap`` and ``gap -> 0`` inside a cluster, no nonzero
    threshold is safe in the limit — which is why
    :data:`GENERALIZED` must be selected deliberately and is never the default.
    """
    return float(np.sqrt(np.finfo(np.dtype(dtype)).eps))


def conditioning_tolerance(dtype=np.float64) -> float:
    """Relative threshold below which a derivative that exists is untrustworthy.

    ``sqrt(eps)``.  The rules carry a factor ``1/gap``, so an input perturbed at
    the ``eps`` level yields a tangent error of order ``eps/gap``: about half the
    significant digits are gone once ``gap ~ sqrt(eps)``.  That is the same
    ``sqrt(eps)`` crossover the finite-difference chapter of the notes derives
    (§4.6), applied to conditioning instead of to step size.

    Between :func:`existence_tolerance` and this value the derivative exists, so
    the rule proceeds — but emits :class:`TesseraDegeneracyWarning`, because a
    silent answer that has lost most of its digits is the failure this whole
    module exists to stop.
    """
    return float(np.sqrt(np.finfo(np.dtype(dtype)).eps))


# ── Spectrum analysis ────────────────────────────────────────────────────────
def singular_value_clusters(
    s: np.ndarray, *, tol: float
) -> list[tuple[int, ...]]:
    """Group indices whose ``s_i^2`` are within ``tol`` (relative to ``s_max^2``).

    Returns only the clusters with **more than one** member — i.e. exactly the
    index groups on which the backward rule's ``1/(s_j^2 - s_i^2)`` is not
    defined.  Grouping is done on ``s^2`` because that is the quantity the rule
    actually divides by.
    """
    s = np.asarray(s, dtype=np.float64).ravel()
    n = s.size
    if n < 2:
        return []
    s2 = s ** 2
    scale = float(s2.max())
    # An all-zero matrix has no scale to be relative to; every value coincides.
    atol = tol * scale if scale > 0.0 else 0.0
    order = np.argsort(-s2, kind="stable")
    clusters: list[list[int]] = []
    current = [int(order[0])]
    for prev, idx in zip(order[:-1], order[1:]):
        if abs(s2[prev] - s2[idx]) <= atol:
            current.append(int(idx))
        else:
            clusters.append(current)
            current = [int(idx)]
    clusters.append(current)
    if scale == 0.0:
        # Degenerate by construction: every singular value is zero.
        return [tuple(sorted(range(n)))]
    return [tuple(sorted(c)) for c in clusters if len(c) > 1]


def _worst_pair(s: np.ndarray, clusters: Sequence[tuple[int, ...]]) -> tuple[int, int, float]:
    """Return ``(i, j, relative_gap)`` for the tightest degenerate pair."""
    s = np.asarray(s, dtype=np.float64).ravel()
    s2 = s ** 2
    scale = float(s2.max())
    best = (int(clusters[0][0]), int(clusters[0][1]), np.inf)
    for cluster in clusters:
        for a in range(len(cluster)):
            for b in range(a + 1, len(cluster)):
                i, j = int(cluster[a]), int(cluster[b])
                gap = abs(s2[i] - s2[j])
                rel = gap / scale if scale > 0.0 else 0.0
                if rel < best[2]:
                    best = (i, j, rel)
    return best


def _degenerate_message(op: str, s: np.ndarray, clusters, tol: float, extra: str = "",
                        mode: str = "backward", batch: str = "") -> str:
    i, j, rel = _worst_pair(s, clusters)
    s = np.asarray(s, dtype=np.float64).ravel()
    where = f" (batch element {batch})" if batch else ""
    body = (
        f"{DEGENERACY_DIAGNOSTIC_CODE}: {op} {mode} is not defined at this "
        f"input{where}. Singular values {i} and {j} coincide "
        f"(s[{i}]={s[i]:.6e}, s[{j}]={s[j]:.6e}; relative gap {rel:.3e} <= "
        f"tol {tol:.3e}), so the coupling term 1/(s_j^2 - s_i^2) has no limit "
        f"and the individual singular vectors are not differentiable. "
        f"Degenerate clusters (index groups): {list(clusters)}."
    )
    if extra:
        body += " " + extra
    return body + (
        " Choose a degeneracy policy explicitly if this input is intended: "
        "`with tessera.autodiff.degeneracy.degeneracy_policy('generalized')` "
        "for the rotation-invariant part only, `'damped:<tau>'` for a "
        "Tikhonov-regularized approximation, or `'unchecked'` to assert "
        "regularity yourself."
    )


def _warn_if_ill_conditioned(op: str, s: np.ndarray, tol: float,
                             mode: str = "backward", batch: str = "") -> None:
    """Warn (never raise) in the band where the derivative exists but is thin.

    Above :func:`existence_tolerance` the coupling term has a finite limit, so
    refusing would reject a real gradient; below :func:`conditioning_tolerance`
    that gradient has lost more than half its digits, so returning it silently
    is the failure this module exists to stop.
    """
    warn_at = conditioning_tolerance(np.float64)
    if tol >= warn_at:
        return
    s_arr = np.asarray(s, dtype=np.float64).ravel()
    if s_arr.size < 2:
        return
    s2 = s_arr ** 2
    scale = float(s2.max())
    if scale <= 0.0:
        return
    order = np.argsort(-s2, kind="stable")
    gaps = np.abs(np.diff(s2[order])) / scale
    worst = int(np.argmin(gaps))
    rel = float(gaps[worst])
    if rel > warn_at:
        return
    i, j = int(order[worst]), int(order[worst + 1])
    warnings.warn(
        f"{DEGENERACY_DIAGNOSTIC_CODE}: {op} {mode} is defined here but ill "
        f"conditioned{f' (batch element {batch})' if batch else ''} — singular "
        f"values {i} and {j} have relative gap "
        f"{rel:.3e}, below the sqrt(eps) half-digits threshold {warn_at:.3e}. "
        f"The coupling term 1/(s_j^2 - s_i^2) has a finite limit, so this is "
        f"not a refusal; expect the result to have lost most of its "
        f"significant digits.",
        TesseraDegeneracyWarning,
        stacklevel=3,
    )


# ── The svd coupling matrix (the thing the dead `eps` was pretending to guard) ─
def svd_coupling(
    s: np.ndarray,
    *,
    ds: np.ndarray | None = None,
    antisymmetric_parts: Iterable[np.ndarray] = (),
    op: str = "svd",
    policy: str | None = None,
    tol: float | None = None,
    admit_generalized: bool = True,
    mode: str = "backward",
) -> np.ndarray:
    """Build ``F`` with ``F[..., i, j] = 1 / (s_j^2 - s_i^2)`` and a zero diagonal.

    **Batched**: ``s`` may carry leading dimensions (``[..., k]``), matching the
    batch-capable `ops.svd` forward and its rule pair; the result is
    ``[..., k, k]``. Each batch element is judged on its own spectrum, and the
    diagnostic names the offending element.

    The diagonal is excluded structurally — the denominator has the identity
    added before the division and the result is multiplied by ``1 - I`` — so no
    division by zero occurs on it. That, and not any regularization, is what the
    old ``np.eye(len(s)) * eps`` term was accidentally doing.

    ``ds`` and ``antisymmetric_parts`` are the caller's cotangent-derived
    quantities; they are read only under :data:`GENERALIZED`, to decide whether
    the requested derivative survives the degeneracy.
    ``admit_generalized=False`` says the caller has no way to express a
    restricted derivative — forward mode returns per-component ``ds``/``dU``,
    none of which exist inside a cluster — so :data:`GENERALIZED` refuses there
    rather than silently returning the rotation-invariant part as if it were the
    answer.
    """
    name, tau, tol_override = active_policy(op, policy)
    s_arr = np.asarray(s, dtype=np.float64)
    if s_arr.ndim == 0:
        s_arr = s_arr.reshape(1)
    k = s_arr.shape[-1]
    s2 = s_arr ** 2
    eye = np.eye(k)

    # `+ eye` keeps the diagonal denominator at 1; `(1 - eye)` then zeroes it.
    den = s2[..., None, :] - s2[..., :, None]

    if name == UNCHECKED:
        with np.errstate(divide="ignore", invalid="ignore"):
            return (1.0 - eye) / (den + eye)

    if tol is None:
        tol = tol_override
    if tol is None:
        tol = existence_tolerance(k, np.float64)

    if name == DAMPED:
        assert tau is not None  # guaranteed by parse_policy
        scale = np.maximum(s2.max(axis=-1, keepdims=True), 0.0)[..., None]
        tau_abs = np.where(scale > 0.0, tau * scale, tau)
        return (1.0 - eye) * den / (den ** 2 + tau_abs ** 2)

    # Judge each batch element on its own spectrum.
    flat_s = s_arr.reshape(-1, k)
    flat_ds = None if ds is None else np.asarray(ds, dtype=np.float64).reshape(-1, k)
    flat_parts = [
        np.asarray(M, dtype=np.float64).reshape(-1, k, k)
        for M in antisymmetric_parts
        if np.asarray(M).shape[-2:] == (k, k)
    ]
    admit_tol = admissibility_tolerance(np.float64)
    masks = np.zeros((flat_s.shape[0], k, k), dtype=bool)
    for b in range(flat_s.shape[0]):
        row = flat_s[b]
        clusters = singular_value_clusters(row, tol=tol)
        if not clusters:
            _warn_if_ill_conditioned(op, row, tol, mode=mode,
                                     batch=_batch_label(s_arr, b))
            continue
        if name == FAIL_CLOSED or not admit_generalized:
            extra = "" if admit_generalized else (
                "Policy 'generalized' admits only quantities invariant under "
                "rotation inside a degenerate cluster, and this rule returns "
                "per-component results (each s_i, each singular vector) — none "
                "of which are differentiable there — so it admits nothing."
            )
            raise TesseraDegeneracyError(_degenerate_message(
                op, row, clusters, tol, extra, mode=mode,
                batch=_batch_label(s_arr, b)))
        _check_generalized_admissible(
            op, row, clusters, tol, admit_tol, b, flat_ds, flat_parts,
            batch=_batch_label(s_arr, b), mode=mode,
        )
        for cluster in clusters:
            idx = np.asarray(cluster, dtype=int)
            masks[b][np.ix_(idx, idx)] = True

    admitted = masks.reshape(s_arr.shape[:-1] + (k, k))
    # Mask the degenerate blocks out of the *denominator* before dividing, so
    # the admitted zero is produced structurally and no `1/0` is ever evaluated
    # (a warning-free zero, not a suppressed inf).
    safe_den = np.where(admitted, 1.0, den) + eye
    F = (1.0 - eye) / safe_den
    return np.where(admitted, 0.0, F)


def _batch_label(s_arr: np.ndarray, b: int) -> str:
    """Human-readable index of batch element `b`, or `""` when unbatched."""
    if s_arr.ndim <= 1:
        return ""
    return str(np.unravel_index(b, s_arr.shape[:-1]))


def _check_generalized_admissible(
    op, row, clusters, tol, admit_tol, b, flat_ds, flat_parts, *, batch: str = "",
    mode: str = "backward",
) -> None:
    """Refuse a cotangent that asks for the part killed by the degeneracy."""
    for cluster in clusters:
        idx = np.asarray(cluster, dtype=int)
        if flat_ds is not None:
            block = flat_ds[b][idx]
            spread = float(np.max(block) - np.min(block))
            scale = max(1.0, float(np.max(np.abs(flat_ds[b]))))
            if spread > admit_tol * scale:
                raise TesseraDegeneracyError(_degenerate_message(
                    op, row, clusters, tol, batch=batch, mode=mode,
                    extra=(
                        f"Policy 'generalized' admits only quantities that are "
                        f"invariant under rotation inside a degenerate cluster, "
                        f"but the requested singular-value cotangent varies "
                        f"across cluster {cluster} (spread {spread:.3e}). "
                        f"Sum-like functionals of a cluster (equal weights) are "
                        f"differentiable; individual s_i are not."
                    ),
                ))
        for M in flat_parts:
            block = M[b][np.ix_(idx, idx)]
            mag = float(np.max(np.abs(block))) if block.size else 0.0
            scale = max(1.0, float(np.max(np.abs(M[b]))))
            if mag > admit_tol * scale:
                raise TesseraDegeneracyError(_degenerate_message(
                    op, row, clusters, tol, batch=batch, mode=mode,
                    extra=(
                        f"Policy 'generalized' admits only quantities that are "
                        f"invariant under rotation inside a degenerate cluster, "
                        f"but the requested singular-vector cotangent couples "
                        f"indices within cluster {cluster} (magnitude {mag:.3e}). "
                        f"Individual singular vectors are not differentiable there."
                    ),
                ))


# ── Rank / conditioning guard for qr, cholesky, tri_solve ────────────────────
def check_factor_rank(
    diagonal: np.ndarray,
    *,
    op: str,
    factor: str,
    policy: str | None = None,
    tol: float | None = None,
    mode: str = "backward",
) -> None:
    """Fail closed when a triangular factor is (numerically) rank deficient.

    ``qr``/``cholesky``/``tri_solve`` backward rules all invert a triangular
    factor.  The derivative exists only while that factor is nonsingular; as
    ``min |diag|`` approaches zero the rule returns a finite, plausible, and
    entirely wrong gradient with nothing raised.  ``factor`` names the factor
    (``"R"``, ``"L"``, ``"T"``) so the diagnostic points at the right object.
    """
    name, _tau, tol_override = active_policy(op, policy)
    if name == UNCHECKED:
        return
    d_all = np.abs(np.asarray(diagonal, dtype=np.float64))
    if d_all.size == 0:
        return
    if d_all.ndim == 0:
        d_all = d_all.reshape(1)
    # A stacked factorization presents one diagonal per batch element; judge the
    # worst one, since a single rank-deficient element poisons the whole batch.
    flat = d_all.reshape(-1, d_all.shape[-1])
    ratios = np.array([
        (row.min() / row.max()) if row.max() > 0.0 else 0.0 for row in flat
    ])
    d = flat[int(np.argmin(ratios))]
    if tol is None:
        tol = tol_override
    if tol is None:
        tol = existence_tolerance(d.size, np.float64)

    scale = float(d.max())
    worst = int(np.argmin(d))
    rel = float(d[worst] / scale) if scale > 0.0 else 0.0
    if rel > tol:
        warn_at = conditioning_tolerance(np.float64)
        if tol < rel <= warn_at:
            warnings.warn(
                f"{DEGENERACY_DIAGNOSTIC_CODE}: {op} {mode} is defined here but "
                f"ill conditioned — |{factor}[{worst},{worst}]| is {rel:.3e} of "
                f"max|diag({factor})|, below the sqrt(eps) half-digits threshold "
                f"{warn_at:.3e}. The gradient exists; expect it to have lost most "
                f"of its significant digits.",
                TesseraDegeneracyWarning,
                stacklevel=3,
            )
        return
    raise TesseraDegeneracyError(
        f"{DEGENERACY_DIAGNOSTIC_CODE}: {op} {mode} is not defined at this "
        f"input. The {factor} factor is numerically rank deficient: "
        f"|{factor}[{worst},{worst}]| = {d[worst]:.6e}, relative to "
        f"max|diag({factor})| = {scale:.6e} that is {rel:.3e} <= tol "
        f"{tol:.3e}. The rule inverts {factor}, and that inverse has no limit "
        f"here, so the returned gradient would be finite and wrong rather than "
        f"absent. Pass an input of full rank, or select "
        f"`degeneracy_policy('unchecked')` to assert regularity yourself."
    )
