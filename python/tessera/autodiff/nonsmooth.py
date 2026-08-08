"""Declared Clarke-subgradient selection policy for nonsmooth primitives.

Motivation (C2, from the *Elements of Differentiable Programming* review).
At a kink of a piecewise-differentiable function — ``relu`` at 0, ``max`` at a
tie, ``clip`` at a bound — the Clarke subdifferential is a *set*, and any element
of it is a valid generalized gradient (Blondel & Roulet §2.7). Which element the
backward pass returns is therefore a **semantic** choice, not an implementation
detail. Under Architecture Decision #21a a semantic key must be *declared* and
must never be chosen silently or inconsistently: if the numpy reference VJP and a
backend kernel pick different subgradients at the kink, they disagree on exactly
the input where a differential oracle probes, and the disagreement reads as a
miscompile.

Before this module the selection was made ad hoc and inconsistently — most
visibly, ``clip`` returned ``0`` at a bound (strict ``>``/``<``) while its alias
``clamp`` returned ``1`` (inclusive ``>=``/``<=``) for the identical operation.
This module makes the convention **one declared thing** per op, gives the
declaration a consumer (the VJPs below call these helpers; a drift test asserts
each op's kink behavior matches its declared policy — Decision #29), and states
the convention in prose so a backend author can match it.

Canonical conventions
---------------------
``SUBGRAD_ZERO``   Pick the value that a central-difference numerical Jacobian
                   sees: a point exactly at the kink is ``eps``-perturbed onto a
                   flat side, so the slope reads ``0``. Used where one side is
                   flat (``relu``/``relu6`` at 0, ``clip``/``clamp`` at a bound,
                   ``abs``/``sign`` at 0 — ``abs`` is a V whose subdifferential at
                   0 is ``[-1, 1]`` and we pick ``0``, the midpoint).
``SUBGRAD_SPLIT``  Distribute the incoming cotangent *equally* among the tied
                   arguments so the total mass is conserved (the subgradient is
                   the barycenter of the argmax simplex). Used for ``max``/``min``
                   families and ``maximum``/``minimum`` at a tie.

The two conventions are internally consistent — ``SUBGRAD_SPLIT`` with a single
maximizer reduces to a hard select, and ``SUBGRAD_ZERO`` is the ``eps→0`` limit
of the smooth surrogate. They differ only in whether tie mass is conserved, which
is why max-family ops (mass matters for reductions) and gate/threshold ops (one
flat side) get different declared policies.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

__all__ = [
    "SUBGRAD_ZERO",
    "SUBGRAD_SPLIT",
    "NONSMOOTH_SELECTION",
    "heaviside_subgrad",
    "sign_subgrad",
    "tie_split_mask",
    "gate_mask",
    "selection_policy",
]

# ── Named policies ───────────────────────────────────────────────────────────
SUBGRAD_ZERO = "subgrad_zero"
SUBGRAD_SPLIT = "subgrad_split"

# ── Declared policy per op (the semantic key; consumed by the VJPs) ──────────
# Every nonsmooth VJP in vjp.py routes its kink behavior through the helpers
# below and is listed here. tests/unit/test_nonsmooth_selection.py is the
# consumer/enforcer (Decision #29): it probes each op *exactly at* the kink and
# asserts the returned subgradient matches the declared policy.
NONSMOOTH_SELECTION: Mapping[str, str] = {
    # One flat side → SUBGRAD_ZERO.
    "relu": SUBGRAD_ZERO,
    "abs": SUBGRAD_ZERO,
    "absolute": SUBGRAD_ZERO,
    "sign": SUBGRAD_ZERO,
    "clip": SUBGRAD_ZERO,
    "clamp": SUBGRAD_ZERO,
    # Tie among competing arguments → mass-conserving equal split.
    "max": SUBGRAD_SPLIT,
    "min": SUBGRAD_SPLIT,
    "amax": SUBGRAD_SPLIT,
    "amin": SUBGRAD_SPLIT,
    "maximum": SUBGRAD_SPLIT,
    "minimum": SUBGRAD_SPLIT,
}


def selection_policy(op_name: str) -> str:
    """Return the declared subgradient policy for ``op_name``.

    Raises ``KeyError`` (fail closed, Decision #21a) rather than defaulting: a
    nonsmooth op with no declared policy is a bug in the registry, not a silent
    ``SUBGRAD_ZERO``.
    """
    try:
        return NONSMOOTH_SELECTION[op_name]
    except KeyError as exc:  # pragma: no cover - guard, exercised by drift test
        raise KeyError(
            f"nonsmooth op {op_name!r} has no declared subgradient-selection "
            f"policy; add it to NONSMOOTH_SELECTION (Decision #21a: a semantic "
            f"key must never be chosen silently)"
        ) from exc


# ── Selection helpers (SUBGRAD_ZERO family) ──────────────────────────────────
def heaviside_subgrad(x: np.ndarray, dtype=None) -> np.ndarray:
    """d/dx relu(x) under SUBGRAD_ZERO: ``1`` for ``x > 0`` else ``0``.

    The value at ``x == 0`` is ``0`` (the flat, left side), matching the
    central-difference oracle and PyTorch/JAX ``relu``.
    """
    a = np.asarray(x)
    out_dtype = dtype if dtype is not None else a.dtype
    return (a > 0).astype(out_dtype)


def sign_subgrad(x: np.ndarray, dtype=None) -> np.ndarray:
    """d/dx abs(x) under SUBGRAD_ZERO: ``sign(x)`` with ``0`` at ``x == 0``.

    ``0`` is the midpoint of the subdifferential ``[-1, 1]`` at the kink, and is
    what ``np.sign`` returns, so ``abs`` and ``absolute`` share one convention
    and one dtype policy (previously ``abs`` forced float64 while ``absolute``
    kept the input dtype).
    """
    a = np.asarray(x)
    out_dtype = dtype if dtype is not None else a.dtype
    return np.sign(a).astype(out_dtype)


def gate_mask(
    x: np.ndarray,
    lo=None,
    hi=None,
    dtype=None,
) -> np.ndarray:
    """Pass-through mask for ``clip``/``clamp``/``hardtanh`` under SUBGRAD_ZERO.

    ``1`` where ``x`` is **strictly** inside ``(lo, hi)``, ``0`` at or beyond a
    bound. Strict comparison is the SUBGRAD_ZERO choice: a point exactly at a
    bound is ``eps``-clipped onto the flat plateau, so the numerical slope is
    ``0``. This is the convention ``clip`` already used; ``clamp`` is brought
    into line here (it previously returned ``1`` at a bound).
    """
    a = np.asarray(x)
    out_dtype = dtype if dtype is not None else a.dtype
    mask = np.ones_like(a, dtype=out_dtype)
    if lo is not None:
        mask = mask * (a > lo).astype(out_dtype)
    if hi is not None:
        mask = mask * (a < hi).astype(out_dtype)
    return mask


# ── Selection helper (SUBGRAD_SPLIT family) ──────────────────────────────────
def tie_split_mask(
    x: np.ndarray,
    *,
    axis=None,
    kind: str = "max",
    dtype=None,
) -> np.ndarray:
    """Mass-conserving tie weights for ``max``/``min`` families under SUBGRAD_SPLIT.

    Returns a mask that is ``1/k`` at each of the ``k`` positions equal to the
    per-slice extreme along ``axis`` and ``0`` elsewhere, so the incoming
    cotangent is divided equally among tied extrema and total mass is preserved.
    With a unique extremum this is a hard one-hot select. ``kind`` is ``"max"``
    or ``"min"``.
    """
    a = np.asarray(x)
    out_dtype = dtype if dtype is not None else a.dtype
    reducer = np.max if kind == "max" else np.min
    extreme = reducer(a, axis=axis, keepdims=True)
    hits = (a == extreme).astype(out_dtype)
    counts = hits.sum(axis=axis, keepdims=True)
    return hits / counts
