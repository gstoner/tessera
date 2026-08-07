"""Smoothing / relaxation operator family (T1).

Motivation (from the *Elements of Differentiable Programming* review). Discrete
choices — argmax, top-k, sorting, sampling — have uninformative (a.e.-zero)
derivatives, so a program that routes gradients through them learns nothing at
the selection (Blondel & Roulet Ch. 4 §"Control flows", Ch. 12–13). The fix is
a continuous relaxation with a non-degenerate Jacobian. Tessera had almost none
of this family: no ``sparsemax``, ``entmax``, ``soft_top_k``, ``gumbel_softmax``,
or perturbed optimizer, and the one relaxation present (``arch.gumbel_softmax``)
was scoped to NAS logits, not tensors.

This module adds the tensor-level family as first-class primitives (registered
through ``custom_primitive`` so they get an ``ops.*`` entry, tape-awareness, and
catalog metadata) each with a VJP:

  * ``sparsemax``   — Euclidean projection onto the simplex (Martins & Astudillo
                      2016); sparse output, exact support-set Jacobian.
  * ``entmax15``    — 1.5-entmax (Peters et al. 2019); between softmax and
                      sparsemax, closed-form threshold + Jacobian.
  * ``soft_top_k``  — temperature-relaxed top-k gate (differentiable selection).
  * ``gumbel_softmax`` — the Gumbel-softmax / concrete relaxation of a sample.
  * ``perturbed_argmax`` — Monte-Carlo perturbed-optimizer relaxation of argmax
                      (Berthet et al. 2020), with the score-function-style VJP.

All VJPs are checked numerically against central differences in
``tests/unit/test_relaxation_ops.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .custom import custom_primitive


# ── helpers ──────────────────────────────────────────────────────────────────
def _move_axis_last(x: np.ndarray, axis: int) -> tuple[np.ndarray, int]:
    x = np.asarray(x, dtype=np.float64)
    axis = axis % x.ndim
    return np.moveaxis(x, axis, -1), axis


# ── sparsemax (Martins & Astudillo 2016) ─────────────────────────────────────
def _sparsemax_forward(z: np.ndarray, *, axis: int = -1) -> np.ndarray:
    x, axis = _move_axis_last(z, axis)
    # sort descending, find support threshold τ(z).
    zs = np.sort(x, axis=-1)[..., ::-1]
    k = np.arange(1, x.shape[-1] + 1)
    cumsum = np.cumsum(zs, axis=-1)
    support = (1 + k * zs) > cumsum
    k_z = np.sum(support, axis=-1, keepdims=True)  # size of support
    tau = (np.take_along_axis(cumsum, k_z - 1, axis=-1) - 1) / k_z
    p = np.clip(x - tau, 0.0, None)
    return np.moveaxis(p, -1, axis)


def _sparsemax_vjp(dout, z, *, axis: int = -1, **_kw):
    # Jacobian of sparsemax: on the support S, J = diag(s) - s sᵀ / |S| where
    # s is the support indicator. So dz = (dout - avg_S(dout)) restricted to S.
    p = _sparsemax_forward(z, axis=axis)
    pm, axis_n = _move_axis_last(p, axis)
    dom, _dax = _move_axis_last(dout, axis)
    support = (pm > 0).astype(np.float64)
    ssize = np.sum(support, axis=-1, keepdims=True)
    ssize = np.where(ssize == 0, 1.0, ssize)
    avg = np.sum(support * dom, axis=-1, keepdims=True) / ssize
    dz = support * (dom - avg)
    return (np.moveaxis(dz, -1, axis_n),)


# ── 1.5-entmax (Peters, Niculae & Martins 2019) ──────────────────────────────
def _entmax15_forward(z: np.ndarray, *, axis: int = -1, n_iter: int = 30) -> np.ndarray:
    x, axis = _move_axis_last(z, axis)
    x = x / 2.0  # alpha=1.5 scaling
    # bisection on the threshold τ so that sum(clip(x-τ,0)²) = 1.
    hi = np.max(x, axis=-1, keepdims=True)
    lo = hi - 1.0
    for _ in range(n_iter):
        tau = (lo + hi) / 2.0
        p = np.clip(x - tau, 0.0, None) ** 2
        s = np.sum(p, axis=-1, keepdims=True)
        lo = np.where(s < 1.0, lo, tau)
        hi = np.where(s < 1.0, tau, hi)
    tau = (lo + hi) / 2.0
    p = np.clip(x - tau, 0.0, None) ** 2
    p = p / np.sum(p, axis=-1, keepdims=True)
    return np.moveaxis(p, -1, axis)


def _entmax15_vjp(dout, z, *, axis: int = -1, **_kw):
    # α=1.5 entmax backward (Peters et al. 2019): with sᵢ = pᵢ^(2-α) = sqrt(pᵢ)
    # on the support, the Jacobian w.r.t. the logits is
    # J dout = s ⊙ (dout - ⟨s, dout⟩ / ⟨s, 1⟩), denominator Σ s (not Σ s²). The
    # (α-1)=½ factor is already absorbed by the s = p^(2-α) substitution, so
    # there is no extra scaling.
    p = _entmax15_forward(z, axis=axis)
    pm, axis_n = _move_axis_last(p, axis)
    dom, _dax = _move_axis_last(dout, axis)
    s = np.sqrt(pm)
    num = np.sum(s * dom, axis=-1, keepdims=True)
    den = np.sum(s, axis=-1, keepdims=True)  # ⟨s, 1⟩
    den = np.where(den == 0, 1.0, den)
    correction = num / den
    dz = s * (dom - correction)
    return (np.moveaxis(dz, -1, axis_n),)


# ── soft top-k (temperature-relaxed selection gate) ──────────────────────────
def _soft_top_k_forward(z: np.ndarray, *, k: int, tau: float = 1.0, axis: int = -1) -> np.ndarray:
    x, axis = _move_axis_last(z, axis)
    n = x.shape[-1]
    if not (1 <= k <= n):
        raise ValueError(f"soft_top_k requires 1<=k<={n}, got k={k}")
    # threshold at the k-th largest; sigmoid gate around it (soft indicator).
    kth = np.sort(x, axis=-1)[..., ::-1][..., k - 1:k]
    gate = 1.0 / (1.0 + np.exp(-(x - kth) / tau))
    return np.moveaxis(gate, -1, axis)


def _soft_top_k_vjp(dout, z, *, k: int, tau: float = 1.0, axis: int = -1, **_kw):
    # Treat the k-th-value threshold as a stop-gradient constant (its own
    # derivative is the discrete part); differentiate the sigmoid gate.
    x, axis_n = _move_axis_last(z, axis)
    dom, _dax = _move_axis_last(dout, axis)
    kth = np.sort(x, axis=-1)[..., ::-1][..., k - 1:k]
    g = 1.0 / (1.0 + np.exp(-(x - kth) / tau))
    dz = dom * g * (1.0 - g) / tau
    return (np.moveaxis(dz, -1, axis_n),)


# ── Gumbel-softmax / concrete relaxation ─────────────────────────────────────
def _gumbel_softmax_forward(
    logits: np.ndarray,
    *,
    tau: float = 1.0,
    axis: int = -1,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    x, axis = _move_axis_last(logits, axis)
    if noise is None:
        # Deterministic (noise-free) relaxation == tempered softmax. Callers
        # wanting a *sample* pass Gumbel noise via `noise` (rng.gumbel).
        g = np.zeros_like(x)
    else:
        g, _ = _move_axis_last(noise, axis)
    y = (x + g) / tau
    y = y - np.max(y, axis=-1, keepdims=True)
    e = np.exp(y)
    p = e / np.sum(e, axis=-1, keepdims=True)
    return np.moveaxis(p, -1, axis)


def _gumbel_softmax_vjp(dout, logits, *, tau: float = 1.0, axis: int = -1, noise=None, **_kw):
    # softmax Jacobian on the tempered logits, scaled by 1/tau.
    p = _gumbel_softmax_forward(logits, tau=tau, axis=axis, noise=noise)
    pm, axis_n = _move_axis_last(p, axis)
    dom, _dax = _move_axis_last(dout, axis)
    dot = np.sum(dom * pm, axis=-1, keepdims=True)
    dz = pm * (dom - dot) / tau
    return (np.moveaxis(dz, -1, axis_n),)


# ── perturbed argmax (Berthet et al. 2020) ───────────────────────────────────
def _perturbed_argmax_forward(
    z: np.ndarray,
    *,
    sigma: float = 1.0,
    n_samples: int = 500,
    seed: int = 0,
    axis: int = -1,
) -> np.ndarray:
    x, axis = _move_axis_last(z, axis)
    rng = np.random.default_rng(seed)
    n = x.shape[-1]
    acc = np.zeros_like(x)
    for _ in range(n_samples):
        g = rng.standard_normal(x.shape)
        idx = np.argmax(x + sigma * g, axis=-1)
        onehot = np.eye(n)[idx]
        acc = acc + onehot
    p = acc / n_samples
    return np.moveaxis(p, -1, axis)


def _perturbed_argmax_vjp(dout, z, *, sigma: float = 1.0, n_samples: int = 500, seed: int = 0, axis: int = -1, **_kw):
    # Score-function Jacobian of the perturbed argmax (Berthet et al. §3):
    # J = E[ y(z+σg) gᵀ ] / σ, estimated with the SAME seed as the forward so
    # the estimate is consistent.
    x, axis_n = _move_axis_last(z, axis)
    dom, _dax = _move_axis_last(dout, axis)
    rng = np.random.default_rng(seed)
    n = x.shape[-1]
    acc = np.zeros_like(x)
    for _ in range(n_samples):
        g = rng.standard_normal(x.shape)
        idx = np.argmax(x + sigma * g, axis=-1)
        onehot = np.eye(n)[idx]
        # contribution to J dout: (onehot·dout) g / σ  (outer-product structure)
        weight = np.sum(onehot * dom, axis=-1, keepdims=True)
        acc = acc + weight * g / sigma
    dz = acc / n_samples
    return (np.moveaxis(dz, -1, axis_n),)


# ── registration ─────────────────────────────────────────────────────────────
def _register() -> dict[str, Any]:
    prims = {}
    prims["sparsemax"] = custom_primitive("sparsemax", vjp=_sparsemax_vjp)(_sparsemax_forward)
    prims["entmax15"] = custom_primitive("entmax15", vjp=_entmax15_vjp)(_entmax15_forward)
    prims["soft_top_k"] = custom_primitive("soft_top_k", vjp=_soft_top_k_vjp)(_soft_top_k_forward)
    prims["gumbel_softmax"] = custom_primitive("gumbel_softmax", vjp=_gumbel_softmax_vjp)(_gumbel_softmax_forward)
    prims["perturbed_argmax"] = custom_primitive(
        "perturbed_argmax", vjp=_perturbed_argmax_vjp
    )(_perturbed_argmax_forward)
    return prims


RELAXATION_PRIMITIVES = _register()

__all__ = ["RELAXATION_PRIMITIVES"]
