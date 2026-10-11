"""Calibration metrics for probabilistic decision heads.

A classifier is *calibrated* when its stated confidence matches its empirical
accuracy: of the predictions made at confidence 0.9, 90% are correct. This
module measures that (binned ECE / MCE and the reliability diagram behind
them), offers a differentiable surrogate usable as a training penalty
(:func:`soft_binned_ece`), and fits the standard post-hoc baseline any
calibration claim should be compared against (:func:`fit_temperature`).

The training-side proper scoring rules live in :mod:`tessera.losses`
(``brier_loss``, ``binary_brier_loss``, ``cross_entropy_loss`` — the
logarithmic score — and ``ordinal_logistic_loss``).

Input conventions for the top-label metrics:

* **multiclass** — ``probs`` of shape ``(..., K)`` on the simplex and integer
  ``labels`` of shape ``(...)``; confidence is the max probability and the
  prediction its argmax.
* **binary** — ``probs`` of shape ``(...)`` holding ``P(True)`` and ``labels``
  of the same shape in ``{0, 1}``; the prediction is ``P(True) >= 0.5`` and its
  confidence ``max(p, 1 - p)``.

Metrics take *probabilities*. Passing logits is the common mistake, so inputs
outside ``[0, 1]`` or off the simplex raise rather than score.

Bins are equal-width and right-closed, ``((m-1)/M, m/M]``, with confidence 0
assigned to the first bin.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

_SIMPLEX_ATOL = 1e-4


def _asarray(x: Any) -> np.ndarray:
    while hasattr(x, "_data"):
        x = x._data
    return np.asarray(x)


def _check_n_bins(n_bins: int) -> int:
    n = int(n_bins)
    if n < 1 or n != n_bins:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
    return n


def top_label_confidence(probs, labels) -> tuple[np.ndarray, np.ndarray]:
    """Flattened ``(confidence, correct)`` pairs for the top-label metrics.

    See the module docstring for the multiclass / binary conventions. Raises
    ``ValueError`` on shape mismatch, non-finite or out-of-range
    probabilities, rows off the simplex, or labels outside the class range.
    """
    p = _asarray(probs).astype(np.float64, copy=False)
    y = _asarray(labels)
    if y.dtype.kind not in "iub":
        raise ValueError("labels must be integer class indices (or {0,1} for binary)")
    if p.size == 0:
        raise ValueError("calibration metrics need at least one prediction")
    if not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError(
            "probabilities must lie in [0, 1] — were logits passed? apply "
            "softmax / sigmoid first")
    y = y.astype(np.int64)
    if p.shape == y.shape:                       # binary: P(True)
        if np.any((y != 0) & (y != 1)):
            raise ValueError("binary labels must be 0 or 1")
        pred = (p >= 0.5).astype(np.int64)
        conf = np.where(pred == 1, p, 1.0 - p)
        return conf.reshape(-1), (pred == y).reshape(-1).astype(np.float64)
    if p.ndim == y.ndim + 1 and p.shape[:-1] == y.shape:
        k = p.shape[-1]
        if k < 2:
            raise ValueError("multiclass probabilities need >= 2 classes")
        if np.any(np.abs(p.sum(axis=-1) - 1.0) > _SIMPLEX_ATOL):
            raise ValueError(
                "multiclass probability rows must sum to 1 — were logits or "
                "unnormalized scores passed?")
        if np.any((y < 0) | (y >= k)):
            raise ValueError("label class index out of range")
        flat = p.reshape(-1, k)
        pred = np.argmax(flat, axis=-1)
        conf = flat[np.arange(flat.shape[0]), pred]
        return conf, (pred == y.reshape(-1)).astype(np.float64)
    raise ValueError(
        f"probs {p.shape} and labels {y.shape} match neither the binary "
        "(same shape) nor the multiclass (probs = labels + trailing class "
        "axis) convention")


@dataclass(frozen=True)
class ReliabilityBins:
    """The reliability diagram behind ECE / MCE.

    ``mean_confidence`` and ``accuracy`` are NaN for empty bins; ``counts``
    says which bins are populated.
    """

    edges: np.ndarray            # (M + 1,)
    counts: np.ndarray           # (M,) int
    mean_confidence: np.ndarray  # (M,)
    accuracy: np.ndarray         # (M,)

    @property
    def total(self) -> int:
        return int(self.counts.sum())

    def gaps(self) -> np.ndarray:
        """``|accuracy − mean_confidence|`` per bin (NaN where empty)."""
        return np.abs(self.accuracy - self.mean_confidence)


def reliability_bins(confidences, correct, n_bins: int = 15) -> ReliabilityBins:
    """Bin ``(confidence, correct)`` pairs into ``n_bins`` equal-width bins."""
    m = _check_n_bins(n_bins)
    c = _asarray(confidences).astype(np.float64, copy=False).reshape(-1)
    a = _asarray(correct).astype(np.float64, copy=False).reshape(-1)
    if c.shape != a.shape:
        raise ValueError(
            f"confidences ({c.size}) and correct ({a.size}) differ in length")
    if c.size == 0:
        raise ValueError("calibration metrics need at least one prediction")
    if not np.all(np.isfinite(c)) or np.any(c < 0.0) or np.any(c > 1.0):
        raise ValueError("confidences must be finite and lie in [0, 1]")
    edges = np.linspace(0.0, 1.0, m + 1)
    # searchsorted(side="left") over the interior edges yields right-closed
    # bins ((m-1)/M, m/M] that agree with the reported `edges` exactly —
    # ceil(c*M) does not (0.3*10 rounds up to bin 4 of 10).
    idx = np.searchsorted(edges[1:-1], c, side="left")
    counts = np.bincount(idx, minlength=m)
    conf_sum = np.bincount(idx, weights=c, minlength=m)
    acc_sum = np.bincount(idx, weights=a, minlength=m)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(counts > 0, conf_sum / counts, np.nan)
        acc = np.where(counts > 0, acc_sum / counts, np.nan)
    return ReliabilityBins(edges=edges, counts=counts,
                           mean_confidence=mean_conf, accuracy=acc)


def expected_calibration_error(probs, labels, n_bins: int = 15) -> float:
    """Top-label ECE: ``Σ_m (|B_m| / N) · |acc(B_m) − conf(B_m)|``."""
    bins = reliability_bins(*top_label_confidence(probs, labels), n_bins=n_bins)
    filled = bins.counts > 0
    return float(np.sum(bins.counts[filled] / bins.total * bins.gaps()[filled]))


def max_calibration_error(probs, labels, n_bins: int = 15) -> float:
    """Top-label MCE: the worst populated bin's ``|acc − conf|``."""
    bins = reliability_bins(*top_label_confidence(probs, labels), n_bins=n_bins)
    return float(np.max(bins.gaps()[bins.counts > 0]))


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable surrogate
# ─────────────────────────────────────────────────────────────────────────────


def _soft_memberships(c: np.ndarray, n_bins: int, temperature: float):
    """Soft bin memberships ``u[i, m]`` and ``∂ℓ[i, m]/∂c_i``.

    ``u = softmax_m(−(c_i − μ_m)² / T)`` over bin centres ``μ_m``.
    """
    if not temperature > 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    centres = (np.arange(n_bins) + 0.5) / n_bins
    diff = c[:, None] - centres[None, :]
    logits = -(diff * diff) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    u = np.exp(logits)
    u /= u.sum(axis=-1, keepdims=True)
    dlogits = -2.0 * diff / temperature
    return u, dlogits


def soft_binned_ece(confidences, correct, *, n_bins: int = 15,
                    temperature: float = 1e-3) -> np.float64:
    """Differentiable soft-binned ECE (Karandikar et al., 2021 style).

    Hard ECE can be rewritten as ``(1/N) Σ_m |Σ_{i∈B_m} (a_i − c_i)|``; this
    replaces the hard membership ``i ∈ B_m`` with a soft one,
    ``u[i, m] = softmax_m(−(c_i − μ_m)² / T)``::

        soft_ece = (1/N) Σ_m | Σ_i u[i, m] · (a_i − c_i) |

    As ``T → 0`` it converges to :func:`expected_calibration_error` on the
    same bins (away from bin edges). ``confidences`` are the predicted
    top-label confidences ``c_i`` (differentiable — compose with a softmax
    max for a training penalty) and ``correct`` the outcomes ``a_i`` in
    ``[0, 1]`` (hard 0/1 in the usual case; the rule differentiates both).
    It is a dataset-level scalar, so there is no ``reduction`` argument.
    """
    m = _check_n_bins(n_bins)
    c = _asarray(confidences).astype(np.float64, copy=False).reshape(-1)
    a = _asarray(correct).astype(np.float64, copy=False).reshape(-1)
    if c.shape != a.shape:
        raise ValueError(
            f"confidences ({c.size}) and correct ({a.size}) differ in length")
    if c.size == 0:
        raise ValueError("soft_binned_ece needs at least one prediction")
    u, _ = _soft_memberships(c, m, float(temperature))
    r = u.T @ (a - c)
    return np.float64(np.sum(np.abs(r)) / c.size)


# ─────────────────────────────────────────────────────────────────────────────
# Post-hoc temperature scaling
# ─────────────────────────────────────────────────────────────────────────────


def _nll_terms(z: np.ndarray, zy: np.ndarray, beta: float):
    """Mean NLL of ``softmax(β z)`` and its first two β-derivatives."""
    s = beta * z
    s_max = s.max(axis=-1, keepdims=True)
    e = np.exp(s - s_max)
    denom = e.sum(axis=-1, keepdims=True)
    p = e / denom
    lse = (s_max + np.log(denom))[:, 0]
    nll = float(np.mean(lse - beta * zy))
    ez = np.sum(p * z, axis=-1)
    grad = float(np.mean(ez - zy))
    hess = float(np.mean(np.sum(p * z * z, axis=-1) - ez * ez))
    return nll, grad, hess


def fit_temperature(logits, labels, *, max_iter: int = 100, tol: float = 1e-10,
                    bounds: tuple[float, float] = (1e-2, 1e2)) -> float:
    """Fit the scalar temperature ``T`` minimizing the mean NLL of
    ``softmax(logits / T)`` on held-out ``(logits, labels)``.

    Solved as a 1-D convex problem in ``β = 1/T`` (the NLL is convex in β)
    with safeguarded Newton steps and a backtracking line search. ``T`` is
    confined to ``bounds``; hitting a bound warns, because it usually means
    the held-out set is separable (NLL keeps falling as ``T → 0``) or
    uninformative (all logits equal). Use the result as
    ``softmax(logits / T)``. ``logits`` are ``(..., K)`` and ``labels``
    integer indices over the leading axes.
    """
    z = _asarray(logits).astype(np.float64, copy=False)
    y = _asarray(labels)
    if z.ndim < 1 or z.shape[-1] < 2:
        raise ValueError("fit_temperature needs logits with >= 2 classes on the last axis")
    if y.dtype.kind not in "iu" or y.shape != z.shape[:-1]:
        raise ValueError(
            f"labels must be integer class indices of shape {z.shape[:-1]}, "
            f"got dtype {y.dtype} shape {y.shape}")
    if not np.all(np.isfinite(z)):
        raise ValueError("logits must be finite")
    k = z.shape[-1]
    z = z.reshape(-1, k)
    y = y.reshape(-1).astype(np.int64)
    if z.shape[0] == 0:
        raise ValueError("fit_temperature needs at least one example")
    if np.any((y < 0) | (y >= k)):
        raise ValueError("label class index out of range")
    t_lo, t_hi = float(bounds[0]), float(bounds[1])
    if not 0.0 < t_lo < t_hi:
        raise ValueError(f"bounds must satisfy 0 < low < high, got {bounds!r}")
    b_lo, b_hi = 1.0 / t_hi, 1.0 / t_lo
    zy = z[np.arange(z.shape[0]), y]

    # Two cases have no finite optimum, and Newton cannot see either: its
    # gradient decays like exp(−margin·β), so it stops at `tol` somewhere
    # arbitrary. Decide them from the data instead.
    others = np.where(np.arange(k)[None, :] == y[:, None], -np.inf, z)
    if np.all(zy > others.max(axis=-1)):
        # Every label is its row's strict argmax: dNLL/dβ < 0 for all β, so
        # the NLL falls monotonically as T → 0.
        warnings.warn(
            f"fit_temperature: the held-out set is separable (every label is "
            f"its row's strict argmax), so the NLL has no finite minimizer; "
            f"returning the lower bound T={t_lo:.6g}. Fit on data the model "
            f"gets wrong sometimes.", RuntimeWarning, stacklevel=2)
        return t_lo
    if np.all(np.ptp(z, axis=-1) == 0.0):
        warnings.warn(
            "fit_temperature: logits are constant within every row, so the "
            "temperature is unidentifiable; returning T=1.0",
            RuntimeWarning, stacklevel=2)
        return 1.0

    beta = min(max(1.0, b_lo), b_hi)
    nll, grad, hess = _nll_terms(z, zy, beta)
    for _ in range(int(max_iter)):
        if abs(grad) <= tol:
            break
        step = -grad / hess if hess > 0.0 else -np.sign(grad) * beta
        target = min(max(beta + step, b_lo), b_hi)
        # Backtrack (Armijo) — Newton on a convex 1-D function can still
        # overshoot where the curvature changes quickly.
        new = target
        for _ in range(60):
            new_nll, new_grad, new_hess = _nll_terms(z, zy, new)
            if new_nll <= nll + 1e-4 * grad * (new - beta):
                break
            new = beta + 0.5 * (new - beta)
        else:
            break
        moved = abs(new - beta)
        beta, nll, grad, hess = new, new_nll, new_grad, new_hess
        if moved <= tol * max(1.0, beta):
            break

    temperature = 1.0 / beta
    # Outward = the NLL still falls past the bound: past β_max (T at its
    # low bound) that is dNLL/dβ < 0; past β_min (T at its high bound),
    # dNLL/dβ > 0.
    at_bound = (np.isclose(beta, b_hi, rtol=1e-9) and grad < -tol) or (
        np.isclose(beta, b_lo, rtol=1e-9) and grad > tol)
    if at_bound:
        warnings.warn(
            f"fit_temperature stopped at the bound T={temperature:.6g} "
            "(gradient still pushing outward): the held-out set may be "
            "separable or the logits uninformative",
            RuntimeWarning, stacklevel=2)
    return float(temperature)


__all__ = [
    "ReliabilityBins",
    "expected_calibration_error",
    "fit_temperature",
    "max_calibration_error",
    "reliability_bins",
    "soft_binned_ece",
    "top_label_confidence",
]
