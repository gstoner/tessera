"""Standalone Tessera loss / criterion library for S11."""

from __future__ import annotations

from typing import Any

import numpy as np


def _asarray(x: Any) -> np.ndarray:
    if hasattr(x, "_data"):
        x = x._data
    if hasattr(x, "_data"):
        x = x._data
    return np.asarray(x)


def _reduce(x: np.ndarray, reduction: str):
    if reduction == "none":
        return x
    if reduction == "mean":
        return np.mean(x)
    if reduction == "sum":
        return np.sum(x)
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def _logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x - _logsumexp(x, axis=axis, keepdims=True)


def mse_loss(pred, target, reduction: str = "mean"):
    return _reduce((_asarray(pred) - _asarray(target)) ** 2, reduction)


def mae_loss(pred, target, reduction: str = "mean"):
    return _reduce(np.abs(_asarray(pred) - _asarray(target)), reduction)


def huber_loss(pred, target, delta: float = 1.0, reduction: str = "mean"):
    err = _asarray(pred) - _asarray(target)
    abs_err = np.abs(err)
    d = float(delta)
    loss = np.where(abs_err <= d, 0.5 * err * err, d * (abs_err - 0.5 * d))
    return _reduce(loss, reduction)


def smooth_l1_loss(pred, target, beta: float = 1.0, reduction: str = "mean"):
    err = np.abs(_asarray(pred) - _asarray(target))
    b = float(beta)
    loss = np.where(err < b, 0.5 * err * err / b, err - 0.5 * b)
    return _reduce(loss, reduction)


def log_cosh_loss(pred, target, reduction: str = "mean"):
    err = _asarray(pred) - _asarray(target)
    loss = err + np.log1p(np.exp(-2.0 * err)) - np.log(2.0)
    return _reduce(loss, reduction)


def cross_entropy_loss(logits, targets, reduction: str = "mean"):
    logits = _asarray(logits).astype(np.float64, copy=False)
    targets = _asarray(targets)
    log_probs = _log_softmax(logits, axis=-1)
    if targets.dtype.kind in "iu":
        flat = log_probs.reshape(-1, log_probs.shape[-1])
        idx = targets.reshape(-1).astype(np.int64)
        loss = -flat[np.arange(idx.size), idx].reshape(targets.shape)
    else:
        loss = -np.sum(targets * log_probs, axis=-1)
    return _reduce(loss, reduction)


def binary_cross_entropy_loss(logits, targets, reduction: str = "mean"):
    logits = _asarray(logits).astype(np.float64, copy=False)
    targets = _asarray(targets).astype(np.float64, copy=False)
    loss = np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    return _reduce(loss, reduction)


def asymmetric_bce(
    logits,
    targets,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
    reduction: str = "mean",
):
    """Asymmetric binary cross-entropy with logits.

    Standard BCE penalizes false-negatives (the positive term) and
    false-positives (the negative term) equally. LDT / lattice candidate-mask
    objectives are *asymmetric*: eliminating a true candidate (false-negative)
    is far more costly than keeping a spurious one (false-positive). This loss
    decouples the two with ``pos_weight`` / ``neg_weight``::

        L = pos_weight · t · softplus(-z)  +  neg_weight · (1-t) · softplus(z)

    which reduces to ``binary_cross_entropy_loss`` when both weights are 1.
    ``softplus`` is evaluated in the numerically stable ``log1p(exp(-|z|)) +
    relu(±z)`` form so large |logits| never overflow.

    Args:
        logits:  pre-sigmoid scores ``z`` (any shape).
        targets: binary targets ``t`` in ``{0, 1}`` (broadcastable to logits).
        pos_weight: multiplier on the positive (false-negative) term.
        neg_weight: multiplier on the negative (false-positive) term.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.
    """
    z = _asarray(logits).astype(np.float64, copy=False)
    t = _asarray(targets).astype(np.float64, copy=False)
    log1p_term = np.log1p(np.exp(-np.abs(z)))          # shared by both softplus
    softplus_neg = np.maximum(-z, 0.0) + log1p_term    # softplus(-z) = -log σ(z)
    softplus_pos = np.maximum(z, 0.0) + log1p_term     # softplus(+z) = -log(1-σ(z))
    loss = pos_weight * t * softplus_neg + neg_weight * (1.0 - t) * softplus_pos
    return _reduce(loss, reduction)


def focal_loss(logits, targets, gamma: float = 2.0, alpha: float | None = None, reduction: str = "mean"):
    logits = _asarray(logits).astype(np.float64, copy=False)
    targets = _asarray(targets)
    probs = np.exp(_log_softmax(logits, axis=-1))
    flat_probs = probs.reshape(-1, probs.shape[-1])
    idx = targets.reshape(-1).astype(np.int64)
    pt = flat_probs[np.arange(idx.size), idx].reshape(targets.shape)
    loss = -((1.0 - pt) ** gamma) * np.log(np.maximum(pt, 1e-12))
    if alpha is not None:
        loss = float(alpha) * loss
    return _reduce(loss, reduction)


def label_smoothed_cross_entropy(logits, targets, smoothing: float = 0.1, reduction: str = "mean"):
    logits = _asarray(logits).astype(np.float64, copy=False)
    targets = _asarray(targets).astype(np.int64)
    n_classes = logits.shape[-1]
    smooth = float(smoothing)
    one_hot = np.full(targets.shape + (n_classes,), smooth / max(1, n_classes - 1), dtype=np.float64)
    np.put_along_axis(one_hot, targets[..., None], 1.0 - smooth, axis=-1)
    return cross_entropy_loss(logits, one_hot, reduction=reduction)


def kl_divergence(p_log_probs, q_probs, reduction: str = "mean"):
    p_log = _asarray(p_log_probs).astype(np.float64, copy=False)
    q = _asarray(q_probs).astype(np.float64, copy=False)
    p = np.exp(p_log)
    loss = p * (p_log - np.log(np.maximum(q, 1e-12)))
    return _reduce(np.sum(loss, axis=-1), reduction)


def js_divergence(p_probs, q_probs, reduction: str = "mean"):
    p = _asarray(p_probs).astype(np.float64, copy=False)
    q = _asarray(q_probs).astype(np.float64, copy=False)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(m, 1e-12))), axis=-1)
    kl_qm = np.sum(q * (np.log(np.maximum(q, 1e-12)) - np.log(np.maximum(m, 1e-12))), axis=-1)
    return _reduce(0.5 * (kl_pm + kl_qm), reduction)


def wasserstein_distance(x, y, reduction: str = "mean"):
    x_sorted = np.sort(_asarray(x), axis=-1)
    y_sorted = np.sort(_asarray(y), axis=-1)
    return _reduce(np.mean(np.abs(x_sorted - y_sorted), axis=-1), reduction)


def cosine_embedding_loss(x1, x2, target, margin: float = 0.0, reduction: str = "mean"):
    a = _asarray(x1).astype(np.float64, copy=False)
    b = _asarray(x2).astype(np.float64, copy=False)
    t = _asarray(target)
    cos = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
    loss = np.where(t > 0, 1.0 - cos, np.maximum(0.0, cos - float(margin)))
    return _reduce(loss, reduction)


def contrastive_loss(x1, x2, target, margin: float = 1.0, reduction: str = "mean"):
    dist = np.linalg.norm(_asarray(x1) - _asarray(x2), axis=-1)
    t = _asarray(target)
    loss = t * dist * dist + (1.0 - t) * np.maximum(0.0, float(margin) - dist) ** 2
    return _reduce(loss, reduction)


def triplet_loss(anchor, positive, negative, margin: float = 1.0, reduction: str = "mean"):
    pos = np.linalg.norm(_asarray(anchor) - _asarray(positive), axis=-1)
    neg = np.linalg.norm(_asarray(anchor) - _asarray(negative), axis=-1)
    return _reduce(np.maximum(0.0, pos - neg + float(margin)), reduction)


def nt_xent_loss(embeddings, labels, temperature: float = 0.5, reduction: str = "mean"):
    z = _asarray(embeddings).astype(np.float64, copy=False)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)
    logits = z @ z.T / float(temperature)
    np.fill_diagonal(logits, -np.inf)
    labels = _asarray(labels)
    positives = labels[:, None] == labels[None, :]
    np.fill_diagonal(positives, False)
    log_probs = _log_softmax(logits, axis=-1)
    denom = np.maximum(positives.sum(axis=-1), 1)
    loss = -np.sum(np.where(positives, log_probs, 0.0), axis=-1) / denom
    return _reduce(loss, reduction)


def info_nce_loss(query, positive, negatives, temperature: float = 0.1, reduction: str = "mean"):
    q = _asarray(query).astype(np.float64, copy=False)
    p = _asarray(positive).astype(np.float64, copy=False)
    n = _asarray(negatives).astype(np.float64, copy=False)
    pos = np.sum(q * p, axis=-1, keepdims=True)
    neg = np.einsum("bd,bkd->bk", q, n)
    logits = np.concatenate([pos, neg], axis=-1) / float(temperature)
    return cross_entropy_loss(logits, np.zeros(q.shape[0], dtype=np.int64), reduction=reduction)


def ddpm_noise_pred_loss(pred_noise, true_noise, reduction: str = "mean"):
    return mse_loss(pred_noise, true_noise, reduction=reduction)


def score_matching_loss(score, target_score, reduction: str = "mean"):
    return 0.5 * mse_loss(score, target_score, reduction=reduction)


def vlb_loss(terms, reduction: str = "mean"):
    return _reduce(_asarray(terms), reduction)


def seq2seq_loss(logits, targets, mask=None, reduction: str = "mean"):
    loss = cross_entropy_loss(logits, targets, reduction="none")
    if mask is not None:
        loss = loss * _asarray(mask)
        if reduction == "mean":
            return np.sum(loss) / max(float(np.sum(_asarray(mask))), 1.0)
    return _reduce(loss, reduction)


# ---------------------------------------------------------------------------
# EBM4 — Energy-based-model training losses.
#
# All four are pre-computed-tensor APIs: the user supplies the necessary
# quantities (energies, scores) and `reduction` controls the per-sample
# reduction. See `docs/audit/domain/DOMAIN_AUDIT.md` § EBM4.
# ---------------------------------------------------------------------------

def contrastive_divergence_loss(energy_pos, energy_neg, reduction: str = "mean"):
    """k-step Contrastive Divergence loss: ``L = E(x⁺) − E(x⁻)``.

    The user is responsible for generating ``x⁻`` via k Langevin / MALA /
    HMC / Gibbs steps from ``x⁺`` (using `tessera.rng.langevin_sample` or
    similar), evaluating both energies, and passing them here. Treating
    ``x⁻`` as detached during the gradient pass is the standard CD
    practice (Hinton 2002).
    """
    diff = _asarray(energy_pos) - _asarray(energy_neg)
    return _reduce(diff, reduction)


def persistent_cd_loss(energy_pos, energy_persistent_neg, reduction: str = "mean"):
    """Persistent Contrastive Divergence — same formula as CD but the
    ``x⁻`` samples come from a chain that persists across batches
    (Tieleman 2008). The user maintains the persistent chain state
    externally; this loss is just the energy difference.
    """
    diff = _asarray(energy_pos) - _asarray(energy_persistent_neg)
    return _reduce(diff, reduction)


def implicit_score_matching_loss(score, divergence_score, reduction: str = "mean"):
    """Implicit (Hyvärinen 2005) Score Matching: ``L = ½‖s(y)‖² + tr(∇·s(y))``.

    Inputs:
        score: model score evaluated at the data points; shape (B, D).
        divergence_score: divergence ``Σ_i ∂s_i/∂y_i`` per sample; shape (B,).

    The trace can be estimated cheaply with Hutchinson's estimator —
    that estimation is left to the caller; this loss just sums the two
    contributions. Per Hyvärinen, minimizing this drives the model
    score toward the true data score even though Z_θ is intractable.
    """
    s = _asarray(score).astype(np.float64, copy=False)
    div = _asarray(divergence_score).astype(np.float64, copy=False)
    # Per-sample: 0.5 * ||s||² + div. Sum over the feature axis for ||s||².
    sum_sq = 0.5 * (s ** 2).sum(axis=-1)
    return _reduce(sum_sq + div, reduction)


def denoising_score_matching_loss(score_noisy, y_clean, y_noisy, sigma: float,
                                  reduction: str = "mean"):
    """Vincent (2011) Denoising Score Matching:
    ``L = ½ ‖s_θ(ỹ) + (ỹ − y) / σ²‖²`` where ``ỹ = y + σ ξ`` is the
    noisy data and ``ξ ~ N(0, I)``.

    Inputs:
        score_noisy: model score at the noisy point, shape (B, D).
        y_clean:     clean data, shape (B, D).
        y_noisy:     noisy version, shape (B, D).
        sigma:       noise std (positive scalar).

    The target score has the closed-form ``-(ỹ − y)/σ²``; minimizing
    matches the model to it without needing Z_θ.
    """
    if sigma <= 0.0:
        raise ValueError(f"denoising_score_matching_loss requires sigma > 0; got {sigma}.")
    s = _asarray(score_noisy).astype(np.float64, copy=False)
    yc = _asarray(y_clean).astype(np.float64, copy=False)
    yn = _asarray(y_noisy).astype(np.float64, copy=False)
    target = -(yn - yc) / (sigma * sigma)
    diff_sq = 0.5 * ((s - target) ** 2).sum(axis=-1)
    return _reduce(diff_sq, reduction)


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank: int = 0, reduction: str = "mean"):
    """Small CPU-reference CTC forward loss."""
    lp = _asarray(log_probs).astype(np.float64, copy=False)
    targets = _asarray(targets).astype(np.int64)
    losses = []
    for b in range(lp.shape[1]):
        t_len = int(target_lengths[b])
        inp_len = int(input_lengths[b])
        target = targets[b, :t_len]
        ext = [blank]
        for token in target:
            ext.extend([int(token), blank])
        s = len(ext)
        alpha = np.full((inp_len, s), -np.inf, dtype=np.float64)
        alpha[0, 0] = lp[0, b, blank]
        if s > 1:
            alpha[0, 1] = lp[0, b, ext[1]]
        for t in range(1, inp_len):
            for i in range(s):
                prev = [alpha[t - 1, i]]
                if i - 1 >= 0:
                    prev.append(alpha[t - 1, i - 1])
                if i - 2 >= 0 and ext[i] != blank and ext[i] != ext[i - 2]:
                    prev.append(alpha[t - 1, i - 2])
                alpha[t, i] = np.logaddexp.reduce(prev) + lp[t, b, ext[i]]
        losses.append(-np.logaddexp(alpha[inp_len - 1, s - 1], alpha[inp_len - 1, s - 2] if s > 1 else -np.inf))
    return _reduce(np.asarray(losses), reduction)


__all__ = [
    "asymmetric_bce",
    "binary_cross_entropy_loss",
    "contrastive_loss",
    "cosine_embedding_loss",
    "cross_entropy_loss",
    "ctc_loss",
    "ddpm_noise_pred_loss",
    "focal_loss",
    "huber_loss",
    "info_nce_loss",
    "js_divergence",
    "kl_divergence",
    "label_smoothed_cross_entropy",
    "log_cosh_loss",
    "mae_loss",
    "mse_loss",
    "nt_xent_loss",
    "score_matching_loss",
    "seq2seq_loss",
    "smooth_l1_loss",
    "triplet_loss",
    "vlb_loss",
    "wasserstein_distance",
]
