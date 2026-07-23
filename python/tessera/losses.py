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


def cross_entropy_loss(
    logits,
    targets,
    reduction: str = "mean",
    *,
    axis: int = -1,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
):
    logits = _asarray(logits).astype(np.float64, copy=False)
    targets = _asarray(targets)
    axis = int(axis)
    if axis < 0:
        axis += logits.ndim
    if axis < 0 or axis >= logits.ndim:
        raise ValueError("axis out of range")
    log_probs = _log_softmax(logits, axis=axis)
    if targets.dtype.kind in "iu":
        moved = np.moveaxis(log_probs, axis, -1)
        if targets.shape != moved.shape[:-1]:
            raise ValueError(
                "integer targets must match logits with class axis removed")
        flat = moved.reshape(-1, moved.shape[-1])
        idx = targets.reshape(-1).astype(np.int64)
        valid = idx != int(ignore_index)
        if np.any(valid & ((idx < 0) | (idx >= flat.shape[-1]))):
            raise ValueError("target class index out of range")
        safe_idx = np.where(valid, idx, 0)
        nll = -flat[np.arange(idx.size), safe_idx]
        smooth = float(label_smoothing)
        if not 0.0 <= smooth < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if smooth:
            if flat.shape[-1] <= 1:
                raise ValueError("label smoothing requires at least 2 classes")
            off_sum = -np.sum(flat, axis=-1) - nll
            nll = (1.0 - smooth) * nll + (
                smooth / (flat.shape[-1] - 1)) * off_sum
        loss = np.where(valid, nll, 0.0).reshape(targets.shape)
        if reduction == "mean":
            return np.sum(loss) / max(int(np.count_nonzero(valid)), 1)
    else:
        if targets.shape != logits.shape:
            raise ValueError("probability targets must match logits")
        loss = -np.sum(targets * log_probs, axis=axis)
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


def z_loss(router_logits, reduction: str = "mean"):
    """Router z-loss (ST-MoE / PaLM): penalize large router logits by the
    squared log-partition so the softmax denominator stays bounded::

        z_loss = reduce( logsumexp(router_logits, axis=-1)² )

    Acts as a numerical-stability regularizer on an MoE router; differentiable
    in ``router_logits``. ``reduction`` is taken over the token / leading axes.
    """
    logits = _asarray(router_logits).astype(np.float64, copy=False)
    lse = _logsumexp(logits, axis=-1)                  # (..,) over experts
    return _reduce(lse * lse, reduction)


def load_balance_loss(router_probs, *, assignment=None, reduction: str = "mean"):
    """Switch-Transformer load-balancing auxiliary loss::

        aux = E · Σ_e  f_e · P_e

    where ``E`` is the expert count, ``f_e`` is the fraction of tokens routed to
    expert ``e`` (a hard top-1 ``argmax`` — treated as a constant, stop-gradient),
    and ``P_e`` is the mean router probability mass on expert ``e``. The gradient
    flows only through ``P_e``, which pushes the router toward a uniform load.

    Args:
        router_probs: post-softmax probabilities, shape ``(..., T, E)``.
        assignment:   optional precomputed top-1 expert indices, shape
                      ``(..., T)``; defaults to ``argmax(router_probs, -1)``.
        reduction:    ``"mean"`` averages the per-leading-group aux losses;
                      ``"sum"`` adds them; ``"none"`` returns them per group.
    """
    p = _asarray(router_probs).astype(np.float64, copy=False)
    n_experts = p.shape[-1]
    n_tokens = p.shape[-2]
    if assignment is None:
        idx = np.argmax(p, axis=-1)
    else:
        idx = _asarray(assignment).astype(np.int64)
    one_hot = np.eye(n_experts, dtype=np.float64)[idx]     # (..., T, E)
    f = one_hot.mean(axis=-2)                              # (..., E) fraction routed
    P = p.mean(axis=-2)                                    # (..., E) mean prob mass
    aux = n_experts * np.sum(f * P, axis=-1)               # (...,) or scalar
    return _reduce(np.asarray(aux), reduction)


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


def label_smoothed_cross_entropy(
    logits,
    targets,
    smoothing: float = 0.1,
    reduction: str = "mean",
    *,
    axis: int = -1,
    ignore_index: int = -100,
):
    return cross_entropy_loss(
        logits, targets, reduction=reduction, axis=axis,
        ignore_index=ignore_index, label_smoothing=smoothing)


def kl_divergence(
    p_log_probs, q_probs, reduction: str = "mean", *,
    axis: int = -1, epsilon: float = 1e-12,
):
    p_log = _asarray(p_log_probs).astype(np.float64, copy=False)
    q = _asarray(q_probs).astype(np.float64, copy=False)
    if p_log.shape != q.shape:
        raise ValueError("KL operands must have identical shapes")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and greater than zero")
    p = np.exp(p_log)
    loss = p * (p_log - np.log(np.maximum(q, float(epsilon))))
    return _reduce(np.sum(loss, axis=int(axis)), reduction)


def js_divergence(
    p_probs, q_probs, reduction: str = "mean", *,
    axis: int = -1, epsilon: float = 1e-12,
):
    p = _asarray(p_probs).astype(np.float64, copy=False)
    q = _asarray(q_probs).astype(np.float64, copy=False)
    if p.shape != q.shape:
        raise ValueError("JS operands must have identical shapes")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and greater than zero")
    m = 0.5 * (p + q)
    axis = int(axis)
    floor = float(epsilon)
    kl_pm = np.sum(
        p * (np.log(np.maximum(p, floor)) - np.log(np.maximum(m, floor))),
        axis=axis)
    kl_qm = np.sum(
        q * (np.log(np.maximum(q, floor)) - np.log(np.maximum(m, floor))),
        axis=axis)
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


def mtp_e2e_tv_loss(
    target_logits,
    draft_logits,
    *,
    mask=None,
    reduction: str = "mean",
    detach_target: bool = True,
    return_metrics: bool = False,
):
    """End-to-end TV loss for multi-step MTP rejection sampling.

    Inputs are ``(batch, positions, mtp_steps, vocab)``.  The loss directly
    optimizes normalized expected accepted length:
    ``1 - gamma^-1 * sum_j prod_{i<=j} (1 - TV(p_i, q_i))``.
    """
    del detach_target  # numpy reference path treats both inputs as arrays.
    target = _asarray(target_logits).astype(np.float64, copy=False)
    draft = _asarray(draft_logits).astype(np.float64, copy=False)
    if target.shape != draft.shape:
        raise ValueError(f"target/draft logits must match; got {target.shape} vs {draft.shape}")
    if target.ndim != 4:
        raise ValueError(
            "mtp_e2e_tv_loss expects (batch, positions, mtp_steps, vocab) logits")
    p = np.exp(_log_softmax(target, axis=-1))
    q = np.exp(_log_softmax(draft, axis=-1))
    tv = 0.5 * np.sum(np.abs(p - q), axis=-1)       # (B,P,G)
    alpha = np.clip(1.0 - tv, 0.0, 1.0)
    prefix = np.cumprod(alpha, axis=-1)
    expected_accept_len = np.sum(prefix, axis=-1)   # (B,P)
    gamma = target.shape[-2]
    loss = 1.0 - expected_accept_len / max(gamma, 1)

    weight = None
    if mask is not None:
        weight = _asarray(mask).astype(np.float64, copy=False)
        if weight.shape != loss.shape:
            raise ValueError(f"mask must have shape {loss.shape}; got {weight.shape}")
        if not np.isfinite(weight).all() or np.any(weight < 0.0):
            raise ValueError("mask must be finite and non-negative")
        loss = loss * weight

    if reduction == "none":
        reduced = loss
    elif reduction == "sum":
        reduced = np.sum(loss)
    elif reduction == "mean":
        denom = np.sum(weight) if weight is not None else loss.size
        reduced = np.sum(loss) / max(float(denom), 1.0)
    else:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    if not return_metrics:
        return reduced
    target_entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=-1)
    draft_entropy = -np.sum(q * np.log(np.maximum(q, 1e-12)), axis=-1)
    metrics = {
        "per_step_tv": tv,
        "per_step_alpha": alpha,
        "expected_accept_len": expected_accept_len,
        "target_entropy": target_entropy,
        "draft_entropy": draft_entropy,
    }
    return reduced, metrics


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
    "load_balance_loss",
    "z_loss",
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
    "mtp_e2e_tv_loss",
    "nt_xent_loss",
    "score_matching_loss",
    "seq2seq_loss",
    "smooth_l1_loss",
    "triplet_loss",
    "vlb_loss",
    "wasserstein_distance",
]
