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
    """Mean squared error, `(pred - target)**2`, reduced by `reduction`.

    No 1/2 factor: `score_matching_loss` supplies its own where its
    definition needs one.
    """
    return _reduce((_asarray(pred) - _asarray(target)) ** 2, reduction)


def mae_loss(pred, target, reduction: str = "mean"):
    """Mean absolute error, `|pred - target|`, reduced by `reduction`.

    Not differentiable at `pred == target`.
    """
    return _reduce(np.abs(_asarray(pred) - _asarray(target)), reduction)


def huber_loss(pred, target, delta: float = 1.0, reduction: str = "mean"):
    """Huber loss with knee at `delta`.

        0.5 * e**2                for |e| <= delta
        delta * (|e| - 0.5*delta) otherwise

    **Not the same function as `smooth_l1_loss`,** though they are related:
    `huber(delta=d) == d * smooth_l1(beta=d)` exactly (pinned in the tests).
    Huber keeps the quadratic branch unscaled, so its curvature near zero is
    independent of `delta`; smooth-L1 divides by `beta`, so the two agree
    only at `delta = beta = 1`. The branch boundary also differs: `<= delta`
    here, `< beta` there.
    """
    err = _asarray(pred) - _asarray(target)
    abs_err = np.abs(err)
    d = float(delta)
    loss = np.where(abs_err <= d, 0.5 * err * err, d * (abs_err - 0.5 * d))
    return _reduce(loss, reduction)


def smooth_l1_loss(pred, target, beta: float = 1.0, reduction: str = "mean"):
    """Smooth L1 loss with transition at `beta`.

        0.5 * e**2 / beta   for |e| < beta
        |e| - 0.5*beta      otherwise

    Equals `huber_loss(delta=beta) / beta` -- see the note there for why the
    two are not interchangeable.
    """
    err = np.abs(_asarray(pred) - _asarray(target))
    b = float(beta)
    loss = np.where(err < b, 0.5 * err * err / b, err - 0.5 * b)
    return _reduce(loss, reduction)


def log_cosh_loss(pred, target, reduction: str = "mean"):
    # log(cosh(e)) = |e| + log1p(exp(-2|e|)) - log 2. The |e| form is required,
    # not cosmetic: writing it on the raw error makes exp(-2e) overflow float64
    # for e < -354.9 and return inf, while log_cosh is even in e.
    """`log(cosh(pred - target))`, reduced by `reduction`.

    A smooth approximation to L1 that stays twice differentiable at zero.
    Computed as `|e| + log1p(exp(-2|e|)) - log 2` on the ABSOLUTE error --
    the even form is required, not cosmetic: on the raw error `exp(-2e)`
    overflows float64 for `e < -354.9` and returns `inf`.
    """
    err = np.abs(_asarray(pred) - _asarray(target))
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
    """Softmax cross-entropy over `axis`, from LOGITS.

    Accepts either integer class indices (shape = logits with `axis`
    removed) or a soft target distribution the same shape as `logits`.
    `ignore_index` drops positions from both the sum and the `mean`
    denominator, so masked tokens do not dilute the average.
    `label_smoothing` in [0, 1) mixes the one-hot target with the uniform
    distribution.

    Takes logits, never probabilities: it applies `log_softmax` itself.
    Passing probabilities computes a finite, wrong number rather than
    raising, because a probability vector is a perfectly valid logit vector.
    """
    logits = _asarray(logits).astype(np.float64, copy=False)
    targets = _asarray(targets)
    axis = int(axis)
    if axis < 0:
        axis += logits.ndim
    if axis < 0 or axis >= logits.ndim:
        raise ValueError("axis out of range")
    smooth = float(label_smoothing)
    if not 0.0 <= smooth < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
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
        # Both are semantic keys (Decision #21a) and neither is expressible
        # against a distribution target: smoothing is already the caller's to
        # apply to `targets`, and there is no single index to mask. Applying
        # smoothing here silently would also diverge from
        # autodiff.vjp.vjp_cross_entropy_loss, which differentiates the
        # unsmoothed objective on this branch.
        if smooth:
            raise ValueError(
                "label_smoothing is not supported for probability targets; "
                "pass label_smoothing=0.0 and smooth `targets` directly")
        if 0 <= int(ignore_index) < logits.shape[axis]:
            raise ValueError(
                f"ignore_index={int(ignore_index)} names a class but targets "
                "are a probability distribution; masking is undefined here")
        loss = -np.sum(targets * log_probs, axis=axis)
    return _reduce(loss, reduction)


def binary_cross_entropy_loss(logits, targets, reduction: str = "mean"):
    """Binary cross-entropy from LOGITS (i.e. BCE-with-logits).

    Computed in the numerically stable form
    `max(x, 0) - x*t + log1p(exp(-|x|))`, which stays exact for large |x|
    where `log(sigmoid(x))` underflows.

    The name omits "with_logits" but the behaviour does not: this applies
    the sigmoid itself. Handing it probabilities in [0, 1] silently computes
    the loss of `sigmoid(p)` -- finite, plausible, and wrong.
    """
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


def focal_loss(logits, targets, gamma: float = 2.0, alpha: float | None = None,
               reduction: str = "mean", *, ignore_index: int = -100):
    """Focal loss over integer class targets.

    Targets are range-checked and ``ignore_index`` is honoured, matching
    :func:`cross_entropy_loss` in this module. Previously the class index was
    fancy-indexed unchecked, so numpy's negative-index wraparound silently
    turned the -100 padding convention into "the 100th class from the end" and
    averaged that probability into the loss.
    """
    logits = _asarray(logits).astype(np.float64, copy=False)
    targets = _asarray(targets)
    probs = np.exp(_log_softmax(logits, axis=-1))
    flat_probs = probs.reshape(-1, probs.shape[-1])
    idx = targets.reshape(-1).astype(np.int64)
    valid = idx != int(ignore_index)
    if np.any(valid & ((idx < 0) | (idx >= flat_probs.shape[-1]))):
        raise ValueError("target class index out of range")
    safe_idx = np.where(valid, idx, 0)
    pt = flat_probs[np.arange(idx.size), safe_idx]
    loss_flat = -((1.0 - pt) ** gamma) * np.log(np.maximum(pt, 1e-12))
    # Ignored positions contribute nothing — neither to the sum nor to the
    # mean's denominator.
    loss_flat = np.where(valid, loss_flat, 0.0)
    if alpha is not None:
        loss_flat = float(alpha) * loss_flat
    if reduction == "mean":
        denom = float(np.count_nonzero(valid))
        return np.float64(loss_flat.sum() / denom) if denom else np.float64(0.0)
    return _reduce(loss_flat.reshape(targets.shape), reduction)


def label_smoothed_cross_entropy(
    logits,
    targets,
    smoothing: float = 0.1,
    reduction: str = "mean",
    *,
    axis: int = -1,
    ignore_index: int = -100,
):
    """`cross_entropy_loss` with `label_smoothing=smoothing`.

    A thin alias kept because the smoothing variant is named separately in
    much of the literature; it is one implementation, not two (#31). Note
    `smoothing` is positional here and keyword-only there.
    """
    return cross_entropy_loss(
        logits, targets, reduction=reduction, axis=axis,
        ignore_index=ignore_index, label_smoothing=smoothing)


def kl_divergence(
    p_log_probs, q_probs, reduction: str = "mean", *,
    axis: int = -1, epsilon: float = 1e-12,
):
    """`KL(p || q) = sum p * (log p - log q)` over `axis`.

    **The argument order is the INVERSE of PyTorch's `F.kl_div`, and getting
    it wrong is silent.** Here `p_log_probs` is the FIRST distribution of
    `KL(p||q)`, in log space, and `q_probs` is the second, in probability
    space. PyTorch takes `kl_div(input, target)` where `input` is the log of
    the SECOND argument and `target` is the first -- so porting an argument
    list across computes `KL(q||p)`. KL is asymmetric, so that is a
    different number rather than a rounding difference: for
    `p = (.1,.2,.7)`, `q = (.3,.3,.4)` it returns 0.2274 instead of 0.2008.
    Both are finite and neither raises. The tests pin both directions.

    Zero-probability entries are dropped rather than propagated: `p log p`
    at `p = 0` is `0 * -inf = NaN` in floating point while its limit is 0,
    and `-inf` arrives routinely from `log_softmax` over masked logits.
    `epsilon` floors `q` only.
    """
    p_log = _asarray(p_log_probs).astype(np.float64, copy=False)
    q = _asarray(q_probs).astype(np.float64, copy=False)
    if p_log.shape != q.shape:
        raise ValueError("KL operands must have identical shapes")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and greater than zero")
    p = np.exp(p_log)
    # A zero-probability entry has p_log = -inf, and 0 * (-inf) is NaN, which
    # then poisons the whole reduction. Its true contribution is
    # lim_{p->0} p*log(p/q) = 0, so drop those terms explicitly. -inf reaches
    # here routinely: log_softmax over masked logits (distillation,
    # constrained decoding) produces it by construction.
    support = p > 0.0
    finite_log_p = np.where(support, p_log, 0.0)  # never multiply 0 by -inf
    loss = np.where(
        support,
        p * (finite_log_p - np.log(np.maximum(q, float(epsilon)))),
        0.0,
    )
    return _reduce(np.sum(loss, axis=int(axis)), reduction)


def js_divergence(
    p_probs, q_probs, reduction: str = "mean", *,
    axis: int = -1, epsilon: float = 1e-12,
):
    """Jensen-Shannon divergence, `0.5*KL(p||m) + 0.5*KL(q||m)`, `m = (p+q)/2`.

    Symmetric and bounded, unlike `kl_divergence` -- and unlike it, BOTH
    arguments are probabilities, not log-probabilities. Natural log, so the
    range is [0, ln 2]; this is the divergence, not its square root (the JS
    *distance*). `epsilon` floors every log argument.
    """
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
    """1-D Wasserstein-1 distance between empirical samples on the last axis.

    `mean(|sort(x) - sort(y)|)` -- the closed form for W1 in one dimension,
    where the optimal transport plan is the order-statistic matching. It
    therefore requires `x` and `y` to have the SAME number of samples and
    reads the last axis as the sample axis, not as a feature vector. This is
    not a general optimal-transport solve and does not become one in higher
    dimensions.
    """
    x_sorted = np.sort(_asarray(x), axis=-1)
    y_sorted = np.sort(_asarray(y), axis=-1)
    return _reduce(np.mean(np.abs(x_sorted - y_sorted), axis=-1), reduction)


def cosine_embedding_loss(x1, x2, target, margin: float = 0.0, reduction: str = "mean"):
    """`1 - cos` for similar pairs, `max(0, cos - margin)` for dissimilar.

    `target` is read as `target > 0`, so the usual +1/-1 convention works
    and so does 1/0. The cosine denominator carries a 1e-12 floor, so a zero
    vector yields cos 0 rather than NaN.
    """
    a = _asarray(x1).astype(np.float64, copy=False)
    b = _asarray(x2).astype(np.float64, copy=False)
    t = _asarray(target)
    cos = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
    loss = np.where(t > 0, 1.0 - cos, np.maximum(0.0, cos - float(margin)))
    return _reduce(loss, reduction)


def contrastive_loss(x1, x2, target, margin: float = 1.0, reduction: str = "mean"):
    """Hadsell-Chopra contrastive loss on Euclidean distance.

        target * d**2 + (1 - target) * max(0, margin - d)**2

    `target = 1` means the pair is SIMILAR (pulled together). Both branches
    are squared, so this is not `triplet_loss`'s hinge on raw distance. And
    `target` is used arithmetically rather than as a predicate, so a -1/+1
    encoding is wrong here even though it works in `cosine_embedding_loss`.
    """
    dist = np.linalg.norm(_asarray(x1) - _asarray(x2), axis=-1)
    t = _asarray(target)
    loss = t * dist * dist + (1.0 - t) * np.maximum(0.0, float(margin) - dist) ** 2
    return _reduce(loss, reduction)


def triplet_loss(anchor, positive, negative, margin: float = 1.0, reduction: str = "mean"):
    """`max(0, d(a, p) - d(a, n) + margin)` on Euclidean distances.

    Raw distances, not squared -- the other common formulation squares them,
    which changes the gradient scale but not the sign.
    """
    pos = np.linalg.norm(_asarray(anchor) - _asarray(positive), axis=-1)
    neg = np.linalg.norm(_asarray(anchor) - _asarray(negative), axis=-1)
    return _reduce(np.maximum(0.0, pos - neg + float(margin)), reduction)


def nt_xent_loss(embeddings, labels, temperature: float = 0.5, reduction: str = "mean"):
    """NT-Xent (SimCLR) loss over an L2-normalised batch.

    Similarities are `z z^T / temperature` with the diagonal set to `-inf`,
    so a sample is never its own positive. Supports MORE THAN ONE positive
    per anchor -- positives are all same-label pairs and their
    log-probabilities are averaged -- which the original two-view
    formulation does not. An anchor with no positive contributes 0 (the
    denominator is floored at 1) rather than NaN.
    """
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
    """InfoNCE: cross-entropy over `[positive, *negatives] / temperature`.

    The positive is concatenated at index 0 and the target is always class
    0, so `negatives` is `(batch, k, dim)` while query and positive are
    `(batch, dim)`. Scores are dot products on the raw vectors: normalise
    beforehand if cosine similarity is intended.
    """
    q = _asarray(query).astype(np.float64, copy=False)
    p = _asarray(positive).astype(np.float64, copy=False)
    n = _asarray(negatives).astype(np.float64, copy=False)
    pos = np.sum(q * p, axis=-1, keepdims=True)
    neg = np.einsum("bd,bkd->bk", q, n)
    logits = np.concatenate([pos, neg], axis=-1) / float(temperature)
    return cross_entropy_loss(logits, np.zeros(q.shape[0], dtype=np.int64), reduction=reduction)


def ddpm_noise_pred_loss(pred_noise, true_noise, reduction: str = "mean"):
    """DDPM simplified objective: MSE between predicted and true noise.

    Exactly `mse_loss` -- named separately because it is the `L_simple` term
    of the diffusion objective, and the name is what makes a training loop
    readable. No 1/2 factor.
    """
    return mse_loss(pred_noise, true_noise, reduction=reduction)


def score_matching_loss(score, target_score, reduction: str = "mean"):
    """`0.5 * ||score - target_score||**2`, reduced by `reduction`.

    The 1/2 IS part of this definition, unlike `mse_loss` which carries
    none -- so this is `0.5 * mse_loss`, not an alias for it.
    """
    return 0.5 * mse_loss(score, target_score, reduction=reduction)


def vlb_loss(terms, reduction: str = "mean"):
    """Reduce PRE-COMPUTED variational-bound terms; it computes no bound.

    A carrier, not a derivation: `terms` already holds the per-element
    KL/NLL contributions and this only applies `reduction`. It exists so a
    training loop can name the quantity it reduces, and so the term appears
    in the op catalog under the same reduction contract as every other loss.
    """
    return _reduce(_asarray(terms), reduction)


def seq2seq_loss(logits, targets, mask=None, reduction: str = "mean"):
    """Token-level cross-entropy with an optional mask.

    With a mask and `reduction="mean"` the denominator is the SUM OF THE
    MASK, not the element count -- a per-token mean over real tokens, which
    is what makes the value comparable across batches with different
    padding. Without a mask it is plain `cross_entropy_loss`. The mask
    multiplies the loss, so float weights are allowed, not only 0/1.
    """
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


# ── Fenchel-Young losses (T2) ────────────────────────────────────────────────
#
# A Fenchel-Young loss (Blondel, Martins & Niculae 2020; book Ch. 15 §4) is
# generated from a prediction map ŷ(θ) = ∇Ω*(θ) and a regularizer Ω on the
# simplex:
#
#     L_Ω(θ, y) = Ω*(θ) + Ω(y) - ⟨θ, y⟩ = ⟨θ, ŷ - y⟩ - Ω(ŷ) + Ω(y),
#
# using Ω*(θ) = ⟨θ, ŷ⟩ - Ω(ŷ). The template's payoff: the gradient in θ is
# *exactly* ``ŷ(θ) - y`` — no hand-derived VJP, no autodiff — and the loss is a
# Bregman divergence, so L ≥ 0 with equality iff ŷ(θ) = y. One template yields:
#
#   * Ω = negative entropy → ŷ = softmax → cross-entropy (for one-hot y);
#   * Ω = ½‖·‖²          → ŷ = sparsemax → the sparsemax loss.
#
# This collapses several bespoke losses into one construction, each with an
# exact gradient rather than a separately-maintained backward.


def _negentropy(p: np.ndarray, axis: int) -> np.ndarray:
    # Ω(p) = Σ p log p, with the convention 0·log0 = 0.
    q = np.where(p > 0, p, 1.0)
    return np.sum(p * np.log(q), axis=axis)


def _fy_prediction(theta: np.ndarray, omega: str, axis: int) -> np.ndarray:
    if omega == "entropy":
        m = np.max(theta, axis=axis, keepdims=True)
        e = np.exp(theta - m)
        return e / np.sum(e, axis=axis, keepdims=True)
    if omega == "l2":
        from .relaxation import _sparsemax_forward
        return _sparsemax_forward(theta, axis=axis)
    raise ValueError(f"unknown Fenchel-Young regularizer {omega!r}; "
                     f"expected 'entropy' or 'l2'")


def _fy_omega(p: np.ndarray, omega: str, axis: int) -> np.ndarray:
    if omega == "entropy":
        return _negentropy(p, axis)
    if omega == "l2":
        return 0.5 * np.sum(p * p, axis=axis)
    raise ValueError(f"unknown Fenchel-Young regularizer {omega!r}")


def fenchel_young_loss(
    theta,
    y,
    *,
    omega: str = "entropy",
    axis: int = -1,
    reduction: str = "mean",
):
    """Fenchel-Young loss ``L_Ω(θ, y)`` for regularizer ``omega``.

    ``theta`` are scores/logits, ``y`` a target *distribution* on the simplex
    (a one-hot vector is the hard-label case). ``omega='entropy'`` gives the
    (softmax) cross-entropy family; ``omega='l2'`` gives the sparsemax loss.
    Non-negative, zero iff the induced prediction equals ``y``.
    """
    theta = _asarray(theta).astype(np.float64, copy=False)
    y = _asarray(y).astype(np.float64, copy=False)
    if y.shape != theta.shape:
        raise ValueError(
            "Fenchel-Young loss expects a target distribution matching theta's "
            "shape (use a one-hot encoding for hard labels)"
        )
    yhat = _fy_prediction(theta, omega, axis)
    loss = (
        np.sum(theta * (yhat - y), axis=axis)
        - _fy_omega(yhat, omega, axis)
        + _fy_omega(y, omega, axis)
    )
    # Numerical floor: the loss is provably ≥ 0; clip tiny negatives from
    # finite-precision cancellation rather than surface them.
    loss = np.maximum(loss, 0.0)
    return _reduce(loss, reduction)


def fy_loss_and_grad(
    theta,
    y,
    *,
    omega: str = "entropy",
    axis: int = -1,
    reduction: str = "mean",
):
    """Return ``(loss, grad_theta)`` where ``grad_theta = ŷ(θ) - y`` exactly.

    The exact gradient is the point of the construction — a training loop can
    use it directly without an autodiff backward. ``grad_theta`` matches the
    reduction: for ``'mean'`` it is scaled by the number of reduced examples.
    """
    theta = _asarray(theta).astype(np.float64, copy=False)
    y = _asarray(y).astype(np.float64, copy=False)
    yhat = _fy_prediction(theta, omega, axis)
    grad = yhat - y
    loss = fenchel_young_loss(theta, y, omega=omega, axis=axis, reduction=reduction)
    if reduction == "mean":
        # loss averaged over the leading (example) axes → scale grad likewise.
        n = int(theta.size // theta.shape[axis]) if theta.ndim > 1 else 1
        grad = grad / max(n, 1)
    return loss, grad


def sparsemax_loss(theta, y, *, axis: int = -1, reduction: str = "mean"):
    """Sparsemax loss — the Fenchel-Young loss with Ω = ½‖·‖² (l2)."""
    return fenchel_young_loss(theta, y, omega="l2", axis=axis, reduction=reduction)


def softmax_fy_loss(theta, y, *, axis: int = -1, reduction: str = "mean"):
    """Softmax Fenchel-Young loss — equals cross-entropy for one-hot ``y``."""
    return fenchel_young_loss(theta, y, omega="entropy", axis=axis, reduction=reduction)


__all__ = [
    "asymmetric_bce",
    "binary_cross_entropy_loss",
    "fenchel_young_loss",
    "fy_loss_and_grad",
    "sparsemax_loss",
    "softmax_fy_loss",
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
