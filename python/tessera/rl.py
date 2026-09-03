"""Reference RL post-training losses for reasoning-model workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _asarray(x: Any) -> np.ndarray:
    if hasattr(x, "_data"):
        x = x._data
    if hasattr(x, "_data"):
        x = x._data
    return np.asarray(x)


def _reduce(x: np.ndarray, reduction: str, mask: np.ndarray | None = None):
    x = np.asarray(x)
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64)
        x = x * mask
    if reduction == "none":
        return x
    if reduction == "sum":
        return np.sum(x)
    if reduction == "mean":
        if mask is None:
            return np.mean(x)
        # Clamp ONLY the zero case. `max(sum, 1.0)` also rescaled every
        # sub-unit mask sum, so a float mask like [0.1, 0.1] normalised by
        # 1.0 instead of 0.2 and returned one fifth of the weighted mean.
        # Clamping the denominator UP makes the loss SMALLER by exactly the
        # ratio of the true sum to the clamp: measured on `ppo_policy_loss`
        # at mask scale 0.1, a mask summing to 0.3 returned 0.30x the
        # correct value. A
        # weighted mean must be invariant to the scale of its weights, and
        # nothing here restricts the mask to 0/1: the spec says only
        # "optional mask", the argument is typed `Any`, and it is cast to
        # float64 and multiplied without validation.
        #
        # At sum == 0 every term is already masked to zero, so dividing by
        # 1.0 yielded 0.0 -- the zero branch keeps that exactly, which is
        # why only the fractional case changes behaviour.
        total = float(np.sum(mask))
        if total == 0.0:
            return np.sum(x)          # every term is masked to zero
        return np.sum(x) / total
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def _kl_penalty(logp_new, ref_logp):
    if ref_logp is None:
        return 0.0
    # Non-negative Schulman-style sample KL approximation.
    delta = np.asarray(ref_logp) - np.asarray(logp_new)
    return np.exp(delta) - delta - 1.0


@dataclass
class RolloutBatch:
    logp_new: Any
    logp_old: Any
    rewards: Any | None = None
    advantages: Any | None = None
    ref_logp: Any | None = None
    mask: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_group_advantages(rewards, *, group_axis: int = 1, eps: float = 1e-8, mask=None):
    r = _asarray(rewards).astype(np.float64, copy=False)
    if mask is None:
        mean = np.mean(r, axis=group_axis, keepdims=True)
        var = np.mean((r - mean) ** 2, axis=group_axis, keepdims=True)
        return (r - mean) / np.sqrt(var + float(eps))
    m = np.asarray(_asarray(mask), dtype=np.float64)
    # Per-group, and the same rule as `_reduce`: clamp only the empty group.
    # `np.maximum(..., 1.0)` rescaled any group whose mask summed below 1, so
    # the group mean and variance -- and therefore every normalised advantage
    # in that group -- moved with the arbitrary scale of the weights.
    # A fully masked group contributes r*m = 0, so mean and var are 0 and the
    # result is 0 either way; dividing by 1.0 there preserves that exactly.
    group_total = np.sum(m, axis=group_axis, keepdims=True)
    denom = np.where(group_total > 0.0, group_total, 1.0)
    mean = np.sum(r * m, axis=group_axis, keepdims=True) / denom
    var = np.sum(((r - mean) ** 2) * m, axis=group_axis, keepdims=True) / denom
    return ((r - mean) / np.sqrt(var + float(eps))) * m


def ppo_policy_loss(
    logp_new,
    logp_old,
    advantages,
    *,
    clip_epsilon: float = 0.2,
    mask=None,
    entropy=None,
    entropy_coef: float = 0.0,
    ref_logp=None,
    kl_coef: float = 0.0,
    reduction: str = "mean",
):
    ln = _asarray(logp_new).astype(np.float64, copy=False)
    lo = _asarray(logp_old).astype(np.float64, copy=False)
    adv = _asarray(advantages).astype(np.float64, copy=False)
    ratio = np.exp(ln - lo)
    clipped = np.clip(ratio, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon))
    surrogate = np.minimum(ratio * adv, clipped * adv)
    loss = -surrogate
    if ref_logp is not None and kl_coef:
        loss = loss + float(kl_coef) * _kl_penalty(ln, _asarray(ref_logp))
    if entropy is not None and entropy_coef:
        loss = loss - float(entropy_coef) * _asarray(entropy)
    return _reduce(loss, reduction, None if mask is None else _asarray(mask))


def grpo_policy_loss(
    logp_new,
    logp_old,
    rewards=None,
    advantages=None,
    *,
    group_axis: int = 1,
    clip_epsilon: float = 0.2,
    mask=None,
    ref_logp=None,
    kl_coef: float = 0.0,
    reduction: str = "mean",
):
    if advantages is None:
        if rewards is None:
            raise ValueError("grpo_policy_loss requires rewards or advantages")
        advantages = normalize_group_advantages(
            rewards, group_axis=group_axis, mask=mask,
        )
    return ppo_policy_loss(
        logp_new,
        logp_old,
        advantages,
        clip_epsilon=clip_epsilon,
        mask=mask,
        ref_logp=ref_logp,
        kl_coef=kl_coef,
        reduction=reduction,
    )


def cispo_policy_loss(
    logp_new,
    logp_old,
    rewards=None,
    advantages=None,
    *,
    group_axis: int = 1,
    epsilon_high: float = 5.0,
    mask=None,
    ref_logp=None,
    kl_coef: float = 0.0,
    reduction: str = "mean",
):
    if advantages is None:
        if rewards is None:
            raise ValueError("cispo_policy_loss requires rewards or advantages")
        advantages = normalize_group_advantages(
            rewards, group_axis=group_axis, mask=mask,
        )
    ln = _asarray(logp_new).astype(np.float64, copy=False)
    lo = _asarray(logp_old).astype(np.float64, copy=False)
    adv = _asarray(advantages).astype(np.float64, copy=False)
    # CISPO clips the importance-sampling weight directly and treats it as
    # detached from the log-prob gradient.
    clipped_weight = np.minimum(np.exp(ln - lo), float(epsilon_high))
    loss = -(clipped_weight * adv * ln)
    if ref_logp is not None and kl_coef:
        loss = loss + float(kl_coef) * _kl_penalty(ln, _asarray(ref_logp))
    return _reduce(loss, reduction, None if mask is None else _asarray(mask))


__all__ = [
    "RolloutBatch",
    "cispo_policy_loss",
    "grpo_policy_loss",
    "normalize_group_advantages",
    "ppo_policy_loss",
]
