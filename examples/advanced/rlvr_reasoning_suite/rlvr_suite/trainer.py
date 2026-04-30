from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from .rollout import Rollout


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 8
    clip_low: float = 0.2
    clip_high: float = 0.28
    resample_on_correct: bool = True


def _advantages(rewards: list[float]) -> list[float]:
    mu = mean(rewards)
    sigma = pstdev(rewards) or 1.0
    return [(r - mu) / sigma for r in rewards]


def _roc_filter(group: list[Rollout]) -> list[Rollout]:
    positives = [r for r in group if r.reward > 0.0]
    negatives = [r for r in group if r.reward <= 0.0]
    if not positives or not negatives:
        return group
    keep_pos = positives[: max(1, len(group) // 2)]
    keep_neg = negatives[: max(1, len(group) - len(keep_pos))]
    return keep_pos + keep_neg


def grpo_accounting_step(group: list[Rollout], cfg: GRPOConfig) -> dict[str, object]:
    kept = _roc_filter(group) if cfg.resample_on_correct else group
    rewards = [r.reward for r in kept]
    advantages = _advantages(rewards)
    best = max(group, key=lambda r: r.reward)
    return {
        "reward_mean": mean(rewards),
        "reward_max": max(rewards),
        "advantage_min": min(advantages),
        "advantage_max": max(advantages),
        "kept_rollouts": len(kept),
        "total_rollouts": len(group),
        "best_answer": best.completion,
    }
