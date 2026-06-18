"""GRPO/CISPO post-training loop over tessera.rl.

This is the RL-first training loop: it computes the GRPO policy loss (DeepSeek-R1
style, group-normalized advantages) using the already-shipped, VJP/JVP-complete
``tessera.rl.grpo_policy_loss``, plus the MoE auxiliary losses from the model.

Integration seam (honest)
-------------------------
``nn.Parameter.grad`` is ``None`` until the Tier-2 reverse-mode tape lands in
the ``nn.Module`` path. The parameter-update half of the loop therefore goes
through ``tessera.autodiff`` (the ``id(buffer) -> Parameter`` weak-ref tape) or
``tessera.control.value_and_grad`` once the model forward is expressed through
the differentiable ``ops.*`` path end-to-end. Until then, ``grpo_step`` computes
and returns the *scalar losses* (fully runnable today) and exposes the
optimizer/update hook so the loop is complete the moment the tape is wired.
This mirrors PithTrain's principle of keeping the seam local and readable rather
than hidden behind indirection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from tessera import ops, rl
from tessera.rl import normalize_group_advantages

from .optimizer import adamw_step


def _np(x) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


@dataclass(frozen=True)
class GRPOConfig:
    clip_epsilon: float = 0.2
    kl_coef: float = 0.0
    group_axis: int = 1
    lb_loss_coef: float = 1e-2   # load-balancing aux weight
    z_loss_coef: float = 1e-3    # router z-loss weight
    lr: float = 0.05             # AdamW learning rate for grpo_train_step


@dataclass(frozen=True)
class RolloutTokenMetadata:
    """Per-token metadata needed to diagnose MTP/DSA RL rollouts."""

    token_id: int
    behavior_logprob: float
    policy_version: int
    token_entropy: float
    draft_entropy: float = 0.0
    target_entropy: float = 0.0
    per_step_acceptance: tuple[float, ...] = ()
    accepted_length: int = 0
    tv_overlap: float = 0.0
    sampler_temperature: float = 1.0
    sampler_top_p: float = 0.0
    index_share_group_id: int | None = None
    deterministic_topk_hash: str = ""
    kv_bytes: int = 0
    cache_hits: int = 0
    cache_reuse_affinity_id: str = ""


@dataclass(frozen=True)
class RolloutDiagnostics:
    """Serializable rollout-level telemetry independent of the GRPO objective."""

    rollout_id: str
    tokens: tuple[RolloutTokenMetadata, ...] = field(default_factory=tuple)

    def summary(self) -> dict[str, float | int | str]:
        n = len(self.tokens)
        if n == 0:
            return {
                "rollout_id": self.rollout_id,
                "num_tokens": 0,
                "mean_entropy": 0.0,
                "mean_target_entropy": 0.0,
                "mean_draft_entropy": 0.0,
                "mean_accepted_length": 0.0,
                "mean_tv_overlap": 0.0,
                "kv_bytes": 0,
                "cache_hits": 0,
            }
        return {
            "rollout_id": self.rollout_id,
            "num_tokens": n,
            "mean_entropy": float(np.mean([t.token_entropy for t in self.tokens])),
            "mean_target_entropy": float(np.mean([t.target_entropy for t in self.tokens])),
            "mean_draft_entropy": float(np.mean([t.draft_entropy for t in self.tokens])),
            "mean_accepted_length": float(np.mean([t.accepted_length for t in self.tokens])),
            "mean_tv_overlap": float(np.mean([t.tv_overlap for t in self.tokens])),
            "kv_bytes": int(sum(t.kv_bytes for t in self.tokens)),
            "cache_hits": int(sum(t.cache_hits for t in self.tokens)),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def grpo_step(
    logp_new,
    logp_old,
    rewards,
    *,
    aux_losses: dict[str, float] | None = None,
    config: GRPOConfig | None = None,
    ref_logp: Any | None = None,
    rollout_diagnostics: RolloutDiagnostics | None = None,
) -> dict[str, Any]:
    """One GRPO objective evaluation, including MoE auxiliary terms.

    Returns a dict of scalar losses: ``policy``, ``load_balancing``,
    ``router_z``, and the combined ``total``. The update step plugs into
    ``tessera.optim`` once gradients flow (see module docstring).
    """
    cfg = config or GRPOConfig()
    policy = float(rl.grpo_policy_loss(
        logp_new, logp_old, rewards=rewards,
        group_axis=cfg.group_axis, clip_epsilon=cfg.clip_epsilon,
        ref_logp=ref_logp, kl_coef=cfg.kl_coef, reduction="mean",
    ))
    aux = aux_losses or {}
    lb = float(aux.get("load_balancing_loss", 0.0))
    z = float(aux.get("router_z_loss", 0.0))
    total = policy + cfg.lb_loss_coef * lb + cfg.z_loss_coef * z
    metrics: dict[str, Any] = {
        "policy": policy,
        "load_balancing": lb,
        "router_z": z,
        "total": total,
    }
    if rollout_diagnostics is not None:
        metrics["rollout"] = rollout_diagnostics.summary()
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable GRPO training step (Tier-2 tape + real AdamW update)
# ─────────────────────────────────────────────────────────────────────────────

def grpo_surrogate(logp_new, logp_old, advantages, *, clip_epsilon: float = 0.2):
    """Traced PPO/GRPO clipped surrogate ``-mean_i min(r_i A_i, clip(r_i) A_i)``.

    ``logp_new`` is a tape-traced tensor (the differentiable output of the
    current policy); ``logp_old`` and ``advantages`` are numpy constants
    (group-normalized advantages and the behavior-policy log-probs). Returns a
    traced scalar so ``adamw_step`` can backprop through it.

    Uses ``ops.clip`` for the clip (its bounds ride in kwargs, so no scalar
    operand detaches the tape) and ``ops.reduce(op="mean")`` for the average.
    The sign is applied via a per-sample ``-1`` array because ``ops.mul`` of a
    *scalar* tensor by a python float does not carry its factor into the VJP
    (array operands are correct) — see the gap notes in
    ``docs/audit/compiler/COMPILER_AUDIT.md``.
    """
    a = np.asarray(advantages, dtype=np.float32).reshape(-1)
    n = a.shape[0]
    old = np.asarray(_np(logp_old), dtype=np.float32).reshape(-1)
    neg = np.full(n, -1.0, np.float32)

    ratio = ops.exp(ops.sub(logp_new, old))                          # (N,)
    clipped = ops.clip(ratio, min=1.0 - clip_epsilon, max=1.0 + clip_epsilon)
    obj = ops.minimum(ops.mul(ratio, a), ops.mul(clipped, a))
    return ops.reduce(ops.mul(obj, neg), op="mean")                  # scalar


def grpo_train_step(policy, batch, opt_state=None, *, config: GRPOConfig | None = None):
    """One on-policy GRPO update of ``policy`` via real gradients + AdamW.

    Args:
        policy: a tape-differentiable module exposing
                ``logp(states, actions_onehot) -> traced (N,)``
                (e.g. ``tessera.train.models.TracedMoEPolicy``).
        batch:  dict with ``states`` and one-hot ``actions``; provide either
                ``advantages`` directly or ``rewards`` (group-normalized here).
                ``logp_old`` is optional (defaults to the current policy →
                on-policy, ratio==1 at step 0).
        opt_state: AdamW state from a prior step.
    Returns:
        ``(metrics, opt_state)`` where ``metrics["surrogate"]`` is the loss.
    """
    cfg = config or GRPOConfig()
    states = batch["states"]
    actions = batch["actions"]

    advantages = batch.get("advantages")
    if advantages is None:
        if "rewards" not in batch:
            raise ValueError("grpo_train_step requires batch['advantages'] or batch['rewards']")
        advantages = normalize_group_advantages(batch["rewards"], group_axis=cfg.group_axis)
    advantages = np.asarray(advantages, dtype=np.float32).reshape(-1)

    logp_old = batch.get("logp_old")
    if logp_old is None:
        logp_old = _np(policy.logp(states, actions))   # on-policy snapshot
    logp_old = np.asarray(logp_old, dtype=np.float32).reshape(-1)

    def loss_fn():
        logp_new = policy.logp(states, actions)
        return grpo_surrogate(logp_new, logp_old, advantages,
                              clip_epsilon=cfg.clip_epsilon)

    surrogate, opt_state = adamw_step(policy, loss_fn, opt_state, lr=cfg.lr)
    return {"surrogate": surrogate}, opt_state
