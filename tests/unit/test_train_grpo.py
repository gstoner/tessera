"""Tier-2 tape + AdamW training tests for tessera.train.

Proves the RL training path is real: gradients come from the reverse-mode tape,
the update comes from tessera.optim.adamw, and the GRPO surrogate strictly
improves a tape-differentiable MoE policy. The surrogate math is cross-checked
against the shipped tessera.rl.grpo_policy_loss as an oracle.
"""

from __future__ import annotations

import numpy as np

import tessera as ts
from tessera import nn, ops, rl
from tessera.train import (
    GRPOConfig,
    RolloutDiagnostics,
    RolloutTokenMetadata,
    TracedMoEPolicy,
    adamw_step,
    grpo_step,
    grpo_surrogate,
    grpo_train_step,
)


# ─────────────────────────────────────────────────────────────────────────────
# adamw_step — the generic Tier-2 wiring
# ─────────────────────────────────────────────────────────────────────────────

class _TinyQuadratic(nn.Module):
    def __init__(self, d: int, seed: int = 0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.W = nn.Parameter((rng.standard_normal((d, d)) * 0.5).astype(np.float32))


def test_adamw_step_minimizes_traced_loss():
    d = 4
    m = _TinyQuadratic(d)
    X = np.random.default_rng(1).standard_normal((6, d)).astype(np.float32)

    def loss_fn():
        z = ops.gemm(X, m.W)               # traced
        return ops.reduce(ops.mul(z, z), op="sum")  # ||XW||^2 -> drives W toward 0

    opt = None
    losses = []
    for _ in range(20):
        loss, opt = adamw_step(m, loss_fn, opt, lr=0.1)
        losses.append(loss)

    assert losses[-1] < losses[0]
    assert losses[-1] < 0.5 * losses[0]   # clear progress
    assert all(np.isfinite(v) for v in losses)


# ─────────────────────────────────────────────────────────────────────────────
# TracedMoEPolicy — gradients flow to all router + expert params
# ─────────────────────────────────────────────────────────────────────────────

def test_traced_policy_grads_flow_to_all_params():
    policy = TracedMoEPolicy(state_dim=6, hidden=8, num_experts=3,
                             expert_ffn=16, num_actions=4, seed=2)
    rng = np.random.default_rng(3)
    states = rng.standard_normal((10, 6)).astype(np.float32)
    actions = np.eye(4, dtype=np.float32)[rng.integers(0, 4, size=10)]

    for p in policy.parameters():
        p._grad = None
    with ts.autodiff.tape() as t:
        logp = policy.logp(states, actions)
        t.backward(ops.reduce(logp, op="sum"))

    params = list(policy.named_parameters())
    assert len(params) >= 7  # w_in, w_router, w_out + per-expert gate/down
    assert all(p.grad is not None for _, p in params), \
        [n for n, p in params if p.grad is None]


def test_traced_policy_logits_shape():
    policy = TracedMoEPolicy(6, 8, 3, 16, 4, seed=0)
    logits = policy.logits(np.random.default_rng(0).standard_normal((5, 6)).astype(np.float32))
    assert np.asarray(logits).shape == (5, 4)


# ─────────────────────────────────────────────────────────────────────────────
# grpo_surrogate — oracle agreement with the shipped loss
# ─────────────────────────────────────────────────────────────────────────────

def test_grpo_surrogate_matches_rl_loss_at_ratio_one():
    # At ratio==1 (logp_new == logp_old) the clipped surrogate is -mean(adv);
    # tessera.rl.grpo_policy_loss must agree. Use non-zero-mean raw advantages.
    rng = np.random.default_rng(4)
    N = 8
    adv = rng.standard_normal(N).astype(np.float32) + 0.5  # non-zero mean
    logp_old = rng.standard_normal(N).astype(np.float32)

    # Build a traced logp_new identical to logp_old via a no-op traced path.
    W = nn.Parameter(np.zeros((1, 1), np.float32))
    with ts.autodiff.tape():
        # logp_new = logp_old + (X @ W) with W=0 -> equals logp_old, but traced.
        bump = ops.reduce(ops.gemm(np.zeros((N, 1), np.float32), W), op="sum", axis=1)
        logp_new = ops.add(logp_old, bump)
        surr = grpo_surrogate(logp_new, logp_old, adv, clip_epsilon=0.2)
    surr_v = float(surr.numpy() if hasattr(surr, "numpy") else surr)

    rl_v = float(rl.grpo_policy_loss(logp_old, logp_old, advantages=adv))
    assert surr_v == _approx(-float(adv.mean()))
    assert surr_v == _approx(rl_v)


def test_grpo_step_telemetry_is_optional_and_serializable():
    logp = np.log(np.array([0.4, 0.6], dtype=np.float32))
    rewards = np.array([1.0, -1.0], dtype=np.float32)
    plain = grpo_step(logp, logp, rewards, config=GRPOConfig(group_axis=0))
    diag = RolloutDiagnostics(
        rollout_id="rollout-a",
        tokens=(
            RolloutTokenMetadata(
                token_id=7,
                behavior_logprob=-0.2,
                policy_version=3,
                token_entropy=1.1,
                draft_entropy=1.0,
                target_entropy=1.2,
                per_step_acceptance=(1.0, 0.5),
                accepted_length=2,
                tv_overlap=0.9,
                sampler_temperature=0.8,
                sampler_top_p=0.95,
                index_share_group_id=2,
                deterministic_topk_hash="abc",
                kv_bytes=128,
                cache_hits=1,
                cache_reuse_affinity_id="agent-1",
            ),
        ),
    )
    with_diag = grpo_step(
        logp, logp, rewards, config=GRPOConfig(group_axis=0),
        rollout_diagnostics=diag,
    )
    for key in ("policy", "load_balancing", "router_z", "total"):
        assert with_diag[key] == plain[key]
    assert with_diag["rollout"]["rollout_id"] == "rollout-a"
    assert with_diag["rollout"]["num_tokens"] == 1
    assert with_diag["rollout"]["mean_accepted_length"] == 2.0
    assert diag.to_dict()["tokens"][0]["deterministic_topk_hash"] == "abc"


def _approx(x, tol=1e-5):
    import pytest
    return pytest.approx(x, abs=tol)


# ─────────────────────────────────────────────────────────────────────────────
# grpo_train_step — real end-to-end PPO-style improvement
# ─────────────────────────────────────────────────────────────────────────────

def test_grpo_train_step_improves_and_updates_params():
    rng = np.random.default_rng(0)
    N, d, A = 24, 6, 4
    policy = TracedMoEPolicy(d, 8, 3, 16, A, seed=1)
    states = rng.standard_normal((N, d)).astype(np.float32)
    actions = np.eye(A, dtype=np.float32)[rng.integers(0, A, size=N)]
    rewards = rng.standard_normal((N,)).astype(np.float32)

    before = {n: p.numpy().copy() for n, p in policy.named_parameters()}

    # PPO-style: fix the behavior-policy log-probs, take K gradient steps.
    logp_old = np.asarray(policy.logp(states, actions))
    logp_old = logp_old.numpy() if hasattr(logp_old, "numpy") else logp_old
    batch = {"states": states, "actions": actions, "rewards": rewards, "logp_old": logp_old}

    opt = None
    losses = []
    for _ in range(15):
        metrics, opt = grpo_train_step(policy, batch, opt, config=GRPOConfig(lr=0.05, group_axis=0))
        losses.append(metrics["surrogate"])

    assert all(np.isfinite(v) for v in losses)
    assert losses[-1] < losses[0] - 1e-3       # surrogate improved
    assert min(losses) < -0.02                  # reached a clearly-negative objective

    after = {n: p.numpy() for n, p in policy.named_parameters()}
    changed = sum(not np.allclose(before[n], after[n]) for n in before)
    assert changed == len(before)               # every parameter moved


def test_grpo_train_step_on_policy_defaults_to_ratio_one():
    # Without logp_old, the step is on-policy: surrogate == -mean(advantages)
    # (~0 for group-normalized) and it still updates params (grad != 0 at ratio 1).
    rng = np.random.default_rng(7)
    N, d, A = 16, 5, 3
    policy = TracedMoEPolicy(d, 6, 2, 12, A, seed=5)
    batch = {
        "states": rng.standard_normal((N, d)).astype(np.float32),
        "actions": np.eye(A, dtype=np.float32)[rng.integers(0, A, size=N)],
        "rewards": rng.standard_normal((N,)).astype(np.float32),
    }
    before = {n: p.numpy().copy() for n, p in policy.named_parameters()}
    metrics, _ = grpo_train_step(policy, batch, None, config=GRPOConfig(group_axis=0))
    assert np.isfinite(metrics["surrogate"])
    after = {n: p.numpy() for n, p in policy.named_parameters()}
    assert any(not np.allclose(before[n], after[n]) for n in before)
