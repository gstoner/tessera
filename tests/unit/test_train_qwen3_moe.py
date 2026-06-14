"""Tests for the agent-native MoE training scaffold (tessera.train).

Covers the Phase-1 runnable surface: model forward, MoE engine auxiliary
losses, the GRPO loop objective, and the agent-native firewall (the train
package must not import the compiler audit/registry machinery).
"""

from __future__ import annotations

import numpy as np
import pytest


def test_qwen3_moe_forward_shapes_and_finite_aux():
    from tessera.train import Qwen3MoEConfig, Qwen3MoEModel

    cfg = Qwen3MoEConfig()
    model = Qwen3MoEModel(cfg)
    ids = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(2, 8))
    logits, aux = model.forward(ids)

    assert np.asarray(logits).shape == (2, 8, cfg.vocab_size)
    assert np.all(np.isfinite(np.asarray(logits)))
    assert set(aux) == {"load_balancing_loss", "router_z_loss"}
    assert all(np.isfinite(float(v)) for v in aux.values())


def test_model_parameters_are_discoverable():
    from tessera.train import Qwen3MoEConfig, Qwen3MoEModel

    model = Qwen3MoEModel(Qwen3MoEConfig())
    params = list(model.parameters())
    # embedding + per-layer (attn 4 + 2 norms + router + 6 expert weights) + final norm + lm_head
    assert len(params) > 10


def test_load_balancing_loss_uniform_is_minimal():
    from tessera.train import load_balancing_loss
    from tessera.train.engine.moe import router_z_loss

    rng = np.random.default_rng(1)
    E, T, k = 8, 256, 2
    # Uniform routing → loss near the ideal value E * sum(1/E * 1/E) = 1.0.
    logits = np.zeros((T, E))
    expert_ids = rng.integers(0, E, size=(T, k))
    lb = load_balancing_loss(logits, expert_ids, E)
    assert 0.8 < lb < 1.3
    assert router_z_loss(logits) >= 0.0


def test_moe_feedforward_matches_naive_reference():
    """The engine block forward must equal the standalone naive reference."""
    from tessera.train.engine.moe import MoEFeedForward, _arr
    from tessera.models.moe_routing import moe_forward_naive

    ff = MoEFeedForward(hidden_size=32, num_experts=4, top_k=2,
                        expert_intermediate=64, shared_intermediate=32, seed=3)
    x = np.random.default_rng(4).standard_normal((10, 32)).astype(np.float32)
    y, aux = ff.forward(x)

    ref = moe_forward_naive(
        x, _arr(ff.router.gate.weight),
        _arr(ff.w_gate), _arr(ff.w_up), _arr(ff.w_down),
        _arr(ff.w_sgate), _arr(ff.w_sup), _arr(ff.w_sdown),
        top_k=2, normalize=True,
    )
    np.testing.assert_allclose(np.asarray(y), ref, rtol=1e-5, atol=1e-5)
    assert np.isfinite(aux["load_balancing_loss"])


def test_grpo_step_combines_policy_and_aux():
    from tessera.train.loop.rl import grpo_step, GRPOConfig

    rng = np.random.default_rng(5)
    logp_new = rng.standard_normal((2, 4))
    logp_old = logp_new + 0.01 * rng.standard_normal((2, 4))
    rewards = rng.standard_normal((2, 4))
    aux = {"load_balancing_loss": 1.5, "router_z_loss": 3.0}
    out = grpo_step(logp_new, logp_old, rewards, aux_losses=aux, config=GRPOConfig())

    assert set(out) == {"policy", "load_balancing", "router_z", "total"}
    assert all(np.isfinite(v) for v in out.values())
    # total = policy + lb_coef*lb + z_coef*z
    expected = out["policy"] + 1e-2 * 1.5 + 1e-3 * 3.0
    assert out["total"] == pytest.approx(expected, rel=1e-6)


def test_agent_native_firewall():
    """tessera.train must not pull in the compiler audit/registry machinery.

    These modules (primitive_coverage / op_catalog / backend_manifest) are the
    implicit indirection PithTrain measures as costly for agents; they belong
    behind @jit, not in the training read-path. See tessera/train/__init__.py.
    """
    import sys

    forbidden = {
        "tessera.compiler.primitive_coverage",
        "tessera.compiler.op_catalog",
        "tessera.compiler.backend_manifest",
    }
    # Drop any that a *prior* test already imported, then import train fresh.
    for name in list(sys.modules):
        if name in forbidden:
            del sys.modules[name]
    for name in [n for n in list(sys.modules) if n.startswith("tessera.train")]:
        del sys.modules[name]

    import tessera.train  # noqa: F401

    leaked = forbidden & set(sys.modules)
    assert not leaked, f"tessera.train leaked compiler registry imports: {leaked}"
