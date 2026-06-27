"""x86 RL policy-loss lane — ppo / cispo / grpo core surrogate (AVX-512
policy-loss kernel) + normalize_group_advantages (layer_norm over the group
axis), loaded from libtessera_x86_elementwise.so. The CPU lane for these S11 RL
losses (previously reference-only on both devices).

Reachable through `runtime.launch()` via `compiler_path="x86_rl_loss_compiled"`.
f32; validated vs the tessera.rl reference (core path: no KL/entropy/mask).

Skip-clean: libtessera_x86_elementwise.so absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import rl


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _artifact(rt, op_name, operands, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_rl_loss_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": list(operands), "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": list(operands),
                 "kwargs": kwargs}],
    })


def _logps(rng, shape):
    lo = (rng.standard_normal(shape) * 0.5).astype(np.float32)
    ln = (lo + rng.standard_normal(shape) * 0.5).astype(np.float32)
    adv = (rng.standard_normal(shape) * 2).astype(np.float32)
    return ln, lo, adv


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("shape", [(64,), (8, 16)])
def test_ppo_matches_reference(reduction, shape):
    rt = _x86_or_skip()
    rng = np.random.default_rng(3 + len(shape) + int(np.prod(shape)))
    ln, lo, adv = _logps(rng, shape)
    res = rt.launch(_artifact(rt, "tessera.ppo_policy_loss",
                              ("ln", "lo", "adv"),
                              {"clip_epsilon": 0.2, "reduction": reduction}),
                    (ln, lo, adv))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_rl_loss_compiled"
    ref = np.asarray(rl.ppo_policy_loss(ln, lo, adv, clip_epsilon=0.2,
                                        reduction=reduction), dtype=np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32), ref,
                               atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_cispo_matches_reference(reduction):
    rt = _x86_or_skip()
    rng = np.random.default_rng(9)
    ln, lo, adv = _logps(rng, (8, 16))
    res = rt.launch(_artifact(rt, "tessera.cispo_policy_loss",
                              ("ln", "lo", "adv"),
                              {"epsilon_high": 5.0, "reduction": reduction}),
                    (ln, lo, adv))
    assert res["ok"] is True, res.get("reason")
    ref = np.asarray(rl.cispo_policy_loss(ln, lo, advantages=adv,
                                          epsilon_high=5.0, reduction=reduction),
                     dtype=np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32), ref,
                               atol=2e-5, rtol=2e-5)


def test_grpo_with_advantages_matches_reference():
    rt = _x86_or_skip()
    rng = np.random.default_rng(5)
    ln, lo, adv = _logps(rng, (4, 8))
    res = rt.launch(_artifact(rt, "tessera.grpo_policy_loss",
                              ("ln", "lo", "adv"),
                              {"clip_epsilon": 0.2, "reduction": "mean"}),
                    (ln, lo, adv))
    assert res["ok"] is True, res.get("reason")
    ref = np.asarray(rl.grpo_policy_loss(ln, lo, advantages=adv,
                                         clip_epsilon=0.2, reduction="mean"),
                     dtype=np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32), ref,
                               atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("group_axis", [1, -1, 0])
def test_normalize_group_advantages_matches_reference(group_axis):
    rt = _x86_or_skip()
    rng = np.random.default_rng(7 + group_axis)
    rewards = (rng.standard_normal((4, 6, 5)) * 3).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.normalize_group_advantages",
                              ("r",), {"group_axis": group_axis}), (rewards,))
    assert res["ok"] is True, res.get("reason")
    ref = np.asarray(rl.normalize_group_advantages(rewards,
                                                   group_axis=group_axis),
                     dtype=np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32), ref,
                               atol=2e-4, rtol=2e-4)


def test_rl_loss_kl_term_diagnoses():
    rt = _x86_or_skip()
    rng = np.random.default_rng(1)
    ln, lo, adv = _logps(rng, (8,))
    res = rt.launch(_artifact(rt, "tessera.ppo_policy_loss",
                              ("ln", "lo", "adv"), {"kl_coef": 0.1}),
                    (ln, lo, adv))
    assert res["ok"] is False
    assert "KL-penalty" in str(res.get("reason"))


def test_rl_loss_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((8,), np.float32)
    with pytest.raises(ValueError, match="x86_rl_loss_compiled executor"):
        rt._execute_x86_compiled_rl_loss(
            _artifact(rt, "tessera.mse_loss", ("a", "b", "c"), {}), (a, a, a))
