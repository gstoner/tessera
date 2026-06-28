"""Compiler-generated RL policy losses on gfx1151 — the ROCm mirror of the
x86_rl_loss lane. ppo/cispo/grpo surrogate on generate-rocm-policy-loss-kernel;
normalize_group_advantages on the rocm norm lane.

Reachable via `compiler_path="rocm_rl_loss_compiled"`. Validated vs tessera.rl on
gfx1151. Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import rl


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name, operands, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_rl_loss_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": list(operands), "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": list(operands),
                 "kwargs": kwargs}],
    })


def _logps(rng, shape):
    lo = (rng.standard_normal(shape) * 0.5).astype(np.float32)
    ln = (lo + rng.standard_normal(shape) * 0.5).astype(np.float32)
    adv = (rng.standard_normal(shape) * 2).astype(np.float32)
    return ln, lo, adv


@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_ppo_matches_reference(reduction):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(3)
    ln, lo, adv = _logps(rng, (8, 16))
    res = rt.launch(_artifact(rt, "tessera.ppo_policy_loss", ("ln", "lo", "adv"),
                              {"clip_epsilon": 0.2, "reduction": reduction}),
                    (ln, lo, adv))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_rl_loss_compiled"
    ref = np.asarray(rl.ppo_policy_loss(ln, lo, adv, clip_epsilon=0.2,
                                        reduction=reduction), dtype=np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32), ref,
                               atol=2e-5, rtol=2e-5)


def test_cispo_matches_reference():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(9)
    ln, lo, adv = _logps(rng, (8, 16))
    res = rt.launch(_artifact(rt, "tessera.cispo_policy_loss",
                              ("ln", "lo", "adv"),
                              {"epsilon_high": 5.0, "reduction": "mean"}),
                    (ln, lo, adv))
    assert res["ok"] is True, res.get("reason")
    ref = np.float32(rl.cispo_policy_loss(ln, lo, advantages=adv,
                                          epsilon_high=5.0))
    np.testing.assert_allclose(np.float32(res["output"]), ref, atol=2e-5,
                               rtol=2e-5)


@pytest.mark.parametrize("group_axis", [1, -1])
def test_normalize_group_advantages_matches_reference(group_axis):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(7 + group_axis)
    rewards = (rng.standard_normal((4, 6, 5)) * 3).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.normalize_group_advantages", ("r",),
                              {"group_axis": group_axis}), (rewards,))
    assert res["ok"] is True, res.get("reason")
    ref = np.asarray(rl.normalize_group_advantages(rewards,
                                                   group_axis=group_axis),
                     dtype=np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32), ref,
                               atol=2e-4, rtol=2e-4)


def test_rl_loss_kl_term_diagnoses():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(1)
    ln, lo, adv = _logps(rng, (8,))
    res = rt.launch(_artifact(rt, "tessera.ppo_policy_loss", ("ln", "lo", "adv"),
                              {"kl_coef": 0.1}), (ln, lo, adv))
    assert res["ok"] is False
    assert "KL-penalty" in str(res.get("reason"))
