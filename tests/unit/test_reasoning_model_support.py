from __future__ import annotations

from pathlib import Path

import numpy as np

import tessera as ts
from tessera.autodiff.jvp import get_jvp
from tessera.autodiff.vjp import get_vjp
from tessera.compiler.op_catalog import OP_SPECS
from tessera.compiler.primitive_coverage import coverage_for


ROOT = Path(__file__).resolve().parents[2]


def _numeric_grad(fn, x, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        plus = x.copy()
        minus = x.copy()
        plus[idx] += eps
        minus[idx] -= eps
        grad[idx] = (np.asarray(fn(plus)).sum() - np.asarray(fn(minus)).sum()) / (2 * eps)
        it.iternext()
    return grad


def _numeric_jvp(fn, x, dx, eps=1e-6):
    return (np.asarray(fn(x + eps * dx)) - np.asarray(fn(x - eps * dx))) / (2 * eps)


def test_tuple_output_tape_backpropagates_from_rope_split_component():
    x = ts.nn.Parameter(np.arange(6.0, dtype=np.float64).reshape(1, 6))
    with ts.autodiff.tape() as tape:
        _rope, no_rope = ts.ops.rope_split(x, rope_dim=2)
        loss = ts.ops.reduce(ts.ops.mul(no_rope, no_rope), op="sum")
        tape.backward(loss)
    np.testing.assert_array_equal(x.grad.numpy(), [[0.0, 0.0, 4.0, 6.0, 8.0, 10.0]])


def test_rope_vjp_and_jvp_match_finite_difference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 4)).astype(np.float64)
    theta = rng.normal(size=(2, 2)).astype(np.float64)
    dx = rng.normal(size=x.shape).astype(np.float64) * 0.1
    dout = np.ones_like(x)

    grad_x, _grad_theta = get_vjp("rope")(dout, x, theta)
    expected = _numeric_grad(lambda v: ts.ops.rope(v, theta), x)
    np.testing.assert_allclose(grad_x, expected, atol=1e-5, rtol=1e-5)

    primal, tangent = get_jvp("rope")((x, theta), (dx, np.zeros_like(theta)))
    np.testing.assert_allclose(primal, ts.ops.rope(x, theta))
    np.testing.assert_allclose(tangent, _numeric_jvp(lambda v: ts.ops.rope(v, theta), x, dx), atol=1e-5)


def test_mla_projection_vjps_are_matmul_shaped():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 3)).astype(np.float64)
    w = rng.normal(size=(3, 4)).astype(np.float64)
    dout = rng.normal(size=(2, 4)).astype(np.float64)
    dx, dw = get_vjp("latent_kv_compress")(dout, x, w)
    np.testing.assert_allclose(dx, dout @ w.T)
    np.testing.assert_allclose(dw, x.T @ dout)


def test_quantization_ste_and_scale_component_is_nondifferentiable():
    x = ts.nn.Parameter(np.array([1.0, -2.0, 3.0], dtype=np.float64))
    with ts.autodiff.tape() as tape:
        x_q, _scale = ts.ops.quantize_fp8(x, format="e4m3")
        loss = ts.ops.reduce(x_q, op="sum")
        tape.backward(loss)
    np.testing.assert_array_equal(x.grad.numpy(), np.ones(3))

    grad, = get_vjp("quantize_fp8")(2.0, np.ones(3), format="e4m3", _output_index=1)
    np.testing.assert_array_equal(grad, np.zeros(3))


def test_sparse_attention_and_moe_transport_rules_are_registered():
    for name in [
        "attn_sliding_window",
        "attn_compressed_blocks",
        "attn_top_k_blocks",
        "moe_dispatch",
        "moe_combine",
    ]:
        assert get_vjp(name) is not None
        assert get_jvp(name) is not None
        assert coverage_for(name).contract_status["vjp"] == "complete"
        assert coverage_for(name).contract_status["jvp"] == "complete"


def test_wrapper_ops_are_catalogued_and_public():
    for name in ["multi_head_attention", "gqa_attention", "mqa_attention", "mla_decode", "alibi", "ntk_rope"]:
        assert name in OP_SPECS
        assert hasattr(ts.ops, name)
        assert coverage_for(name).contract_status["vjp"] == "complete"
        assert coverage_for(name).contract_status["jvp"] == "complete"


def test_kv_cache_mutations_are_not_differentiable_contracts():
    for name in ["kv_cache_append", "kv_cache_prune"]:
        entry = coverage_for(name)
        assert entry.contract_status["vjp"] == "not_applicable"
        assert entry.contract_status["jvp"] == "not_applicable"
        assert entry.contract_status["transpose_rule"] == "not_applicable"


def test_normal_lowering_pipelines_run_attention_fusions_before_backend_lowering():
    text = (ROOT / "src/transforms/lib/Passes.cpp").read_text()
    x86 = text[text.index('lowerToX86("tessera-lower-to-x86"'):text.index("// ── Phase 3 passes")]
    gpu = text[text.index('lowerToGPU("tessera-lower-to-gpu"'):]
    for block in [x86, gpu]:
        assert block.index("createSwigluFusionPass") < block.index("createDistributionLoweringPass")
        assert block.index("createMLAFusionPass") < block.index("createDistributionLoweringPass")
        assert block.index("createNativeSparseAttnFusionPass") < block.index("createDistributionLoweringPass")


def test_rl_api_losses_and_jvp_vjp_registry():
    rewards = np.array([[1.0, 2.0, 3.0], [3.0, 3.0, 5.0]])
    adv = ts.rl.normalize_group_advantages(rewards, group_axis=1)
    np.testing.assert_allclose(adv.mean(axis=1), np.zeros(2), atol=1e-7)

    logp_new = np.array([[0.0, 0.2, 3.0]], dtype=np.float64)
    logp_old = np.zeros_like(logp_new)
    advantages = np.ones_like(logp_new)
    ppo = ts.rl.ppo_policy_loss(logp_new, logp_old, advantages, clip_epsilon=0.2, reduction="none")
    np.testing.assert_allclose(ppo[0, 2], -1.2)

    cispo = ts.rl.cispo_policy_loss(
        logp_new,
        logp_old,
        advantages=advantages,
        epsilon_high=2.0,
        reduction="none",
    )
    np.testing.assert_allclose(cispo[0, 2], -6.0)

    for name in ["normalize_group_advantages", "ppo_policy_loss", "grpo_policy_loss", "cispo_policy_loss"]:
        assert get_vjp(name) is not None
        assert get_jvp(name) is not None
        assert coverage_for(name).contract_status["vjp"] == "complete"
        assert coverage_for(name).contract_status["jvp"] == "complete"


def test_graph_ir_recognizes_promoted_wrapper_names():
    @ts.jit
    def wrappers(q, k, v, x, theta):
        a = ts.ops.gqa_attention(q, k, v, num_query_heads=2, num_kv_heads=1)
        b = ts.ops.ntk_rope(x, theta, scale=2.0)
        return ts.ops.add(a, b)

    text = wrappers.ir_text()
    assert "tessera.gqa_attention" in text
    assert "tessera.ntk_rope" in text
