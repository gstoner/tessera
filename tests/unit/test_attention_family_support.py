"""Modern attention-family coverage: Lightning, Delta, DeepSeek sparse, hybrids."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import tessera as ts
from tessera.autodiff.jvp import get_jvp
from tessera.autodiff.vjp import get_vjp
from tessera.compiler.op_catalog import get_op_spec
from tessera.compiler.primitive_coverage import coverage_for


ROOT = Path(__file__).resolve().parents[2]


def _tiny_inputs(dtype=np.float64):
    rng = np.random.default_rng(7)
    q = rng.normal(size=(1, 1, 4, 3)).astype(dtype)
    k = rng.normal(size=(1, 1, 4, 3)).astype(dtype)
    v = rng.normal(size=(1, 1, 4, 2)).astype(dtype)
    return q, k, v


def _numeric_grad(fn, x, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        plus = x.copy()
        minus = x.copy()
        plus[idx] += eps
        minus[idx] -= eps
        grad[idx] = (fn(plus) - fn(minus)) / (2.0 * eps)
        it.iternext()
    return grad


def _numeric_jvp(fn, x, dx, eps=1e-6):
    return (fn(x + eps * dx) - fn(x - eps * dx)) / (2.0 * eps)


def test_requested_attention_ops_are_public_and_autodiff_complete():
    names = [
        "multi_head_attention",
        "gqa_attention",
        "mla_decode",
        "attn_sliding_window",
        "deepseek_sparse_attention",
        "gated_attention",
        "hybrid_attention",
        "lightning_attention",
        "gated_deltanet",
        "kimi_delta_attention",
        "modified_delta_attention",
        "power_attn",
        "retention",
    ]
    for name in names:
        assert hasattr(ts.ops, name), name
        assert get_op_spec(name) is not None, name
        assert get_vjp(name) is not None, name
        assert get_jvp(name) is not None, name
        entry = coverage_for(name)
        assert entry.contract_status["vjp"] == "complete", name
        assert entry.contract_status["jvp"] == "complete", name


def test_gated_attention_multiplies_flash_attention_by_sigmoid_gate():
    q, k, v = _tiny_inputs()
    gate = np.zeros((1, 1, 4, 2), dtype=np.float64)
    out = ts.ops.gated_attention(q, k, v, gate, causal=True)
    ref = 0.5 * ts.ops.flash_attn(q, k, v, causal=True)
    np.testing.assert_allclose(out, ref)


def test_deepseek_sparse_attention_matches_equal_weight_branch_composition():
    q, k, v = _tiny_inputs()
    k_c, v_c = ts.ops.compress_blocks(k, v, block_size=2)
    scores = np.matmul(q, np.swapaxes(k_c, -1, -2))
    ref = (
        ts.ops.attn_sliding_window(q, k, v, window_size=2, causal=True)
        + ts.ops.attn_compressed_blocks(q, k_c, v_c)
        + ts.ops.attn_top_k_blocks(q, k, v, scores=scores, top_k=1, block_size=2, causal=True)
    ) / 3.0
    out = ts.ops.deepseek_sparse_attention(q, k, v, window_size=2, block_size=2, top_k=1)
    np.testing.assert_allclose(out, ref)


def test_lightning_attention_matches_identity_linear_attention_and_chunks_state():
    q, k, v = _tiny_inputs()
    out = ts.ops.lightning_attention(q, k, v, causal=True)
    ref = ts.ops.linear_attn(q, k, v, feature_map="identity", causal=True)
    np.testing.assert_allclose(out, ref)

    full = ts.ops.lightning_attention(q, k, v, causal=True)
    first, state = ts.ops.lightning_attention(q[:, :, :2], k[:, :, :2], v[:, :, :2], return_state=True)
    second = ts.ops.lightning_attention(q[:, :, 2:], k[:, :, 2:], v[:, :, 2:], state=state)
    np.testing.assert_allclose(np.concatenate([first, second], axis=2), full)
    assert state.dtype == np.float32


def test_delta_attention_family_shapes_gates_and_dtype_policy():
    q, k, v = _tiny_inputs(np.float16)
    gate = np.zeros((1, 1, 4, 2), dtype=np.float16)
    beta = np.full((1, 1, 4), 0.5, dtype=np.float16)
    out, state = ts.ops.gated_deltanet(q, k, v, gate, beta, return_state=True)
    assert out.shape == (1, 1, 4, 2)
    assert out.dtype == np.float16
    assert state.dtype == np.float32
    assert ts.ops.kimi_delta_attention(q, k, v, gate, beta).shape == out.shape
    assert ts.ops.modified_delta_attention(q, k, v, gate, beta).shape == out.shape


def test_hybrid_attention_ling_and_kimi_policy_schedule():
    q, k, v = _tiny_inputs()
    ling0 = ts.ops.hybrid_attention(q, k, v, pattern="ling_1_7_mla_lightning", layer_index=0)
    ling7 = ts.ops.hybrid_attention(q, k, v, pattern="ling_1_7_mla_lightning", layer_index=7)
    np.testing.assert_allclose(ling0, ts.ops.lightning_attention(q, k, v))
    np.testing.assert_allclose(ling7, ts.ops.flash_attn(q, k, v, causal=True))

    kimi0 = ts.ops.hybrid_attention(q, k, v, pattern="kimi_kda_mla", layer_index=0)
    kimi1 = ts.ops.hybrid_attention(q, k, v, pattern="kimi_kda_mla", layer_index=1)
    np.testing.assert_allclose(kimi0, ts.ops.kimi_delta_attention(q, k, v))
    np.testing.assert_allclose(kimi1, ts.ops.flash_attn(q, k, v, causal=True))


def test_attention_family_vjp_and_jvp_match_finite_difference():
    q, k, v = _tiny_inputs()
    dout = np.ones((1, 1, 4, 2), dtype=np.float64)
    d_q, _, _ = get_vjp("lightning_attention")(dout, q, k, v)
    expected = _numeric_grad(lambda qq: float(ts.ops.lightning_attention(qq, k, v).sum()), q)
    np.testing.assert_allclose(d_q, expected, atol=1e-3)

    dq = np.ones_like(q) * 0.1
    _, tangent = get_jvp("gated_deltanet")((q, k, v), (dq, np.zeros_like(k), np.zeros_like(v)))
    expected_tan = _numeric_jvp(lambda qq: ts.ops.gated_deltanet(qq, k, v), q, dq)
    np.testing.assert_allclose(tangent, expected_tan, atol=1e-3)


def test_attention_family_passes_are_declared_and_pipeline_wired():
    passes_h = (ROOT / "src/transforms/include/Tessera/Transforms/Passes.h").read_text()
    passes_cpp = (ROOT / "src/transforms/lib/Passes.cpp").read_text()
    pass_impl = (ROOT / "src/transforms/lib/AttentionFamilyPasses.cpp").read_text()
    for name in [
        "tessera-lightning-attn-fusion",
        "tessera-delta-attn-chunking",
        "tessera-hybrid-attn-expand",
    ]:
        assert name in pass_impl
    for factory in [
        "createLightningAttnFusionPass",
        "createDeltaAttnChunkingPass",
        "createHybridAttnExpandPass",
    ]:
        assert factory in passes_h
        assert factory in passes_cpp
