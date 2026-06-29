"""flash_attn lane on x86 AVX-512 (P10 of S_SERIES_GAP_CLOSURE_PLAN) — the
AVX-512 partner to the ROCm WMMA flash_attn. FA-style online-softmax forward
(tessera_x86_flash_attn_f32); the S×S score matrix is never materialized.
Reachable via `compiler_path="x86_flash_attn_compiled"`. Validated vs the dense
attention reference. Skip-clean: libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, kwargs, op="tessera.flash_attn"):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_flash_attn_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["q", "k", "v"], "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": ["q", "k", "v"],
                 "kwargs": kwargs}]})


def _run(rt, q, k, v, op="tessera.flash_attn", **kwargs):
    res = rt.launch(_art(rt, kwargs, op), (q, k, v))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_flash_attn_compiled"
    return np.asarray(res["output"])


def _ref(q, k, v, scale=None, causal=False):
    """Dense attention reference (matches tessera.flash_attn)."""
    d = q.shape[-1]
    s = scale if scale is not None else 1.0 / np.sqrt(d)
    scores = (q @ np.swapaxes(k, -1, -2)) * s
    if causal:
        sq, sk = scores.shape[-2], scores.shape[-1]
        mask = np.triu(np.ones((sq, sk), bool), k=1 + max(sk - sq, 0))
        scores = np.where(mask, -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return (w @ v).astype(np.float32)


_RNG = np.random.default_rng(11)


def _qkv(shape):
    return (_RNG.standard_normal(shape).astype(np.float32),
            _RNG.standard_normal(shape).astype(np.float32),
            _RNG.standard_normal(shape).astype(np.float32))


def test_attention_2d():
    rt = _rt_or_skip()
    q, k, v = _qkv((6, 16))
    np.testing.assert_allclose(_run(rt, q, k, v), _ref(q, k, v),
                               rtol=1e-4, atol=1e-4)


def test_attention_batched_heads():
    rt = _rt_or_skip()
    q, k, v = _qkv((2, 4, 8, 16))      # [B, H, S, D]
    np.testing.assert_allclose(_run(rt, q, k, v), _ref(q, k, v),
                               rtol=1e-4, atol=1e-4)


def test_causal():
    rt = _rt_or_skip()
    q, k, v = _qkv((3, 12, 32))
    np.testing.assert_allclose(_run(rt, q, k, v, causal=True),
                               _ref(q, k, v, causal=True), rtol=1e-4, atol=1e-4)


def test_custom_scale():
    rt = _rt_or_skip()
    q, k, v = _qkv((5, 24))
    np.testing.assert_allclose(_run(rt, q, k, v, scale=0.25),
                               _ref(q, k, v, scale=0.25), rtol=1e-4, atol=1e-4)


def test_cross_attention_unequal_seq():
    rt = _rt_or_skip()
    # Sq != Sk (encoder-decoder cross attention), non-causal.
    q = _RNG.standard_normal((2, 7, 16)).astype(np.float32)
    k = _RNG.standard_normal((2, 11, 16)).astype(np.float32)
    v = _RNG.standard_normal((2, 11, 16)).astype(np.float32)
    np.testing.assert_allclose(_run(rt, q, k, v), _ref(q, k, v),
                               rtol=1e-4, atol=1e-4)


def test_distinct_value_width():
    """V's value width may differ from the QK head dim — the output takes V's
    width (the dense reference allows V's last dim to be independent)."""
    rt = _rt_or_skip()
    q = _RNG.standard_normal((2, 5, 16)).astype(np.float32)
    k = _RNG.standard_normal((2, 7, 16)).astype(np.float32)
    v = _RNG.standard_normal((2, 7, 24)).astype(np.float32)   # value width 24
    got = _run(rt, q, k, v)
    assert got.shape == (2, 5, 24)
    np.testing.assert_allclose(got, _ref(q, k, v), rtol=1e-4, atol=1e-4)


def _ref_extra(q, k, v, scale=None, causal=False, window=0, softcap=0.0,
               bias=None):
    """Dense attention reference with the P10 extras (GQA pre-expanded)."""
    d = q.shape[-1]
    sc = scale if scale is not None else 1.0 / np.sqrt(d)
    s = (q @ np.swapaxes(k, -1, -2)) * sc
    if softcap > 0:
        s = softcap * np.tanh(s / softcap)
    if bias is not None:
        s = s + bias
    sq, sk = s.shape[-2], s.shape[-1]
    off = max(sk - sq, 0)
    i = np.arange(sq)[:, None]
    j = np.arange(sk)[None, :]
    mask = np.zeros((sq, sk), bool)
    if window > 0:
        if causal:                         # causal band (i-W, i]
            mask |= (j > i + off) | (j < i + off - window + 1)
        else:                              # symmetric local window |i-j| <= W/2
            mask |= (j > i + off + window // 2) | (j < i + off - window // 2)
    elif causal:
        mask |= j > i + off
    s = np.where(mask, -np.inf, s)
    s = s - s.max(axis=-1, keepdims=True)
    w = np.exp(s)
    w = w / w.sum(axis=-1, keepdims=True)
    return (w @ v).astype(np.float32)


def test_gqa_expands_kv_heads():
    """GQA: Hq query heads share Hkv<Hq kv heads — the x86 lane expands the kv
    heads on the host and runs MHA."""
    rt = _rt_or_skip()
    q = _RNG.standard_normal((1, 4, 8, 16)).astype(np.float32)   # 4 q-heads
    k = _RNG.standard_normal((1, 2, 8, 16)).astype(np.float32)   # 2 kv-heads
    v = _RNG.standard_normal((1, 2, 8, 16)).astype(np.float32)
    got = _run(rt, q, k, v, op="tessera.gqa_attention")
    k_e, v_e = np.repeat(k, 2, axis=1), np.repeat(v, 2, axis=1)
    np.testing.assert_allclose(got, _ref(q, k_e, v_e), rtol=1e-4, atol=1e-4)


def test_sliding_window():
    rt = _rt_or_skip()
    q, k, v = _qkv((3, 12, 24))
    got = _run(rt, q, k, v, op="tessera.attn_sliding_window",
               window=4, causal=True)
    np.testing.assert_allclose(
        got, _ref_extra(q, k, v, window=4, causal=True), rtol=1e-4, atol=1e-4)


def test_sliding_window_non_causal_symmetric():
    """Non-causal sliding window is a SYMMETRIC local band (|i-j| <= W/2) — it
    must NOT attend future keys outside the window."""
    rt = _rt_or_skip()
    q, k, v = _qkv((2, 16, 16))
    got = _run(rt, q, k, v, op="tessera.attn_sliding_window",
               window=6, causal=False)
    np.testing.assert_allclose(
        got, _ref_extra(q, k, v, window=6, causal=False), rtol=1e-4, atol=1e-4)


def test_attn_bias_rank4():
    """A rank-4 [B, H, Sq, Sk] additive bias flattens consistently with the
    [B, H, S, D] Q/K/V."""
    rt = _rt_or_skip()
    q = _RNG.standard_normal((2, 3, 5, 8)).astype(np.float32)
    k = _RNG.standard_normal((2, 3, 5, 8)).astype(np.float32)
    v = _RNG.standard_normal((2, 3, 5, 8)).astype(np.float32)
    bias = _RNG.standard_normal((2, 3, 5, 5)).astype(np.float32)
    art = rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_flash_attn_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["q", "k", "v", "b"], "output_name": "o",
        "ops": [{"op_name": "tessera.flash_attn", "result": "o",
                 "operands": ["q", "k", "v", "b"], "kwargs": {}}]})
    res = rt.launch(art, (q, k, v, bias))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]),
                               _ref_extra(q, k, v, bias=bias),
                               rtol=1e-4, atol=1e-4)


def test_logit_softcap():
    rt = _rt_or_skip()
    q, k, v = _qkv((2, 10, 16))
    got = _run(rt, q, k, v, logit_softcap=30.0, causal=True)
    np.testing.assert_allclose(
        got, _ref_extra(q, k, v, softcap=30.0, causal=True), rtol=1e-4, atol=1e-4)


def test_attn_bias_operand_applied():
    """A 4th operand is the additive attn_bias — the x86 lane applies it."""
    rt = _rt_or_skip()
    q, k, v = _qkv((4, 16))
    bias = _RNG.standard_normal((4, 4)).astype(np.float32)
    art = rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_flash_attn_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["q", "k", "v", "bias"], "output_name": "o",
        "ops": [{"op_name": "tessera.flash_attn", "result": "o",
                 "operands": ["q", "k", "v", "bias"], "kwargs": {}}]})
    res = rt.launch(art, (q, k, v, bias))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]),
                               _ref_extra(q, k, v, bias=bias),
                               rtol=1e-4, atol=1e-4)


def test_multi_head_attention_op_rejected():
    """tessera.multi_head_attention ([B,S,H*D] + num_heads) is a different
    contract — the x86 lane only accepts tessera.flash_attn ([..., S, D])."""
    rt = _rt_or_skip()
    q, k, v = _qkv((2, 8, 32))
    art = rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_flash_attn_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["q", "k", "v"], "output_name": "o",
        "ops": [{"op_name": "tessera.multi_head_attention", "result": "o",
                 "operands": ["q", "k", "v"], "kwargs": {"num_heads": 4}}]})
    res = rt.launch(art, (q, k, v))
    assert res["ok"] is False
    assert "tessera.flash_attn" in res.get("reason", "")
