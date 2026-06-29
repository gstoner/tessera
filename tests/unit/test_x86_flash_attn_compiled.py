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


def test_gqa_rejected_with_stable_diagnostic():
    rt = _rt_or_skip()
    q = _RNG.standard_normal((4, 8, 16)).astype(np.float32)   # 4 q-heads
    k = _RNG.standard_normal((2, 8, 16)).astype(np.float32)   # 2 kv-heads
    v = _RNG.standard_normal((2, 8, 16)).astype(np.float32)
    res = rt.launch(_art(rt, {}), (q, k, v))
    assert res["ok"] is False
    assert "GQA" in res.get("reason", "") or "MHA" in res.get("reason", "")
