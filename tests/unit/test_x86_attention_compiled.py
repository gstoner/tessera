"""x86 softmax-attention lane — multi_head / gqa / mqa / mla_decode, composed
from the AVX-512 f32 GEMM (QKᵀ, probs·V) + the AVX-512 row-softmax kernel. The
CPU analog of the ROCm WMMA flash-attention family.

Reachable through `runtime.launch()` via `compiler_path="x86_attention_compiled"`.
f32; validated vs an independent numpy SDPA reference at rtol 1e-3 (f32 GEMM +
polynomial softmax accumulation).

Skip-clean: libtessera_x86_elementwise.so absent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _artifact(rt, op_name, operands, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_attention_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": list(operands), "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": list(operands),
                 "kwargs": kwargs}],
    })


_TOL = dict(atol=1e-3, rtol=1e-3)


def _sdpa_ref(Q, K, V, nq, nkv, scale=None, causal=False):
    # Q [.., nq, Sq, D]; K/V [.., nkv, Sk, D]
    D = Q.shape[-1]
    Sq, Sk = Q.shape[-2], K.shape[-2]
    sc = scale if scale is not None else 1.0 / math.sqrt(D)
    g = nq // nkv
    Kx = np.repeat(K, g, axis=-3) if g > 1 else K
    Vx = np.repeat(V, g, axis=-3) if g > 1 else V
    scores = (Q @ np.swapaxes(Kx, -1, -2)) * sc
    if causal:
        i = np.arange(Sq).reshape(Sq, 1)
        j = np.arange(Sk).reshape(1, Sk)
        scores = np.where(j > i + (Sk - Sq), -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    probs = e / e.sum(axis=-1, keepdims=True)
    return probs @ Vx


@pytest.mark.parametrize("causal", [False, True])
def test_multi_head_attention(causal):
    rt = _x86_or_skip()
    rng = np.random.default_rng(1 + int(causal))
    B, S, H, D = 2, 6, 4, 8
    q = rng.standard_normal((B, S, H * D)).astype(np.float32)
    k = rng.standard_normal((B, S, H * D)).astype(np.float32)
    v = rng.standard_normal((B, S, H * D)).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.multi_head_attention", ("q", "k", "v"),
                              {"num_heads": H, "causal": causal}), (q, k, v))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_attention_compiled"

    def split(t):
        return np.ascontiguousarray(t.reshape(B, S, H, D).transpose(0, 2, 1, 3))
    ref = _sdpa_ref(split(q), split(k), split(v), H, H, causal=causal)
    ref = ref.transpose(0, 2, 1, 3).reshape(B, S, H * D)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               ref.astype(np.float32), **_TOL)


@pytest.mark.parametrize("causal", [False, True])
def test_gqa_attention(causal):
    rt = _x86_or_skip()
    rng = np.random.default_rng(7 + int(causal))
    B, nq, nkv, S, D = 2, 8, 2, 5, 16
    q = rng.standard_normal((B, nq, S, D)).astype(np.float32)
    k = rng.standard_normal((B, nkv, S, D)).astype(np.float32)
    v = rng.standard_normal((B, nkv, S, D)).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.gqa_attention", ("q", "k", "v"),
                              {"num_query_heads": nq, "num_kv_heads": nkv,
                               "causal": causal}), (q, k, v))
    assert res["ok"] is True, res.get("reason")
    ref = _sdpa_ref(q, k, v, nq, nkv, causal=causal)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               ref.astype(np.float32), **_TOL)


def test_mqa_attention():
    rt = _x86_or_skip()
    rng = np.random.default_rng(3)
    B, nq, S, D = 2, 6, 7, 8
    q = rng.standard_normal((B, nq, S, D)).astype(np.float32)
    k = rng.standard_normal((B, 1, S, D)).astype(np.float32)
    v = rng.standard_normal((B, 1, S, D)).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.mqa_attention", ("q", "k", "v"),
                              {"causal": True}), (q, k, v))
    assert res["ok"] is True, res.get("reason")
    ref = _sdpa_ref(q, k, v, nq, 1, causal=True)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               ref.astype(np.float32), **_TOL)


def test_mla_decode_with_projection():
    rt = _x86_or_skip()
    rng = np.random.default_rng(5)
    B, H, Sq, Sk, Dk, D = 2, 4, 1, 6, 12, 8
    q = rng.standard_normal((B, H, Sq, D)).astype(np.float32)
    k = rng.standard_normal((B, H, Sk, Dk)).astype(np.float32)
    v = rng.standard_normal((B, H, Sk, Dk)).astype(np.float32)
    w_k = rng.standard_normal((Dk, D)).astype(np.float32)
    w_v = rng.standard_normal((Dk, D)).astype(np.float32)
    res = rt.launch(
        _artifact(rt, "tessera.mla_decode", ("q", "k", "v", "wk", "wv"), {}),
        (q, k, v, w_k, w_v))
    assert res["ok"] is True, res.get("reason")
    ref = _sdpa_ref(q, k @ w_k, v @ w_v, H, H)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               ref.astype(np.float32), **_TOL)


def test_attention_explicit_scale():
    rt = _x86_or_skip()
    rng = np.random.default_rng(11)
    B, nq, nkv, S, D = 1, 4, 4, 5, 8
    q = rng.standard_normal((B, nq, S, D)).astype(np.float32)
    k = rng.standard_normal((B, nkv, S, D)).astype(np.float32)
    v = rng.standard_normal((B, nkv, S, D)).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.gqa_attention", ("q", "k", "v"),
                              {"num_query_heads": nq, "num_kv_heads": nkv,
                               "scale": 0.1}), (q, k, v))
    assert res["ok"] is True, res.get("reason")
    ref = _sdpa_ref(q, k, v, nq, nkv, scale=0.1)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               ref.astype(np.float32), **_TOL)


def test_attention_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((1, 2, 3, 4), np.float32)
    with pytest.raises(ValueError, match="x86_attention_compiled executor"):
        rt._execute_x86_compiled_attention(
            _artifact(rt, "tessera.softmax", ("q", "k", "v"), {}), (a, a, a))


def test_attention_rejects_non_f32():
    rt = _x86_or_skip()
    a = np.zeros((1, 2, 3, 4), np.float64)
    res = rt.launch(_artifact(rt, "tessera.mqa_attention", ("q", "k", "v"), {}),
                    (a, a, a))
    assert res["ok"] is False
    assert "f32 only" in str(res.get("reason"))
