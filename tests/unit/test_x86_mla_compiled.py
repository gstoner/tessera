"""MLA latent-KV lane on x86 AVX-512 (P11 of S_SERIES_GAP_CLOSURE_PLAN) — the
DeepSeek Multi-head Latent Attention building blocks composed on the AVX-512
GEMM (latent_kv_compress/expand_k/expand_v = batched matmul) + the P10
flash_attn lane (mla_decode_fused chains compress→expand→flash_attn). No new
kernel. Reachable via `compiler_path="x86_mla_compiled"`. Validated vs the numpy
MLA reference. Skip-clean: libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op, n_operands, kwargs=None):
    names = [f"a{i}" for i in range(n_operands)]
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_mla_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": kwargs or {}}]})


def _run(rt, op, *arrs, **kwargs):
    res = rt.launch(_art(rt, op, len(arrs), kwargs), arrs)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_mla_compiled"
    return np.asarray(res["output"])


_RNG = np.random.default_rng(13)
HID, LAT, HD = 32, 16, 24       # hidden, latent, head dim


def test_latent_kv_compress():
    rt = _rt_or_skip()
    x = _RNG.standard_normal((2, 6, HID)).astype(np.float32)
    w = _RNG.standard_normal((HID, LAT)).astype(np.float32)
    np.testing.assert_allclose(
        _run(rt, "tessera.latent_kv_compress", x, w),
        tessera.ops.latent_kv_compress(x, w), rtol=1e-4, atol=1e-4)


def test_latent_kv_expand_k_and_v():
    rt = _rt_or_skip()
    c = _RNG.standard_normal((2, 6, LAT)).astype(np.float32)
    w_uk = _RNG.standard_normal((LAT, HD)).astype(np.float32)
    w_uv = _RNG.standard_normal((LAT, HD)).astype(np.float32)
    np.testing.assert_allclose(
        _run(rt, "tessera.latent_kv_expand_k", c, w_uk),
        tessera.ops.latent_kv_expand_k(c, w_uk), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        _run(rt, "tessera.latent_kv_expand_v", c, w_uv),
        tessera.ops.latent_kv_expand_v(c, w_uv), rtol=1e-4, atol=1e-4)


def test_mla_decode_fused():
    rt = _rt_or_skip()
    B, S = 2, 7
    x = _RNG.standard_normal((B, S, HID)).astype(np.float32)
    w_dkv = _RNG.standard_normal((HID, LAT)).astype(np.float32)
    w_uk = _RNG.standard_normal((LAT, HD)).astype(np.float32)
    w_uv = _RNG.standard_normal((LAT, HD)).astype(np.float32)
    q = _RNG.standard_normal((B, S, HD)).astype(np.float32)
    got = _run(rt, "tessera.mla_decode_fused", x, w_dkv, w_uk, w_uv, q)
    ref = tessera.ops.mla_decode_fused(x, w_dkv, w_uk, w_uv, q)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-4, atol=1e-4)


def test_mla_decode_fused_distinct_value_width():
    """MLA config with v_head_dim != qk_head_dim — V (W_uv) expands to a
    different width than K (W_uk); the output takes V's width."""
    rt = _rt_or_skip()
    B, S = 2, 6
    qk_dim, v_dim = HD, HD + 8        # distinct value width
    x = _RNG.standard_normal((B, S, HID)).astype(np.float32)
    w_dkv = _RNG.standard_normal((HID, LAT)).astype(np.float32)
    w_uk = _RNG.standard_normal((LAT, qk_dim)).astype(np.float32)
    w_uv = _RNG.standard_normal((LAT, v_dim)).astype(np.float32)
    q = _RNG.standard_normal((B, S, qk_dim)).astype(np.float32)
    got = _run(rt, "tessera.mla_decode_fused", x, w_dkv, w_uk, w_uv, q)
    ref = tessera.ops.mla_decode_fused(x, w_dkv, w_uk, w_uv, q)
    assert got.shape == (B, S, v_dim)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-4, atol=1e-4)


def test_mla_decode_fused_causal():
    rt = _rt_or_skip()
    B, S = 2, 7
    x = _RNG.standard_normal((B, S, HID)).astype(np.float32)
    w_dkv = _RNG.standard_normal((HID, LAT)).astype(np.float32)
    w_uk = _RNG.standard_normal((LAT, HD)).astype(np.float32)
    w_uv = _RNG.standard_normal((LAT, HD)).astype(np.float32)
    q = _RNG.standard_normal((B, S, HD)).astype(np.float32)
    got = _run(rt, "tessera.mla_decode_fused", x, w_dkv, w_uk, w_uv, q,
               causal=True)
    ref = tessera.ops.mla_decode_fused(x, w_dkv, w_uk, w_uv, q, causal=True)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-4, atol=1e-4)
