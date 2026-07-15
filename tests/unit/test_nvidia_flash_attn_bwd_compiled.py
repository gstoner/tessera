"""sm_120 generated CUDA Flash-Attention backward contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from _nvidia_testutil import require_nvidia_mma_runtime


def _artifact(bias: bool, **kwargs):
    from tessera import runtime as rt
    names = ["do", "q", "k", "v"] + (["bias"] if bias else [])
    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_flash_attn_bwd_compiled",
        "executable": True, "execution_kind": "native_gpu", "arg_names": names,
        "output_name": "grads", "ops": [{"op_name": "tessera.flash_attn",
        "result": "grads", "operands": names, "kwargs": kwargs}],
    })


def _ref(do, q, k, v, *, scale, causal=False, window=None, bias=None, softcap=None):
    B, Hq, Sq, D = q.shape; Hkv, Sk, Dv = k.shape[1], k.shape[2], v.shape[-1]
    dq, dk, dv = np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)
    ratio = Hq // Hkv
    for b in range(B):
        for hq in range(Hq):
            hk = hq // ratio
            for m in range(Sq):
                raw = q[b,hq,m] @ k[b,hk].T * scale
                legal = np.ones(Sk, bool)
                if causal: legal &= np.arange(Sk) <= m
                if window is not None:
                    wl, wr = window if isinstance(window, tuple) else (window, window)
                    legal &= (np.arange(Sk) >= m-wl) & (np.arange(Sk) <= m+wr)
                s = raw + (0 if bias is None else bias[b,hq,m])
                deriv = np.ones(Sk, np.float32)
                if softcap is not None:
                    t = np.tanh(s / softcap); s = softcap * t; deriv = 1 - t*t
                s = np.where(legal, s, -np.inf); p = np.exp(s - s.max()); p /= p.sum()
                o = p @ v[b,hk]; delta = do[b,hq,m] @ o
                for n in np.flatnonzero(legal):
                    ds = p[n] * (do[b,hq,m] @ v[b,hk,n] - delta) * deriv[n]
                    dv[b,hk,n] += p[n] * do[b,hq,m]
                    dq[b,hq,m] += ds * scale * k[b,hk,n]
                    dk[b,hk,n] += ds * scale * q[b,hq,m]
    return dq, dk, dv


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("hkv", [2, 1], ids=["gqa", "mqa"])
def test_live_nvidia_flash_attn_bwd_gqa_mqa(hkv):
    rt = require_nvidia_mma_runtime(); rng = np.random.default_rng(911 + hkv)
    q = rng.standard_normal((1, 4, 4, 4), dtype=np.float32)
    k = rng.standard_normal((1, hkv, 5, 4), dtype=np.float32)
    v = rng.standard_normal((1, hkv, 5, 3), dtype=np.float32)
    do = rng.standard_normal((1, 4, 4, 3), dtype=np.float32); scale = .5
    got = rt.launch(_artifact(False, scale=scale), (do, q, k, v))
    assert got["ok"], got.get("reason")
    for actual, expected in zip(got["output"], _ref(do, q, k, v, scale=scale)):
        np.testing.assert_allclose(actual, expected, atol=7e-5, rtol=0)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
def test_live_nvidia_flash_attn_bwd_mask_bias_softcap():
    rt = require_nvidia_mma_runtime(); rng = np.random.default_rng(919)
    q = rng.standard_normal((1, 2, 5, 4), dtype=np.float32)
    k = rng.standard_normal((1, 1, 6, 4), dtype=np.float32)
    v = rng.standard_normal((1, 1, 6, 3), dtype=np.float32)
    do = rng.standard_normal((1, 2, 5, 3), dtype=np.float32)
    bias = (rng.standard_normal((1, 2, 5, 6)) * .1).astype(np.float32)
    kw = dict(scale=.5, causal=True, window=(2, 1), softcap=1.4)
    got = rt.launch(_artifact(True, **kw), (do, q, k, v, bias))
    assert got["ok"], got.get("reason")
    for actual, expected in zip(got["output"], _ref(do, q, k, v, bias=bias, **kw)):
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
def test_live_nvidia_flash_attn_bwd_f16_storage_f32_accumulation():
    rt = require_nvidia_mma_runtime(); rng = np.random.default_rng(923)
    q = rng.standard_normal((1, 2, 4, 4), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((1, 1, 5, 4), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((1, 1, 5, 3), dtype=np.float32).astype(np.float16)
    do = rng.standard_normal((1, 2, 4, 3), dtype=np.float32).astype(np.float16)
    got = rt.launch(_artifact(False, scale=.5), (do, q, k, v))
    assert got["ok"], got.get("reason")
    ref = _ref(do.astype(np.float32), q.astype(np.float32), k.astype(np.float32),
               v.astype(np.float32), scale=.5)
    for actual, expected in zip(got["output"], ref):
        assert actual.dtype == np.float16
        np.testing.assert_allclose(actual.astype(np.float32), expected, atol=3e-3, rtol=0)
