"""sm_120 CUDA Flash-Attention forward contract: MHA/GQA/MQA and masks."""

from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import require_nvidia_mma_runtime


def _artifact(*, with_bias: bool, **kwargs):
    from tessera import runtime as rt
    operands = ["q", "k", "v"] + (["bias"] if with_bias else [])
    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_flash_attn_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": operands, "output_name": "o",
        "ops": [{"op_name": "tessera.flash_attn", "result": "o",
                 "operands": operands, "kwargs": kwargs}],
    })


def _reference(q, k, v, *, scale, causal=False, window=None, bias=None, softcap=None):
    B, Hq, Sq, D = q.shape
    _, Hkv, Sk, _ = k.shape
    out = np.zeros((B, Hq, Sq, v.shape[-1]), np.float32)
    ratio = Hq // Hkv
    if window is None:
        wl = wr = None
    elif isinstance(window, tuple):
        wl, wr = window
    else:
        wl = wr = window
    for b in range(B):
        for hq in range(Hq):
            hk = hq // ratio
            for m in range(Sq):
                s = q[b, hq, m] @ k[b, hk].T * scale
                keep = np.ones(Sk, dtype=bool)
                if causal:
                    keep &= np.arange(Sk) <= m
                if wl is not None:
                    keep &= np.arange(Sk) >= m - wl
                if wr is not None:
                    keep &= np.arange(Sk) <= m + wr
                if bias is not None:
                    s = s + bias[b, hq, m]
                if softcap is not None:
                    s = softcap * np.tanh(s / softcap)
                # The mask is dominant: soft-cap applies to legal logits only;
                # it must never turn a masked -inf score back into a finite one.
                s = np.where(keep, s, -np.inf)
                p = np.exp(s - np.max(s))
                p /= p.sum()
                out[b, hq, m] = p @ v[b, hk]
    return out


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("hkv", [4, 2, 1], ids=["mha", "gqa", "mqa"])
def test_live_nvidia_flash_attn_mha_gqa_mqa(hkv):
    rt = require_nvidia_mma_runtime()
    rng = np.random.default_rng(817 + hkv)
    q = rng.standard_normal((2, 4, 5, 8), dtype=np.float32)
    k = rng.standard_normal((2, hkv, 7, 8), dtype=np.float32)
    v = rng.standard_normal((2, hkv, 7, 6), dtype=np.float32)
    scale = 1.0 / np.sqrt(8.0)
    got = rt.launch(_artifact(with_bias=False, scale=scale), (q, k, v))
    assert got["ok"] is True, got.get("reason")
    np.testing.assert_allclose(got["output"], _reference(q, k, v, scale=scale),
                               atol=3e-5, rtol=0)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
def test_live_nvidia_flash_attn_causal_window_bias_softcap():
    rt = require_nvidia_mma_runtime()
    rng = np.random.default_rng(818)
    q = rng.standard_normal((1, 4, 6, 8), dtype=np.float32)
    k = rng.standard_normal((1, 2, 8, 8), dtype=np.float32)
    v = rng.standard_normal((1, 2, 8, 8), dtype=np.float32)
    bias = (rng.standard_normal((1, 4, 6, 8)) * 0.2).astype(np.float32)
    kw = dict(scale=0.35, causal=True, window=(3, 1), softcap=1.7)
    got = rt.launch(_artifact(with_bias=True, **kw), (q, k, v, bias))
    assert got["ok"] is True, got.get("reason")
    np.testing.assert_allclose(got["output"], _reference(q, k, v, bias=bias, **kw),
                               atol=4e-5, rtol=0)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
def test_live_nvidia_flash_attn_f16_storage_f32_accumulation():
    rt = require_nvidia_mma_runtime()
    rng = np.random.default_rng(821)
    q = rng.standard_normal((1, 4, 5, 8), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((1, 2, 7, 8), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((1, 2, 7, 6), dtype=np.float32).astype(np.float16)
    scale = 1.0 / np.sqrt(8.0)
    got = rt.launch(_artifact(with_bias=False, scale=scale), (q, k, v))
    assert got["ok"] is True, got.get("reason")
    assert got["output"].dtype == np.float16
    np.testing.assert_allclose(got["output"].astype(np.float32),
                               _reference(q.astype(np.float32), k.astype(np.float32),
                                          v.astype(np.float32), scale=scale),
                               atol=2e-3, rtol=0)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("warps_per_cta", [4, 8])
def test_live_nvidia_flash_attn_multiwarp_d128_mha_gqa_mqa(warps_per_cta):
    require_nvidia_mma_runtime()
    from tessera.compiler.emit.nvidia_cuda import (
        run_flash_attention_forward_schedule,
    )
    rng = np.random.default_rng(20260721 + warps_per_cta)
    for hkv in (4, 2, 1):
        q = rng.standard_normal((1, 4, 17, 128), dtype=np.float32) * 0.25
        k = rng.standard_normal((1, hkv, 19, 128), dtype=np.float32) * 0.25
        v = rng.standard_normal((1, hkv, 19, 128), dtype=np.float32) * 0.25
        scale = 1.0 / np.sqrt(128.0)
        got = run_flash_attention_forward_schedule(
            q, k, v, scale=scale, warps_per_cta=warps_per_cta)
        np.testing.assert_allclose(
            got, _reference(q, k, v, scale=scale), atol=5e-5, rtol=0)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("warps_per_cta", [4, 8])
def test_live_nvidia_flash_attn_multiwarp_ragged_mask_bias_softcap(
        warps_per_cta):
    require_nvidia_mma_runtime()
    from tessera.compiler.emit.nvidia_cuda import (
        run_flash_attention_forward_schedule,
    )
    rng = np.random.default_rng(20260730 + warps_per_cta)
    q = rng.standard_normal((1, 4, 17, 128), dtype=np.float32) * 0.2
    k = rng.standard_normal((1, 2, 37, 128), dtype=np.float32) * 0.2
    v = rng.standard_normal((1, 2, 37, 128), dtype=np.float32) * 0.2
    bias = rng.standard_normal((1, 4, 17, 37), dtype=np.float32) * 0.05
    kwargs = dict(scale=1.0 / np.sqrt(128.0), causal=True,
                  window_left=11, window_right=1, bias=bias, softcap=1.7)
    got = run_flash_attention_forward_schedule(
        q, k, v, warps_per_cta=warps_per_cta, **kwargs)
    ref = _reference(q, k, v, scale=kwargs["scale"], causal=True,
                     window=(11, 1), bias=bias, softcap=1.7)
    np.testing.assert_allclose(got, ref, atol=6e-5, rtol=0)
