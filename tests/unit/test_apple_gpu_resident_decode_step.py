"""Apple GPU R2 (ops) — rowop + gumbel encode variants; a full decode step in
one command buffer.

With the bmm encode (PR #35) plus the rowop (norms/softmax) and gumbel encode
variants here, an entire per-token decode step — norm → projections/attention →
norm → logits → sample — encodes into a single command buffer and commits once.
Validated stage-by-stage against numpy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tessera import runtime as R
from tessera.runtime import DeviceTensor, AppleGPUEncodeSession
from tessera import rng as TR


def _require():
    s = AppleGPUEncodeSession()
    if not s.available:
        s.commit()
        pytest.skip("encode-session ABI unavailable")
    return s


def _softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def _rmsnorm(x, gamma, eps=1e-6):
    d = x.astype(np.float64)
    norm = d / np.sqrt((d * d).mean(-1, keepdims=True) + eps)
    return (norm * gamma.astype(np.float64)) if gamma is not None else norm


def _layernorm(x, eps=1e-6):
    d = x.astype(np.float64)
    m = d.mean(-1, keepdims=True)
    return (d - m) / np.sqrt(((d - m) ** 2).mean(-1, keepdims=True) + eps)


def test_rmsnorm_encoded():
    s = _require()
    rng = np.random.RandomState(0)
    x = rng.randn(4, 16).astype(np.float32)
    gamma = rng.randn(16).astype(np.float32)
    dx, dg = DeviceTensor.from_numpy(x), DeviceTensor.from_numpy(gamma)
    with s:
        o = s.rmsnorm(dx, dg)
        assert o is not None
    np.testing.assert_allclose(o.numpy(), _rmsnorm(x, gamma), rtol=1e-4, atol=1e-4)


def test_softmax_encoded():
    s = _require()
    rng = np.random.RandomState(1)
    x = rng.randn(3, 20).astype(np.float32)
    dx = DeviceTensor.from_numpy(x)
    with s:
        o = s.softmax(dx)
    np.testing.assert_allclose(o.numpy(), _softmax(x), rtol=1e-4, atol=1e-4)


def test_layernorm_and_logsoftmax_encoded():
    s = _require()
    rng = np.random.RandomState(2)
    x = rng.randn(2, 12).astype(np.float32)
    dx = DeviceTensor.from_numpy(x)
    with s:
        ln = s.rowop(dx, 0)        # unweighted layer_norm
        ls = s.rowop(dx, 3)        # log_softmax
    np.testing.assert_allclose(ln.numpy(), _layernorm(x), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ls.numpy(), np.log(_softmax(x)), rtol=1e-4, atol=1e-4)


def test_gumbel_encoded_greedy():
    s = _require()
    rng = np.random.RandomState(3)
    logits = rng.randn(5, 40).astype(np.float32)
    zero = np.zeros((5, 40), np.float32)
    dl, dz = DeviceTensor.from_numpy(logits), DeviceTensor.from_numpy(zero)
    with s:
        ids = s.gumbel(dl, dz, inv_temp=1.0)   # zero noise => greedy argmax
        assert ids is not None
    np.testing.assert_array_equal(ids.numpy(), np.argmax(logits, axis=-1).astype(np.int32))


def test_full_decode_step_one_command_buffer():
    """A per-token decode step — norm → attention (Q@Kᵀ → softmax → @V) → logits
    → sample — encoded into ONE command buffer, committed once. Shape changes
    between ops use zero-copy `reshape_view` (same buffer), so NO host read
    happens mid-chain; only the final token ids are read after commit. Validated
    end-to-end against numpy."""
    s = _require()
    rng = np.random.RandomState(8)
    H, D, Skv, V = 2, 8, 6, 50
    x = rng.randn(H, D).astype(np.float32)
    g1 = rng.randn(D).astype(np.float32)
    Kt = rng.randn(H, D, Skv).astype(np.float32)     # per-head Kᵀ
    Vv = rng.randn(H, Skv, D).astype(np.float32)
    Wlogit = rng.randn(1, D, V).astype(np.float32)   # logit proj (broadcast)
    noise = R._gumbel_noise_from_key((H, V), TR.RNGKey.from_seed(1), np)

    dx = DeviceTensor.from_numpy(x)
    dg1 = DeviceTensor.from_numpy(g1)
    dKt = DeviceTensor.from_numpy(Kt)
    dV = DeviceTensor.from_numpy(Vv)
    dW = DeviceTensor.from_numpy(Wlogit)
    dnoise = DeviceTensor.from_numpy(noise)

    with s:
        qn = s.rmsnorm(dx, dg1)                        # [H, D]
        scores = s.bmm(qn.reshape_view(H, 1, D), dKt)  # [H, 1, Skv]
        attn = s.softmax(scores.reshape_view(H, Skv))  # [H, Skv]
        ctx = s.bmm(attn.reshape_view(H, 1, Skv), dV)  # [H, 1, D]
        logits = s.bmm(ctx.reshape_view(H, 1, D), dW)  # [H, 1, V]
        ids = s.gumbel(logits.reshape_view(H, V), dnoise, inv_temp=1.0)  # [H]
        assert ids is not None
    # only now (post-commit) is any result read back to host

    qn_ref = _rmsnorm(x, g1)
    sc_ref = np.einsum("hd,hdk->hk", qn_ref, Kt.astype(np.float64))
    attn_ref = _softmax(sc_ref)
    ctx_ref = np.einsum("hk,hkd->hd", attn_ref, Vv.astype(np.float64))
    logit_ref = np.einsum("hd,dv->hv", ctx_ref, Wlogit[0].astype(np.float64))
    ids_ref = np.argmax(logit_ref + noise.astype(np.float64), axis=-1)
    np.testing.assert_array_equal(ids.numpy(), ids_ref.astype(np.int32))


def test_symbols_exported():
    rt = R._load_apple_gpu_runtime()
    for name in ("tessera_apple_gpu_rowop_dev_f32",
                 "tessera_apple_gpu_rowop_dev_f32_enc",
                 "tessera_apple_gpu_gumbel_argmax_dev_f32",
                 "tessera_apple_gpu_gumbel_argmax_dev_f32_enc"):
        assert hasattr(rt, name), name
