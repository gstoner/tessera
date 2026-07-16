"""Phase-G attention sprint, Sub-sprint B — linear / recurrent attention on GPU.

linear_attn (+_state), lightning_attention, power_attn, retention all share the
quadratic-parallel form O = (φ(Q)φ(K)ᵀ ⊙ causal[⊙decay]) @ V — two GPU batched
matmuls + an elementwise mask multiply, validated bit-equivalent to the
sequential recurrence reference.  The causal/decay mask is a structured
host-side constant; the QKᵀ / PV matmuls run on the GPU.  metal_runtime.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import runtime as _runtime
from tessera.compiler import driver as _driver

R = _runtime
_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.hardware_apple_gpu

_B, _H, _S, _D = 2, 3, 16, 16


def _qkv(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((_B, _H, _S, _D)).astype(np.float32),
            rng.standard_normal((_B, _H, _S, _D)).astype(np.float32),
            rng.standard_normal((_B, _H, _S, _D)).astype(np.float32))


def _decay(seed=9):
    return np.random.default_rng(seed).uniform(0.85, 0.99, (_B, _H, _S)).astype(np.float32)


def test_in_envelope():
    for op in ("tessera.linear_attn", "tessera.linear_attn_state",
               "tessera.lightning_attention", "tessera.power_attn",
               "tessera.retention"):
        assert op in _driver._APPLE_GPU_RUNTIME_OPS, op
        assert op in _runtime._APPLE_GPU_RUNTIME_OPS, op
        assert op in _driver._APPLE_GPU_LINEAR_ATTN_OPS, op


def test_driver_runtime_linear_attn_envelopes_agree():
    assert _driver._APPLE_GPU_LINEAR_ATTN_OPS == _runtime._APPLE_GPU_LINEAR_ATTN_OPS


@gpu
@pytest.mark.parametrize("fmap", ["identity", "elu", "relu", "polynomial_2"])
@pytest.mark.parametrize("causal", [True, False])
def test_linear_attn(fmap, causal):
    Q, K, V = _qkv(1)
    out = R._apple_gpu_dispatch_linear_attn(
        "tessera.linear_attn", [Q, K, V], {"feature_map": fmap, "causal": causal}, np)
    ref = ts.ops.linear_attn(Q, K, V, feature_map=fmap, causal=causal)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-3, atol=1e-3)


@gpu
def test_linear_attn_decay():
    Q, K, V = _qkv(2)
    d = _decay()
    out = R._apple_gpu_dispatch_linear_attn(
        "tessera.linear_attn", [Q, K, V], {"feature_map": "identity", "decay": d, "causal": True}, np)
    ref = ts.ops.linear_attn(Q, K, V, feature_map="identity", decay=d, causal=True)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-3, atol=1e-3)


@gpu
@pytest.mark.parametrize("decay_on", [False, True])
def test_linear_attn_state(decay_on):
    Q, K, V = _qkv(3)
    kw = {"feature_map": "identity", "causal": True}
    if decay_on:
        kw["decay"] = _decay()
    out = R._apple_gpu_dispatch_linear_attn("tessera.linear_attn_state", [Q, K, V], kw, np)
    ref = ts.ops.linear_attn_state(Q, K, V, **kw)
    assert np.asarray(out).shape == (_B, _H, _D, _D)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-3, atol=1e-3)


@gpu
@pytest.mark.parametrize("decay_on", [False, True])
def test_lightning_attention(decay_on):
    Q, K, V = _qkv(4)
    kw = {"causal": True}
    if decay_on:
        kw["decay"] = _decay()
    out = R._apple_gpu_dispatch_linear_attn("tessera.lightning_attention", [Q, K, V], kw, np)
    ref = ts.ops.lightning_attention(Q, K, V, **kw)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-3, atol=1e-3)


@gpu
@pytest.mark.parametrize("deg", [2, 3])
def test_power_attn(deg):
    Q, K, V = _qkv(5)
    out = R._apple_gpu_dispatch_linear_attn("tessera.power_attn", [Q, K, V], {"deg": deg, "causal": True}, np)
    ref = ts.ops.power_attn(Q, K, V, deg=deg, causal=True)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=2e-3, atol=2e-3)


@gpu
def test_retention():
    Q, K, V = _qkv(6)
    log_g = np.log(_decay())
    out = R._apple_gpu_dispatch_linear_attn(
        "tessera.retention", [Q, K, V], {"deg": 2, "log_g": log_g, "causal": True}, np)
    ref = ts.ops.retention(Q, K, V, log_g=log_g, deg=2, causal=True)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-3, atol=1e-3)


@gpu
def test_linear_attn_jit_metal_runtime():
    Q, K, V = _qkv(7)

    @ts.jit(target="apple_gpu")
    def f(q, k, v):
        return ts.ops.linear_attn(q, k, v, feature_map="identity", causal=True)

    np.testing.assert_allclose(
        np.asarray(f(Q, K, V)),
        np.asarray(ts.ops.linear_attn(Q, K, V, feature_map="identity", causal=True)),
        rtol=1e-3, atol=1e-3)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


@gpu
def test_lightning_jit_metal_runtime():
    Q, K, V = _qkv(8)

    @ts.jit(target="apple_gpu")
    def f(q, k, v):
        return ts.ops.lightning_attention(q, k, v, causal=True)

    np.testing.assert_allclose(
        np.asarray(f(Q, K, V)), np.asarray(ts.ops.lightning_attention(Q, K, V, causal=True)),
        rtol=1e-3, atol=1e-3)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
