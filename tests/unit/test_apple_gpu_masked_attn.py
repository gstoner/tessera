"""Phase-G attention sprint, Sub-sprint C — NSA masked-softmax attention on GPU.

attn_compressed_blocks is plain softmax(QK_cᵀ·scale)@V_c (the proven batched-
attention kernel); attn_sliding_window adds a structured causal/window additive
mask before the softmax.  QKᵀ / softmax / PV run on the GPU; the window mask is
a host-side structured constant.  metal_runtime.
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


def _qkv(seed=0, sk=None):
    rng = np.random.default_rng(seed)
    sk = sk or _S
    return (rng.standard_normal((_B, _H, _S, _D)).astype(np.float32),
            rng.standard_normal((_B, _H, sk, _D)).astype(np.float32),
            rng.standard_normal((_B, _H, sk, _D)).astype(np.float32))


def test_in_envelope():
    for op in ("tessera.attn_compressed_blocks", "tessera.attn_sliding_window"):
        assert op in _driver._APPLE_GPU_RUNTIME_OPS, op
        assert op in _runtime._APPLE_GPU_RUNTIME_OPS, op
        assert op in _driver._APPLE_GPU_MASKED_ATTN_OPS, op


def test_driver_runtime_masked_attn_envelopes_agree():
    assert _driver._APPLE_GPU_MASKED_ATTN_OPS == _runtime._APPLE_GPU_MASKED_ATTN_OPS


@gpu
@pytest.mark.parametrize("num_blocks", [4, 8])
def test_compressed_blocks(num_blocks):
    Q, Kc, Vc = _qkv(1, sk=num_blocks)
    out = R._apple_gpu_dispatch_masked_attn(
        "tessera.attn_compressed_blocks", [Q, Kc, Vc], {}, np)
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(ts.ops.attn_compressed_blocks(Q, Kc, Vc)), atol=1e-4)


@gpu
@pytest.mark.parametrize("window,causal", [(4, True), (8, True), (6, False), (4, False)])
def test_sliding_window(window, causal):
    Q, K, V = _qkv(2)
    out = R._apple_gpu_dispatch_masked_attn(
        "tessera.attn_sliding_window", [Q, K, V],
        {"window_size": window, "causal": causal}, np)
    ref = ts.ops.attn_sliding_window(Q, K, V, window_size=window, causal=causal)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-4)


@gpu
def test_compressed_blocks_jit_metal_runtime():
    Q, Kc, Vc = _qkv(3, sk=4)

    @ts.jit(target="apple_gpu")
    def f(q, kc, vc):
        return ts.ops.attn_compressed_blocks(q, kc, vc)

    np.testing.assert_allclose(
        np.asarray(f(Q, Kc, Vc)), np.asarray(ts.ops.attn_compressed_blocks(Q, Kc, Vc)), atol=1e-4)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


@gpu
def test_sliding_window_jit_metal_runtime():
    Q, K, V = _qkv(4)

    @ts.jit(target="apple_gpu")
    def f(q, k, v):
        return ts.ops.attn_sliding_window(q, k, v, window_size=4, causal=True)

    ref = ts.ops.attn_sliding_window(Q, K, V, window_size=4, causal=True)
    np.testing.assert_allclose(np.asarray(f(Q, K, V)), np.asarray(ref), atol=1e-4)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
