"""Phase-G attention sprint, Sub-sprint E — data-dependent NSA attention on GPU.

The selection / gather is data-dependent and runs on the host (consistent with
how argmax/argmin fall to host); the attention FLOPs run on the GPU.

  * attn_top_k_blocks   — host argpartition top-k block select + gather →
                          GPU per-query dense attention.
  * deepseek_sparse_attention — weighted blend of the sliding-window (GPU),
                          compressed-block (GPU) and top-k (GPU) branches.
  * attn_local_window_2d — host im2col gather of the 2D neighbourhood →
                          GPU per-position masked dense attention.
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


def test_in_envelope():
    for op in ("tessera.attn_top_k_blocks", "tessera.deepseek_sparse_attention",
               "tessera.attn_local_window_2d"):
        assert op in _driver._APPLE_GPU_RUNTIME_OPS, op
        assert op in _runtime._APPLE_GPU_RUNTIME_OPS, op
        assert op in _driver._APPLE_GPU_SPARSE_ATTN_OPS, op


def test_driver_runtime_sparse_envelopes_agree():
    assert _driver._APPLE_GPU_SPARSE_ATTN_OPS == _runtime._APPLE_GPU_SPARSE_ATTN_OPS


@gpu
@pytest.mark.parametrize("block_size,top_k", [(4, 2), (4, 1), (8, 1)])
def test_attn_top_k_blocks(block_size, top_k):
    rng = np.random.default_rng(block_size * 10 + top_k)
    Q = rng.standard_normal((_B, _H, _S, _D)).astype(np.float32)
    K = rng.standard_normal((_B, _H, _S, _D)).astype(np.float32)
    V = rng.standard_normal((_B, _H, _S, _D)).astype(np.float32)
    nb = _S // block_size
    scores = rng.standard_normal((_B, _H, _S, nb)).astype(np.float32)
    out = R._apple_gpu_dispatch_sparse_attn(
        "tessera.attn_top_k_blocks", [Q, K, V, scores],
        {"top_k": top_k, "block_size": block_size, "causal": True}, np)
    ref = ts.ops.attn_top_k_blocks(Q, K, V, scores=scores, top_k=top_k,
                                   block_size=block_size, causal=True)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-4)


@gpu
@pytest.mark.parametrize("gated", [True, False])
def test_deepseek_sparse_attention(gated):
    rng = np.random.default_rng(7 if gated else 8)
    Q = rng.standard_normal((_B, _H, _S, _D)).astype(np.float32)
    K = rng.standard_normal((_B, _H, _S, _D)).astype(np.float32)
    V = rng.standard_normal((_B, _H, _S, _D)).astype(np.float32)
    kw = {"window_size": 6, "block_size": 4, "top_k": 2, "causal": True}
    operands = [Q, K, V]
    if gated:
        gl = rng.standard_normal((_B, _H, _S, 3)).astype(np.float32)
        operands = [Q, K, V, gl]
        ref = ts.ops.deepseek_sparse_attention(Q, K, V, gate_logits=gl, **kw)
    else:
        ref = ts.ops.deepseek_sparse_attention(Q, K, V, **kw)
    out = R._apple_gpu_dispatch_sparse_attn("tessera.deepseek_sparse_attention", operands, kw, np)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-4)


@gpu
@pytest.mark.parametrize("window", [(1, 1), (2, 1), (0, 2)])
def test_attn_local_window_2d(window):
    rng = np.random.default_rng(hash(window) % 99)
    Hq, Wq = 4, 4
    Q = rng.standard_normal((_B, _H, Hq, Wq, _D)).astype(np.float32)
    K = rng.standard_normal((_B, _H, Hq, Wq, _D)).astype(np.float32)
    V = rng.standard_normal((_B, _H, Hq, Wq, _D)).astype(np.float32)
    out = R._apple_gpu_dispatch_sparse_attn(
        "tessera.attn_local_window_2d", [Q, K, V], {"window": window}, np)
    ref = ts.ops.attn_local_window_2d(Q, K, V, window=window)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-4)


@gpu
def test_deepseek_jit_metal_runtime():
    rng = np.random.default_rng(21)
    Q = rng.standard_normal((_B, _H, _S, _D)).astype(np.float32)
    K = rng.standard_normal((_B, _H, _S, _D)).astype(np.float32)
    V = rng.standard_normal((_B, _H, _S, _D)).astype(np.float32)

    @ts.jit(target="apple_gpu")
    def f(q, k, v):
        return ts.ops.deepseek_sparse_attention(q, k, v, window_size=6, block_size=4, top_k=2, causal=True)

    ref = ts.ops.deepseek_sparse_attention(Q, K, V, window_size=6, block_size=4, top_k=2, causal=True)
    np.testing.assert_allclose(np.asarray(f(Q, K, V)), np.asarray(ref), atol=1e-4)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
