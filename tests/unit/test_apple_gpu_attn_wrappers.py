"""Phase-G attention sprint, Sub-sprint A — standard attention family on Apple GPU.

multi_head_attention / gqa_attention / mqa_attention / mla_decode /
gated_attention are thin wrappers over softmax((QKᵀ)·scale)@V.  They route
through the proven native GQA flash-attention kernel (no new MSL): reshapes are
host-side views, the attention FLOPs run on the GPU.  metal_runtime.
"""

import numpy as np
import pytest

import tessera as ts
import tessera.nn.functional as F
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import runtime as _runtime
from tessera.compiler import driver as _driver

R = _runtime
_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.hardware_apple_gpu


def _r(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def test_in_envelope():
    for op in ("tessera.multi_head_attention", "tessera.gqa_attention",
               "tessera.mqa_attention", "tessera.mla_decode",
               "tessera.gated_attention"):
        assert op in _driver._APPLE_GPU_RUNTIME_OPS, op
        assert op in _runtime._APPLE_GPU_RUNTIME_OPS, op
        assert op in _driver._APPLE_GPU_ATTN_WRAPPER_OPS, op


def test_driver_runtime_attn_envelopes_agree():
    assert _driver._APPLE_GPU_ATTN_WRAPPER_OPS == _runtime._APPLE_GPU_ATTN_WRAPPER_OPS


@gpu
@pytest.mark.parametrize("causal", [False, True])
def test_multi_head_attention(causal):
    B, S, H, D = 2, 16, 4, 32
    Q, K, V = _r(B, S, H * D, seed=1), _r(B, S, H * D, seed=2), _r(B, S, H * D, seed=3)
    out = R._apple_gpu_dispatch_attn_wrapper(
        "tessera.multi_head_attention", [Q, K, V], {"num_heads": H, "causal": causal}, np)
    ref = F.multi_head_attention(Q, K, V, num_heads=H, causal=causal)
    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-4)


@gpu
@pytest.mark.parametrize("nq,nkv", [(4, 2), (8, 2), (6, 3)])
def test_gqa_attention(nq, nkv):
    B, S, D = 2, 16, 32
    Q, K, V = _r(B, nq, S, D, seed=nq), _r(B, nkv, S, D, seed=nkv), _r(B, nkv, S, D, seed=nkv + 1)
    out = R._apple_gpu_dispatch_attn_wrapper(
        "tessera.gqa_attention", [Q, K, V], {"num_query_heads": nq, "num_kv_heads": nkv}, np)
    ref = F.gqa_attention(Q, K, V, num_query_heads=nq, num_kv_heads=nkv)
    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-4)


@gpu
def test_mqa_attention():
    B, H, S, D = 2, 4, 16, 32
    Q, K, V = _r(B, H, S, D, seed=1), _r(B, 1, S, D, seed=2), _r(B, 1, S, D, seed=3)
    out = R._apple_gpu_dispatch_attn_wrapper("tessera.mqa_attention", [Q, K, V], {}, np)
    np.testing.assert_allclose(np.asarray(out), F.mqa_attention(Q, K, V), atol=1e-4)


@gpu
def test_mla_decode():
    B, H, S, D = 2, 4, 16, 32
    Q, K, V = _r(B, H, S, D, seed=1), _r(B, H, S, D, seed=2), _r(B, H, S, D, seed=3)
    out = R._apple_gpu_dispatch_attn_wrapper("tessera.mla_decode", [Q, K, V], {}, np)
    np.testing.assert_allclose(np.asarray(out), F.mla_decode(Q, K, V), atol=1e-4)


@gpu
@pytest.mark.parametrize("act", ["sigmoid", "identity"])
def test_gated_attention(act):
    B, H, S, D = 2, 4, 16, 32
    Q, K, V = _r(B, H, S, D, seed=1), _r(B, H, S, D, seed=2), _r(B, H, S, D, seed=3)
    gate = _r(B, H, S, D, seed=4)
    out = R._apple_gpu_dispatch_attn_wrapper(
        "tessera.gated_attention", [Q, K, V, gate], {"causal": True, "gate_activation": act}, np)
    ref = ts.ops.gated_attention(Q, K, V, gate, causal=True, gate_activation=act)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-4)


@gpu
def test_mha_jit_metal_runtime():
    B, S, H, D = 2, 16, 4, 32
    Q, K, V = _r(B, S, H * D, seed=1), _r(B, S, H * D, seed=2), _r(B, S, H * D, seed=3)

    @ts.jit(target="apple_gpu")
    def f(q, k, v):
        return ts.ops.multi_head_attention(q, k, v, num_heads=4)

    np.testing.assert_allclose(
        np.asarray(f(Q, K, V)), F.multi_head_attention(Q, K, V, num_heads=4), atol=1e-4)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


@gpu
@pytest.mark.parametrize("causal", [False, True])
def test_mla_decode_fused(causal):
    B, S, Dx, Dc, Dm = 2, 12, 16, 8, 16
    rng = np.random.default_rng(1 if causal else 2)
    x = rng.standard_normal((B, S, Dx)).astype(np.float32)
    w_dkv = rng.standard_normal((Dx, Dc)).astype(np.float32)
    w_uk = rng.standard_normal((Dc, Dm)).astype(np.float32)
    w_uv = rng.standard_normal((Dc, Dm)).astype(np.float32)
    q = rng.standard_normal((B, S, Dm)).astype(np.float32)
    out = R._apple_gpu_dispatch_attn_wrapper(
        "tessera.mla_decode_fused", [x, w_dkv, w_uk, w_uv, q], {"causal": causal}, np)
    ref = ts.ops.mla_decode_fused(x, w_dkv, w_uk, w_uv, q, causal=causal)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-3)


@gpu
def test_mla_decode_jit_metal_runtime():
    B, H, S, D = 2, 4, 16, 32
    Q, K, V = _r(B, H, S, D, seed=1), _r(B, H, S, D, seed=2), _r(B, H, S, D, seed=3)

    @ts.jit(target="apple_gpu")
    def f(q, k, v):
        return ts.ops.mla_decode(q, k, v)

    np.testing.assert_allclose(np.asarray(f(Q, K, V)), F.mla_decode(Q, K, V), atol=1e-4)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
