"""Local SwiGLU-fused MoE expert-FFN block (MegaMoE precursor).

moe_swiglu_block composes the grouped-GEMM + silu_mul lanes into the MoE expert
feed-forward core: grouped_gemm(x,W_gate)/grouped_gemm(x,W_up) -> silu_mul ->
grouped_gemm(.,W_down).  It inherits the grouped-layout contract (kind /
alignment) + the dequant-on-host quant path from grouped_gemm (Rungs A/B), and
on Apple GPU it routes to metal_runtime by composing the proven lanes.
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
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")


def _silu(z):
    return z / (1.0 + np.exp(-z))


def _ref(x, wg, wu, wd, gs):
    """numpy reference: per-group SwiGLU expert FFN, f64 accumulation."""
    T, N = x.shape[0], wd.shape[2]
    out = np.zeros((T, N), np.float64)
    off = 0
    for e in range(len(gs)):
        n = int(gs[e])
        if n:
            xb = x[off:off + n].astype(np.float64)
            h = _silu(xb @ wg[e]) * (xb @ wu[e])
            out[off:off + n] = h @ wd[e]
        off += n
    return out


def _inputs(seed, gs, K=8, F=10, N=6):
    rng = np.random.default_rng(seed)
    gs = np.asarray(gs, np.int64)
    T, E = int(gs.sum()), len(gs)
    x = rng.standard_normal((T, K)).astype(np.float32)
    wg = rng.standard_normal((E, K, F)).astype(np.float32)
    wu = rng.standard_normal((E, K, F)).astype(np.float32)
    wd = rng.standard_normal((E, F, N)).astype(np.float32)
    return x, wg, wu, wd, gs


def test_in_envelope():
    assert "tessera.moe_swiglu_block" in _driver._APPLE_GPU_RUNTIME_OPS
    assert "tessera.moe_swiglu_block" in _runtime._APPLE_GPU_RUNTIME_OPS
    assert _driver._APPLE_GPU_MOE_OPS == _runtime._APPLE_GPU_MOE_OPS


@pytest.mark.parametrize("gs", [[5, 3, 4], [16, 16, 32], [12], [0, 7, 5]])
def test_eager_matches_numpy_reference(gs):
    x, wg, wu, wd, g = _inputs(hash(tuple(gs)) % 99, gs)
    got = np.asarray(ts.ops.moe_swiglu_block(x, wg, wu, wd, g))
    np.testing.assert_allclose(got, _ref(x, wg, wu, wd, g), rtol=1e-4, atol=1e-4)


def test_single_expert_reduces_to_dense_swiglu():
    rng = np.random.default_rng(3)
    K, F, N = 8, 10, 6
    x = rng.standard_normal((7, K)).astype(np.float32)
    wg = rng.standard_normal((1, K, F)).astype(np.float32)
    wu = rng.standard_normal((1, K, F)).astype(np.float32)
    wd = rng.standard_normal((1, F, N)).astype(np.float32)
    moe = np.asarray(ts.ops.moe_swiglu_block(x, wg, wu, wd, np.array([7])))
    dense = np.asarray(ts.ops.swiglu(x, wg[0], wu[0], wd[0]))
    np.testing.assert_allclose(moe, dense, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("quant,bound", [("fp8_e4m3", 0.10), ("nvfp4", 0.30)])
def test_quantized_block_within_budget(quant, bound):
    x, wg, wu, wd, gs = _inputs(5, [16, 16, 32], K=64, F=32, N=16)
    ref = _ref(x, wg, wu, wd, gs)
    got = np.asarray(ts.ops.moe_swiglu_block(x, wg, wu, wd, gs, quant=quant))
    rel = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-9)
    assert rel < bound, f"{quant} moe_swiglu rel {rel:.4f} > {bound}"


@pytest.mark.parametrize("kind", ["masked", "k_grouped"])
def test_eager_rejects_unsupported_kinds(kind):
    x, wg, wu, wd, gs = _inputs(7, [5, 3, 4])
    with pytest.raises(NotImplementedError):
        ts.ops.moe_swiglu_block(x, wg, wu, wd, gs, kind=kind)


def test_runtime_dispatch_matches_eager():
    x, wg, wu, wd, gs = _inputs(11, [16, 16, 32], K=64, F=32, N=16)
    eager = np.asarray(ts.ops.moe_swiglu_block(x, wg, wu, wd, gs))
    rt = np.asarray(R._apple_gpu_dispatch_moe_swiglu_block([x, wg, wu, wd, gs], {}, np))
    np.testing.assert_allclose(rt, eager, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("kind", ["masked", "k_grouped"])
def test_runtime_dispatch_rejects_unsupported_kinds(kind):
    x, wg, wu, wd, gs = _inputs(13, [5, 3, 4])
    with pytest.raises(ValueError):
        R._apple_gpu_dispatch_moe_swiglu_block([x, wg, wu, wd, gs], {"kind": kind}, np)


@gpu
def test_jit_metal_runtime():
    x, wg, wu, wd, gs = _inputs(17, [16, 16, 32], K=64, F=32, N=16)

    @ts.jit(target="apple_gpu")
    def f(x, wg, wu, wd, gs):
        return ts.ops.moe_swiglu_block(x, wg, wu, wd, gs)

    got = np.asarray(f(x, wg, wu, wd, gs))
    np.testing.assert_allclose(got, _ref(x, wg, wu, wd, gs), rtol=1e-3, atol=1e-3)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
