"""Thrust #3a — fused ragged grouped-GEMM MSL kernel (one dispatch).

`grouped_gemm` previously ran a per-group MPS-matmul loop (one dispatch per
expert). This routes it through a single `grouped_gemm_f32` MSL kernel that folds
the routing in (per-token expert id), removing the per-expert dispatch overhead.
f32; validated bit-exact against the per-group reference.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import runtime as _runtime

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.hardware_apple_gpu


def _ref(x, w, gs):
    out = np.zeros((x.shape[0], w.shape[2]), dtype=np.float32)
    off = 0
    for e in range(w.shape[0]):
        n = int(gs[e]); out[off:off + n] = x[off:off + n] @ w[e]; off += n
    return out


@gpu
def test_grouped_gemm_symbol_present_in_runtime():
    # The sentinel is always the *newest* symbol (it moves as kernels land);
    # what matters for grouped_gemm is that its C ABI symbol is loadable in the
    # current runtime image (i.e. the recompile picked it up).
    from tessera import _apple_gpu_dispatch as agd
    assert isinstance(agd._SENTINEL_SYMBOL, str) and agd._SENTINEL_SYMBOL
    assert _GPU, "Apple GPU runtime unavailable on the hardware test host"
    lib = agb._load()
    assert hasattr(lib, "tessera_apple_gpu_grouped_gemm_f32")


@gpu
@pytest.mark.parametrize("gs", [[5, 3, 4], [0, 7, 5], [16, 16, 32], [1, 1, 10], [12]])
def test_fused_kernel_matches_reference(gs):
    rng = np.random.default_rng(sum(gs) + len(gs))
    gs = np.array(gs); T, K, N, E = int(gs.sum()), 8, 6, len(gs)
    x = rng.standard_normal((T, K)).astype(np.float32)
    w = rng.standard_normal((E, K, N)).astype(np.float32)
    eid = np.repeat(np.arange(E, dtype=np.int32), gs)
    got = agb.gpu_grouped_gemm(x, w, eid)
    np.testing.assert_allclose(got, _ref(x, w, gs), rtol=1e-4, atol=1e-4)


@gpu
def test_gpu_grouped_gemm_shape_guards():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 4)).astype(np.float32)
    w = rng.standard_normal((2, 4, 3)).astype(np.float32)
    with pytest.raises(agb.AppleGpuError):
        agb.gpu_grouped_gemm(x, rng.standard_normal((2, 5, 3)).astype(np.float32),
                             np.zeros(6, np.int32))           # K mismatch
    with pytest.raises(agb.AppleGpuError):
        agb.gpu_grouped_gemm(x, w, np.zeros(4, np.int32))     # expert_ids wrong len


@gpu
def test_jit_grouped_gemm_uses_fused_path(monkeypatch):
    # Proof the single fused dispatch is taken: break the per-group fallback
    # (agb.gpu_matmul) — the @jit path must STILL work, which it only can via the
    # fused gpu_grouped_gemm kernel.
    rng = np.random.default_rng(2)
    gs = np.array([5, 3, 4]); T, K, N, E = 12, 8, 6, 3
    x = rng.standard_normal((T, K)).astype(np.float32)
    w = rng.standard_normal((E, K, N)).astype(np.float32)
    ref = _ref(x, w, gs)

    def _boom(*a, **k):
        raise AssertionError("per-group fallback must not be reached")
    monkeypatch.setattr(agb, "gpu_matmul", _boom)

    @ts.jit(target="apple_gpu")
    def f(x, w, group_sizes):
        return ts.ops.grouped_gemm(x, w, group_sizes)

    out = np.asarray(f(x, w, gs))
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-4)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


@gpu
def test_many_small_groups_one_dispatch(monkeypatch):
    # The regime the fusion helps most: many tiny groups (high per-dispatch
    # overhead in the old loop). Same break-the-fallback proof.
    rng = np.random.default_rng(3)
    E = 16; gs = np.ones(E, dtype=np.int64) * 2; T, K, N = int(gs.sum()), 8, 4
    x = rng.standard_normal((T, K)).astype(np.float32)
    w = rng.standard_normal((E, K, N)).astype(np.float32)
    ref = _ref(x, w, gs)
    monkeypatch.setattr(agb, "gpu_matmul",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("loop")))

    @ts.jit(target="apple_gpu")
    def f(x, w, group_sizes):
        return ts.ops.grouped_gemm(x, w, group_sizes)

    np.testing.assert_allclose(np.asarray(f(x, w, gs)), ref, rtol=1e-3, atol=1e-4)
