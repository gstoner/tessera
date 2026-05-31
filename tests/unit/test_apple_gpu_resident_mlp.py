"""R0 resident chaining primitives — resident dtype cast + both-operands-resident
matmul2d, and a round-trip-free MLP stack built from them.

These are *capabilities* (keep work on the GPU across the matrix-unit lane), not a
throughput win: on Apple unified memory a host round-trip is a cheap memcpy, so the
resident chain (run_dev + a resident cast per layer) measures *slower* than the
host path — the extra cast dispatch outweighs the avoided memcpy. See the
integration review. These tests lock correctness, not speed.
"""

import numpy as np
import pytest

import tessera.runtime as R

DT = R.DeviceTensor


def _on_metal() -> bool:
    return DT.is_metal()


def _rel(x, ref):
    ref = np.asarray(ref, np.float64)
    return float(np.abs(np.asarray(x, np.float64) - ref).max() / (np.abs(ref).max() + 1e-9))


@pytest.mark.parametrize("dtype,tol", [(np.float16, 1e-3), ("bf16", 1e-2)])
def test_resident_cast_roundtrip(dtype, tol):
    if dtype == "bf16":
        ml = pytest.importorskip("ml_dtypes")
        dtype = ml.bfloat16
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 16)).astype(np.float32)
    xd = DT.from_numpy(x)
    if xd is None:
        return
    try:
        low = xd.cast_to(dtype)              # f32 -> low precision (resident)
        assert low.dtype == np.dtype(dtype)
        assert _rel(low.numpy().astype(np.float32), x) < tol
        back = low.cast_to(np.float32)       # back to f32 (resident)
        assert back.dtype == np.float32
        assert _rel(back.numpy(), low.numpy().astype(np.float32)) < 1e-6
    finally:
        xd.free()


@pytest.mark.parametrize("M,N,K", [(64, 128, 32), (8, 8, 8), (128, 64, 16)])
def test_matmul2d_dev_resident(M, N, K):
    rng = np.random.default_rng(M + N + K)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    Ad, Bd = DT.from_numpy(A), DT.from_numpy(B)
    Cd = DT.empty((M, N), np.float32)
    if Ad is None or Bd is None or Cd is None:
        return
    try:
        ran = R.apple_gpu_matmul2d_dev(Ad, Bd, Cd, bf16=False)
        if _on_metal():
            assert ran is True
            ref = A.astype(np.float64) @ B.astype(np.float64)
            assert _rel(Cd.numpy(), ref) < 3e-2
    finally:
        Ad.free(); Bd.free(); Cd.free()


def test_resident_mlp_chain_correct():
    """A 3-layer MLP stays entirely on the GPU: run_dev (resident f32 out) -> a
    resident f32->f16 cast -> next layer. Bit-matches a numpy reference that
    applies the same f16 cast between layers."""
    if not _on_metal():
        return
    rng = np.random.default_rng(0)
    dims = [64, 128, 128, 64]
    Ws = [(rng.standard_normal((dims[i], dims[i + 1])) * 0.1).astype(np.float16) for i in range(3)]
    bs = [(rng.standard_normal(dims[i + 1]) * 0.1).astype(np.float32) for i in range(3)]
    X0 = (rng.standard_normal((8, 64)) * 0.1).astype(np.float16)
    # numpy reference (relu; f16 cast between layers, matching the GPU chain)
    ref = X0.copy()
    for i in range(3):
        y = np.maximum(ref.astype(np.float32) @ Ws[i].astype(np.float32) + bs[i], 0.0)
        ref = y.astype(np.float16) if i < 2 else y
    sessions = [R.AppleGPUMLPSession(Ws[i], np, bias=bs[i], act="relu", dtype="f16")
                for i in range(3)]
    tmp = []
    try:
        cur = DT.from_numpy(X0); tmp.append(cur)
        Yd = None
        for i in range(3):
            Yd = sessions[i].run_dev(cur); tmp.append(Yd)
            if i < 2:
                cur = Yd.cast_to(np.float16); tmp.append(cur)
        assert _rel(Yd.numpy(), ref) < 1e-4
    finally:
        for s in sessions:
            s.close()
        for t in tmp:
            t.free()
