"""Apple GPU — Metal 4 capability probe (M0) + MTLTensor round-trip (M1).

Metal 4 is an *additive* lane alongside MPSGraph (which still runs on the
classic command model). These tests lock the live capability probe — which
actually creates the Metal 4 objects on-device — and the native ``MTLTensor``
typed-resource round-trip. Everything degrades cleanly to ``available=False`` /
a numpy copy off Tahoe / non-Darwin, so the contract is checked everywhere. See
docs/apple_gpu_metal4_adoption.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as R


def test_metal4_probe_reports_consistent_caps():
    caps = R.apple_gpu_metal4_caps()
    for k in ("available", "command_queue", "command_allocator", "compiler",
              "tensor", "msl4"):
        assert isinstance(caps[k], bool), k
    assert isinstance(caps["bits"], int)
    # If the Metal 4 stack is up at all, the command queue + MTLTensor are the
    # core bits the lane depends on.
    if caps["available"]:
        assert caps["command_queue"]
        assert caps["tensor"]


def test_metal4_probe_matches_bits():
    caps = R.apple_gpu_metal4_caps()
    expect = (caps["command_queue"] * 1 + caps["command_allocator"] * 2
              + caps["compiler"] * 4 + caps["tensor"] * 8 + caps["msl4"] * 16)
    assert caps["bits"] == expect
    assert caps["available"] == (caps["bits"] != 0)


@pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
def test_metal4_tensor_roundtrip_is_exact(dtype):
    if dtype == "bf16":
        ml = pytest.importorskip("ml_dtypes")
        np_dtype = ml.bfloat16
    else:
        np_dtype = {"f32": np.float32, "f16": np.float16}[dtype]
    rng = np.random.default_rng(0)
    a = rng.standard_normal(19).astype(np_dtype)
    rt = R.apple_gpu_metal4_tensor_roundtrip(a, np)
    # Round-trip through the native MTLTensor (or numpy fallback) is a storage
    # copy — bit-exact, dtype preserved.
    assert rt.dtype == a.dtype
    assert np.array_equal(rt.astype(np.float32), a.astype(np.float32))


def test_metal4_tensor_roundtrip_preserves_shape_size():
    a = np.arange(33, dtype=np.float32) * 0.5
    rt = R.apple_gpu_metal4_tensor_roundtrip(a, np)
    assert rt.shape == a.shape
    np.testing.assert_array_equal(rt, a)


def _numpy_scan(Wh, Wx, xseq, init):
    c = init.astype(np.float64)
    ys = np.empty((xseq.shape[0], Wh.shape[0]), np.float64)
    for t in range(xseq.shape[0]):
        c = np.tanh(c @ Wh.astype(np.float64)
                    + xseq[t].astype(np.float64) @ Wx.astype(np.float64))
        ys[t] = c
    return ys


def test_mtl4_scan_msl_loop_matches_numpy_and_mpsgraph():
    """M2 + Phase-G->MSL4: the scan recurrence as a hand-written MSL kernel with
    a native in-kernel for-loop, dispatched through the full MTL4 command model.
    Matches numpy and agrees with the MPSGraph forLoop scan (Rung 0)."""
    rng = np.random.default_rng(0)
    T, d, m = 6, 8, 4
    Wh = rng.standard_normal((d, d)).astype(np.float32) * 0.3
    Wx = rng.standard_normal((m, d)).astype(np.float32) * 0.3
    xseq = rng.standard_normal((T, m)).astype(np.float32) * 0.3
    init = rng.standard_normal(d).astype(np.float32) * 0.1

    ys, ran = R.apple_gpu_mtl4_scan(Wh, Wx, xseq, init, np)
    np.testing.assert_allclose(ys.astype(np.float64),
                               _numpy_scan(Wh, Wx, xseq, init),
                               rtol=1e-4, atol=1e-5)
    # The MSL-loop lowering and the MPSGraph-forLoop lowering must agree.
    ys_mps = R.apple_gpu_cf_scan(Wh, Wx, xseq, init, np)
    np.testing.assert_allclose(ys.astype(np.float64), ys_mps.astype(np.float64),
                               rtol=1e-4, atol=1e-5)
    # On a Tahoe machine with Metal 4, the real MTL4 dispatch must have run.
    if R.apple_gpu_metal4_caps()["available"]:
        assert ran


@pytest.mark.parametrize("M,N,K", [(8, 8, 8), (16, 24, 32), (64, 32, 16)])
def test_mtl4_matmul_cooperative_matches_numpy(M, N, K):
    """M3: matmul via MSL cooperative-matrix ops (simdgroup_matrix → matrix
    units), dispatched through the MTL4 command model. Bit-close to numpy."""
    rng = np.random.default_rng(M + N + K)
    A = rng.standard_normal((M, K)).astype(np.float32) * 0.1
    B = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    C, ran = R.apple_gpu_mtl4_matmul_sg(A, B, np)
    np.testing.assert_allclose(C.astype(np.float64),
                               A.astype(np.float64) @ B.astype(np.float64),
                               rtol=1e-4, atol=1e-4)
    if R.apple_gpu_metal4_caps()["available"]:
        assert ran


def test_mtl4_matmul_non_multiple_of_8_falls_back():
    # The simdgroup kernel needs M/N/K multiples of 8; otherwise numpy fallback.
    A = np.ones((7, 8), np.float32)
    B = np.ones((8, 8), np.float32)
    C, ran = R.apple_gpu_mtl4_matmul_sg(A, B, np)
    assert not ran
    np.testing.assert_allclose(C.astype(np.float64),
                               A.astype(np.float64) @ B.astype(np.float64),
                               rtol=1e-5)


def test_mtl4_pipeline_caching_survives_many_calls():
    """The MTL4 pipeline + command queue are cached/shared. Many repeated calls
    must stay correct — before the per-call residency-set removal this tripped
    the queue's 32-residency-set limit around call ~33."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((16, 16)).astype(np.float32) * 0.1
    B = rng.standard_normal((16, 16)).astype(np.float32) * 0.1
    ref = A.astype(np.float64) @ B.astype(np.float64)
    C = None
    ran = False
    for _ in range(40):
        C, ran = R.apple_gpu_mtl4_matmul_sg(A, B, np)
    np.testing.assert_allclose(C.astype(np.float64), ref, rtol=1e-4, atol=1e-4)
    if R.apple_gpu_metal4_caps()["available"]:
        assert ran


def test_m4_routing_default_off_and_toggle():
    # M4 capability-gated routing is OFF by default (the MTL4 matmul kernel is
    # correct but slower than MPS today); the flag toggles cleanly.
    import tessera as ts

    @ts.jit(target="apple_gpu")
    def mm(a, b):
        return ts.ops.matmul(a, b)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((16, 8)).astype(np.float32) * 0.1
    B = rng.standard_normal((8, 16)).astype(np.float32) * 0.1
    ref = A.astype(np.float64) @ B.astype(np.float64)
    prev = R.apple_gpu_mtl4_routing_enabled()
    try:
        R.set_apple_gpu_mtl4_routing(False)
        assert not R.apple_gpu_mtl4_routing_enabled()
        np.testing.assert_allclose(np.asarray(mm(A, B)).astype(np.float64), ref,
                                   rtol=1e-4, atol=1e-4)          # MPS path
        R.set_apple_gpu_mtl4_routing(True)
        assert R.apple_gpu_mtl4_routing_enabled()
        # 8-multiple f32 -> routed onto MTL4 when capable, else MPS; either way
        # the result must be correct.
        np.testing.assert_allclose(np.asarray(mm(A, B)).astype(np.float64), ref,
                                   rtol=1e-4, atol=1e-4)
    finally:
        R.set_apple_gpu_mtl4_routing(prev)


def test_m4_route_predicate_gates_envelope():
    a = np.ones((8, 8), np.float32)
    b = np.ones((8, 8), np.float32)
    prev = R.apple_gpu_mtl4_routing_enabled()
    try:
        R.set_apple_gpu_mtl4_routing(False)
        assert R._mtl4_route_matmul_f32(a, b, np) is None          # disabled
        R.set_apple_gpu_mtl4_routing(True)
        # ineligible: f16 dtype, and non-8-multiple dims -> None (MPS fallback).
        assert R._mtl4_route_matmul_f32(a.astype(np.float16),
                                        b.astype(np.float16), np) is None
        assert R._mtl4_route_matmul_f32(np.ones((7, 8), np.float32),
                                        np.ones((8, 8), np.float32), np) is None
    finally:
        R.set_apple_gpu_mtl4_routing(prev)


def test_mtl4_scan_falls_back_cleanly():
    # Even without Metal 4 the contract holds (numpy fallback), correct + shaped.
    rng = np.random.default_rng(3)
    Wh = rng.standard_normal((4, 4)).astype(np.float32) * 0.3
    Wx = rng.standard_normal((4, 4)).astype(np.float32) * 0.3
    xseq = rng.standard_normal((3, 4)).astype(np.float32) * 0.3
    init = rng.standard_normal(4).astype(np.float32)
    ys, _ran = R.apple_gpu_mtl4_scan(Wh, Wx, xseq, init, np)
    assert ys.shape == (3, 4)
    np.testing.assert_allclose(ys.astype(np.float64),
                               _numpy_scan(Wh, Wx, xseq, init),
                               rtol=1e-4, atol=1e-5)
