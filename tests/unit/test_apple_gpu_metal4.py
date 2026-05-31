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


@pytest.mark.parametrize(
    "M,N,K",
    [
        # general (bounds-checked 32x32) path — non-64/16-aligned shapes
        (8, 8, 8),       # single 8x8 sub-tile (< one 32x32 threadgroup tile)
        (16, 24, 32),    # partial 32x32 tile, K spans 2 BK slabs
        (64, 32, 16),    # multi-threadgroup in M
        (40, 56, 24),    # M,N not 32-multiples -> exercises zero-pad + store mask
        (128, 96, 32),   # N=96 not a 64-multiple -> general path
        # fast (register-blocked vectorized 64x64) path — M%64,N%64,K%16 aligned
        (64, 64, 16),    # single fast threadgroup tile
        (128, 128, 128), # several fast tiles each way, multi-slab K
        (192, 256, 64),  # non-square aligned, exercises 2x4 SIMD-group grid
    ],
)
def test_mtl4_matmul_cooperative_matches_numpy(M, N, K):
    """M3/M5: matmul via MSL cooperative-matrix ops (simdgroup_matrix → matrix
    units). Aligned shapes (M%64,N%64,K%16) take the register-blocked vectorized
    64x64 fast kernel; others take the bounds-checked 32x32 double-buffered
    kernel. Both dispatch through the MTL4 command model and are bit-close to
    numpy across partial- and multi-tile shapes."""
    rng = np.random.default_rng(M + N + K)
    A = rng.standard_normal((M, K)).astype(np.float32) * 0.1
    B = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    C, ran = R.apple_gpu_mtl4_matmul_sg(A, B, np)
    np.testing.assert_allclose(C.astype(np.float64),
                               A.astype(np.float64) @ B.astype(np.float64),
                               rtol=1e-4, atol=1e-4)
    if R.apple_gpu_metal4_caps()["available"]:
        assert ran


@pytest.mark.parametrize(
    "M,N,K",
    [
        (8, 8, 8),        # tiny (< one 64x64 tile)
        (64, 64, 64),     # single MPP tile
        (128, 96, 80),    # partial tile, K spans multiple chunks
        (100, 72, 48),    # none 64/8-aligned -> matmul2d slice() edge-checks
        (256, 256, 256),  # several full tiles
        (512, 128, 320),  # non-square, large K
    ],
)
def test_mtl4_matmul2d_f16_matches_numpy(M, N, K):
    """M6: fp16 matmul via the MSL 4.0 cooperative `tensor` op
    (MetalPerformancePrimitives matmul2d) on the GPU matrix units, with real
    MTLTensor arguments bound through an MTL4ArgumentTable. f16 in, f32 out;
    accumulation is fp32 so it stays bit-close to the fp16-reference product
    across aligned, partial, and non-square shapes."""
    rng = np.random.default_rng(M + N + K)
    A = (rng.standard_normal((M, K)) * 0.25).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.25).astype(np.float16)
    C, ran = R.apple_gpu_mtl4_matmul2d_f16(A, B, np)
    assert C.dtype == np.float32
    ref = A.astype(np.float64) @ B.astype(np.float64)
    den = np.maximum(1e-2, np.abs(ref))
    assert float(np.max(np.abs(C.astype(np.float64) - ref) / den)) < 3e-2
    # On a Tahoe machine with Metal 4, the MPP tensor-op must have run on-GPU.
    if R.apple_gpu_metal4_caps()["available"]:
        assert ran


@pytest.mark.parametrize("M,N,K", [(64, 64, 64), (128, 96, 80), (100, 72, 48), (256, 256, 256)])
def test_mtl4_matmul2d_bf16_matches_numpy(M, N, K):
    """M6 bf16 sibling: bf16 matmul via MPP matmul2d on the matrix units, same
    MTLTensor-bound MTL4 path as the f16 kernel. fp32 accumulation keeps it close
    to the bf16-reference product."""
    ml = pytest.importorskip("ml_dtypes")
    bf16 = ml.bfloat16
    rng = np.random.default_rng(M + N + K)
    A = (rng.standard_normal((M, K)) * 0.25).astype(bf16)
    B = (rng.standard_normal((K, N)) * 0.25).astype(bf16)
    C, ran = R.apple_gpu_mtl4_matmul2d_bf16(A, B, np)
    assert C.dtype == np.float32
    ref = A.astype(np.float64) @ B.astype(np.float64)
    den = np.maximum(1e-2, np.abs(ref))
    assert float(np.max(np.abs(C.astype(np.float64) - ref) / den)) < 6e-2
    if R.apple_gpu_metal4_caps()["available"]:
        assert ran


def _epi_ref(ref, bias, act, np):
    if bias is not None:
        ref = ref + bias[None, :].astype(np.float64)
    if act == "relu":
        return np.maximum(0.0, ref)
    if act == "gelu":
        t = 0.7978845608028654 * (ref + 0.044715 * ref ** 3)
        return 0.5 * ref * (1.0 + np.tanh(t))
    if act == "silu":
        return ref / (1.0 + np.exp(-ref))
    return ref


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("act", ["none", "relu", "gelu", "silu"])
def test_mtl4_matmul2d_epilogue_fuses_bias_and_activation(dtype, act):
    """M7: bias (per output column) + activation fused IN-REGISTER on the float
    cooperative_tensor before the single store — one MPP matmul2d dispatch, no
    extra device round-trip. Matches the numpy reference for both input dtypes
    and all four activations."""
    if dtype == "bf16":
        ml = pytest.importorskip("ml_dtypes")
        cast = ml.bfloat16
        tol = 6e-2
    else:
        cast = np.float16
        tol = 3e-2
    M, N, K = 128, 96, 64
    rng = np.random.default_rng(hash((dtype, act)) % 1000)
    A = (rng.standard_normal((M, K)) * 0.25).astype(cast)
    B = (rng.standard_normal((K, N)) * 0.25).astype(cast)
    bias = (rng.standard_normal(N) * 0.5).astype(np.float32)
    C, ran = R.apple_gpu_mtl4_matmul2d_epilogue(A, B, np, bias=bias, act=act, dtype=dtype)
    assert C.dtype == np.float32
    ref = _epi_ref(A.astype(np.float64) @ B.astype(np.float64), bias, act, np)
    den = np.maximum(1e-2, np.abs(ref))
    assert float(np.max(np.abs(C.astype(np.float64) - ref) / den)) < tol
    if R.apple_gpu_metal4_caps()["available"]:
        assert ran


def test_mtl4_matmul2d_epilogue_bias_is_per_output_column():
    """Lock the bias axis: with A=B=0 the fused result is exactly the per-column
    bias broadcast across rows (catches a row/col index swap in the kernel)."""
    M, N, K = 64, 128, 32
    A = np.zeros((M, K), np.float16)
    B = np.zeros((K, N), np.float16)
    bias = np.arange(N).astype(np.float32)
    C, _ran = R.apple_gpu_mtl4_matmul2d_epilogue(A, B, np, bias=bias, act="none", dtype="f16")
    np.testing.assert_array_equal(C, np.broadcast_to(bias, (M, N)))


def test_mtl4_matmul2d_epilogue_no_bias_relu():
    # bias=None path (binds a dummy buffer, has_bias=0) + relu still correct.
    rng = np.random.default_rng(5)
    A = (rng.standard_normal((72, 48)) * 0.3).astype(np.float16)
    B = (rng.standard_normal((48, 80)) * 0.3).astype(np.float16)
    C, _ = R.apple_gpu_mtl4_matmul2d_epilogue(A, B, np, bias=None, act="relu", dtype="f16")
    ref = np.maximum(0.0, A.astype(np.float64) @ B.astype(np.float64))
    den = np.maximum(1e-2, np.abs(ref))
    assert float(np.max(np.abs(C.astype(np.float64) - ref) / den)) < 3e-2


def test_mtl4_matmul2d_epilogue_rejects_bad_act():
    with pytest.raises(ValueError):
        R.apple_gpu_mtl4_matmul2d_epilogue(np.ones((8, 8), np.float16),
                                           np.ones((8, 8), np.float16), np, act="elu")


def _mlp_ref(X, W, bias, act, np):
    y = X.astype(np.float64) @ W.astype(np.float64)
    if bias is not None:
        y = y + bias[None, :].astype(np.float64)
    t = 0.7978845608028654 * (y + 0.044715 * y ** 3)
    return {"none": y, "relu": np.maximum(0.0, y), "gelu": 0.5 * y * (1.0 + np.tanh(t)),
            "silu": y / (1.0 + np.exp(-y))}[act]


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_mtl4_mlp_session_resident_weights_matches_reference(dtype):
    """M8: a resident-weight MLP-block session (Linear + bias + GELU) — the weight
    is uploaded once and reused across run() steps. Each step is the fused
    matmul2d epilogue; results must match the composed reference across the kind
    of varying-M decode steps the session is built for."""
    if dtype == "bf16":
        ml = pytest.importorskip("ml_dtypes")
        cast, tol = ml.bfloat16, 6e-2
    else:
        cast, tol = np.float16, 3e-2
    K, N = 256, 512
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((K, N)) * 0.05).astype(cast)
    bias = (rng.standard_normal(N) * 0.1).astype(np.float32)
    sess = R.AppleGPUMLPSession(W, np, bias=bias, act="gelu", dtype=dtype)
    try:
        if R.apple_gpu_metal4_caps()["available"]:
            assert sess.ran_on_gpu
        for M in (1, 8, 64, 100):  # decode-step shapes, incl. partial tile
            X = (rng.standard_normal((M, K)) * 0.1).astype(cast)
            Y = sess.run(X)
            assert Y.shape == (M, N) and Y.dtype == np.float32
            ref = _mlp_ref(X, W, bias, "gelu", np)
            den = np.maximum(1e-2, np.abs(ref))
            assert float(np.max(np.abs(Y.astype(np.float64) - ref) / den)) < tol
    finally:
        sess.close()


def test_mtl4_mlp_session_matches_oneshot_epilogue():
    # The session and the one-shot fused epilogue compute the same thing.
    rng = np.random.default_rng(7)
    K, N = 128, 256
    W = (rng.standard_normal((K, N)) * 0.05).astype(np.float16)
    bias = (rng.standard_normal(N) * 0.1).astype(np.float32)
    X = (rng.standard_normal((32, K)) * 0.1).astype(np.float16)
    with R.AppleGPUMLPSession(W, np, bias=bias, act="silu", dtype="f16") as sess:
        Y_sess = sess.run(X)
    Y_one, _ = R.apple_gpu_mtl4_matmul2d_epilogue(X, W, np, bias=bias, act="silu", dtype="f16")
    np.testing.assert_allclose(Y_sess, Y_one, rtol=1e-3, atol=1e-3)


def test_mtl4_mlp_session_run_after_close_uses_fallback():
    # close() releases resident weights; further run() still returns correct
    # values via the numpy fallback (no crash, no stale handle use).
    rng = np.random.default_rng(3)
    K, N = 64, 128
    W = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    sess = R.AppleGPUMLPSession(W, np, act="relu", dtype="f16")
    sess.close()
    assert not sess.ran_on_gpu
    X = (rng.standard_normal((8, K)) * 0.1).astype(np.float16)
    Y = sess.run(X)
    ref = _mlp_ref(X, W, None, "relu", np)
    den = np.maximum(1e-2, np.abs(ref))
    assert float(np.max(np.abs(Y.astype(np.float64) - ref) / den)) < 3e-2


def test_mtl4_matmul2d_f16_falls_back_cleanly():
    # Off Tahoe / non-Darwin the contract still holds via the numpy fp16 ref.
    rng = np.random.default_rng(11)
    A = (rng.standard_normal((24, 40)) * 0.2).astype(np.float16)
    B = (rng.standard_normal((40, 16)) * 0.2).astype(np.float16)
    C, _ran = R.apple_gpu_mtl4_matmul2d_f16(A, B, np)
    assert C.shape == (24, 16) and C.dtype == np.float32
    ref = A.astype(np.float64) @ B.astype(np.float64)
    den = np.maximum(1e-2, np.abs(ref))
    assert float(np.max(np.abs(C.astype(np.float64) - ref) / den)) < 3e-2


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
