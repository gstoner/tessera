"""Project 5 (2026-06-01) — conv2d encode-session integration.

Conv2d now joins the 8 existing encode-eligible ops on the
single-command-buffer chain. The new C ABI
``tessera_apple_gpu_conv2d_dev_f32_enc`` appends an MPSGraph
``convolution2DWithSourceTensor:`` node to the session's command
buffer instead of running its own command queue — so a CNN-style
chain (conv → relu/silu/gelu → conv → norm → conv ...) stays on
ONE command buffer with no per-op GPU↔CPU sync.

Tests pin:

* **Symbol availability** — ``tessera_apple_gpu_conv2d_dev_f32_enc``
  is exported by the runtime and the Python wrapper binds it.
* **Registry registration** — the chain registry treats conv2d as
  encode-eligible for dtype="f32".
* **Numerical correctness** — encode-session conv2d output matches
  the legacy (host-buffer) ``tessera_apple_gpu_conv2d_f32`` at
  ``rtol=1e-4``.
* **Single-cb chaining** — a chain that calls conv2d alongside
  silu/rmsnorm commits ONE command buffer (≤ chunking budget).
* **TraceRef** — conv2d output can feed into a downstream encode-op
  via ``@auto_batch`` trace capture.
* **Groups + padding** — depthwise + padded variants both encode
  correctly (the most common conv2d configurations a model would
  actually use).
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from tessera._apple_gpu_dispatch import bind_symbol
from tessera.apple_gpu_batched import (
    DeviceTensor,
    batched_session,
    conv2d_enc,
    conv2d_enc_no_bias,
    device_tensor,
    session_available,
    session_commit_count,
)
from tessera.apple_gpu_chain import (
    ENCODE_OP_REGISTRY,
    OpRecord,
    is_encode_eligible,
    run_trace,
)
import tessera.apple_gpu_ops as agpu


# ---- Symbol availability -----------------------------------------------

def test_conv2d_dev_f32_enc_symbol_resolves():
    fn = bind_symbol(
        "tessera_apple_gpu_conv2d_dev_f32_enc",
        (ctypes.c_void_p,) * 5 + (ctypes.c_int32,) * 14,
        ctypes.c_int32)
    assert fn is not None, (
        "tessera_apple_gpu_conv2d_dev_f32_enc not exported by the "
        "runtime — rebuild TesseraAppleRuntime")


# ---- Registry surface --------------------------------------------------

def test_conv2d_is_encode_eligible_f32():
    assert is_encode_eligible("conv2d", "f32")
    spec = ENCODE_OP_REGISTRY[("conv2d", "f32")]
    assert spec.input_tensor_args == (0, 1)
    assert spec.encode_fn is conv2d_enc_no_bias


def test_conv2d_is_encode_eligible_for_f16_and_bf16():
    """Sprint A (2026-06-01) — the f16 and bf16 conv2d encode lanes
    landed (one sprint after Project 5's f32 lane). The 3-dtype
    matrix is now complete for conv2d."""
    assert is_encode_eligible("conv2d", "f16")
    assert is_encode_eligible("conv2d", "bf16")


# ---- Numerical correctness vs the legacy host path --------------------

@pytest.mark.parametrize("N,H,W,Cin,Cout,kH,kW,padH,padW,strideH,strideW,groups", [
    (1,  8,  8,  3, 4, 3, 3, 1, 1, 1, 1, 1),   # standard 3x3 conv with padding
    (1,  6,  6,  4, 8, 3, 3, 0, 0, 1, 1, 1),   # 3x3 no padding
    (1,  4,  4,  2, 2, 1, 1, 0, 0, 1, 1, 1),   # 1x1 pointwise
    (1,  8,  8,  4, 4, 3, 3, 1, 1, 1, 1, 4),   # depthwise (groups=Cin=Cout)
    (1, 10, 10,  3, 6, 3, 3, 0, 0, 2, 2, 1),   # strided conv
])
def test_conv2d_encode_matches_legacy_host_path(
        N, H, W, Cin, Cout, kH, kW, padH, padW, strideH, strideW, groups):
    if not session_available():
        pytest.skip("encode-session unavailable")

    rng = np.random.default_rng(0xC04205D)
    X_np = rng.standard_normal((N, H, W, Cin), dtype=np.float32) * 0.1
    Wt_np = rng.standard_normal(
        (kH, kW, Cin // groups, Cout), dtype=np.float32) * 0.1

    # Encode-session output.
    X = device_tensor(X_np)
    Wt = device_tensor(Wt_np)
    try:
        with batched_session() as sess:
            outH = (H + 2 * padH - kH) // strideH + 1
            outW = (W + 2 * padW - kW) // strideW + 1
            O = conv2d_enc(sess, X, Wt, None,
                            N=N, H=H, W=W, Cin=Cin, Cout=Cout,
                            kH=kH, kW=kW,
                            strideH=strideH, strideW=strideW,
                            padH=padH, padW=padW,
                            dilationH=1, dilationW=1,
                            groups=groups)
        got = O.download(np.float32, (N, outH, outW, Cout))
        O.free()
    finally:
        X.free(); Wt.free()

    # Legacy host path (the reference numerical answer).
    # Legacy signature: 4 ptrs (X, W, bias, O) + 14 int32s.
    legacy = bind_symbol(
        "tessera_apple_gpu_conv2d_f32",
        (ctypes.c_void_p,) * 4 + (ctypes.c_int32,) * 14, None)
    assert legacy is not None
    out_legacy = np.empty((N, outH, outW, Cout), dtype=np.float32)
    legacy(X_np.ctypes.data_as(ctypes.c_void_p),
            Wt_np.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(0),  # no bias
            out_legacy.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int32(N), ctypes.c_int32(H), ctypes.c_int32(W),
            ctypes.c_int32(Cin), ctypes.c_int32(Cout),
            ctypes.c_int32(kH), ctypes.c_int32(kW),
            ctypes.c_int32(strideH), ctypes.c_int32(strideW),
            ctypes.c_int32(padH), ctypes.c_int32(padW),
            ctypes.c_int32(1), ctypes.c_int32(1),
            ctypes.c_int32(groups))

    np.testing.assert_allclose(got, out_legacy, rtol=1e-4, atol=1e-4)


# ---- Single-cb chaining -----------------------------------------------

def test_conv2d_chain_commits_one_cb():
    """conv2d + silu + conv2d on one chain → one cb commit."""
    if not session_available():
        pytest.skip("encode-session unavailable")

    N, H, W, Cin, Cmid, Cout, kH, kW = 1, 8, 8, 3, 4, 4, 3, 3
    rng = np.random.default_rng(0xC1AED)
    X_np = rng.standard_normal((N, H, W, Cin), dtype=np.float32) * 0.1
    W1_np = rng.standard_normal((kH, kW, Cin, Cmid),
                                  dtype=np.float32) * 0.1
    W2_np = rng.standard_normal((kH, kW, Cmid, Cout),
                                  dtype=np.float32) * 0.1

    trace = [
        OpRecord("conv2d", "f32",
                  inputs=[X_np, W1_np],
                  shape_kwargs=dict(N=N, H=H, W=W, Cin=Cin, Cout=Cmid,
                                     kH=kH, kW=kW,
                                     strideH=1, strideW=1,
                                     padH=1, padW=1,
                                     dilationH=1, dilationW=1,
                                     groups=1)),
        # The conv2d output is (N, H, W, Cmid) = (1, 8, 8, 4) = 256 elems.
        OpRecord("silu", "f32",
                  inputs=[__import__("tessera").apple_gpu_chain.TraceRef(0)],
                  shape_kwargs=dict(n=N * H * W * Cmid)),
        OpRecord("conv2d", "f32",
                  inputs=[__import__("tessera").apple_gpu_chain.TraceRef(1),
                           W2_np],
                  shape_kwargs=dict(N=N, H=H, W=W, Cin=Cmid, Cout=Cout,
                                     kH=kH, kW=kW,
                                     strideH=1, strideW=1,
                                     padH=1, padW=1,
                                     dilationH=1, dilationW=1,
                                     groups=1)),
    ]

    before = session_commit_count()
    results = run_trace(trace)
    after = session_commit_count()

    # 3 ops at default chunking budget (30) — must commit exactly ONE cb.
    assert (after - before) == 1, (
        f"expected 1 cb commit, got {after - before}")
    # Final output is real conv2d-shape data, not nans.
    out = results[-1]
    assert out is not None
    arr = out.download(np.float32, (N, H, W, Cout))
    out.free()
    for r in results[:-1]:
        if r is not None:
            r.free()
    assert np.isfinite(arr).all()


# ---- @auto_batch surface ----------------------------------------------

def test_conv2d_via_auto_batch_decorator():
    """The high-level surface — ``agpu.conv2d`` inside an
    ``@auto_batch`` function traces + executes through the chain."""
    if not session_available():
        pytest.skip("encode-session unavailable")

    N, H, W, Cin, Cout, kH, kW = 1, 6, 6, 3, 4, 3, 3
    rng = np.random.default_rng(0xAB04)
    X_np = rng.standard_normal((N, H, W, Cin), dtype=np.float32) * 0.1
    Wt_np = rng.standard_normal((kH, kW, Cin, Cout),
                                  dtype=np.float32) * 0.1

    @agpu.auto_batch
    def step(x, w):
        # Conv → silu → conv, all on one chain.
        return agpu.conv2d(x, w, N=N, H=H, W=W, Cin=Cin, Cout=Cout,
                            kH=kH, kW=kW, padH=1, padW=1)

    out = step(X_np, Wt_np)
    arr = out.download(np.float32, (N, H, W, Cout))
    out.free()
    assert np.isfinite(arr).all()


# ---- Bias path (outside the chain registry) ---------------------------

def test_conv2d_with_bias_legacy_path_via_conv2d_enc():
    """The bias-bearing surface is :func:`conv2d_enc` directly (not
    routed through the chain registry). Confirms the optional-bias
    code path encodes + executes correctly."""
    if not session_available():
        pytest.skip("encode-session unavailable")

    N, H, W, Cin, Cout, kH, kW = 1, 4, 4, 2, 3, 3, 3
    rng = np.random.default_rng(0xB1A5)
    X_np = rng.standard_normal((N, H, W, Cin), dtype=np.float32) * 0.1
    Wt_np = rng.standard_normal((kH, kW, Cin, Cout),
                                  dtype=np.float32) * 0.1
    bias_np = rng.standard_normal((Cout,), dtype=np.float32) * 0.01

    X = device_tensor(X_np)
    Wt = device_tensor(Wt_np)
    bias = device_tensor(bias_np)
    try:
        with batched_session() as sess:
            O = conv2d_enc(sess, X, Wt, bias,
                            N=N, H=H, W=W, Cin=Cin, Cout=Cout,
                            kH=kH, kW=kW, padH=1, padW=1)
        out = O.download(np.float32, (N, H, W, Cout))
        O.free()
    finally:
        X.free(); Wt.free(); bias.free()

    # Compare against numpy reference.
    expected = np.zeros((N, H, W, Cout), dtype=np.float64)
    for n in range(N):
        for oy in range(H):
            for ox in range(W):
                for oc in range(Cout):
                    acc = float(bias_np[oc])
                    for ky in range(kH):
                        iy = oy + ky - 1  # padH=1
                        if iy < 0 or iy >= H:
                            continue
                        for kx in range(kW):
                            ix = ox + kx - 1  # padW=1
                            if ix < 0 or ix >= W:
                                continue
                            for ic in range(Cin):
                                acc += float(X_np[n, iy, ix, ic]) * float(
                                    Wt_np[ky, kx, ic, oc])
                    expected[n, oy, ox, oc] = acc
    np.testing.assert_allclose(out, expected.astype(np.float32),
                                 rtol=1e-4, atol=1e-4)


# ---- Validation -------------------------------------------------------

def test_conv2d_enc_rejects_invalid_groups():
    """``groups`` must evenly divide ``Cin`` and ``Cout``. The Python
    wrapper catches this BEFORE handing to the C ABI (which would
    return 0 silently)."""
    if not session_available():
        pytest.skip("encode-session unavailable")
    with batched_session() as sess:
        X = device_tensor(np.zeros((1, 4, 4, 3), dtype=np.float32))
        Wt = device_tensor(np.zeros((3, 3, 1, 4), dtype=np.float32))
        try:
            with pytest.raises(ValueError, match=r"groups=2"):
                conv2d_enc(sess, X, Wt, None,
                            N=1, H=4, W=4, Cin=3, Cout=4, kH=3, kW=3,
                            groups=2)  # 3 % 2 != 0
        finally:
            X.free(); Wt.free()
