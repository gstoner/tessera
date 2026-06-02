"""Sprint A (2026-06-01) — f16 / bf16 conv2d encode-session lanes.

Project 5 (2026-06-01) shipped the f32 encode-session conv2d. This
sprint extends it to the full 3-dtype matrix:
* ``tessera_apple_gpu_conv2d_dev_f16_enc`` — MPSGraph f16 (native on
  every recent macOS).
* ``tessera_apple_gpu_conv2d_dev_bf16_enc`` — MPSGraph bf16 (native
  on macOS 26+ / Apple Silicon M2+; falls back to fp32 conversion
  internally on older hosts via the existing ``mpsg_up``/``mpsg_down``
  helpers).

Tests pin:

* **Symbol availability** — both new C ABI symbols resolve from the
  runtime.
* **Registry surface** — ``("conv2d", "f16")`` and ``("conv2d",
  "bf16")`` are encode-eligible; the 3-dtype matrix is complete for
  conv2d.
* **f16 numerical equivalence** vs the legacy f16 host path at
  ``rtol=1e-2`` (fp16 has ~3 decimal digits).
* **bf16 numerical equivalence** vs an f32 reference at bf16
  tolerance (~1% — bf16 keeps fp32's range but only 7 mantissa
  bits). Test skips honestly if the host doesn't support bf16
  MPSGraph.
* **Output buffer sizing** — f16 / bf16 outputs are 2 bytes/elem
  (vs f32's 4), so DeviceTensor.nbytes must match.
* **Chain composition** — a 3-op chain (conv2d_f16 → silu_f16 →
  conv2d_f16) shares one cb.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

try:
    from ml_dtypes import bfloat16 as _ml_bfloat16
    _HAS_BF16 = True
except ImportError:
    _HAS_BF16 = False

from tessera._apple_gpu_dispatch import bind_symbol
from tessera.apple_gpu_batched import (
    batched_session,
    bf16_session_available,
    conv2d_enc_bf16,
    conv2d_enc_f16,
    device_tensor,
    session_available,
    session_commit_count,
)
from tessera.apple_gpu_chain import (
    ENCODE_OP_REGISTRY,
    is_encode_eligible,
)


# ---- Symbol availability -----------------------------------------------

def test_conv2d_dev_f16_enc_symbol_resolves():
    fn = bind_symbol(
        "tessera_apple_gpu_conv2d_dev_f16_enc",
        (ctypes.c_void_p,) * 5 + (ctypes.c_int32,) * 14,
        ctypes.c_int32)
    assert fn is not None


def test_conv2d_dev_bf16_enc_symbol_resolves():
    fn = bind_symbol(
        "tessera_apple_gpu_conv2d_dev_bf16_enc",
        (ctypes.c_void_p,) * 5 + (ctypes.c_int32,) * 14,
        ctypes.c_int32)
    assert fn is not None


# ---- Registry surface --------------------------------------------------

def test_conv2d_3_dtype_matrix_is_complete():
    """Sprint A — conv2d is now encode-eligible for f32 / f16 / bf16.
    Before Sprint A, conv2d was the only encode-eligible op without
    the full dtype matrix."""
    assert is_encode_eligible("conv2d", "f32")
    assert is_encode_eligible("conv2d", "f16")
    assert is_encode_eligible("conv2d", "bf16")
    # All other encode-eligible ops cover the same 3 dtypes; the
    # invariant "every op has all 3 dtypes" now holds for conv2d
    # too. Verify by counting per-dtype entries.
    f32_ops = {name for (name, d) in ENCODE_OP_REGISTRY if d == "f32"}
    f16_ops = {name for (name, d) in ENCODE_OP_REGISTRY if d == "f16"}
    bf16_ops = {name for (name, d) in ENCODE_OP_REGISTRY if d == "bf16"}
    # Symmetric: f32 ⊇ f16 ⊇ bf16, and conv2d ∈ all three.
    assert "conv2d" in f32_ops and "conv2d" in f16_ops and "conv2d" in bf16_ops
    assert f16_ops == bf16_ops, (
        f"f16 / bf16 dtype matrices diverged: "
        f"f16={f16_ops} bf16={bf16_ops}")
    assert f32_ops == f16_ops, (
        f"f32 / f16 dtype matrices diverged: "
        f"f32={f32_ops} f16={f16_ops}")


# ---- f16 numerical correctness vs legacy host path --------------------

def test_conv2d_f16_encode_matches_legacy_host_path():
    if not session_available():
        pytest.skip("encode-session unavailable")

    N, H, W, Cin, Cout, kH, kW = 1, 6, 6, 3, 4, 3, 3
    rng = np.random.default_rng(0xF16C0)
    X_f32 = rng.standard_normal((N, H, W, Cin), dtype=np.float32) * 0.1
    Wt_f32 = rng.standard_normal((kH, kW, Cin, Cout),
                                   dtype=np.float32) * 0.1
    X_f16 = X_f32.astype(np.float16)
    Wt_f16 = Wt_f32.astype(np.float16)

    X = device_tensor(X_f16)
    Wt = device_tensor(Wt_f16)
    try:
        outH = H + 2 - kH + 1
        outW = W + 2 - kW + 1
        with batched_session() as sess:
            O = conv2d_enc_f16(sess, X, Wt, None,
                                N=N, H=H, W=W, Cin=Cin, Cout=Cout,
                                kH=kH, kW=kW, padH=1, padW=1)
        # Output nbytes must be N*outH*outW*Cout * 2 (fp16).
        assert O.nbytes == N * outH * outW * Cout * 2
        got = O.download(np.float16, (N, outH, outW, Cout))
        O.free()
    finally:
        X.free(); Wt.free()

    # Legacy f16 host path.
    legacy = bind_symbol(
        "tessera_apple_gpu_conv2d_f16",
        (ctypes.c_void_p,) * 4 + (ctypes.c_int32,) * 14, None)
    assert legacy is not None
    out_legacy = np.empty((N, outH, outW, Cout), dtype=np.float16)
    legacy(X_f16.ctypes.data_as(ctypes.c_void_p),
            Wt_f16.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(0),
            out_legacy.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int32(N), ctypes.c_int32(H), ctypes.c_int32(W),
            ctypes.c_int32(Cin), ctypes.c_int32(Cout),
            ctypes.c_int32(kH), ctypes.c_int32(kW),
            ctypes.c_int32(1), ctypes.c_int32(1),
            ctypes.c_int32(1), ctypes.c_int32(1),
            ctypes.c_int32(1), ctypes.c_int32(1),
            ctypes.c_int32(1))

    # f16 has limited precision — both paths use MPSGraph so they
    # should produce identical bytes, but allow a tiny tolerance to
    # be robust across host versions.
    np.testing.assert_allclose(got.astype(np.float32),
                                 out_legacy.astype(np.float32),
                                 rtol=1e-2, atol=1e-2)


# ---- bf16 numerical correctness vs f32 reference ----------------------

@pytest.mark.skipif(not _HAS_BF16, reason="ml_dtypes not installed")
def test_conv2d_bf16_encode_matches_fp32_reference():
    if not session_available():
        pytest.skip("encode-session unavailable")
    if not bf16_session_available():
        pytest.skip("MPSGraph bf16 not supported on this host "
                     "(macOS<26 or pre-M2)")

    N, H, W, Cin, Cout, kH, kW = 1, 6, 6, 3, 4, 3, 3
    rng = np.random.default_rng(0xBF16C0)
    X_f32 = rng.standard_normal((N, H, W, Cin), dtype=np.float32) * 0.1
    Wt_f32 = rng.standard_normal((kH, kW, Cin, Cout),
                                   dtype=np.float32) * 0.1
    X_bf16 = X_f32.astype(_ml_bfloat16)
    Wt_bf16 = Wt_f32.astype(_ml_bfloat16)

    X = device_tensor(X_bf16)
    Wt = device_tensor(Wt_bf16)
    try:
        outH = H + 2 - kH + 1
        outW = W + 2 - kW + 1
        with batched_session() as sess:
            O = conv2d_enc_bf16(sess, X, Wt, None,
                                 N=N, H=H, W=W, Cin=Cin, Cout=Cout,
                                 kH=kH, kW=kW, padH=1, padW=1)
        assert O.nbytes == N * outH * outW * Cout * 2  # bf16 = 2 bytes
        got = O.download(_ml_bfloat16, (N, outH, outW, Cout))
        O.free()
    finally:
        X.free(); Wt.free()

    # bf16 reference — compute in f32 then cast to bf16 (matching
    # MPSGraph's internal up-cast).
    f32_ref = np.zeros((N, outH, outW, Cout), dtype=np.float32)
    for n in range(N):
        for oy in range(outH):
            for ox in range(outW):
                for oc in range(Cout):
                    acc = 0.0
                    for ky in range(kH):
                        iy = oy + ky - 1
                        if iy < 0 or iy >= H:
                            continue
                        for kx in range(kW):
                            ix = ox + kx - 1
                            if ix < 0 or ix >= W:
                                continue
                            for ic in range(Cin):
                                acc += float(X_bf16[n, iy, ix, ic]) * float(
                                    Wt_bf16[ky, kx, ic, oc])
                    f32_ref[n, oy, ox, oc] = acc

    got_f32 = got.astype(np.float32)
    # bf16 has ~7 mantissa bits → ~1% precision.
    np.testing.assert_allclose(got_f32, f32_ref, rtol=2e-2, atol=2e-2)


# ---- Chain composition -------------------------------------------------

def test_conv2d_f16_chain_commits_one_cb():
    """f16 conv2d + f16 silu + f16 conv2d → one cb commit."""
    if not session_available():
        pytest.skip("encode-session unavailable")

    from tessera.apple_gpu_chain import OpRecord, TraceRef, run_trace

    N, H, W, Cin, Cmid, Cout, kH, kW = 1, 6, 6, 3, 4, 4, 3, 3
    rng = np.random.default_rng(0xF16CAED)
    X_np = rng.standard_normal((N, H, W, Cin),
                                 dtype=np.float32).astype(np.float16) * 0.1
    W1_np = rng.standard_normal((kH, kW, Cin, Cmid),
                                  dtype=np.float32).astype(np.float16) * 0.1
    W2_np = rng.standard_normal((kH, kW, Cmid, Cout),
                                  dtype=np.float32).astype(np.float16) * 0.1

    trace = [
        OpRecord("conv2d", "f16",
                  inputs=[X_np, W1_np],
                  shape_kwargs=dict(N=N, H=H, W=W, Cin=Cin, Cout=Cmid,
                                     kH=kH, kW=kW,
                                     strideH=1, strideW=1,
                                     padH=1, padW=1,
                                     dilationH=1, dilationW=1,
                                     groups=1)),
        OpRecord("silu", "f16",
                  inputs=[TraceRef(0)],
                  shape_kwargs=dict(n=N * H * W * Cmid)),
        OpRecord("conv2d", "f16",
                  inputs=[TraceRef(1), W2_np],
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

    assert (after - before) == 1, (
        f"f16 chain should commit one cb, got {after - before}")
    out = results[-1]
    assert out is not None
    arr = out.download(np.float16, (N, H, W, Cout))
    out.free()
    for r in results[:-1]:
        if r is not None:
            r.free()
    assert np.isfinite(arr.astype(np.float32)).all()
