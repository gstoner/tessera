"""Apple GPU Tier-3 conv2d via the MPSGraph convolution2D node (2026-05-30).

`tessera_apple_gpu_conv2d_{f32,f16}` run an NHWC-source / HWIO-weight 2-D
convolution (full stride / padding / dilation / groups) through MPSGraph's
`convolution2DWithSourceTensor:weightsTensor:descriptor:`, with a portable
fp32 reference fallback when Metal is unavailable. Validated against a numpy
reference. See docs/audit/backend/apple/archive/apple_gpu_tier2_tier3_plan.md.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tessera import runtime as R

DARWIN = sys.platform == "darwin"


def _ref_conv2d(X, W, bias, stride, padding, dilation, groups):
    """NHWC X [N,H,W,Cin]; HWIO weight [kH,kW,Cin/groups,Cout]; bias [Cout]."""
    X = X.astype(np.float64)
    W = W.astype(np.float64)
    sH, sW = (stride, stride) if isinstance(stride, int) else stride
    pH, pW = (padding, padding) if isinstance(padding, int) else padding
    dH, dW = (dilation, dilation) if isinstance(dilation, int) else dilation
    N, H, Wd, Cin = X.shape
    kH, kW, cinG, Cout = W.shape
    outH = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    outW = (Wd + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    coutG = Cout // groups
    Xp = np.pad(X, ((0, 0), (pH, pH), (pW, pW), (0, 0)))
    O = np.zeros((N, outH, outW, Cout), dtype=np.float64)
    for n in range(N):
        for oy in range(outH):
            for ox in range(outW):
                for oc in range(Cout):
                    grp = oc // coutG
                    acc = float(bias[oc]) if bias is not None else 0.0
                    for ky in range(kH):
                        iy = oy * sH + ky * dH
                        for kx in range(kW):
                            ix = ox * sW + kx * dW
                            for ic in range(cinG):
                                acc += (Xp[n, iy, ix, grp * cinG + ic]
                                        * W[ky, kx, ic, oc])
                    O[n, oy, ox, oc] = acc
    return O


# N, H, W, Cin, Cout, kH, kW, stride, pad, dilation, groups, bias
_CASES = [
    pytest.param((1, 5, 5, 3, 4, 3, 3, 1, 0, 1, 1, True), id="basic_k3"),
    pytest.param((2, 7, 6, 4, 8, 3, 3, 1, 1, 1, 1, True), id="same_pad_bias"),
    pytest.param((1, 8, 8, 2, 6, 3, 3, 2, 1, 1, 1, False), id="stride2_nobias"),
    pytest.param((1, 9, 9, 3, 3, 3, 3, 1, 2, 2, 1, True), id="dilation2"),
    pytest.param((1, 6, 6, 4, 4, 3, 3, 1, 1, 1, 2, True), id="groups2"),
    pytest.param((1, 5, 5, 4, 4, 1, 1, 1, 0, 1, 4, False), id="depthwise_1x1"),
    pytest.param((2, 6, 5, 3, 5, 2, 3, 1, 0, 1, 1, True), id="rect_kernel"),
]


def _params(case):
    N, H, W, Cin, Cout, kH, kW, stride, pad, dil, groups, has_bias = case
    rng = np.random.RandomState(abs(hash(case)) % (2**31))
    X = rng.randn(N, H, W, Cin).astype(np.float32)
    Wt = (rng.randn(kH, kW, Cin // groups, Cout) * 0.3).astype(np.float32)
    bias = rng.randn(Cout).astype(np.float32) if has_bias else None
    kw = dict(stride=stride, padding=pad, dilation=dil, groups=groups)
    return X, Wt, bias, kw


@pytest.mark.parametrize("case", _CASES)
def test_conv2d_f32_matches_numpy(case):
    X, Wt, bias, kw = _params(case)
    ops = [X, Wt] + ([bias] if bias is not None else [])
    out = R._apple_gpu_dispatch_conv2d(ops, kw, np)
    assert out is not None
    ref = _ref_conv2d(X, Wt, bias, kw["stride"], kw["padding"],
                      kw["dilation"], kw["groups"])
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_conv2d_f16_native():
    N, H, W, Cin, Cout = 1, 6, 6, 3, 4
    rng = np.random.RandomState(7)
    X = (rng.randn(N, H, W, Cin) * 0.5).astype(np.float16)
    Wt = (rng.randn(3, 3, Cin, Cout) * 0.3).astype(np.float16)
    bias = (rng.randn(Cout) * 0.1).astype(np.float16)
    out = R._apple_gpu_dispatch_conv2d([X, Wt, bias],
                                       dict(stride=1, padding=1), np)
    assert out is not None and out.dtype == np.float16
    ref = _ref_conv2d(X.astype(np.float32), Wt.astype(np.float32),
                      bias.astype(np.float32), 1, 1, 1, 1)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=3e-2, atol=3e-2)


def test_conv2d_bf16_round_trip():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    bf16 = ml_dtypes.bfloat16
    rng = np.random.RandomState(11)
    Xf = (rng.randn(1, 5, 5, 3) * 0.5).astype(np.float32)
    Wf = (rng.randn(3, 3, 3, 4) * 0.3).astype(np.float32)
    out = R._apple_gpu_dispatch_conv2d([Xf.astype(bf16), Wf.astype(bf16)],
                                       dict(stride=1, padding=1), np)
    assert out is not None and out.dtype == bf16
    ref = _ref_conv2d(Xf, Wf, None, 1, 1, 1, 1)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=6e-2, atol=6e-2)


def test_conv2d_symbols_exported():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_conv2d_f32")
    assert hasattr(rt, "tessera_apple_gpu_conv2d_f16")
    assert R._apple_gpu_conv2d_f32() is not None
    assert R._apple_gpu_conv2d_f16() is not None


def test_conv2d_rejects_non_4d():
    X = np.zeros((5, 5, 3), np.float32)   # missing N
    Wt = np.zeros((3, 3, 3, 4), np.float32)
    assert R._apple_gpu_dispatch_conv2d([X, Wt], {}, np) is None


def test_conv2d_in_runtime_envelope():
    assert "tessera.conv2d" in R._APPLE_GPU_CONV_OPS
    assert "tessera.conv2d" in R._APPLE_GPU_RUNTIME_OPS


def test_conv2d_matches_reference_ops_conv2d():
    """The GPU dispatcher agrees with the eager numpy ops.conv2d reference."""
    import tessera

    rng = np.random.RandomState(3)
    X = rng.randn(2, 7, 7, 3).astype(np.float32)
    Wt = (rng.randn(3, 3, 3, 5) * 0.3).astype(np.float32)
    bias = rng.randn(5).astype(np.float32)
    ref = tessera.ops.conv2d(X, Wt, bias=bias, stride=1, padding=1)
    out = R._apple_gpu_dispatch_conv2d([X, Wt, bias],
                                       dict(stride=1, padding=1), np)
    assert out is not None
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
