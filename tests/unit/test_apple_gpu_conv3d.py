"""Apple GPU Tier-3 conv3d via im2col + a GPU MPSGraph batched matmul.

MPSGraph has no 3-D convolution node, so `tessera_apple_gpu_conv3d_{f32,f16}`
lower conv3d to the classic im2col + GEMM decomposition: spatial patches are
gathered into a per-group column matrix and the dominant GEMM runs on-GPU as a
single MPSGraph batched matmul (batch = groups, fp32 accumulation). Bias +
scatter run on the host. Validated against a numpy reference. NDHWC source,
DHWIO weights. See docs/apple_gpu_tier2_tier3_plan.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as R


def _ref_conv3d(X, W, bias, stride, padding, dilation, groups):
    """NDHWC X [N,D,H,W,Cin]; DHWIO weight [kD,kH,kW,Cin/groups,Cout]."""
    X = X.astype(np.float64)
    W = W.astype(np.float64)
    sD, sH, sW = (stride,) * 3 if isinstance(stride, int) else stride
    pD, pH, pW = (padding,) * 3 if isinstance(padding, int) else padding
    dD, dH, dW = (dilation,) * 3 if isinstance(dilation, int) else dilation
    N, iD, iH, iW, Cin = X.shape
    kD, kH, kW, cinG, Cout = W.shape
    oD = (iD + 2 * pD - dD * (kD - 1) - 1) // sD + 1
    oH = (iH + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    oW = (iW + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    coutG = Cout // groups
    Xp = np.pad(X, ((0, 0), (pD, pD), (pH, pH), (pW, pW), (0, 0)))
    O = np.zeros((N, oD, oH, oW, Cout), dtype=np.float64)
    for n in range(N):
        for od in range(oD):
            for oh in range(oH):
                for ow in range(oW):
                    for oc in range(Cout):
                        grp = oc // coutG
                        acc = float(bias[oc]) if bias is not None else 0.0
                        for kd in range(kD):
                            iz = od * sD + kd * dD
                            for ky in range(kH):
                                iy = oh * sH + ky * dH
                                for kx in range(kW):
                                    ix = ow * sW + kx * dW
                                    for ic in range(cinG):
                                        acc += (Xp[n, iz, iy, ix, grp * cinG + ic]
                                                * W[kd, ky, kx, ic, oc])
                        O[n, od, oh, ow, oc] = acc
    return O


# N, D, H, W, Cin, Cout, kD, kH, kW, stride, pad, dil, groups, bias
_CASES = [
    pytest.param((1, 4, 5, 5, 3, 4, 2, 3, 3, 1, 0, 1, 1, True), id="basic"),
    pytest.param((2, 5, 5, 4, 4, 6, 3, 3, 3, 1, 1, 1, 1, True), id="pad_bias"),
    pytest.param((1, 6, 6, 6, 2, 4, 2, 2, 2, 2, 0, 1, 1, False), id="stride2"),
    pytest.param((1, 7, 7, 7, 3, 3, 3, 3, 3, 1, 2, 2, 1, True), id="dilation2"),
    pytest.param((1, 4, 4, 4, 4, 4, 2, 3, 3, 1, 1, 1, 2, True), id="groups2"),
    pytest.param((1, 4, 4, 4, 4, 4, 1, 1, 1, 1, 0, 1, 4, False), id="depthwise"),
]


def _params(case):
    (N, D, H, W, Cin, Cout, kD, kH, kW,
     stride, pad, dil, groups, has_bias) = case
    rng = np.random.RandomState(abs(hash(case)) % (2**31))
    X = rng.randn(N, D, H, W, Cin).astype(np.float32)
    Wt = (rng.randn(kD, kH, kW, Cin // groups, Cout) * 0.3).astype(np.float32)
    bias = rng.randn(Cout).astype(np.float32) if has_bias else None
    kw = dict(stride=stride, padding=pad, dilation=dil, groups=groups)
    return X, Wt, bias, kw


@pytest.mark.parametrize("case", _CASES)
def test_conv3d_f32_matches_numpy(case):
    X, Wt, bias, kw = _params(case)
    ops = [X, Wt] + ([bias] if bias is not None else [])
    out = R._apple_gpu_dispatch_conv3d(ops, kw, np)
    assert out is not None
    ref = _ref_conv3d(X, Wt, bias, kw["stride"], kw["padding"],
                      kw["dilation"], kw["groups"])
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_conv3d_f16_native():
    N, D, H, W, Cin, Cout = 1, 4, 5, 5, 3, 4
    rng = np.random.RandomState(7)
    X = (rng.randn(N, D, H, W, Cin) * 0.5).astype(np.float16)
    Wt = (rng.randn(2, 3, 3, Cin, Cout) * 0.3).astype(np.float16)
    bias = (rng.randn(Cout) * 0.1).astype(np.float16)
    out = R._apple_gpu_dispatch_conv3d([X, Wt, bias],
                                       dict(stride=1, padding=1), np)
    assert out is not None and out.dtype == np.float16
    ref = _ref_conv3d(X.astype(np.float32), Wt.astype(np.float32),
                      bias.astype(np.float32), 1, 1, 1, 1)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=4e-2, atol=4e-2)


def test_conv3d_bf16_round_trip():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    bf16 = ml_dtypes.bfloat16
    rng = np.random.RandomState(11)
    Xf = (rng.randn(1, 4, 5, 5, 3) * 0.5).astype(np.float32)
    Wf = (rng.randn(2, 3, 3, 3, 4) * 0.3).astype(np.float32)
    out = R._apple_gpu_dispatch_conv3d([Xf.astype(bf16), Wf.astype(bf16)],
                                       dict(stride=1, padding=1), np)
    assert out is not None and out.dtype == bf16
    ref = _ref_conv3d(Xf, Wf, None, 1, 1, 1, 1)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=8e-2, atol=8e-2)


def test_conv3d_symbols_exported():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_conv3d_f32")
    assert hasattr(rt, "tessera_apple_gpu_conv3d_f16")
    assert R._apple_gpu_conv3d_f32() is not None
    assert R._apple_gpu_conv3d_f16() is not None


def test_conv3d_rejects_non_5d():
    X = np.zeros((4, 5, 5, 3), np.float32)   # 4-D (conv2d shape)
    Wt = np.zeros((2, 3, 3, 3, 4), np.float32)
    assert R._apple_gpu_dispatch_conv3d([X, Wt], {}, np) is None


def test_conv3d_in_runtime_envelope():
    assert "tessera.conv3d" in R._APPLE_GPU_CONV_OPS
    assert "tessera.conv3d" in R._APPLE_GPU_RUNTIME_OPS


def test_conv3d_matches_reference_ops_conv3d():
    """The GPU dispatcher agrees with the eager numpy ops.conv3d reference."""
    import tessera

    rng = np.random.RandomState(3)
    X = rng.randn(2, 5, 6, 6, 3).astype(np.float32)
    Wt = (rng.randn(3, 3, 3, 3, 5) * 0.3).astype(np.float32)
    bias = rng.randn(5).astype(np.float32)
    ref = tessera.ops.conv3d(X, Wt, bias=bias, stride=1, padding=1)
    out = R._apple_gpu_dispatch_conv3d([X, Wt, bias],
                                       dict(stride=1, padding=1), np)
    assert out is not None
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
