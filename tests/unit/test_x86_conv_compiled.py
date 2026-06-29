"""Convolution lane on x86 AVX-512 (P13 of S_SERIES_GAP_CLOSURE_PLAN) — conv2d
(NHWC/HWIO) + conv3d (NDHWC/DHWIO) via im2col + the AVX-512 f32 GEMM (host lays
out the patch matrix; the device runs the GEMM). Reachable via
`compiler_path="x86_conv_compiled"`. Validated vs the numpy conv reference.
Skip-clean: libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op, n_operands, kwargs):
    names = [f"a{i}" for i in range(n_operands)]
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_conv_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": kwargs}]})


def _run(rt, op, *arrs, **kwargs):
    res = rt.launch(_art(rt, op, len(arrs), kwargs), arrs)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_conv_compiled"
    return np.asarray(res["output"])


def _ref2d(X, W, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """numpy NHWC/HWIO conv2d reference."""
    def _pair(v):
        return (v, v) if isinstance(v, int) else tuple(v)
    sH, sW = _pair(stride)
    pH, pW = _pair(padding)
    dH, dW = _pair(dilation)
    N, H, Wd, Cin = X.shape
    kH, kW, cinG, Cout = W.shape
    Xp = np.pad(X, ((0, 0), (pH, pH), (pW, pW), (0, 0))) if (pH or pW) else X
    outH = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    outW = (Wd + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    Y = np.zeros((N, outH, outW, Cout), np.float32)
    cinPerG, coutPerG = Cin // groups, Cout // groups
    for g in range(groups):
        ci0, co0 = g * cinPerG, g * coutPerG
        for i in range(outH):
            for j in range(outW):
                hs, ws = i * sH, j * sW
                patch = Xp[:, hs:hs + dH * (kH - 1) + 1:dH,
                           ws:ws + dW * (kW - 1) + 1:dW, ci0:ci0 + cinPerG]
                wg = W[:, :, :, co0:co0 + coutPerG]
                Y[:, i, j, co0:co0 + coutPerG] = np.tensordot(
                    patch, wg, axes=([1, 2, 3], [0, 1, 2]))
    if bias is not None:
        Y = Y + bias.reshape(1, 1, 1, -1)
    return Y


_RNG = np.random.default_rng(37)


def test_conv2d_basic():
    rt = _rt_or_skip()
    X = _RNG.standard_normal((2, 7, 7, 3)).astype(np.float32)
    W = _RNG.standard_normal((3, 3, 3, 5)).astype(np.float32)
    got = _run(rt, "tessera.conv2d", X, W)
    np.testing.assert_allclose(got, _ref2d(X, W), rtol=1e-4, atol=1e-4)


def test_conv2d_stride_pad_bias():
    rt = _rt_or_skip()
    X = _RNG.standard_normal((2, 8, 8, 4)).astype(np.float32)
    W = _RNG.standard_normal((3, 3, 4, 6)).astype(np.float32)
    bias = _RNG.standard_normal((6,)).astype(np.float32)
    got = _run(rt, "tessera.conv2d", X, W, bias,
               stride=2, padding=1)
    np.testing.assert_allclose(
        got, _ref2d(X, W, bias, stride=2, padding=1), rtol=1e-4, atol=1e-4)


def test_conv2d_dilation():
    rt = _rt_or_skip()
    X = _RNG.standard_normal((1, 9, 9, 2)).astype(np.float32)
    W = _RNG.standard_normal((3, 3, 2, 4)).astype(np.float32)
    got = _run(rt, "tessera.conv2d", X, W, dilation=2)
    np.testing.assert_allclose(got, _ref2d(X, W, dilation=2),
                               rtol=1e-4, atol=1e-4)


def test_conv2d_groups():
    rt = _rt_or_skip()
    X = _RNG.standard_normal((2, 6, 6, 4)).astype(np.float32)
    W = _RNG.standard_normal((3, 3, 2, 6)).astype(np.float32)   # groups=2
    got = _run(rt, "tessera.conv2d", X, W, groups=2)
    np.testing.assert_allclose(got, _ref2d(X, W, groups=2),
                               rtol=1e-4, atol=1e-4)


def _ref3d(X, W, bias=None, stride=1, padding=0):
    """numpy NDHWC/DHWIO conv3d reference (independent tensordot loop)."""
    s = (stride,) * 3 if isinstance(stride, int) else tuple(stride)
    p = (padding,) * 3 if isinstance(padding, int) else tuple(padding)
    N, D, H, Wd, Cin = X.shape
    kD, kH, kW, cinG, Cout = W.shape
    Xp = np.pad(X, ((0, 0), (p[0], p[0]), (p[1], p[1]),
                    (p[2], p[2]), (0, 0))) if any(p) else X
    oD = (D + 2 * p[0] - kD) // s[0] + 1
    oH = (H + 2 * p[1] - kH) // s[1] + 1
    oW = (Wd + 2 * p[2] - kW) // s[2] + 1
    Y = np.zeros((N, oD, oH, oW, Cout), np.float32)
    for a in range(oD):
        for b in range(oH):
            for c in range(oW):
                patch = Xp[:, a * s[0]:a * s[0] + kD, b * s[1]:b * s[1] + kH,
                           c * s[2]:c * s[2] + kW, :]
                Y[:, a, b, c, :] = np.tensordot(
                    patch, W, axes=([1, 2, 3, 4], [0, 1, 2, 3]))
    if bias is not None:
        Y = Y + bias.reshape(1, 1, 1, 1, -1)
    return Y


def test_conv3d_basic():
    rt = _rt_or_skip()
    X = _RNG.standard_normal((2, 5, 5, 5, 2)).astype(np.float32)
    W = _RNG.standard_normal((3, 3, 3, 2, 4)).astype(np.float32)
    got = _run(rt, "tessera.conv3d", X, W, padding=1)
    np.testing.assert_allclose(got, _ref3d(X, W, padding=1),
                               rtol=1e-4, atol=1e-4)


def test_conv3d_stride():
    rt = _rt_or_skip()
    X = _RNG.standard_normal((1, 6, 6, 6, 3)).astype(np.float32)
    W = _RNG.standard_normal((2, 2, 2, 3, 5)).astype(np.float32)
    got = _run(rt, "tessera.conv3d", X, W, stride=2)
    np.testing.assert_allclose(got, _ref3d(X, W, stride=2),
                               rtol=1e-4, atol=1e-4)
