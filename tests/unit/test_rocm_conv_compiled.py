"""Convolution lane on AMD ROCm gfx1151 (P13 of S_SERIES_GAP_CLOSURE_PLAN) —
conv2d (NHWC/HWIO) + conv3d (NDHWC/DHWIO) via im2col + the COMPILER-GENERATED
WMMA GEMM (host lays out the patch matrix; f16/bf16 storage, f32 accumulate).
Reachable via `compiler_path="rocm_conv_compiled"`. Validated vs the numpy conv
reference on gfx1151 to WMMA-f16 tolerance. Skip-clean: tessera-opt not built /
no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, op, n_operands, kwargs):
    names = [f"a{i}" for i in range(n_operands)]
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_conv_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": kwargs}]})


def _run(rt, op, *arrs, **kwargs):
    res = rt.launch(_art(rt, op, len(arrs), kwargs), arrs)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_conv_compiled"
    return np.asarray(res["output"])


def _ref2d(X, W, bias=None, stride=1, padding=0):
    sH, sW = (stride, stride) if isinstance(stride, int) else stride
    pH, pW = (padding, padding) if isinstance(padding, int) else padding
    N, H, Wd, Cin = X.shape
    kH, kW, _, Cout = W.shape
    Xp = np.pad(X, ((0, 0), (pH, pH), (pW, pW), (0, 0))) if (pH or pW) else X
    oH = (H + 2 * pH - kH) // sH + 1
    oW = (Wd + 2 * pW - kW) // sW + 1
    Y = np.zeros((N, oH, oW, Cout), np.float32)
    for i in range(oH):
        for j in range(oW):
            patch = Xp[:, i * sH:i * sH + kH, j * sW:j * sW + kW, :]
            Y[:, i, j, :] = np.tensordot(patch, W, axes=([1, 2, 3], [0, 1, 2]))
    if bias is not None:
        Y = Y + bias.reshape(1, 1, 1, -1)
    return Y


_RNG = np.random.default_rng(37)


def test_conv2d_basic():
    rt = _rocm_or_skip()
    # smaller magnitudes keep the f16 im2col/GEMM quantization error bounded
    X = (0.5 * _RNG.standard_normal((2, 7, 7, 3))).astype(np.float32)
    W = (0.5 * _RNG.standard_normal((3, 3, 3, 8))).astype(np.float32)
    got = _run(rt, "tessera.conv2d", X, W)
    np.testing.assert_allclose(got, _ref2d(X, W), rtol=2e-2, atol=3e-2)


def test_conv2d_stride_pad_bias():
    rt = _rocm_or_skip()
    X = (0.5 * _RNG.standard_normal((2, 8, 8, 4))).astype(np.float32)
    W = (0.5 * _RNG.standard_normal((3, 3, 4, 8))).astype(np.float32)
    bias = _RNG.standard_normal((8,)).astype(np.float32)
    got = _run(rt, "tessera.conv2d", X, W, bias, stride=2, padding=1)
    np.testing.assert_allclose(got, _ref2d(X, W, bias, stride=2, padding=1),
                               rtol=2e-2, atol=3e-2)
