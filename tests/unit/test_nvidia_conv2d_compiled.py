"""Guarded direct NHWC/HWIO convolution on NVIDIA sm_120."""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest


def _cuda_or_skip():
    if not (shutil.which("nvcc") or os.path.exists("/usr/local/cuda/bin/nvcc")):
        pytest.skip("nvcc not installed")
    from tessera import runtime as rt
    if not rt._nvidia_mma_runtime_available():
        pytest.skip("no usable NVIDIA CUDA device")
    return rt


def _artifact(*, bias: bool, stride=1, padding=0, dilation=1, groups=1,
              route=None):
    from tessera import runtime as rt
    names = ["x", "w"] + (["bias"] if bias else [])
    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_conv2d_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "y",
        "ops": [{"op_name": "tessera.conv2d", "result": "y",
                 "operands": names,
                 "kwargs": {"stride": stride, "padding": padding,
                            "dilation": dilation, "groups": groups,
                            **({"route": route} if route else {})}}],
    })


def _pair(v):
    return v if isinstance(v, tuple) else (v, v)


def _reference(x, w, bias, stride, padding, dilation):
    sh, sw = _pair(stride); ph, pw = _pair(padding); dh, dw = _pair(dilation)
    B, IH, IW, CI = x.shape
    KH, KW, _, CO = w.shape
    OH = (IH + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    OW = (IW + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    out = np.empty((B, OH, OW, CO), np.float32)
    for b in range(B):
        for oh in range(OH):
            for ow in range(OW):
                acc = np.zeros(CO, np.float32) if bias is None else bias.copy()
                for kh in range(KH):
                    ih = oh * sh - ph + kh * dh
                    if not 0 <= ih < IH: continue
                    for kw in range(KW):
                        iw = ow * sw - pw + kw * dw
                        if 0 <= iw < IW:
                            acc += x[b, ih, iw] @ w[kh, kw]
                out[b, oh, ow] = acc
    return out


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize(
    "shape,kernel,stride,padding,dilation,bias",
    [((1, 8, 7, 3), (3, 3, 3, 5), 1, 1, 1, False),
     ((2, 9, 8, 4), (2, 3, 4, 6), (2, 1), (1, 0), 1, True),
     ((1, 11, 10, 2), (3, 2, 2, 4), 1, (2, 1), (2, 2), True)],
)
def test_live_nvidia_conv2d_matches_reference(
    shape, kernel, stride, padding, dilation, bias
):
    rt = _cuda_or_skip()
    rng = np.random.default_rng(sum(shape) + sum(kernel))
    x = (rng.standard_normal(shape) * .2).astype(np.float32)
    w = (rng.standard_normal(kernel) * .2).astype(np.float32)
    b = (rng.standard_normal(kernel[-1]) * .1).astype(np.float32) if bias else None
    inputs = (x, w, b) if b is not None else (x, w)
    result = rt.launch(_artifact(
        bias=bias, stride=stride, padding=padding, dilation=dilation), inputs)
    assert result["ok"] is True, result.get("reason")
    expected = _reference(x, w, b, stride, padding, dilation)
    np.testing.assert_allclose(result["output"], expected, rtol=2e-5, atol=2e-6)


def test_nvidia_conv2d_rejects_groups_before_cuda():
    from tessera import runtime as rt
    x = np.zeros((1, 4, 4, 4), np.float32)
    w = np.zeros((3, 3, 2, 4), np.float32)
    with pytest.raises(ValueError, match="groups=1"):
        rt._execute_nvidia_conv2d_compiled(
            _artifact(bias=False, groups=2), (x, w))


def test_nvidia_conv2d_rejects_bad_channels_before_cuda():
    from tessera import runtime as rt
    x = np.zeros((1, 4, 4, 3), np.float32)
    w = np.zeros((3, 3, 2, 4), np.float32)
    with pytest.raises(ValueError, match="channels"):
        rt._execute_nvidia_conv2d_compiled(
            _artifact(bias=False), (x, w))


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.performance
@pytest.mark.parametrize("route,atol", [
    ("direct", 2e-5), ("shared", 2e-5), ("im2col_tf32", 2e-2)])
def test_live_nvidia_conv2d_performance_routes(route, atol):
    rt = _cuda_or_skip()
    rng = np.random.default_rng(1400 + len(route))
    x = (rng.standard_normal((1, 12, 11, 8)) * .1).astype(np.float32)
    w = (rng.standard_normal((3, 3, 8, 16)) * .1).astype(np.float32)
    bias = (rng.standard_normal((16,)) * .05).astype(np.float32)
    result = rt.launch(
        _artifact(bias=True, padding=1, route=route), (x, w, bias))
    assert result["ok"] is True, result.get("reason")
    expected = _reference(x, w, bias, 1, 1, 1)
    np.testing.assert_allclose(result["output"], expected, rtol=0, atol=atol)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.performance
def test_live_nvidia_conv2d_size_aware_dispatch_records_measurements():
    rt = _cuda_or_skip()
    rt._nvidia_conv2d_route_cache.clear()
    rt._nvidia_conv2d_route_evidence.clear()
    rng = np.random.default_rng(1500)
    x = (rng.standard_normal((1, 16, 16, 8)) * .1).astype(np.float32)
    w = (rng.standard_normal((3, 3, 8, 16)) * .1).astype(np.float32)
    result = rt.launch(_artifact(bias=False, padding=1), (x, w))
    assert result["ok"] is True
    assert len(rt._nvidia_conv2d_route_cache) == 1
    evidence = next(iter(rt._nvidia_conv2d_route_evidence.values()))
    assert set(evidence) == {"direct", "shared", "im2col_tf32"}
    winner = next(iter(rt._nvidia_conv2d_route_cache.values()))
    valid_names = {"direct", "shared", "im2col_tf32"}
    assert winner in valid_names and all(v > 0 for v in evidence.values())
