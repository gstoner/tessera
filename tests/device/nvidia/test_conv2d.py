"""Native NVIDIA NHWC/HWIO convolution execute-and-compare proofs."""

from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import require_nvidia_mma_runtime


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
    rt = require_nvidia_mma_runtime()
    rt._nvidia_conv2d_route_cache.clear()
    rt._nvidia_conv2d_route_evidence.clear()
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
    assert set(rt._nvidia_conv2d_route_cache.values()) <= {"direct", "shared"}
