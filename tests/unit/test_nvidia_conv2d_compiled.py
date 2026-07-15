"""Guarded direct NHWC/HWIO convolution on NVIDIA sm_120."""

from __future__ import annotations

import numpy as np
import pytest


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
