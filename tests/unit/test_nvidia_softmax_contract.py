"""Host-free NVIDIA softmax rejection contracts."""

from __future__ import annotations

import numpy as np
import pytest


def _artifact(*, axis: int = -1):
    from tessera import runtime as rt

    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_softmax_compiled",
        "executable": True, "execution_kind": "native_gpu", "arg_names": ["x"],
        "output_name": "o", "ops": [{"op_name": "tessera.softmax", "result": "o", "operands": ["x"], "kwargs": {"axis": axis}}],
    })


def test_nvidia_softmax_rejects_nonlast_axis_without_gpu():
    from tessera import runtime as rt

    with pytest.raises(ValueError, match="axis=-1"):
        rt._execute_nvidia_compiled_softmax(_artifact(axis=0), (np.zeros((4, 8), np.float32),))


def test_nvidia_softmax_rejects_unsupported_dtype_without_gpu():
    from tessera import runtime as rt

    with pytest.raises(ValueError, match="f32/f16"):
        rt._execute_nvidia_compiled_softmax(_artifact(), (np.zeros((4, 8), np.float64),))
