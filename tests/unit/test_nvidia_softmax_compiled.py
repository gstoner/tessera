"""Generated CUDA row-softmax parity lane for the live sm_120 target."""

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


def _artifact(op_name: str = "tessera.softmax", *, axis: int = -1):
    from tessera import runtime as rt
    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_softmax_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["x"],
                 "kwargs": {"axis": axis}}],
    })


def _ref(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("dtype,tol", [(np.float32, 2e-6), (np.float16, 4e-3)])
@pytest.mark.parametrize("shape", [(1, 16), (8, 64), (4, 300), (32, 17), (2, 3, 48)])
def test_live_nvidia_softmax_matches_numpy(dtype, tol, shape):
    rt = _cuda_or_skip()
    rng = np.random.default_rng(100 + len(shape) + shape[-1])
    x = (rng.standard_normal(shape) * 3.0).astype(dtype)
    result = rt.launch(_artifact(), (x,))
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == "nvidia_softmax_compiled"
    out = result["output"]
    np.testing.assert_allclose(out.astype(np.float32), _ref(x.astype(np.float32)), atol=tol, rtol=0)
    np.testing.assert_allclose(out.astype(np.float32).sum(axis=-1), np.ones(shape[:-1]), atol=tol * 3)


def test_nvidia_softmax_rejects_nonlast_axis_without_gpu():
    from tessera import runtime as rt
    with pytest.raises(ValueError, match="axis=-1"):
        rt._execute_nvidia_compiled_softmax(_artifact(axis=0), (np.zeros((4, 8), np.float32),))


def test_nvidia_softmax_rejects_unsupported_dtype_without_gpu():
    from tessera import runtime as rt
    with pytest.raises(ValueError, match="f32/f16"):
        rt._execute_nvidia_compiled_softmax(_artifact(), (np.zeros((4, 8), np.float64),))
