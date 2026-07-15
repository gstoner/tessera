"""Generated CUDA f32-accumulating reduction parity lane for sm_120."""

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


def _artifact(op: str, **kwargs):
    from tessera import runtime as rt
    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_reduce_compiled",
        "executable": True, "execution_kind": "native_gpu", "arg_names": ["x"],
        "output_name": "o", "ops": [{"op_name": op, "result": "o",
        "operands": ["x"], "kwargs": kwargs}],
    })


_NP = {"tessera.sum": np.sum, "tessera.mean": np.mean, "tessera.max": np.max,
       "tessera.min": np.min, "tessera.amax": np.max, "tessera.amin": np.min}


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("op", list(_NP))
@pytest.mark.parametrize("shape,axis", [((8, 64), -1), ((4, 130), -1),
                                          ((3, 5, 16), 1), ((6, 48), None)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_live_nvidia_reduce_matches_numpy(op, shape, axis, dtype):
    rt = _cuda_or_skip()
    rng = np.random.default_rng(300 + len(shape) + shape[-1])
    x = (rng.standard_normal(shape) * 1.5).astype(dtype)
    result = rt.launch(_artifact(op, axis=axis), (x,))
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_allclose(result["output"], _NP[op](x.astype(np.float32), axis=axis),
                               atol=2e-5, rtol=0)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
def test_live_nvidia_reduce_keepdims_and_nan_propagation():
    rt = _cuda_or_skip()
    x = np.array([[1.0, np.nan, 3.0], [2.0, 4.0, 1.0]], np.float32)
    out = rt.launch(_artifact("tessera.max", axis=-1, keepdims=True), (x,))["output"]
    assert out.shape == (2, 1)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(np.max(x, axis=-1, keepdims=True)))
    np.testing.assert_allclose(out[1], np.max(x, axis=-1, keepdims=True)[1], atol=1e-6)


def test_nvidia_reduce_rejects_non_float_without_gpu():
    from tessera import runtime as rt
    with pytest.raises(ValueError, match="f32/f16"):
        rt._execute_nvidia_compiled_reduce(_artifact("tessera.sum", axis=-1),
                                            (np.zeros((2, 8), np.int32),))
