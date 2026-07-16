"""Generated CUDA row-softmax parity lane for the live sm_120 target."""

from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import require_nvidia_mma_runtime


def _artifact():
    from tessera import runtime as rt

    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_softmax_compiled",
        "executable": True, "execution_kind": "native_gpu", "arg_names": ["x"],
        "output_name": "o", "ops": [{"op_name": "tessera.softmax", "result": "o", "operands": ["x"], "kwargs": {"axis": -1}}],
    })


def _ref(x: np.ndarray) -> np.ndarray:
    exponent = np.exp(x - x.max(axis=-1, keepdims=True))
    return exponent / exponent.sum(axis=-1, keepdims=True)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("dtype,tol", [(np.float32, 2e-6), (np.float16, 4e-3)])
@pytest.mark.parametrize("shape", [(1, 16), (8, 64), (4, 300), (32, 17), (2, 3, 48)])
def test_live_nvidia_softmax_matches_numpy(dtype, tol, shape):
    rt = require_nvidia_mma_runtime()
    rng = np.random.default_rng(100 + len(shape) + shape[-1])
    x = (rng.standard_normal(shape) * 3.0).astype(dtype)
    result = rt.launch(_artifact(), (x,))
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == "nvidia_softmax_compiled"
    out = result["output"]
    np.testing.assert_allclose(out.astype(np.float32), _ref(x.astype(np.float32)), atol=tol, rtol=0)
    np.testing.assert_allclose(out.astype(np.float32).sum(axis=-1), np.ones(shape[:-1]), atol=tol * 3)
