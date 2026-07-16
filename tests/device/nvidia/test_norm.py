"""Generated CUDA f32-accumulating RMSNorm / LayerNorm parity lane for sm_120."""
from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import require_nvidia_mma_runtime


def _artifact(op_name: str, eps: float | None = None):
    from tessera import runtime as rt
    return rt.RuntimeArtifact(metadata={"target": "nvidia_sm120", "compiler_path": "nvidia_norm_compiled", "executable": True, "execution_kind": "native_gpu", "arg_names": ["x"], "output_name": "o", "ops": [{"op_name": op_name, "result": "o", "operands": ["x"], "kwargs": {} if eps is None else {"eps": eps}}]})


def _ref(x: np.ndarray, op: str, eps: float) -> np.ndarray:
    x = x.astype(np.float32)
    if op == "tessera.layer_norm":
        mean = x.mean(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(((x - mean) ** 2).mean(axis=-1, keepdims=True) + eps)
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("op,eps", [("tessera.rmsnorm", 1e-5), ("tessera.rmsnorm_safe", 1e-6), ("tessera.layer_norm", 1e-5)])
@pytest.mark.parametrize("shape", [(1, 16), (8, 64), (4, 300), (32, 17), (2, 3, 48)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_live_nvidia_norm_matches_numpy(op, eps, shape, dtype):
    rt = require_nvidia_mma_runtime()
    rng = np.random.default_rng(200 + len(shape) + shape[-1])
    x = (rng.standard_normal(shape) * 2.0 + 0.5).astype(dtype)
    result = rt.launch(_artifact(op), (x,))
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_allclose(result["output"], _ref(x, op, eps).astype(dtype), atol=3e-5 if dtype == np.float32 else 5e-3, rtol=0)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
def test_live_nvidia_layer_norm_large_offset_small_variance():
    rt = require_nvidia_mma_runtime()
    rng = np.random.default_rng(2024)
    x = (1e4 + rng.standard_normal((4, 128))).astype(np.float32)
    out = rt.launch(_artifact("tessera.layer_norm"), (x,))["output"]
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, _ref(x, "tessera.layer_norm", 1e-5), atol=2e-3, rtol=0)
