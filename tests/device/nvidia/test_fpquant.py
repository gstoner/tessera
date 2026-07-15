"""Exact-device NVIDIA FP quantization execute/compare proofs."""
from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import nvidia_mma_runtime_available
from tessera import ops


def _artifact(rt, op, fmt):
    return rt.RuntimeArtifact(metadata={"target": "nvidia_sm120", "compiler_path": "nvidia_fpquant_compiled", "executable": True, "execution_kind": "native_gpu", "arg_names": ["x"], "output_name": "o", "ops": [{"op_name": op, "result": "o", "operands": ["x"], "kwargs": {"format": fmt}}]})


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and NVIDIA GPU")
@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("op,fn,formats", [("tessera.quantize_fp8", ops.quantize_fp8, ["e4m3", "e5m2"]), ("tessera.quantize_fp6", ops.quantize_fp6, ["e2m3", "e3m2"]), ("tessera.quantize_fp4", ops.quantize_fp4, ["e2m1"])])
def test_quantize_grid_matches_reference(op, fn, formats):
    from tessera import runtime as rt
    rng = np.random.default_rng(91 + len(op)); x = (rng.standard_normal((8, 16)) * 3).astype(np.float32)
    for fmt in formats:
        output = rt.launch(_artifact(rt, op, fmt), (x,))["output"]; reference, _ = fn(x, format=fmt); np.testing.assert_allclose(output, np.asarray(reference, np.float32), rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and NVIDIA GPU")
@pytest.mark.hardware_nvidia
def test_dequantize_is_native_shape_preserving_passthrough():
    from tessera import runtime as rt
    x = np.linspace(-3, 3, 33, dtype=np.float32); result = rt.launch(_artifact(rt, "tessera.dequantize_fp8", "e4m3"), (x,)); assert result["execution_kind"] == "native_gpu"; np.testing.assert_array_equal(result["output"], x)
