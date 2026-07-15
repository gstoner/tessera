"""Exact-device execution proof for the shipped NVIDIA MMA runtime lane."""
from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import require_nvidia_mma_runtime


def _matmul_artifact(rt):
    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_mma",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a", "b"], "output_name": "c",
        "ops": [{"op_name": "tessera.matmul", "result": "c",
                 "operands": ["a", "b"], "kwargs": {}}],
    })


@pytest.mark.parametrize("shape", [(16, 16, 16), (64, 48, 32), (128, 96, 64)])
@pytest.mark.hardware_nvidia
def test_launch_nvidia_mma_matmul_f16_matches_numpy(shape):
    rt = require_nvidia_mma_runtime()
    m, n, k = shape
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((m, k)) * 0.5).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.5).astype(np.float16)
    result = rt.launch(_matmul_artifact(rt), (a, b))
    assert result["ok"] is True, result.get("reason")
    assert result["runtime_status"] == "success"
    assert result["compiler_path"] == "nvidia_mma"
    assert result["execution_kind"] == "native_gpu"
    reference = a.astype(np.float32) @ b.astype(np.float32)
    max_error = float(np.max(np.abs(result["output"] - reference)))
    assert max_error < 1e-2, f"nvidia_mma launch{shape} maxerr={max_error}"


@pytest.mark.hardware_nvidia
def test_launch_nvidia_mma_matmul_bf16_matches_numpy():
    rt = require_nvidia_mma_runtime()
    bf16 = rt._bfloat16_dtype()
    if bf16 is None:
        pytest.skip("no bfloat16 dtype available")
    rng = np.random.default_rng(1)
    a = (rng.standard_normal((64, 48)) * 0.5).astype(bf16)
    b = (rng.standard_normal((48, 64)) * 0.5).astype(bf16)
    result = rt.launch(_matmul_artifact(rt), (a, b))
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == "nvidia_mma"
    reference = a.astype(np.float32) @ b.astype(np.float32)
    assert float(np.max(np.abs(result["output"] - reference))) < 2e-1


@pytest.mark.hardware_nvidia
def test_jit_nvidia_sm120_matmul_dispatches_to_shipped_symbol():
    import tessera

    rt = require_nvidia_mma_runtime()

    @tessera.jit(target="nvidia_sm120")
    def mm(a, b):
        return tessera.ops.matmul(a, b)

    artifact = mm.runtime_artifact()
    metadata = artifact.metadata or {}
    assert metadata.get("executable") is True
    assert metadata.get("compiler_path") == "nvidia_mma"
    assert metadata.get("execution_kind") == "native_gpu"

    rng = np.random.default_rng(3)
    a = (rng.standard_normal((128, 64)) * 0.4).astype(np.float16)
    b = (rng.standard_normal((64, 96)) * 0.4).astype(np.float16)
    result = rt.launch(artifact, (a, b))
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == "nvidia_mma"
    reference = a.astype(np.float32) @ b.astype(np.float32)
    assert float(np.max(np.abs(result["output"] - reference))) < 1e-2
