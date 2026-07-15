"""Native Apple GPU dense-block f32 BSMM via the existing matmul ABI.

The public BSMM primitive currently carries two dense operands and no BSR block
metadata.  This suite locks the honest dense-block interpretation; it does not
claim a general block-sparse kernel.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera import runtime as rt


@ts.jit(target="apple_gpu")
def _jit_bsmm(a, b):
    return ts.ops.bsmm(a, b)


def _art():
    return rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_bsmm_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a", "b"], "output_name": "o",
        "ops": [{"op_name": "tessera.bsmm", "result": "o",
                 "operands": ["a", "b"], "kwargs": {}}],
    })


def _launch(a, b):
    result = rt.launch(_art(), (a, b))
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == "apple_gpu_bsmm_compiled"
    return np.asarray(result["output"]), result


@pytest.mark.parametrize("shape", [(5, 7, 4), (3, 4, 9)])
def test_bsmm_dense_block_f32_matches_matmul(shape):
    m, k, n = shape
    rng = np.random.default_rng(m * 100 + n)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    out, _ = _launch(a, b)
    np.testing.assert_allclose(out, a @ b, rtol=2e-5, atol=2e-5)


def test_bsmm_jit_routes_through_the_apple_gpu_compiler_envelope():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.arange(20, dtype=np.float32).reshape(4, 5)
    np.testing.assert_allclose(np.asarray(_jit_bsmm(a, b)), a @ b)
    metadata = _jit_bsmm.runtime_artifact().metadata
    assert metadata["compiler_path"] == "apple_gpu_mps"
    assert metadata["execution_mode"] == "metal_runtime"
    assert "JIT_COMPILED_TARGET_RUNTIME" in metadata["diagnostics"][0]


def test_bsmm_execution_matrix_declares_native_dense_block_lane():
    from tessera.compiler.execution_matrix import lookup
    row = lookup("apple_gpu", "apple_gpu_bsmm_compiled")
    assert row is not None
    assert row.execution_kind == "native_gpu"


def test_bsmm_non_f32_uses_reference_cpu_override():
    a = np.ones((2, 3), np.float64)
    b = np.ones((3, 4), np.float64)
    out, result = _launch(a, b)
    assert result["execution_kind"] == "reference_cpu"
    np.testing.assert_allclose(out, a @ b)


@pytest.mark.hardware_apple_gpu
def test_bsmm_f32_reports_native_gpu_on_metal():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.arange(20, dtype=np.float32).reshape(4, 5)
    out, result = _launch(a, b)
    assert result["execution_kind"] == "native_gpu"
    assert result["execution_mode"] == "metal_runtime"
    np.testing.assert_allclose(out, a @ b, rtol=2e-5, atol=2e-5)


def test_bsmm_f32_demotes_to_reference_cpu_without_metal(monkeypatch):
    """No Metal device -> the f32 matmul ABI would resolve to the CPU reference
    symbol, so BSMM must fall back and report reference_cpu, never native_gpu."""
    monkeypatch.setattr(rt.DeviceTensor, "is_metal", staticmethod(lambda: False))
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.arange(20, dtype=np.float32).reshape(4, 5)
    out, result = _launch(a, b)
    assert result["execution_kind"] == "reference_cpu"
    np.testing.assert_allclose(out, a @ b, rtol=2e-5, atol=2e-5)
