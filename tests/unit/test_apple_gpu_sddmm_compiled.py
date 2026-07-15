"""Native Apple GPU f32 SDDMM: sampled dense-dense multiplication."""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera import runtime as rt


@ts.jit(target="apple_gpu")
def _jit_sddmm(a, b, mask):
    return ts.ops.sddmm(a, b, mask)


def _art():
    return rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_sddmm_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a", "b", "m"], "output_name": "o",
        "ops": [{"op_name": "tessera.sddmm", "result": "o",
                 "operands": ["a", "b", "m"], "kwargs": {}}],
    })


def _launch(a, b, mask):
    result = rt.launch(_art(), (a, b, mask))
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == "apple_gpu_sddmm_compiled"
    return np.asarray(result["output"]), result


@pytest.mark.parametrize("shape", [(5, 7, 4), (3, 4, 9)])
def test_sddmm_f32_matches_masked_dense_product(shape):
    m, k, n = shape
    rng = np.random.default_rng(m * 100 + n)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    mask = (rng.random((m, n)) > 0.65).astype(np.float32)
    out, _ = _launch(a, b, mask)
    np.testing.assert_allclose(out, (a @ b) * mask, rtol=2e-5, atol=2e-5)


def test_sddmm_preserves_nonbinary_mask_weights_and_zero_samples():
    a = np.array([[1, 2], [-1, 3]], np.float32)
    b = np.array([[2, -4, 5], [3, 1, -2]], np.float32)
    mask = np.array([[0, -0.5, 2], [1.25, 0, -3]], np.float32)
    out, _ = _launch(a, b, mask)
    np.testing.assert_allclose(out, (a @ b) * mask, rtol=0, atol=0)


def test_sddmm_jit_routes_through_the_apple_gpu_compiler_envelope():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.arange(20, dtype=np.float32).reshape(4, 5)
    mask = np.eye(3, 5, dtype=np.float32)
    np.testing.assert_allclose(np.asarray(_jit_sddmm(a, b, mask)), (a @ b) * mask)
    metadata = _jit_sddmm.runtime_artifact().metadata
    assert metadata["compiler_path"] == "apple_gpu_mps"
    assert metadata["execution_mode"] == "metal_runtime"
    assert "JIT_COMPILED_TARGET_RUNTIME" in metadata["diagnostics"][0]


def test_sddmm_execution_matrix_declares_native_lane():
    from tessera.compiler.execution_matrix import lookup
    row = lookup("apple_gpu", "apple_gpu_sddmm_compiled")
    assert row is not None
    assert row.execution_kind == "native_gpu"


def test_sddmm_non_f32_uses_reference_cpu_override():
    a = np.ones((2, 3), np.float64)
    b = np.ones((3, 4), np.float64)
    mask = np.ones((2, 4), np.float64)
    out, result = _launch(a, b, mask)
    assert result["execution_kind"] == "reference_cpu"
    np.testing.assert_allclose(out, (a @ b) * mask)


@pytest.mark.hardware_apple_gpu
def test_sddmm_f32_reports_native_gpu_on_metal():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.arange(20, dtype=np.float32).reshape(4, 5)
    mask = np.eye(3, 5, dtype=np.float32)
    out, result = _launch(a, b, mask)
    assert result["execution_kind"] == "native_gpu"
    assert result["execution_mode"] == "metal_runtime"
    np.testing.assert_allclose(out, (a @ b) * mask, rtol=2e-5, atol=2e-5)
