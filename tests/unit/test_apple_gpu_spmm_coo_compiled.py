"""Apple GPU f32 COO SpMM adapter: host canonicalization, native CSR compute."""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera import ops as O
from tessera import runtime as rt


@ts.jit(target="apple_gpu")
def _jit_spmm_coo(coo, rhs):
    return ts.ops.spmm_coo(coo, rhs)


def _coo(dense: np.ndarray):
    rows, cols = np.nonzero(dense)
    return (np.ascontiguousarray(np.stack((rows, cols), axis=1), np.int64),
            np.ascontiguousarray(dense[rows, cols], np.float32), tuple(dense.shape))


def _art():
    return rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_spmm_coo_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a", "b"], "output_name": "o",
        "ops": [{"op_name": "tessera.spmm_coo", "result": "o",
                 "operands": ["a", "b"], "kwargs": {}}],
    })


def _launch(coo, rhs):
    result = rt.launch(_art(), (coo, rhs))
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == "apple_gpu_spmm_coo_compiled"
    return np.asarray(result["output"]), result


@pytest.mark.parametrize("shape", [(6, 5, 4), (3, 7, 9)])
def test_spmm_coo_f32_adapter_matches_dense_product(shape):
    m, k, n = shape
    rng = np.random.default_rng(m * 100 + n)
    dense = rng.standard_normal((m, k)).astype(np.float32)
    dense[rng.random((m, k)) < 0.6] = 0
    rhs = rng.standard_normal((k, n)).astype(np.float32)
    out, _ = _launch(_coo(dense), rhs)
    np.testing.assert_allclose(out, dense @ rhs, rtol=2e-5, atol=2e-5)


def test_spmm_coo_duplicate_coordinates_preserve_last_write_semantics():
    # The public COO oracle writes coordinates in input order; canonicalization
    # must retain the last value, not sum duplicate entries in the CSR kernel.
    coo = (np.array([[0, 1], [1, 2], [0, 1], [2, 0]], np.int64),
           np.array([2.0, -1.0, 4.5, 3.0], np.float32), (3, 3))
    rhs = np.arange(12, dtype=np.float32).reshape(3, 4)
    out, _ = _launch(coo, rhs)
    np.testing.assert_allclose(out, np.asarray(O.spmm_coo(coo, rhs)), rtol=0, atol=0)


def test_spmm_coo_jit_routes_through_the_apple_gpu_compiler_envelope():
    dense = np.array([[1, 0, 2], [0, -1, 0]], np.float32)
    rhs = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_allclose(np.asarray(_jit_spmm_coo(_coo(dense), rhs)), dense @ rhs)
    metadata = _jit_spmm_coo.runtime_artifact().metadata
    assert metadata["compiler_path"] == "apple_gpu_mps"
    assert metadata["execution_mode"] == "metal_runtime"
    assert "JIT_COMPILED_TARGET_RUNTIME" in metadata["diagnostics"][0]


def test_spmm_coo_execution_matrix_declares_native_compute_lane():
    from tessera.compiler.execution_matrix import lookup
    row = lookup("apple_gpu", "apple_gpu_spmm_coo_compiled")
    assert row is not None
    assert row.execution_kind == "native_gpu"


def test_spmm_coo_non_f32_uses_reference_cpu_override():
    dense = np.array([[1, 0], [0, 2]], np.float32)
    coo = _coo(dense)
    coo = (coo[0], coo[1].astype(np.float64), coo[2])
    rhs = np.eye(2, dtype=np.float64)
    out, result = _launch(coo, rhs)
    assert result["execution_kind"] == "reference_cpu"
    np.testing.assert_allclose(out, dense @ rhs)


@pytest.mark.hardware_apple_gpu
def test_spmm_coo_f32_reports_native_gpu_on_metal():
    dense = np.array([[1, 0, 2], [0, -1, 0]], np.float32)
    rhs = np.arange(12, dtype=np.float32).reshape(3, 4)
    out, result = _launch(_coo(dense), rhs)
    assert result["execution_kind"] == "native_gpu"
    assert result["execution_mode"] == "metal_runtime"
    np.testing.assert_allclose(out, dense @ rhs, rtol=2e-5, atol=2e-5)
