"""Native Apple GPU f32 CSR SpMM vertical slice.

The supported ABI is deliberately narrow: contiguous i64 CSR ``indptr`` /
``indices`` and f32 values, with a contiguous row-major f32 dense RHS.  The
kernel is genuinely sparse (one output element walks its CSR row); unsupported
contracts remain explicit CPU-reference overrides.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import tessera as ts
from tessera import runtime as rt


@ts.jit(target="apple_gpu")
def _jit_spmm_csr(csr, rhs):
    return ts.ops.spmm_csr(csr, rhs)


def _csr(dense: np.ndarray):
    indptr, indices, values = [0], [], []
    for row in dense:
        for col, value in enumerate(row):
            if value != 0:
                indices.append(col)
                values.append(value)
        indptr.append(len(indices))
    return (np.asarray(indptr, np.int64), np.asarray(indices, np.int64),
            np.asarray(values, np.float32), tuple(dense.shape))


def _art():
    return rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_spmm_csr_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a", "b"], "output_name": "o",
        "ops": [{"op_name": "tessera.spmm_csr", "result": "o",
                 "operands": ["a", "b"], "kwargs": {}}],
    })


def _launch(csr, rhs):
    result = rt.launch(_art(), (csr, rhs))
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == "apple_gpu_spmm_csr_compiled"
    return np.asarray(result["output"]), result


@pytest.mark.parametrize("shape", [(5, 7, 4), (3, 4, 9)])
def test_spmm_csr_f32_matches_dense_product(shape):
    m, k, n = shape
    rng = np.random.default_rng(m * 100 + n)
    dense = rng.standard_normal((m, k)).astype(np.float32)
    dense[rng.random((m, k)) < 0.55] = 0
    rhs = rng.standard_normal((k, n)).astype(np.float32)
    out, _ = _launch(_csr(dense), rhs)
    np.testing.assert_allclose(out, dense @ rhs, rtol=2e-5, atol=2e-5)


def test_spmm_csr_handles_empty_rows():
    dense = np.array([[0, 0, 0], [1, 0, -2], [0, 0, 0], [0, 3, 0]], np.float32)
    rhs = np.arange(15, dtype=np.float32).reshape(3, 5)
    out, _ = _launch(_csr(dense), rhs)
    np.testing.assert_allclose(out, dense @ rhs, rtol=0, atol=0)


def test_spmm_csr_execution_matrix_declares_native_base_lane():
    from tessera.compiler.execution_matrix import lookup
    row = lookup("apple_gpu", "apple_gpu_spmm_csr_compiled")
    assert row is not None
    assert row.execution_kind == "native_gpu"


def test_spmm_csr_is_in_the_apple_gpu_compiler_envelope():
    from tessera.compiler.apple_gpu_envelope import runtime_ops
    assert "tessera.spmm_csr" in runtime_ops()


def test_spmm_csr_jit_routes_through_the_apple_gpu_compiler_envelope():
    dense = np.array([[1, 0, 2], [0, -1, 0]], np.float32)
    rhs = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_allclose(np.asarray(_jit_spmm_csr(_csr(dense), rhs)), dense @ rhs)
    metadata = _jit_spmm_csr.runtime_artifact().metadata
    assert metadata["compiler_path"] == "apple_gpu_mps"
    assert metadata["execution_mode"] == "metal_runtime"
    assert "JIT_COMPILED_TARGET_RUNTIME" in metadata["diagnostics"][0]


def test_spmm_csr_non_f32_uses_reference_cpu_override():
    dense = np.array([[1, 0], [0, 2]], np.float64)
    csr = _csr(dense.astype(np.float32))
    csr = (csr[0], csr[1], csr[2].astype(np.float64), csr[3])
    rhs = np.eye(2, dtype=np.float64)
    out, result = _launch(csr, rhs)
    assert result["execution_kind"] == "reference_cpu"
    np.testing.assert_allclose(out, dense @ rhs)


def test_spmm_csr_strided_rhs_uses_reference_cpu_override():
    dense = np.array([[1, 0, 2], [0, -1, 0]], np.float32)
    rhs = np.arange(24, dtype=np.float32).reshape(3, 8)[:, ::2]
    assert not rhs.flags.c_contiguous
    out, result = _launch(_csr(dense), rhs)
    assert result["execution_kind"] == "reference_cpu"
    np.testing.assert_allclose(out, dense @ rhs)


def test_spmm_csr_f32_reports_native_gpu_on_metal():
    if sys.platform != "darwin" or not rt.DeviceTensor.is_metal():
        pytest.skip("requires an available Apple Metal runtime")
    dense = np.array([[1, 0, 2], [0, -1, 0]], np.float32)
    rhs = np.arange(12, dtype=np.float32).reshape(3, 4)
    out, result = _launch(_csr(dense), rhs)
    assert result["execution_kind"] == "native_gpu"
    assert result["execution_mode"] == "metal_runtime"
    np.testing.assert_allclose(out, dense @ rhs, rtol=2e-5, atol=2e-5)
