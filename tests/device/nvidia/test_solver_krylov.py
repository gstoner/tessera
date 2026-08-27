"""Exact-SM120 device-resident Krylov and dedicated-CG certificates."""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.implicit_solver import (
    build_nvidia_dense_krylov_contract,
    build_nvidia_krylov_contract,
    package_nvidia_dense_krylov_solver,
    package_nvidia_krylov_solver,
)
from tests._support.nvidia import assert_native_gpu, require_nvidia_mma_runtime


pytestmark = [pytest.mark.slow, pytest.mark.hardware_nvidia]


def _storage(name: str, value: np.ndarray):
    if name == "f32":
        return value.astype(np.float32)
    if name == "f16":
        return value.astype(np.float16)
    ml_dtypes = pytest.importorskip("ml_dtypes")
    return value.astype(ml_dtypes.bfloat16)


@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
def test_device_resident_diagonal_cg_matches_oracle(storage: str) -> None:
    rt = require_nvidia_mma_runtime()
    diagonal_f32 = np.linspace(0.75, 3.5, 257, dtype=np.float32)
    rhs_f32 = np.sin(np.linspace(-2.0, 2.0, 257, dtype=np.float32))
    diagonal = _storage(storage, diagonal_f32)
    rhs = _storage(storage, rhs_f32)
    package = package_nvidia_krylov_solver(
        shape=diagonal.shape, storage=storage, algorithm="cg",
        tolerance=2.0e-6, max_iterations=128,
    )
    result = rt.launch(
        rt.RuntimeArtifact(metadata=package.runtime_metadata()),
        (diagonal, rhs),
    )
    assert_native_gpu(result)
    solution, residual, direction, matvec, info = result["output"]
    expected = rhs.astype(np.float32) / diagonal.astype(np.float32)
    tolerance = 2e-5 if storage == "f32" else (2e-3 if storage == "f16" else 1e-2)
    np.testing.assert_allclose(solution, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        diagonal.astype(np.float32) * solution - rhs.astype(np.float32),
        0.0, atol=max(tolerance, 3e-6),
    )
    assert residual.shape == direction.shape == matvec.shape == diagonal.shape
    assert info["algorithm"] == "cg"
    assert info["state_residency"] == "single_launch_device_resident"
    assert info["storage"] == storage and info["accumulation"] == "f32"
    assert info["converged"] is True
    assert 0 < info["iterations"] <= 128


def test_krylov_contract_fails_closed_for_algorithm_and_lineage() -> None:
    rt = require_nvidia_mma_runtime()
    with pytest.raises(ValueError, match="dedicated CG only"):
        build_nvidia_krylov_contract(shape=(17,), algorithm="gmres")
    package = package_nvidia_krylov_solver(shape=(17,))
    metadata = package.runtime_metadata()
    stale = dict(metadata)
    contract = dict(metadata["nvidia_krylov"])
    contract["artifact_hash"] = "0" * 64
    stale["nvidia_krylov"] = contract
    value = np.ones(17, dtype=np.float32)
    result = rt.launch(rt.RuntimeArtifact(metadata=stale), (value, value))
    assert result["ok"] is False
    assert "stale physical lineage" in result["reason"]


def test_krylov_rejects_non_spd_diagonal_before_launch() -> None:
    rt = require_nvidia_mma_runtime()
    package = package_nvidia_krylov_solver(shape=(4,))
    diagonal = np.array([1.0, 2.0, 0.0, 4.0], dtype=np.float32)
    result = rt.launch(
        rt.RuntimeArtifact(metadata=package.runtime_metadata()),
        (diagonal, np.ones_like(diagonal)),
    )
    assert result["ok"] is False
    assert "finite positive diagonal" in result["reason"]


def _tridiagonal(order: int, *, nonsymmetric: bool = False) -> np.ndarray:
    matrix = np.eye(order, dtype=np.float32) * np.float32(3.5)
    index = np.arange(order - 1)
    matrix[index, index + 1] = np.float32(0.35 if nonsymmetric else -0.75)
    matrix[index + 1, index] = np.float32(-0.55 if nonsymmetric else -0.75)
    return matrix


@pytest.mark.parametrize("algorithm", ["cg", "gmres"])
def test_cooperative_dense_krylov_matches_independent_solve_and_uses_multiple_ctas(
    algorithm: str,
) -> None:
    rt = require_nvidia_mma_runtime()
    order = 513
    matrix = _tridiagonal(order, nonsymmetric=algorithm == "gmres")
    expected = np.sin(np.linspace(-1.0, 2.0, order, dtype=np.float32))
    rhs = matrix @ expected
    package = package_nvidia_dense_krylov_solver(
        order=order, algorithm=algorithm, tolerance=2.0e-6,
        max_iterations=96, restart=16, reduction_ctas=8,
    )
    result = rt.launch(
        rt.RuntimeArtifact(metadata=package.runtime_metadata()), (matrix, rhs)
    )
    assert_native_gpu(result)
    solution, residual, matvec, info = result["output"]
    np.testing.assert_allclose(solution, expected, rtol=2e-5, atol=2e-5)
    true_residual = rhs - matrix @ solution
    np.testing.assert_allclose(residual, true_residual, rtol=2e-3, atol=2e-5)
    np.testing.assert_allclose(matvec, matrix @ solution, rtol=2e-5, atol=2e-5)
    assert info["algorithm"] == algorithm
    assert info["state_residency"] == "single_cooperative_launch_device_resident"
    assert info["reduction"] == "deterministic_multi_cta_two_level"
    assert info["reduction_ctas"] >= 2
    assert info["converged"] is True
    assert info["residual_norm"] <= 2.0e-6 * max(1.0, float(np.linalg.norm(rhs)))


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_dense_gmres_low_precision_storage_has_fp32_true_residual(storage: str) -> None:
    rt = require_nvidia_mma_runtime()
    order = 257
    matrix_f32 = _tridiagonal(order, nonsymmetric=True)
    expected = np.cos(np.linspace(-0.5, 1.5, order, dtype=np.float32))
    rhs_f32 = matrix_f32 @ expected
    matrix = _storage(storage, matrix_f32)
    rhs = _storage(storage, rhs_f32)
    package = package_nvidia_dense_krylov_solver(
        order=order, storage=storage, algorithm="gmres", tolerance=2.0e-4,
        max_iterations=96, restart=16, reduction_ctas=4,
    )
    result = rt.launch(
        rt.RuntimeArtifact(metadata=package.runtime_metadata()), (matrix, rhs)
    )
    assert_native_gpu(result)
    solution, residual, _matvec, info = result["output"]
    operator = matrix.astype(np.float32)
    stored_rhs = rhs.astype(np.float32)
    oracle = np.linalg.solve(operator.astype(np.float64), stored_rhs.astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(solution, oracle, rtol=7e-4, atol=7e-4)
    np.testing.assert_allclose(residual, stored_rhs - operator @ solution, atol=8e-5)
    assert info["storage"] == storage and info["accumulation"] == "f32"
    assert info["reduction_ctas"] >= 2


def test_dense_krylov_rejects_stale_lineage_and_invalid_operator_promises() -> None:
    rt = require_nvidia_mma_runtime()
    package = package_nvidia_dense_krylov_solver(order=17, algorithm="cg")
    stale = package.runtime_metadata()
    contract = dict(stale["nvidia_dense_krylov"])
    contract["artifact_hash"] = "0" * 64
    stale["nvidia_dense_krylov"] = contract
    matrix = np.eye(17, dtype=np.float32)
    rhs = np.ones(17, dtype=np.float32)
    result = rt.launch(rt.RuntimeArtifact(metadata=stale), (matrix, rhs))
    assert result["ok"] is False
    assert "stale physical lineage" in result["reason"]

    indefinite = np.eye(17, dtype=np.float32)
    indefinite[0, 0] = -1.0
    rhs = np.zeros(17, dtype=np.float32)
    rhs[0] = 1.0
    result = rt.launch(
        rt.RuntimeArtifact(metadata=package.runtime_metadata()), (indefinite, rhs)
    )
    assert result["ok"] is False
    assert "breakdown_or_non_spd" in result["reason"]


def test_dense_gmres_fails_closed_on_arnoldi_breakdown() -> None:
    rt = require_nvidia_mma_runtime()
    package = package_nvidia_dense_krylov_solver(
        order=33, algorithm="gmres", tolerance=1.0e-6,
        max_iterations=32, restart=8, reduction_ctas=2,
    )
    result = rt.launch(
        rt.RuntimeArtifact(metadata=package.runtime_metadata()),
        (np.zeros((33, 33), np.float32), np.ones(33, np.float32)),
    )
    assert result["ok"] is False
    assert "breakdown_or_non_spd" in result["reason"]


def test_dense_krylov_contract_records_reduction_and_orthogonalization() -> None:
    contract = build_nvidia_dense_krylov_contract(
        order=1025, algorithm="gmres", reduction_ctas=7, restart=24,
    )
    assert contract["reduction"] == "deterministic_multi_cta_two_level"
    assert contract["requested_reduction_ctas"] == 7
    assert contract["orthogonalization"] == "twice_modified_gram_schmidt"
    assert contract["numeric_policy"]["residual_check"] == "fp32_true_residual"
