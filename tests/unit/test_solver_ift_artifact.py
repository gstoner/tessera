from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.implicit_solver import (
    build_solver_ift_contract,
    diagonal_sqrt_ift_reference,
    lower_scheduled_solver_ift,
)
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt


def test_solver_ift_contract_is_content_addressed_and_fail_closed() -> None:
    first = build_solver_ift_contract(target="x86", shape=(64,))
    second = build_solver_ift_contract(target="x86", shape=(64,))
    rocm = build_solver_ift_contract(target="rocm_gfx1151", shape=(64,))
    assert first == second
    assert first["artifact_hash"] != rocm["artifact_hash"]
    assert first["linear_solve"]["materializes_matrix"] is False
    with pytest.raises(ValueError, match="fail closed"):
        build_solver_ift_contract(target="rocm_gfx1200", shape=(64,))


def test_diagonal_sqrt_reference_exposes_all_ift_phases() -> None:
    theta = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    solution = np.sqrt(theta)
    cotangent = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    residual, linear_solution, parameter_cotangent = diagonal_sqrt_ift_reference(theta, solution, cotangent)
    np.testing.assert_array_equal(residual, np.zeros_like(theta))
    np.testing.assert_allclose(linear_solution, cotangent / (2.0 * solution))
    np.testing.assert_array_equal(parameter_cotangent, linear_solution)


def test_solver_forward_product_has_distinct_nontransposed_lineage() -> None:
    vjp = build_solver_ift_contract(target="x86", shape=(17,))
    jvp = build_solver_ift_contract(target="x86", shape=(17,), product_mode="jvp")
    assert jvp["product_mode"] == "jvp"
    assert jvp["linear_solve"]["transpose"] is False
    assert jvp["artifact_hash"] != vjp["artifact_hash"]
    with pytest.raises(ValueError, match="product_mode"):
        build_solver_ift_contract(target="x86", shape=(17,), product_mode="hvp")


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
def test_solver_forward_product_consumes_residual_jvp(target: str) -> None:
    artifact = lower_scheduled_solver_ift(
        target=target, shape=(17,), product_mode="jvp"
    )
    artifact.validate()
    assert "tessera_solver.residual_jvp" in artifact.shared_solver_ir
    assert 'product_mode = "jvp"' in artifact.schedule_ir
    assert "transpose = false" in artifact.schedule_ir


@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
def test_solver_forward_product_executes_parameter_tangent(target: str) -> None:
    from tessera import runtime as rt

    if find_tessera_opt() is None:
        pytest.skip("production tessera-opt unavailable")
    if target == "x86" and not rt._x86_elementwise_available():
        pytest.skip("production x86 AVX-512 image unavailable")
    if target == "rocm_gfx1151" and (
        rt._rocm_chip() != "gfx1151" or not rt._rocm_wmma_runtime_available()
    ):
        pytest.skip("exact gfx1151 runtime unavailable")
    parameter = np.linspace(0.5, 8.5, 257, dtype=np.float32)
    solution = np.sqrt(parameter).astype(np.float32)
    dparameter = np.linspace(-1.0, 1.0, 257, dtype=np.float32)
    artifact = lower_scheduled_solver_ift(
        target=target, shape=parameter.shape, product_mode="jvp"
    )
    result = rt.launch(
        rt.RuntimeArtifact(metadata=artifact.runtime_metadata()),
        (parameter, solution, dparameter),
    )
    assert result["ok"], result.get("reason")
    residual, solution_tangent, parameter_product = result["output"]
    np.testing.assert_allclose(residual, 0.0, atol=2e-6)
    expected = dparameter / (2.0 * solution)
    np.testing.assert_allclose(solution_tangent, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(parameter_product, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
def test_solver_ift_consumes_shared_chain_into_one_tile_artifact(target: str) -> None:
    artifact = lower_scheduled_solver_ift(target=target, shape=(64,))
    artifact.validate()
    assert artifact.shared_solver_ir.count("tessera_solver.linear_solve") == 1
    assert artifact.tile_ir.count("tile.solver_ift_kernel") == 1
    assert artifact.artifact_hash in artifact.tile_ir


@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
def test_solver_ift_compiled_package_matches_numerical_oracle(target: str) -> None:
    from tessera import runtime as rt

    if find_tessera_opt() is None:
        pytest.skip("production tessera-opt unavailable")
    if target == "x86" and not rt._x86_elementwise_available():
        pytest.skip("production x86 AVX-512 image unavailable")
    if target == "rocm_gfx1151" and not rt._rocm_wmma_runtime_available():
        pytest.skip("gfx1151 HIP runtime unavailable")

    rng = np.random.default_rng(20260809)
    parameter = rng.uniform(0.25, 16.0, size=(3, 257)).astype(np.float32)
    solution = np.sqrt(parameter).astype(np.float32)
    cotangent = rng.standard_normal(parameter.shape).astype(np.float32)
    scheduled = lower_scheduled_solver_ift(target=target, shape=parameter.shape)
    result = rt.launch(
        rt.RuntimeArtifact(metadata=scheduled.runtime_metadata()),
        (parameter, solution, cotangent),
    )
    assert result["ok"] is True, result.get("reason")
    expected = diagonal_sqrt_ift_reference(parameter, solution, cotangent)
    for actual, reference in zip(result["output"], expected):
        np.testing.assert_allclose(np.asarray(actual), reference, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
def test_solver_ift_tile_artifacts_reach_architecture_owned_lowering() -> None:
    tool = find_tessera_opt()
    assert tool is not None
    x86 = lower_scheduled_solver_ift(target="x86", shape=(17,))
    x86_ir = run_tessera_opt(
        tool,
        x86.tile_ir,
        "--pass-pipeline=builtin.module(tessera-x86-executable{"
        "family=solver_ift input=tile output=target arch=x86_64_avx512})",
    )
    assert "tessera_x86_avx512_solver_ift_sqrt_f32" in x86_ir
    assert "tessera_x86.abi_call" in x86_ir
    assert "tile.solver_ift_kernel" not in x86_ir

    rocm = lower_scheduled_solver_ift(target="rocm_gfx1151", shape=(17,))
    target_ir = run_tessera_opt(tool, rocm.tile_ir, "--lower-tile-to-rocm=arch=gfx1151")
    assert target_ir.count("tessera_rocm.solver_ift") == 1
    gpu_ir = run_tessera_opt(tool, target_ir, "--generate-rocm-solver-ift-kernel")
    assert "gpu.func @tessera_solver_ift" in gpu_ir
    assert "tessera_rocm.solver_ift" not in gpu_ir


def test_solver_ift_rocm_runtime_fails_closed_off_gfx1151(monkeypatch) -> None:
    from tessera import runtime as rt

    shape = (4,)
    contract = build_solver_ift_contract(target="rocm_gfx1151", shape=shape)
    metadata = {
        "target": "rocm",
        "compiler_path": "rocm_solver_ift_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": ["parameter", "solution", "cotangent"],
        "scheduled_solver_ift": contract,
    }
    monkeypatch.setattr(rt, "_rocm_chip", lambda: "gfx1200")
    value = np.ones(shape, dtype=np.float32)
    result = rt.launch(rt.RuntimeArtifact(metadata=metadata), (value, value, value))
    assert result["ok"] is False
    assert "verified for gfx1151" in result["reason"]
