"""Exact-SM120 certificate for the diagonal-sqrt solver/IFT pilot."""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.implicit_solver import (
    build_solver_ift_contract,
    compile_physical_general_solver_from_graph,
    diagonal_sqrt_ift_reference,
    lower_scheduled_solver_ift,
)
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tests._support.nvidia import assert_native_gpu, require_nvidia_mma_runtime


pytestmark = [pytest.mark.slow, pytest.mark.hardware_nvidia]


def _affine_residual_graph(shape: tuple[int, ...]) -> GraphIRModule:
    ty = IRType("tensor<" + "x".join(map(str, shape)) + "xf32>")
    return GraphIRModule(functions=[GraphIRFunction(
        name="affine_residual",
        args=[IRArg("theta", ty), IRArg("x", ty)], result_types=[ty],
        body=[IROp(
            "residual", "tessera.sub", ["%x", "%theta"],
            [str(ty), str(ty)], str(ty),
        )],
        return_values=["%residual"],
    )])


@pytest.mark.parametrize("product_mode", ["vjp", "jvp"])
def test_diagonal_sqrt_solver_ift_matches_exact_device_oracle(product_mode: str) -> None:
    rt = require_nvidia_mma_runtime()
    rng = np.random.default_rng(20260827)
    parameter = rng.uniform(0.25, 16.0, size=(3, 257)).astype(np.float32)
    solution = np.sqrt(parameter).astype(np.float32)
    product = rng.standard_normal(parameter.shape).astype(np.float32)
    scheduled = lower_scheduled_solver_ift(
        target="nvidia_sm120", shape=parameter.shape, product_mode=product_mode
    )
    result = rt.launch(
        rt.RuntimeArtifact(metadata=scheduled.runtime_metadata()),
        (parameter, solution, product),
    )
    assert_native_gpu(result)
    expected = diagonal_sqrt_ift_reference(parameter, solution, product)
    for actual, reference in zip(result["output"], expected):
        np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)


def test_solver_ift_contract_and_runtime_fail_closed() -> None:
    rt = require_nvidia_mma_runtime()
    shape = (17,)
    scheduled = lower_scheduled_solver_ift(target="nvidia_sm120", shape=shape)
    metadata = scheduled.runtime_metadata()
    stale = dict(metadata)
    stale_contract = dict(stale["scheduled_solver_ift"])
    stale_contract["artifact_hash"] = "0" * 64
    stale["scheduled_solver_ift"] = stale_contract
    value = np.ones(shape, dtype=np.float32)
    result = rt.launch(rt.RuntimeArtifact(metadata=stale), (value, value, value))
    assert result["ok"] is False
    assert "stale physical lineage" in result["reason"]

    with pytest.raises(ValueError, match="unmeasured architectures fail closed"):
        build_solver_ift_contract(target="nvidia_sm90", shape=shape)


def test_solver_ift_schedule_to_tile_binds_sm120_lineage() -> None:
    require_nvidia_mma_runtime()
    artifact = lower_scheduled_solver_ift(target="nvidia_sm120", shape=(5, 13))
    assert artifact.architecture == "sm120"
    assert 'tessera.target = "nvidia"' in artifact.schedule_ir
    assert artifact.tile_ir.count("tile.solver_ift_kernel") == 1
    assert artifact.artifact_hash in artifact.tile_ir
    assert "schedule." not in artifact.tile_ir


@pytest.mark.parametrize("product_mode", ["vjp", "jvp"])
def test_general_affine_solver_replays_cuda_children(product_mode: str) -> None:
    rt = require_nvidia_mma_runtime()
    shape = (37,)
    package = compile_physical_general_solver_from_graph(
        _affine_residual_graph(shape), target="nvidia_sm120", shape=shape,
        product_mode=product_mode, tolerance=1.0e-7,
        max_iterations=8, restart=4,
    )
    for child in package.contract["children"].values():
        assert child["metadata"]["target"] == "nvidia_sm120"
        assert child["metadata"]["compiler_path"] == "nvidia_solver_graph_compiled"
    parameter = np.linspace(-2.0, 3.0, shape[0], dtype=np.float32)
    solution = parameter.copy()
    product = np.linspace(-1.0, 1.0, shape[0], dtype=np.float32)
    result = rt.launch(
        rt.RuntimeArtifact(metadata=package.runtime_metadata()),
        (parameter, solution, product),
    )
    assert_native_gpu(result)
    residual, linear, parameter_product = result["output"]
    np.testing.assert_array_equal(residual, np.zeros_like(solution))
    np.testing.assert_allclose(linear, product, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(parameter_product, product, rtol=1e-6, atol=1e-6)
    assert package.runtime_metadata()["physical_general_solver"]["artifact_hash"] == package.artifact_hash


def test_general_solver_executes_nonlinear_unary_products() -> None:
    rt = require_nvidia_mma_runtime()
    shape = (7,)
    ty = IRType("tensor<7xf32>")
    module = GraphIRModule(functions=[GraphIRFunction(
        name="nonlinear_unary_residual",
        args=[IRArg("theta", ty), IRArg("x", ty)], result_types=[ty],
        body=[
            IROp("xx", "tessera.mul", ["%x", "%x"],
                 [str(ty), str(ty)], str(ty)),
            IROp("sx", "tessera.sin", ["%x"], [str(ty)], str(ty)),
            IROp("sum", "tessera.add", ["%xx", "%sx"],
                 [str(ty), str(ty)], str(ty)),
            IROp("r", "tessera.sub", ["%sum", "%theta"],
                 [str(ty), str(ty)], str(ty)),
        ],
        return_values=["%r"],
    )])
    package = compile_physical_general_solver_from_graph(
        module, target="nvidia_sm120", shape=shape,
        tolerance=2e-6, max_iterations=16, restart=8,
    )
    solution = np.linspace(0.75, 1.75, shape[0], dtype=np.float32)
    parameter = (solution * solution + np.sin(solution)).astype(np.float32)
    product = np.linspace(-1.0, 1.0, shape[0], dtype=np.float32)
    result = rt.launch(
        rt.RuntimeArtifact(metadata=package.runtime_metadata()),
        (parameter, solution, product),
    )
    assert_native_gpu(result)
    residual, linear, parameter_product = result["output"]
    expected = product / (2.0 * solution + np.cos(solution))
    np.testing.assert_allclose(residual, 0.0, atol=3e-6)
    np.testing.assert_allclose(linear, expected, rtol=2e-5, atol=3e-6)
    np.testing.assert_allclose(parameter_product, expected, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_general_solver_low_precision_storage_widens_at_bound_boundary(storage: str) -> None:
    rt = require_nvidia_mma_runtime()
    shape = (31,)
    storage_ty = IRType(f"tensor<31x{storage}>")
    result_ty = IRType("tensor<31xf32>")
    module = GraphIRModule(functions=[GraphIRFunction(
        name="lowp_affine_residual",
        args=[IRArg("theta", storage_ty), IRArg("x", storage_ty)],
        result_types=[result_ty],
        body=[IROp("r", "tessera.sub", ["%x", "%theta"],
                   [str(storage_ty), str(storage_ty)], str(result_ty))],
        return_values=["%r"],
    )])
    package = compile_physical_general_solver_from_graph(
        module, target="nvidia_sm120", shape=shape,
        tolerance=2e-6, max_iterations=8, restart=4,
    )
    base = np.linspace(-2.0, 3.0, shape[0], dtype=np.float32)
    if storage == "f16":
        parameter = base.astype(np.float16)
    else:
        ml_dtypes = pytest.importorskip("ml_dtypes")
        parameter = base.astype(ml_dtypes.bfloat16)
    solution = parameter.copy()
    product = np.linspace(-1.0, 1.0, shape[0], dtype=np.float32)
    result = rt.launch(
        rt.RuntimeArtifact(metadata=package.runtime_metadata()),
        (parameter, solution, product),
    )
    assert_native_gpu(result)
    residual, linear, parameter_product = result["output"]
    np.testing.assert_array_equal(residual, np.zeros_like(product))
    np.testing.assert_allclose(linear, product, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(parameter_product, product, rtol=2e-6, atol=2e-6)
    policy = package.contract["value_contract"]
    assert policy["storage_dtypes"]["parameter"] == storage
    assert policy["storage_dtypes"]["solution"] == storage
    assert policy["accumulation_dtype"] == "f32"
    assert policy["conversion"] == "explicit_package_boundary_widen"
