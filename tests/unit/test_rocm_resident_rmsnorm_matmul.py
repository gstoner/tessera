"""Graph-to-Tile resident RMSNorm/matmul edge proof on gfx1201."""
from __future__ import annotations

import os

import numpy as np
import pytest

from tessera.compiler import scheduled_kernel, rocm_native
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests.unit.test_rocm_gfx1201_scheduled import _rmsnorm_graph

@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native compiler")
def test_gfx1201_bf16_resident_edge_packages_through_schedule_and_tile():
    """The narrow bf16 edge builds matching native packages."""
    from tessera.compiler.resident_rocm_norm_matmul import lower_graph_rmsnorm_matmul
    from tests.unit.test_rocm_gfx1201_scheduled import _rmsnorm_graph
    from tests.unit.test_scheduled_matmul_consumers import _module as matmul_module

    m, k, n = 5, 32, 16
    norm_graph = _rmsnorm_graph(dtype="bf16", shape=(m, k))
    gemm_graph = matmul_module(
        target="rocm", shape=(m, k, n), dtype="bf16", output_dtype="fp32"
    )
    norm, gemm = lower_graph_rmsnorm_matmul(norm_graph, gemm_graph)
    norm_package = rocm_native.package_scheduled_kernel(
        norm, pipeline_name="tessera-lower-to-rocm"
    )
    gemm_package = rocm_native.package_scheduled_matmul(
        gemm, pipeline_name="tessera-lower-to-rocm"
    )
    assert norm.storage == gemm.a_dtype == gemm.b_dtype == "bf16"
    assert norm.accum == gemm.accum == "f32"
    assert norm_package.descriptor.abi_id == rocm_native.GFX_NORM_BF16_ABI
    assert gemm_package.descriptor.abi_id == rocm_native.GFX_MATMUL_BF16_F32_ABI
    assert norm_package.image.payload and gemm_package.image.payload
    assert "tile.norm_kernel" in norm.tile_ir and "tile.matmul_kernel" in gemm.tile_ir


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
def test_gfx1201_bf16_resident_edge_exact_device_numerics_and_lifetime():
    """Check generic ABI dispatch, resident values, and allocation lifetime."""
    import ml_dtypes
    from tessera import runtime as rt
    from tessera.compiler import scheduled_matmul
    from tessera.compiler.resident_rocm_norm_matmul import ResidentROCmNormMatmul
    from tests.unit.test_rocm_gfx1201_scheduled import _rmsnorm_graph
    from tests.unit.test_scheduled_matmul_consumers import _module as matmul_module

    assert rt._rocm_live_arch() == rt._rocm_chip() == "gfx1201"
    m, k, n = 5, 32, 16
    norm_graph = _rmsnorm_graph(dtype="bf16", shape=(m, k))
    gemm_graph = matmul_module(
        target="rocm", shape=(m, k, n), dtype="bf16", output_dtype="fp32"
    )
    norm_artifact = scheduled_kernel.lower_scheduled_kernel(
        norm_graph, target="rocm_gfx1201"
    )
    gemm_artifact = scheduled_matmul.lower_scheduled_matmul(
        gemm_graph, target="rocm_gfx1201"
    )
    norm_package = rocm_native.package_scheduled_kernel(
        norm_artifact, pipeline_name="tessera-lower-to-rocm"
    )
    gemm_package = rocm_native.package_scheduled_matmul(
        gemm_artifact, pipeline_name="tessera-lower-to-rocm"
    )
    rng = np.random.default_rng(1201_53017)
    x = rng.normal(0.0, 0.5, (m, k)).astype(ml_dtypes.bfloat16)
    rhs = rng.normal(0.0, 0.25, (k, n)).astype(ml_dtypes.bfloat16)
    x32 = x.astype(np.float32)
    normalized = (
        x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + 1e-5)
    ).astype(ml_dtypes.bfloat16)
    expected = normalized.astype(np.float32) @ rhs.astype(np.float32)

    def runtime_artifact(package):
        return rt.RuntimeArtifact(
            metadata={"target": package.image.target},
            native_image=package.image,
            launch_descriptor=package.descriptor,
            tile_ir=package.tile_ir,
            target_ir=package.target_ir,
        )

    norm_input = next(
        item.name for item in norm_package.descriptor.buffers
        if item.direction == "input"
    )
    norm_output = next(
        item.name for item in norm_package.descriptor.buffers
        if item.direction == "output"
    )
    generic_norm = np.empty_like(x)
    norm_receipt = rt.launch(runtime_artifact(norm_package), {
        "buffers": {norm_input: x, norm_output: generic_norm},
        "scalars": {"Rows": m, "K": k, "Epsilon": 1e-5},
    })
    assert norm_receipt["ok"] and norm_receipt["execution_kind"] == "native_gpu", norm_receipt
    np.testing.assert_allclose(generic_norm, normalized, rtol=3e-2, atol=3e-2)

    gemm_inputs = sorted(
        (item for item in gemm_package.descriptor.buffers if item.direction == "input"),
        key=lambda item: item.ordinal,
    )
    gemm_output = next(
        item.name for item in gemm_package.descriptor.buffers
        if item.direction == "output"
    )
    generic_output = np.empty((m, n), dtype=np.float32)
    gemm_receipt = rt.launch(runtime_artifact(gemm_package), {
        "buffers": {gemm_inputs[0].name: generic_norm,
                    gemm_inputs[1].name: rhs, gemm_output: generic_output},
        "scalars": {"M": m, "N": n, "K": k},
    })
    assert gemm_receipt["ok"] and gemm_receipt["execution_kind"] == "native_gpu", gemm_receipt
    np.testing.assert_allclose(generic_output, expected, rtol=3e-2, atol=3e-2)

    with ResidentROCmNormMatmul(norm_package, gemm_package, x, rhs) as session:
        first = session.run(warmup=2, iterations=5)
        addresses = first["buffer_addresses"]
        assert addresses["intermediate"] not in {
            addresses["x"], addresses["rhs"], addresses["output"]
        }
        assert first["producer_median_ms"] > 0
        assert first["consumer_median_ms"] > 0
        for output in first["outputs"]:
            np.testing.assert_allclose(output, expected, rtol=3e-2, atol=3e-2)
        second = session.run(warmup=0, iterations=1)
        assert second["buffer_addresses"] == addresses
        np.testing.assert_allclose(second["outputs"][0], expected, rtol=3e-2, atol=3e-2)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
def test_gfx1201_resident_rmsnorm_matmul_edge_lifetime_and_numerics():
    """Keep an RMSNorm result resident for a scheduled matmul consumer."""
    from tessera import runtime as rt
    from tessera.compiler import scheduled_matmul
    from tessera.compiler.resident_rocm_norm_matmul import ResidentROCmNormMatmul
    from tests.unit.test_scheduled_matmul_consumers import _module as matmul_module

    assert rt._rocm_live_arch() == "gfx1201"
    assert rt._rocm_chip() == "gfx1201"
    m, k, n = 5, 32, 16
    epsilon = 1.0e-5
    norm_ir = _rmsnorm_graph(dtype="fp16", shape=(m, k))
    norm_artifact = scheduled_kernel.lower_scheduled_kernel(norm_ir, target="rocm_gfx1201")
    norm_package = rocm_native.package_scheduled_kernel(
        norm_artifact, pipeline_name="tessera-lower-to-rocm"
    )
    gemm_ir = matmul_module(target="rocm", shape=(m, k, n), dtype="fp16")
    gemm_artifact = scheduled_matmul.lower_scheduled_matmul(
        gemm_ir, target="rocm_gfx1201"
    )
    scheduled_matmul.verify_matmul_projection(gemm_artifact)
    gemm_package = rocm_native.package_scheduled_matmul(
        gemm_artifact, pipeline_name="tessera-lower-to-rocm"
    )
    assert norm_package.descriptor.provenance["workgroup"] == [256, 1, 1]
    assert np.isclose(norm_package.descriptor.provenance["epsilon"], epsilon)
    epsilon = float(norm_package.descriptor.provenance["epsilon"])

    rng = np.random.default_rng(1201_5032)
    x = rng.normal(0.0, 0.5, (m, k)).astype(np.float16)
    rhs = rng.normal(0.0, 0.25, (k, n)).astype(np.float16)
    x32 = x.astype(np.float32)
    normalized = (
        x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + epsilon)
    ).astype(np.float16)
    expected = normalized.astype(np.float32) @ rhs.astype(np.float32)

    with ResidentROCmNormMatmul(norm_package, gemm_package, x, rhs) as session:
        first = session.run(warmup=2, iterations=5)
        addresses = first["buffer_addresses"]
        assert set(addresses) == {"x", "rhs", "intermediate", "output"}
        assert addresses["intermediate"] not in {
            addresses["x"], addresses["rhs"], addresses["output"]
        }
        assert first["producer_median_ms"] > 0
        assert first["consumer_median_ms"] > 0
        assert len(first["outputs"]) == 5
        for output in first["outputs"]:
            np.testing.assert_allclose(output, expected, rtol=2e-3, atol=2e-3)
        second = session.run(warmup=0, iterations=1)
        assert second["buffer_addresses"] == addresses
        np.testing.assert_allclose(second["outputs"][0], expected, rtol=2e-3, atol=2e-3)
    with pytest.raises(RuntimeError, match="session is closed"):
        session.run(warmup=0, iterations=1)

@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
@pytest.mark.parametrize("storage_dtype", ["fp16", "bf16"])
def test_gfx1201_resident_rmsnorm_to_bounded_dynamic_n_matmul_reuses_package(storage_dtype):
    """Reuse one resident edge package as the consumer N extent changes."""
    from tessera import runtime as rt
    from tessera.compiler import scheduled_matmul
    from tessera.compiler.resident_rocm_norm_matmul import ResidentROCmNormMatmul
    from tessera.compiler.graph_ir import IRType
    from tests.unit.test_scheduled_matmul_consumers import _module as matmul_module

    assert rt._rocm_live_arch() == rt._rocm_chip() == "gfx1201"
    m, k, n_bound = 5, 32, 32
    if storage_dtype == "bf16":
        import ml_dtypes
        storage_np_dtype = np.dtype(ml_dtypes.bfloat16)
        storage_element = "bf16"
    else:
        storage_np_dtype = np.dtype(np.float16)
        storage_element = "f16"
    norm_ir = _rmsnorm_graph(dtype=storage_dtype, shape=(m, k))
    norm_package = rocm_native.package_scheduled_kernel(
        scheduled_kernel.lower_scheduled_kernel(norm_ir, target="rocm_gfx1201"),
        pipeline_name="tessera-lower-to-rocm",
    )

    gemm_ir = matmul_module(target="rocm", shape=(m, k, n_bound), dtype=storage_dtype, output_dtype="fp32")
    function = gemm_ir.functions[0]
    function.args[1].ir_type = IRType(f"tensor<32x?x{storage_element}>", ("32", "?"), storage_dtype)
    function.result_types[0] = IRType(
        f"tensor<{m}x?xf32>", (str(m), "?"), "fp32"
    )
    op = function.body[0]
    op.operand_types[1] = str(function.args[1].ir_type)
    op.result_type = str(function.result_types[0])
    op.inferred_type = function.result_types[0]
    op.kwargs["shape_bounds"] = [m, n_bound, k]
    gemm_artifact = scheduled_matmul.lower_scheduled_matmul(
        gemm_ir, target="rocm_gfx1201"
    )
    assert gemm_artifact.dynamic_n
    assert not gemm_artifact.dynamic_m and not gemm_artifact.dynamic_k
    gemm_package = rocm_native.package_scheduled_matmul(
        gemm_artifact, pipeline_name="tessera-lower-to-rocm"
    )

    rng = np.random.default_rng(1201_17032)
    x = rng.normal(0.0, 0.5, (m, k)).astype(storage_np_dtype)
    rhs_bound = rng.normal(0.0, 0.25, (k, n_bound)).astype(storage_np_dtype)
    x32 = x.astype(np.float32)
    epsilon = float(norm_package.descriptor.provenance["epsilon"])
    normalized = (
        x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + epsilon)
    ).astype(storage_np_dtype)
    with ResidentROCmNormMatmul(norm_package, gemm_package, x, rhs_bound) as session:
        addresses = session.run(warmup=1, iterations=1)["buffer_addresses"]
        for active_n in (17, n_bound):
            rhs = rhs_bound[:, :active_n].copy()
            result = session.run(warmup=1, iterations=3, rhs=rhs)
            expected = normalized.astype(np.float32) @ rhs.astype(np.float32)
            assert all(output.shape == (m, active_n) for output in result["outputs"])
            assert result["buffer_addresses"] == addresses
            for output in result["outputs"]:
                np.testing.assert_allclose(output, expected, rtol=2e-3, atol=2e-3)
        with pytest.raises(ValueError, match="N bound"):
            session.run(warmup=0, iterations=1, rhs=np.ones((k, n_bound + 1), dtype=storage_np_dtype))


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
def test_gfx1201_public_bf16_dynamic_n_resident_edge_reuses_package():
    """Reuse one public-traced BF16 package under bounded runtime N guards."""
    import ml_dtypes
    from tessera import runtime as rt
    from tessera.compiler.resident_rocm_norm_matmul import package_graph_rmsnorm_matmul

    assert rt._rocm_live_arch() == rt._rocm_chip() == "gfx1201"
    m, k, n_bound = 5, 32, 32
    norm_graph, matmul_graph = _public_rmsnorm_matmul_graphs(
        m, k, n_bound, storage_dtype="bf16"
    )
    storage_dtype = np.dtype(ml_dtypes.bfloat16)
    rng = np.random.default_rng(1201_32041)
    x = rng.normal(0.0, 0.5, (m, k)).astype(storage_dtype)
    rhs_bound = rng.normal(0.0, 0.25, (k, n_bound)).astype(storage_dtype)
    with package_graph_rmsnorm_matmul(
        norm_graph, matmul_graph, x, rhs_bound, dynamic_n_bound=n_bound
    ) as session:
        assert session._dynamic_n
        assert any(
            guard.predicate == "max" and guard.binding == session._gemm_b
            and guard.dimension == 1 and guard.value == n_bound
            for guard in session._gemm.shape_guards
        )
        addresses = session.run(warmup=0, iterations=1, rhs=rhs_bound)["buffer_addresses"]
        epsilon = float(session._norm.provenance["epsilon"])
        x32 = x.astype(np.float32)
        normalized = (
            x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + epsilon)
        ).astype(storage_dtype)
        for active_n in (17, n_bound):
            rhs = rhs_bound[:, :active_n].copy()
            result = session.run(warmup=1, iterations=3, rhs=rhs)
            expected = normalized.astype(np.float32) @ rhs.astype(np.float32)
            assert result["buffer_addresses"] == addresses
            assert result["producer_median_ms"] > 0
            assert result["consumer_median_ms"] > 0
            assert all(output.shape == (m, active_n) for output in result["outputs"])
            for output in result["outputs"]:
                np.testing.assert_allclose(output, expected, rtol=3e-2, atol=3e-2)
        with pytest.raises(ValueError, match="N bound"):
            session.run(
                warmup=0, iterations=1,
                rhs=np.ones((k, n_bound + 1), dtype=storage_dtype),
            )




@pytest.mark.compiler_route
@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native compiler")
@pytest.mark.parametrize("storage_dtype", ["fp16", "bf16"])
def test_gfx1201_dynamic_m_graph_projects_to_bounded_schedule(storage_dtype):
    """The compiler lane checks dynamic-M projection without ROCm libraries."""
    from tessera.compiler.scheduled_matmul import (
        lower_scheduled_matmul, with_bounded_dynamic_m,
    )

    m_bound, k, n = 8, 32, 16
    _, graph = _public_rmsnorm_matmul_graphs(
        m_bound, k, n, storage_dtype=storage_dtype
    )
    if storage_dtype == "bf16":
        import ml_dtypes
        storage_np_dtype = np.dtype(ml_dtypes.bfloat16)
    else:
        storage_np_dtype = np.dtype(np.float16)
    graph = with_bounded_dynamic_m(graph, m_bound)
    artifact = lower_scheduled_matmul(graph, target="rocm_gfx1201")
    assert artifact.dynamic_m
    assert not artifact.dynamic_n and not artifact.dynamic_k
    assert (artifact.m, artifact.k, artifact.n) == (m_bound, k, n)
    assert "schedule.matmul" in artifact.schedule_ir
    assert "tile.matmul_kernel" in artifact.tile_ir


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
@pytest.mark.parametrize("storage_dtype", ["fp16", "bf16"])
def test_gfx1201_public_dynamic_m_resident_edge_reuses_package(storage_dtype):
    """Reuse max-capacity norm/matmul packages for shorter row prefixes."""
    from tessera import runtime as rt
    from tessera.compiler.resident_rocm_norm_matmul import package_graph_rmsnorm_matmul

    assert rt._rocm_live_arch() == rt._rocm_chip() == "gfx1201"
    m_bound, k, n = 8, 32, 16
    norm_graph, matmul_graph = _public_rmsnorm_matmul_graphs(
        m_bound, k, n, storage_dtype=storage_dtype
    )
    if storage_dtype == "bf16":
        import ml_dtypes
        storage_np_dtype = np.dtype(ml_dtypes.bfloat16)
        tolerance = 3e-2
    else:
        storage_np_dtype = np.dtype(np.float16)
        tolerance = 2e-3
    rng = np.random.default_rng(1201_64008)
    x_bound = rng.normal(0.0, 0.5, (m_bound, k)).astype(storage_np_dtype)
    rhs = rng.normal(0.0, 0.25, (k, n)).astype(storage_np_dtype)
    with package_graph_rmsnorm_matmul(
        norm_graph, matmul_graph, x_bound, rhs, dynamic_m_bound=m_bound
    ) as session:
        assert session._dynamic_m and not session._dynamic_n
        assert any(
            guard.binding == session._gemm_a and guard.dimension == 0
            and guard.predicate == "max" and guard.value == m_bound
            for guard in session._gemm.shape_guards
        )
        assert any(
            guard.binding == session._norm_input and guard.dimension == 0
            and guard.predicate == "max" and guard.value == m_bound
            for guard in session._norm.shape_guards
        )
        addresses = session.run(warmup=0, iterations=1)["buffer_addresses"]
        epsilon = float(session._norm.provenance["epsilon"])
        for active_m in (3, m_bound):
            x = x_bound[:active_m].copy()
            result = session.run(warmup=1, iterations=3, x=x)
            x32 = x.astype(np.float32)
            normalized = (
                x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + epsilon)
            ).astype(storage_np_dtype)
            expected = normalized.astype(np.float32) @ rhs.astype(np.float32)
            assert result["buffer_addresses"] == addresses
            assert result["producer_median_ms"] > 0
            assert result["consumer_median_ms"] > 0
            assert all(output.shape == (active_m, n) for output in result["outputs"])
            for output in result["outputs"]:
                np.testing.assert_allclose(
                    output, expected, rtol=tolerance, atol=tolerance
                )
        with pytest.raises(ValueError, match="M bound"):
            session.run(
                warmup=0, iterations=1,
                x=np.ones((m_bound + 1, k), dtype=storage_np_dtype),
            )


def test_public_bf16_rmsnorm_trace_preserves_graph_storage_dtype():
    norm_graph, matmul_graph = _public_rmsnorm_matmul_graphs(
        5, 32, 16, storage_dtype="bf16"
    )
    assert norm_graph.functions[0].result_types[0].dtype == "bf16"
    assert matmul_graph.functions[0].args[0].ir_type.dtype == "bf16"
    assert matmul_graph.functions[0].result_types[0].dtype == "fp32"


def _public_rmsnorm_matmul_graphs(
    m: int = 5, k: int = 32, n: int = 16, *, storage_dtype: str = "fp16"
):
    from tessera.compiler.from_text import from_text

    producer = from_text("""
        def rmsnorm_frontend(x):
            return ts.ops.rmsnorm(x, eps=1e-5)
    """)
    consumer = from_text("""
        def matmul_frontend(normalized, weights):
            return ts.ops.matmul(normalized, weights, output_dtype="fp32")
    """)
    if storage_dtype == "bf16":
        import ml_dtypes
        array_dtype = np.dtype(ml_dtypes.bfloat16)
    elif storage_dtype == "fp16":
        array_dtype = np.dtype(np.float16)
    else:
        raise ValueError(f"unsupported resident test storage {storage_dtype!r}")
    source = np.ones((m, k), dtype=array_dtype)
    normalized = producer(source)
    assert np.asarray(normalized).dtype == array_dtype
    rhs = np.ones((k, n), dtype=array_dtype)
    consumer(normalized, rhs)
    assert consumer.graph_ir.functions[0].result_types[0].dtype == "fp32"
    assert producer.frontend_authority == consumer.frontend_authority == "tracer"
    return producer.graph_ir, consumer.graph_ir


@pytest.mark.parametrize(
    "output_dtype,expected_dtype",
    [("fp16", np.float16), ("fp32", np.float32)],
)
def test_public_matmul_output_dtype_uses_fp32_accumulation(
    output_dtype, expected_dtype,
):
    import tessera as ts

    a = np.asarray([[0.3333, 0.6667, -0.1429]], dtype=np.float16)
    b = np.asarray([[0.3333, 0.1429], [0.1111, -0.6667], [0.7777, 0.2222]], dtype=np.float16)
    result = ts.ops.matmul(a, b, output_dtype=output_dtype)
    expected = a.astype(np.float32) @ b.astype(np.float32)
    if output_dtype == "fp16":
        expected = expected.astype(np.float16)
    assert result.dtype == expected_dtype
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("epilogue_mode", ["explicit", "mapping"])
def test_public_matmul_fp16_cast_happens_after_epilogue(epilogue_mode):
    import tessera as ts

    # The fp32 dot product is just above 1.0. Rounding it to fp16 before
    # adding the bias would leave a positive value after the bias, unlike the fp32 result.
    a = np.asarray([[1.0, 0.001]], dtype=np.float16)
    b = np.asarray([[1.0], [0.5]], dtype=np.float16)
    bias = np.asarray([-1.0007], dtype=np.float32)
    accumulator = a.astype(np.float32) @ b.astype(np.float32)
    expected = np.maximum(accumulator + bias, 0.0).astype(np.float16)

    if epilogue_mode == "explicit":
        result = ts.ops.matmul(
            a, b, bias=bias, activation="relu", output_dtype="fp16"
        )
    else:
        result = ts.ops.matmul(
            a, b, epilogue={"bias": bias, "activation": "relu"},
            output_dtype="fp16",
        )

    assert result.dtype == np.float16
    np.testing.assert_array_equal(result, expected)
    assert result.item() == 0.0


def test_public_matmul_output_dtype_is_explicit_in_abstract_trace():
    import tessera as ts
    from tessera.compiler.trace import to_graph_ir_module, trace

    def matmul(a, b):
        return ts.ops.matmul(a, b, output_dtype="fp32")

    traced = trace(matmul, ((5, 32), "fp16"), ((32, 16), "fp16"))
    assert traced.output_specs == (((5, 16), "fp32"),)
    module = to_graph_ir_module(traced, name="output_dtype_matmul")
    assert module.functions[0].result_types[0].dtype == "fp32"
    assert module.functions[0].body[0].kwargs["output_dtype"] == "fp32"


@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native compiler")
def test_gfx1201_public_frontend_graph_lowers_through_schedule_and_tile():
    from tessera.compiler.resident_rocm_norm_matmul import lower_graph_rmsnorm_matmul

    norm_graph, matmul_graph = _public_rmsnorm_matmul_graphs()
    norm, matmul = lower_graph_rmsnorm_matmul(norm_graph, matmul_graph)
    assert norm.target == matmul.target == "rocm"
    assert norm.architecture == matmul.architecture == "gfx1201"
    assert "schedule.norm" in norm.schedule_ir
    assert "tile.norm_kernel" in norm.tile_ir
    assert "schedule.matmul" in matmul.schedule_ir
    assert "tile.matmul_kernel" in matmul.tile_ir
    assert matmul.output_dtype == "fp32"


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
def test_gfx1201_public_frontend_packages_and_executes_resident_edge():
    from tessera import runtime as rt
    from tessera.compiler.resident_rocm_norm_matmul import package_graph_rmsnorm_matmul

    assert rt._rocm_live_arch() == rt._rocm_chip() == "gfx1201"
    norm_graph, matmul_graph = _public_rmsnorm_matmul_graphs()
    m, k, n = 5, 32, 16
    rng = np.random.default_rng(1201_53016)
    x = rng.normal(0.0, 0.5, (m, k)).astype(np.float16)
    rhs = rng.normal(0.0, 0.25, (k, n)).astype(np.float16)
    x32 = x.astype(np.float32)
    normalized = (
        x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + 1e-5)
    ).astype(np.float16)
    expected = normalized.astype(np.float32) @ rhs.astype(np.float32)

    with package_graph_rmsnorm_matmul(norm_graph, matmul_graph, x, rhs) as session:
        result = session.run(warmup=2, iterations=5)
        addresses = result["buffer_addresses"]
        assert addresses["intermediate"] not in {
            addresses["x"], addresses["rhs"], addresses["output"]
        }
        assert result["producer_median_ms"] > 0
        assert result["consumer_median_ms"] > 0
        assert len(result["outputs"]) == 5
        for output in result["outputs"]:
            np.testing.assert_allclose(output, expected, rtol=2e-3, atol=2e-3)
        repeated = session.run(warmup=0, iterations=1)
        assert repeated["buffer_addresses"] == addresses
        np.testing.assert_allclose(repeated["outputs"][0], expected, rtol=2e-3, atol=2e-3)



@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate",
)
@pytest.mark.parametrize("storage_dtype", ["fp16", "bf16"])
def test_gfx1201_resident_rmsnorm_to_bounded_dynamic_k_reuses_capacity(storage_dtype):
    """Reuse both images and resident allocations as the contracted K prefix changes."""
    from tessera import runtime as rt
    from tessera.compiler.resident_rocm_norm_matmul import package_graph_rmsnorm_matmul
    from tests.unit.test_scheduled_matmul_consumers import _module as matmul_module

    assert rt._rocm_live_arch() == rt._rocm_chip() == "gfx1201"
    m, k_bound, n = 5, 32, 16
    if storage_dtype == "bf16":
        import ml_dtypes
        storage_np_dtype = np.dtype(ml_dtypes.bfloat16)
    else:
        storage_np_dtype = np.dtype(np.float16)
    norm_graph = _rmsnorm_graph(dtype=storage_dtype, shape=(m, k_bound))
    matmul_graph = matmul_module(
        target="rocm", shape=(m, k_bound, n), dtype=storage_dtype, output_dtype="fp32"
    )
    rng = np.random.default_rng(1201_64013)
    x_bound = rng.normal(0.0, 0.5, (m, k_bound)).astype(storage_np_dtype)
    rhs_bound = rng.normal(0.0, 0.25, (k_bound, n)).astype(storage_np_dtype)

    with package_graph_rmsnorm_matmul(
        norm_graph, matmul_graph, x_bound, rhs_bound, dynamic_k_bound=k_bound
    ) as session:
        assert session._dynamic_k
        assert not session._dynamic_m and not session._dynamic_n
        assert session._norm.provenance["dynamic_columns_bound"] == k_bound
        assert any(
            guard.predicate == "max" and guard.binding == session._gemm_a
            and guard.dimension == 1 and guard.value == k_bound
            for guard in session._gemm.shape_guards
        )
        addresses = session.run(warmup=0, iterations=1)["buffer_addresses"]
        epsilon = float(session._norm.provenance["epsilon"])
        for active_k in (13, 21, k_bound):
            x_backing = np.zeros((m, active_k + 3), dtype=storage_np_dtype)
            x_backing[:, :active_k] = x_bound[:, :active_k]
            x = x_backing[:, :active_k]
            rhs_backing = np.zeros((k_bound + 5, n + 3), dtype=storage_np_dtype)
            rhs_backing[:active_k, :n] = rhs_bound[:active_k, :]
            rhs = rhs_backing[:active_k, :n]
            assert not x.flags.c_contiguous
            assert not rhs.flags.c_contiguous
            result = session.run(warmup=1, iterations=3, x=x, rhs=rhs)
            x32 = x.astype(np.float32)
            normalized = (
                x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + epsilon)
            ).astype(storage_np_dtype)
            expected = normalized.astype(np.float32) @ rhs.astype(np.float32)
            assert all(output.shape == (m, n) for output in result["outputs"])
            assert result["buffer_addresses"] == addresses
            assert result["producer_median_ms"] > 0
            assert result["consumer_median_ms"] > 0
            for output in result["outputs"]:
                np.testing.assert_allclose(output, expected, rtol=3e-2 if storage_dtype == "bf16" else 2e-3,
                                           atol=3e-2 if storage_dtype == "bf16" else 2e-3)
        with pytest.raises(ValueError, match="dynamic K bound"):
            package_graph_rmsnorm_matmul(
                norm_graph, matmul_graph, x_bound, rhs_bound, dynamic_k_bound=k_bound + 1
            )
