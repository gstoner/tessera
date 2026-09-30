"""Graph-to-Tile resident RMSNorm/matmul edge proof on gfx1201."""
from __future__ import annotations

import os

import numpy as np
import pytest

from tessera.compiler import scheduled_kernel, rocm_native
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests.unit.test_rocm_gfx1201_scheduled import _rmsnorm_graph

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
def test_gfx1201_resident_rmsnorm_to_bounded_dynamic_n_matmul_reuses_package():
    """Reuse one resident edge package as the consumer N extent changes."""
    from tessera import runtime as rt
    from tessera.compiler import scheduled_matmul
    from tessera.compiler.resident_rocm_norm_matmul import ResidentROCmNormMatmul
    from tessera.compiler.graph_ir import IRType
    from tests.unit.test_scheduled_matmul_consumers import _module as matmul_module

    assert rt._rocm_live_arch() == rt._rocm_chip() == "gfx1201"
    m, k, n_bound = 5, 32, 32
    norm_ir = _rmsnorm_graph(dtype="fp16", shape=(m, k))
    norm_package = rocm_native.package_scheduled_kernel(
        scheduled_kernel.lower_scheduled_kernel(norm_ir, target="rocm_gfx1201"),
        pipeline_name="tessera-lower-to-rocm",
    )

    gemm_ir = matmul_module(target="rocm", shape=(m, k, n_bound), dtype="fp16")
    function = gemm_ir.functions[0]
    function.args[1].ir_type = IRType("tensor<32x?xf16>", ("32", "?"), "fp16")
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
    x = rng.normal(0.0, 0.5, (m, k)).astype(np.float16)
    rhs_bound = rng.normal(0.0, 0.25, (k, n_bound)).astype(np.float16)
    x32 = x.astype(np.float32)
    epsilon = float(norm_package.descriptor.provenance["epsilon"])
    normalized = (
        x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + epsilon)
    ).astype(np.float16)
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
            session.run(warmup=0, iterations=1, rhs=np.ones((k, n_bound + 1), np.float16))


def _public_rmsnorm_matmul_graphs(m: int = 5, k: int = 32, n: int = 16):
    from tessera.compiler.from_text import from_text

    producer = from_text("""
        def rmsnorm_frontend(x):
            return ts.ops.rmsnorm(x, eps=1e-5)
    """)
    consumer = from_text("""
        def matmul_frontend(normalized, weights):
            return ts.ops.matmul(normalized, weights, output_dtype="fp32")
    """)
    source = np.ones((m, k), dtype=np.float16)
    normalized = producer(source)
    rhs = np.ones((k, n), dtype=np.float16)
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
