"""X86-E2E-2 cohort 3/4 stable-ABI registry and packaging gates."""

from __future__ import annotations

import pytest
import numpy as np
import ctypes
from typing import cast

from tessera import runtime as rt
from tessera.compiler.canonical_compile import canonical_compile
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.x86_breadth import (
    X86_BREADTH_ABIS,
    cohort_specs,
    emit_abi_tile_ir,
    graph_breadth_contract,
    package_abi,
    package_graph_breadth,
    supports_graph_breadth,
    supports_promoted_graph_breadth,
)
from tessera.compiler.x86_native import _library_path, tools_available


COHORT3_SYMBOLS = {
    "tessera_x86_avx512_spmm_csr_f32", "tessera_x86_avx512_sddmm_f32",
    "tessera_x86_gather_f32", "tessera_x86_scatter_f32",
    "tessera_x86_bitonic_sort_kv_f32", "tessera_x86_fft_c2c_f32",
    "tessera_x86_cholesky_f32", "tessera_x86_tri_solve_f32",
    "tessera_x86_lu_f32", "tessera_x86_qr_f32", "tessera_x86_svd_f32",
    "tessera_x86_clifford_bilinear_f32",
}

COHORT4_SYMBOLS = {
    "tessera_x86_avx512_pointwise_loss_f32", "tessera_x86_avx512_binary_loss_f32",
    "tessera_x86_avx512_policy_loss_f32", "tessera_x86_avx512_fpquant_f32",
    "tessera_x86_avx512_selective_ssm_f32", "tessera_x86_avx512_selective_ssm_f16",
    "tessera_x86_avx512_selective_ssm_bf16", "tessera_x86_selective_ssm_bwd_f32",
    "tessera_x86_moe_f32", "tessera_x86_optimizer_f32", "tessera_x86_deltanet_f32",
    "tessera_x86_kv_cache_append_f32", "tessera_x86_kv_cache_read_f32",
    "tessera_x86_kv_cache_prune_f32", "tessera_x86_philox_uniform_f32",
    "tessera_x86_ebm_affine_langevin_f32",
    "tessera_x86_ebm_decode_init_noise_apply_f32", "tessera_x86_ebm_ebt_tiny_f32",
    "tessera_x86_ebm_energy_quadratic_f32", "tessera_x86_ebm_langevin_philox_f32",
    "tessera_x86_ebm_partition_exact_f32",
}


def _type(shape: tuple[int, ...], dtype: str = "fp32") -> IRType:
    spelling = {"fp32": "f32", "int64": "i64"}[dtype]
    extents = "x".join(map(str, shape))
    return IRType(f"tensor<{extents}x{spelling}>", tuple(map(str, shape)), dtype)


def _graph_module(family: str, size: int = 1_048_576) -> GraphIRModule:
    if family == "gather":
        source, indices, output = _type((size * 2,)), _type((size,), "int64"), _type((size,))
        args = [IRArg("x", source), IRArg("ids", indices)]
        op_name, operands, kwargs, result = "tessera.gather", ["%x", "%ids"], {"axis": 0}, output
    elif family == "pointwise_loss":
        source = output = _type((size,))
        args = [IRArg("prediction", source), IRArg("target", source)]
        op_name, operands, kwargs, result = (
            "tessera.mse_loss", ["%prediction", "%target"], {"reduction": "none"}, output,
        )
    elif family == "cholesky":
        source = output = _type((2, 32, 32))
        args = [IRArg("matrix", source)]
        op_name, operands, kwargs, result = "tessera.cholesky", ["%matrix"], {}, output
    else:
        matrix, rhs = _type((2, 64, 64)), _type((2, 64, 4))
        args = [IRArg("matrix", matrix), IRArg("rhs", rhs)]
        op_name, operands, kwargs, result = "tessera.tri_solve", ["%matrix", "%rhs"], {"lower": True}, rhs
    return GraphIRModule(functions=[GraphIRFunction(
        name=f"x86_{family}", args=args, result_types=[result],
        body=[IROp(result="result", op_name=op_name, operands=operands,
                   operand_types=[str(arg.ir_type) for arg in args],
                   result_type=str(result), kwargs=kwargs)], return_values=["%result"],
    )])


def test_breadth_registry_is_total_for_inventory() -> None:
    assert {spec.symbol for spec in cohort_specs(3)} == COHORT3_SYMBOLS
    assert {spec.symbol for spec in cohort_specs(4)} == COHORT4_SYMBOLS
    assert len(X86_BREADTH_ABIS) == len(COHORT3_SYMBOLS | COHORT4_SYMBOLS)
    assert all(spec.abi_id.startswith("tessera.x86.") and spec.abi_id.endswith(".v1")
               for spec in X86_BREADTH_ABIS.values())
    for key in ("selective_ssm_f32", "selective_ssm_f16", "selective_ssm_bf16"):
        spec = X86_BREADTH_ABIS[key]
        assert spec.effects == "readwrite"
        assert next(arg for arg in spec.args if arg.name == "state").direction == "inout"


@pytest.mark.skipif(not tools_available(), reason="x86 compiler/shared library unavailable")
def test_breadth_registry_symbols_are_exported_by_exact_host_image() -> None:
    path = _library_path()
    assert path is not None
    library = ctypes.CDLL(str(path))
    missing = [spec.symbol for spec in X86_BREADTH_ABIS.values()
               if getattr(library, spec.symbol, None) is None]
    assert missing == []


@pytest.mark.parametrize("key", ["spmm_csr_f32", "fft_c2c_f32", "optimizer_f32",
                                  "kv_cache_append_f32", "selective_ssm_bf16"])
def test_breadth_carrier_has_explicit_abi_and_effects(key: str) -> None:
    spec = X86_BREADTH_ABIS[key]
    text = emit_abi_tile_ir(spec, entry="entry")
    assert "tile.x86_abi_kernel" in text
    assert f'symbol = "{spec.symbol}"' in text
    assert f'abi = "{spec.abi_id}"' in text
    assert f'effects = "{spec.effects}"' in text
    assert f"returns_status = {str(spec.returns_status).lower()}" in text


def test_package_preserves_interleaved_c_abi_order(monkeypatch) -> None:
    spec = X86_BREADTH_ABIS["gather_f32"]

    def fake_lower(tile_ir: str, symbol: str):
        assert symbol == spec.symbol
        assert "!llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr" in tile_ir
        return f"module {{ func.call @{symbol}() : () -> () }}", b"\x7fELF-breadth", "cc", "tc"

    monkeypatch.setattr("tessera.compiler.x86_breadth._lower", fake_lower)
    package = package_abi(
        "gather_f32", pipeline_name="tessera-lower-to-x86",
        buffer_shapes={"source": (17,), "indices": (5,), "output": (5,)},
    )
    assert package.descriptor.abi_id == spec.abi_id
    assert package.descriptor.provenance["cohort"] == 3
    assert [arg["name"] for arg in package.descriptor.provenance["abi_arguments"]] == [
        "source", "SourceN", "indices", "N", "output",
    ]


def test_package_rejects_unaccounted_buffers() -> None:
    with pytest.raises(ValueError, match="buffer shape mismatch"):
        package_abi(
            "kv_cache_append_f32", pipeline_name="tessera-lower-to-x86",
            buffer_shapes={"cache": (8, 4)},
        )


@pytest.mark.parametrize("family", ["gather", "pointwise_loss", "cholesky", "tri_solve"])
def test_graph_breadth_contract_is_isomorphic_and_packageable(monkeypatch, family: str) -> None:
    module = _graph_module(family)
    contract = graph_breadth_contract(module)
    assert contract is not None
    assert supports_graph_breadth(module)
    assert supports_promoted_graph_breadth(module)

    def fake_lower(tile_ir: str, symbol: str):
        assert "tile.x86_abi_kernel" in tile_ir
        return f"module {{ func.call @{symbol}() : () -> () }}", b"\x7fELF-graph", "cc", "tc"

    monkeypatch.setattr("tessera.compiler.x86_breadth._lower", fake_lower)
    package = package_graph_breadth(module, pipeline_name="tessera-lower-to-x86")
    assert package.descriptor.provenance["graph_level"] is True
    assert package.descriptor.provenance["selector_family"] == family
    assert package.descriptor.provenance["graph_scalars"] == contract["scalars"]
    assert package.descriptor.buffers[-1].name == "result"


def test_graph_breadth_rejects_composite_variants() -> None:
    loss = _graph_module("pointwise_loss")
    loss.functions[0].body[0].kwargs["reduction"] = "mean"
    assert not supports_graph_breadth(loss)
    gather = _graph_module("gather")
    gather.functions[0].args[0].ir_type = _type((8, 8))
    assert not supports_graph_breadth(gather)


@pytest.mark.parametrize("family", ["gather", "pointwise_loss", "cholesky", "tri_solve"])
def test_canonical_selector_promotes_measured_graph_breadth(monkeypatch, family: str) -> None:
    def fake_lower(tile_ir: str, symbol: str):
        return f"module {{ func.call @{symbol}() : () -> () }}", b"\x7fELF-graph", "cc", "tc"

    monkeypatch.setattr("tessera.compiler.x86_breadth._lower", fake_lower)
    monkeypatch.setattr("tessera.compiler.x86_native.tools_available", lambda: True)
    result = canonical_compile(
        _graph_module(family), target="x86", enable_tool_validation=False,
    )
    assert result.bundle.execution_mode == "native_descriptor"
    assert result.bundle.execution_kind == "native_cpu"
    assert result.launch_descriptor is not None
    assert result.launch_descriptor.provenance["selector_family"] == family


def test_canonical_selector_retains_small_unmeasured_breadth(monkeypatch) -> None:
    monkeypatch.setattr("tessera.compiler.x86_native.tools_available", lambda: True)
    result = canonical_compile(
        _graph_module("pointwise_loss", 130), target="x86",
        enable_tool_validation=False,
    )
    assert result.launch_descriptor is None


def _launch(package, values: dict[str, object]) -> dict[str, object]:
    artifact = rt.RuntimeArtifact(
        metadata={"target": "x86"}, native_image=package.image,
        launch_descriptor=package.descriptor, tile_ir=package.tile_ir,
        target_ir=package.target_ir,
    )
    return rt.launch(artifact, values)


@pytest.mark.skipif(not tools_available(), reason="x86 compiler/shared library unavailable")
def test_cohort3_gather_descriptor_executes_on_exact_host() -> None:
    source = np.ascontiguousarray([2.0, -1.0, 4.5, 8.0, 3.0], dtype=np.float32)
    indices = np.ascontiguousarray([4, 0, 3], dtype=np.int64)
    output = np.zeros(3, dtype=np.float32)
    package = package_abi(
        "gather_f32", pipeline_name="tessera-lower-to-x86",
        buffer_shapes={"source": source.shape, "indices": indices.shape,
                       "output": output.shape},
    )
    result = _launch(package, {
        "source": source, "indices": indices, "output": output,
        "SourceN": source.size, "N": indices.size,
    })
    assert result["ok"] is True, result.get("reason")
    assert result["execution_kind"] == "native_cpu"
    np.testing.assert_array_equal(output, source[indices])


@pytest.mark.skipif(not tools_available(), reason="x86 compiler/shared library unavailable")
def test_cohort4_loss_and_stateful_cache_descriptors_execute_on_exact_host() -> None:
    prediction = np.ascontiguousarray([1.0, -2.0, 0.5, 4.0], dtype=np.float32)
    target = np.ascontiguousarray([0.0, -1.0, 1.5, 2.0], dtype=np.float32)
    loss = np.zeros_like(prediction)
    loss_package = package_abi(
        "pointwise_loss_f32", pipeline_name="tessera-lower-to-x86",
        buffer_shapes={"prediction": prediction.shape, "target": target.shape,
                       "output": loss.shape},
    )
    result = _launch(loss_package, {
        "prediction": prediction, "target": target, "output": loss,
        "N": prediction.size, "Kind": 0, "Parameter": 0.0,
    })
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_allclose(loss, (prediction - target) ** 2, rtol=0.0, atol=1e-7)

    cache = np.zeros((8, 4), dtype=np.float32)
    rows = np.ascontiguousarray(np.arange(8, dtype=np.float32).reshape(2, 4))
    cache_package = package_abi(
        "kv_cache_append_f32", pipeline_name="tessera-lower-to-x86",
        buffer_shapes={"cache": cache.shape, "rows": rows.shape},
    )
    result = _launch(cache_package, {
        "cache": cache, "rows": rows, "MaxSequence": 8, "RowLength": 4,
        "Start": 3, "RowCount": 2,
    })
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_array_equal(cache[3:5], rows)


@pytest.mark.skipif(not tools_available(), reason="x86 compiler/shared library unavailable")
@pytest.mark.parametrize("family", ["gather", "pointwise_loss", "cholesky", "tri_solve"])
def test_promoted_graph_breadth_executes_on_exact_host(family: str) -> None:
    module = _graph_module(family)
    contract = graph_breadth_contract(module)
    assert contract is not None
    package = package_graph_breadth(module, pipeline_name="tessera-lower-to-x86")
    rng = np.random.default_rng(86500 + len(family))
    if family == "gather":
        source = np.ascontiguousarray(rng.normal(size=2_097_152), dtype=np.float32)
        indices = np.ascontiguousarray(rng.integers(0, source.size, size=1_048_576), dtype=np.int64)
        output = np.zeros(indices.shape, dtype=np.float32)
        values = {"x": source, "ids": indices, "result": output}
        expected = source[indices]
    elif family == "pointwise_loss":
        prediction = np.ascontiguousarray(rng.normal(size=1_048_576), dtype=np.float32)
        target = np.ascontiguousarray(rng.normal(size=1_048_576), dtype=np.float32)
        output = np.zeros_like(prediction)
        values = {"prediction": prediction, "target": target, "result": output}
        expected = (prediction - target) ** 2
    elif family == "cholesky":
        raw = rng.normal(size=(2, 32, 32)).astype(np.float32)
        matrix = np.ascontiguousarray(raw @ raw.transpose(0, 2, 1) + 2.0 * np.eye(32, dtype=np.float32))
        output = np.zeros_like(matrix)
        values = {"matrix": matrix, "result": output}
        expected = np.linalg.cholesky(matrix)
    else:
        lower = np.tril(rng.normal(size=(2, 64, 64))).astype(np.float32)
        lower[:, np.arange(64), np.arange(64)] += np.float32(65.0)
        matrix = np.ascontiguousarray(lower)
        rhs = np.ascontiguousarray(rng.normal(size=(2, 64, 4)), dtype=np.float32)
        output = np.zeros_like(rhs)
        values = {"matrix": matrix, "rhs": rhs, "result": output}
        expected = np.stack([np.linalg.solve(matrix[b], rhs[b]) for b in range(2)])
    values.update(cast(dict[str, object], contract["scalars"]))
    result = _launch(package, values)
    assert result["ok"] is True, result.get("reason")
    assert result["execution_kind"] == "native_cpu"
    np.testing.assert_allclose(output, expected, rtol=3e-4, atol=3e-4)
