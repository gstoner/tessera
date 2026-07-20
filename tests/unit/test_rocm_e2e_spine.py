"""Contracts for ROCM-E2E typed softmax and reduction packages."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from tessera.compiler.driver import compile_graph_module
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.native_artifact import (
    ArtifactContractError,
    BufferArgument,
    DeviceLibraryRecord,
)
from tessera.compiler.rocm_native import (
    GFX1151_MOE_DISPATCH_F32_ABI,
    GFX1151_PAGED_KV_F32_ABI,
    GFX1151_REDUCE_BF16_ABI,
    GFX1151_REDUCE_F16_ABI,
    GFX1151_REDUCE_F32_ABI,
    GFX1151_SOFTMAX_F32_ABI,
    _driver_selected_device_libraries,
    emit_moe_dispatch_tile_ir,
    emit_reduce_tile_ir,
    emit_paged_kv_read_tile_ir,
    emit_softmax_tile_ir,
    package_moe_dispatch,
    package_reduction,
    package_paged_kv_read,
    package_softmax,
    requests_moe_dispatch,
    requests_reduction,
    requests_paged_kv_read,
    requests_softmax,
    supports_moe_dispatch,
    supports_reduction,
    supports_paged_kv_read,
    supports_softmax,
)


def _moe_dispatch_module(*, tokens: int = 7, slots: int = 9, hidden: int = 13) -> GraphIRModule:
    x = IRType(f"tensor<{tokens}x{hidden}xf32>", (str(tokens), str(hidden)), "fp32")
    token = IRType(f"tensor<{slots}xi32>", (str(slots),), "int32")
    output = IRType(f"tensor<{slots}x{hidden}xf32>", (str(slots), str(hidden)), "fp32")
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="gfx1151_moe_dispatch",
                args=[IRArg("x", x), IRArg("token", token)],
                result_types=[output],
                body=[
                    IROp(
                        result="o",
                        op_name="tessera.moe_dispatch",
                        operands=["%x", "%token"],
                        operand_types=[str(x), str(token)],
                        result_type=str(output),
                        kwargs={},
                    )
                ],
                return_values=["%o"],
            )
        ]
    )


def _paged_kv_module(*, start: int = 3, end: int = 10) -> GraphIRModule:
    pages = IRType("tensor<4x4x3x8xf32>", ("4", "4", "3", "8"), "fp32")
    table = IRType("tensor<4xi32>", ("4",), "int32")
    output = IRType(f"tensor<{end - start}x3x8xf32>", (str(end - start), "3", "8"), "fp32")
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="gfx1151_paged_kv",
                args=[IRArg("pages", pages), IRArg("page_table", table)],
                result_types=[output],
                body=[
                    IROp(
                        result="slice",
                        op_name="tessera.kv_cache.read",
                        operands=["%pages", "%page_table"],
                        operand_types=[str(pages), str(table)],
                        result_type=str(output),
                        kwargs={"start": start, "end": end},
                    )
                ],
                return_values=["%slice"],
            )
        ]
    )


def _reduction_module(
    *,
    dtype: str = "fp32",
    kind: str = "sum",
    shape: tuple[int, ...] = (2, 3, 5),
    axis: int = 1,
    keepdims: bool = False,
) -> GraphIRModule:
    element = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}[dtype]
    tensor = IRType(
        f"tensor<{'x'.join(map(str, shape))}x{element}>",
        tuple(map(str, shape)),
        dtype,
    )
    normalized = axis + len(shape) if axis < 0 else axis
    output_shape = shape[:normalized] + ((1,) if keepdims else ()) + shape[normalized + 1 :]
    output = IRType(
        f"tensor<{'x'.join(map(str, output_shape))}xf32>",
        tuple(map(str, output_shape)),
        "fp32",
    )
    op_name = {"sum": "tessera.sum", "mean": "tessera.mean", "max": "tessera.max"}[kind]
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="gfx1151_reduce",
                args=[IRArg("x", tensor)],
                result_types=[output],
                body=[
                    IROp(
                        result="o",
                        op_name=op_name,
                        operands=["%x"],
                        operand_types=[str(tensor)],
                        result_type=str(output),
                        kwargs={"axis": axis, "keepdims": keepdims},
                    )
                ],
                return_values=["%o"],
            )
        ]
    )


def _fake_reduce_compile(tile_ir: str):
    assert "tile.reduce_kernel" in tile_ir
    return (
        'module { "tessera_rocm.reduce"() {layout = "outer_axis_inner"} : () -> () }',
        "gpu.binary @binary",
        b"\x7fELFrocm-e2e-2",
        "compiler",
        "toolchain",
        (DeviceLibraryRecord("rocm.ocml", "1" * 64, "compiler_driver"),),
        "cold",
    )


def _fake_paged_kv_compile(tile_ir: str):
    assert "tile.paged_kv_read_kernel" in tile_ir
    return (
        'module { "tessera_rocm.paged_kv_read"() {route = "direct"} : () -> () }',
        "gpu.binary @binary",
        b"\x7fELFrocm-e2e-2-paged-kv",
        "compiler",
        "toolchain",
        (DeviceLibraryRecord("rocm.ocml", "1" * 64, "compiler_driver"),),
        "cold",
    )


def _fake_moe_dispatch_compile(tile_ir: str):
    assert "tile.moe_dispatch_kernel" in tile_ir
    return (
        'module { "tessera_rocm.moe_dispatch"() {route = "direct_gather"} : () -> () }',
        "gpu.binary @binary",
        b"\x7fELFrocm-e2e-2-moe",
        "compiler",
        "toolchain",
        (DeviceLibraryRecord("rocm.ocml", "1" * 64, "compiler_driver"),),
        "cold",
    )


def _softmax_module(dtype: str = "fp32", shape: tuple[int, ...] = (3, 17)) -> GraphIRModule:
    element = "f16" if dtype == "fp16" else "f32"
    extents = "x".join(str(value) for value in shape)
    tensor = IRType(f"tensor<{extents}x{element}>", tuple(str(value) for value in shape), dtype)
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="gfx1151_softmax",
                args=[IRArg("x", tensor)],
                result_types=[tensor],
                body=[
                    IROp(
                        result="o",
                        op_name="tessera.softmax",
                        operands=["%x"],
                        operand_types=[str(tensor)],
                        result_type=str(tensor),
                        kwargs={"axis": -1},
                    )
                ],
                return_values=["%o"],
            )
        ]
    )


def _fake_compile(tile_ir: str):
    assert "tile.softmax_kernel" in tile_ir
    target = (
        "module { llvm.func @tessera_tile_softmax_f32() { "
        '"tessera_rocm.softmax"() {name = "tessera_tile_softmax_f32", '
        'dtype = "f32"} : () -> () llvm.return } }'
    )
    libraries = (
        DeviceLibraryRecord("rocm.ocml", "1" * 64, "compiler_driver"),
        DeviceLibraryRecord("rocm.ockl", "2" * 64, "compiler_driver"),
        DeviceLibraryRecord("rocm.oclc_isa_version_1151", "3" * 64, "compiler_driver"),
    )
    return (
        target,
        "gpu.binary @binary",
        b"\x7fELFrocm-e2e-1",
        "compiler",
        "toolchain",
        libraries,
        "cold",
    )


def test_rocm_softmax_emitter_uses_shared_typed_envelope() -> None:
    source = emit_softmax_tile_ir(entry="tessera_tile_softmax_f32", storage="f32")
    assert "tile.softmax_kernel" in source
    assert 'storage = "f32", accum = "f32", axis = -1' in source
    assert 'exp_mode = "accurate", ftz = false' in source
    assert "tessera_rocm.softmax" not in source


def test_rocm_softmax_request_is_narrow_and_static() -> None:
    module = _softmax_module()
    assert requests_softmax(module)
    assert supports_softmax(module)
    module.functions[0].body[0].kwargs["axis"] = 0
    assert requests_softmax(module)
    assert not supports_softmax(module)


@pytest.mark.parametrize("shape", [(1,), (1, 1), (4, 256), (3, 17), (2, 257)])
def test_rocm_softmax_contract_accepts_boundary_aligned_and_ragged_shapes(
    shape,
) -> None:
    assert supports_softmax(_softmax_module(shape=shape))


def test_rocm_softmax_contract_rejects_invalid_dtype_dynamic_shape_and_result() -> None:
    unsupported_dtype = _softmax_module()
    unsupported_dtype.functions[0].args[0].ir_type = IRType("tensor<3x17xbf16>", ("3", "17"), "bf16")
    assert not supports_softmax(unsupported_dtype)

    dynamic = _softmax_module()
    dynamic.functions[0].args[0].ir_type = IRType("tensor<?x17xf32>", ("?", "17"), "fp32")
    assert not supports_softmax(dynamic)

    mismatched_result = _softmax_module()
    mismatched_result.functions[0].result_types[0] = IRType("tensor<3x17xf16>", ("3", "17"), "fp16")
    assert not supports_softmax(mismatched_result)


def test_rocm_driver_selected_device_libraries_are_content_addressed(monkeypatch, tmp_path) -> None:
    names = (
        "ocml.bc",
        "ockl.bc",
        "oclc_unsafe_math_off.bc",
        "oclc_finite_only_off.bc",
        "oclc_wavefrontsize64_off.bc",
        "oclc_isa_version_1151.bc",
        "oclc_abi_version_600.bc",
    )
    paths = []
    for index, name in enumerate(names):
        path = tmp_path / name
        path.write_bytes(f"library-{index}".encode())
        paths.append(path)
    transcript = " ".join(f'"-mlink-builtin-bitcode" "{path}"' for path in paths)
    monkeypatch.setattr("tessera.compiler.rocm_native._rocm_path", lambda: tmp_path)
    monkeypatch.setattr("tessera.compiler.rocm_native._rocm_clang", lambda _path: tmp_path / "clang")
    monkeypatch.setattr(
        "tessera.compiler.rocm_native.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=transcript),
    )

    records = _driver_selected_device_libraries()
    assert [record.logical_name for record in records] == [f"rocm.{path.stem}" for path in paths]
    assert all(record.link_mode == "compiler_driver" for record in records)
    assert all(len(record.content_digest) == 64 for record in records)
    assert not any(str(tmp_path) in str(record.to_dict()) for record in records)


def test_rocm_softmax_package_owns_hsaco_and_descriptor(monkeypatch) -> None:
    monkeypatch.setattr("tessera.compiler.rocm_native._compile_tile_ir", _fake_compile)
    package = package_softmax(_softmax_module(), pipeline_name="tessera-lower-to-rocm")
    assert package.image.target == "rocm_gfx1151"
    assert package.image.architecture == "gfx1151"
    assert package.image.binary_format == "hsaco"
    assert [item.logical_name for item in package.image.device_libraries] == [
        "rocm.ocml",
        "rocm.ockl",
        "rocm.oclc_isa_version_1151",
    ]
    assert package.descriptor.abi_id == GFX1151_SOFTMAX_F32_ABI
    assert package.descriptor.entry_symbol == "tessera_tile_softmax_f32"
    assert [item.name for item in package.descriptor.buffers] == ["x", "o"]
    assert [item.name for item in package.descriptor.scalars] == ["Rows", "K"]
    assert package.descriptor.provenance["work_item"] == "ROCM-E2E-1"
    assert package.descriptor.provenance["schedule"] == "workgroup_per_row_256"


@pytest.mark.parametrize("failure", ["dtype", "shape", "scalar"])
def test_rocm_softmax_descriptor_rejects_invalid_invocations(monkeypatch, failure) -> None:
    monkeypatch.setattr("tessera.compiler.rocm_native._compile_tile_ir", _fake_compile)
    package = package_softmax(_softmax_module(), pipeline_name="tessera-lower-to-rocm")
    x = BufferArgument("fp32", (3, 17), "row_major", 64)
    output = BufferArgument("fp32", (3, 17), "row_major", 64)
    scalars = {"Rows": 3, "K": 17}
    if failure == "dtype":
        x = BufferArgument("fp16", (3, 17), "row_major", 64)
    elif failure == "shape":
        output = BufferArgument("fp32", (3, 18), "row_major", 64)
    else:
        scalars["K"] = True
    with pytest.raises(ArtifactContractError, match="E_LAUNCH_BINDING_MISMATCH"):
        package.descriptor.validate_invocation(package.image, {"x": x, "o": output}, scalars)


def test_driver_joins_exact_gfx1151_native_package(monkeypatch) -> None:
    monkeypatch.setattr("tessera.compiler.rocm_native._compile_tile_ir", _fake_compile)
    bundle = compile_graph_module(
        _softmax_module(),
        source_origin="unit",
        target="rocm_gfx1151",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None
    assert bundle.launch_descriptor is not None
    assert bundle.orchestration_state == "launchable"
    assert bundle.tile is not None and "tile.softmax_kernel" in bundle.tile.text
    assert bundle.target_ir is not None and "tessera_rocm.softmax" in bundle.target_ir.text
    assert any(event.pass_name == "rocm-gfx1151-native-package" for event in bundle.trace_events)


def test_rocm_reduction_emitter_and_contract_are_typed_and_arbitrary_axis() -> None:
    source = emit_reduce_tile_ir(
        entry="tessera_tile_reduce_sum_f32",
        storage="f32",
        kind="sum",
        axis=1,
        keepdims=False,
    )
    assert "tile.reduce_kernel" in source
    assert 'storage = "f32", accum = "f32", kind = "sum"' in source
    assert 'schedule = "serial", nan_mode = "propagate"' in source
    module = _reduction_module(axis=1)
    assert requests_reduction(module)
    assert supports_reduction(module)


def test_rocm_reduction_package_owns_outer_axis_inner_descriptor(monkeypatch) -> None:
    monkeypatch.setattr(
        "tessera.compiler.rocm_native._compile_reduction_tile_ir",
        _fake_reduce_compile,
    )
    package = package_reduction(_reduction_module(axis=1), pipeline_name="tessera-lower-to-rocm")
    assert package.descriptor.abi_id == GFX1151_REDUCE_F32_ABI
    assert [item.name for item in package.descriptor.scalars] == ["Outer", "AxisExtent", "Inner"]
    assert package.descriptor.provenance["work_item"] == "ROCM-E2E-2"
    assert package.descriptor.provenance["outer"] == 2
    assert package.descriptor.provenance["axis_extent"] == 3
    assert package.descriptor.provenance["inner"] == 5


@pytest.mark.parametrize(
    "dtype,abi",
    [
        ("fp16", GFX1151_REDUCE_F16_ABI),
        ("bf16", GFX1151_REDUCE_BF16_ABI),
        ("fp32", GFX1151_REDUCE_F32_ABI),
    ],
)
def test_rocm_reduction_package_has_storage_keyed_f32_output_abi(monkeypatch, dtype, abi) -> None:
    monkeypatch.setattr(
        "tessera.compiler.rocm_native._compile_reduction_tile_ir",
        _fake_reduce_compile,
    )
    package = package_reduction(_reduction_module(dtype=dtype), pipeline_name="tessera-lower-to-rocm")
    assert package.descriptor.abi_id == abi
    assert package.descriptor.buffers[0].dtype == dtype
    assert package.descriptor.buffers[1].dtype == "fp32"


def test_driver_joins_gfx1151_reduction_native_package(monkeypatch) -> None:
    monkeypatch.setattr(
        "tessera.compiler.rocm_native._compile_reduction_tile_ir",
        _fake_reduce_compile,
    )
    bundle = compile_graph_module(
        _reduction_module(axis=1),
        source_origin="unit",
        target="rocm_gfx1151",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.orchestration_state == "launchable"
    assert bundle.tile is not None and "tile.reduce_kernel" in bundle.tile.text
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.provenance["work_item"] == "ROCM-E2E-2"


def test_rocm_paged_kv_owns_typed_direct_descriptor(monkeypatch) -> None:
    module = _paged_kv_module()
    assert requests_paged_kv_read(module)
    assert supports_paged_kv_read(module)
    assert not supports_paged_kv_read(_paged_kv_module(start=-1, end=2))
    source = emit_paged_kv_read_tile_ir(entry="paged")
    assert "tile.paged_kv_read_kernel" in source
    assert 'table_storage = "i32"' in source
    monkeypatch.setattr(
        "tessera.compiler.rocm_native._compile_paged_kv_tile_ir",
        _fake_paged_kv_compile,
    )
    package = package_paged_kv_read(module, pipeline_name="tessera-lower-to-rocm")
    assert package.descriptor.abi_id == GFX1151_PAGED_KV_F32_ABI
    assert [item.name for item in package.descriptor.buffers] == [
        "pages",
        "page_table",
        "slice",
    ]
    assert [item.name for item in package.descriptor.scalars] == [
        "P",
        "LP",
        "PageSize",
        "H",
        "D",
        "Start",
        "Tokens",
    ]


def test_rocm_paged_kv_contract_rejects_bounds_dtype_and_output_drift() -> None:
    assert not supports_paged_kv_read(_paged_kv_module(start=0, end=17))

    table_dtype = _paged_kv_module()
    table_dtype.functions[0].args[1].ir_type = IRType("tensor<4xi64>", ("4",), "int64")
    assert not supports_paged_kv_read(table_dtype)

    result_shape = _paged_kv_module()
    result_shape.functions[0].result_types[0] = IRType(
        "tensor<7x3x9xf32", ("7", "3", "9"), "fp32"
    )
    assert not supports_paged_kv_read(result_shape)


def test_driver_joins_gfx1151_paged_kv_native_package(monkeypatch) -> None:
    monkeypatch.setattr(
        "tessera.compiler.rocm_native._compile_paged_kv_tile_ir",
        _fake_paged_kv_compile,
    )
    bundle = compile_graph_module(
        _paged_kv_module(),
        source_origin="unit",
        target="rocm_gfx1151",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.orchestration_state == "launchable"
    assert bundle.tile is not None and "tile.paged_kv_read_kernel" in bundle.tile.text
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.abi_id == GFX1151_PAGED_KV_F32_ABI


def test_rocm_moe_dispatch_owns_typed_direct_descriptor(monkeypatch) -> None:
    module = _moe_dispatch_module()
    assert requests_moe_dispatch(module)
    assert supports_moe_dispatch(module)
    source = emit_moe_dispatch_tile_ir(entry="dispatch")
    assert "tile.moe_dispatch_kernel" in source
    assert 'storage = "f32", index_storage = "i32"' in source
    monkeypatch.setattr(
        "tessera.compiler.rocm_native._compile_moe_dispatch_tile_ir",
        _fake_moe_dispatch_compile,
    )
    package = package_moe_dispatch(module, pipeline_name="tessera-lower-to-rocm")
    assert package.descriptor.abi_id == GFX1151_MOE_DISPATCH_F32_ABI
    assert [item.name for item in package.descriptor.buffers] == ["x", "token", "o"]
    assert [item.name for item in package.descriptor.scalars] == ["T", "S", "H"]
    assert package.descriptor.provenance["route"] == "direct_gather"


def test_driver_joins_gfx1151_moe_dispatch_native_package(monkeypatch) -> None:
    monkeypatch.setattr(
        "tessera.compiler.rocm_native._compile_moe_dispatch_tile_ir",
        _fake_moe_dispatch_compile,
    )
    bundle = compile_graph_module(
        _moe_dispatch_module(),
        source_origin="unit",
        target="rocm_gfx1151",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.orchestration_state == "launchable"
    assert bundle.tile is not None and "tile.moe_dispatch_kernel" in bundle.tile.text
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.abi_id == GFX1151_MOE_DISPATCH_F32_ABI


def test_rocm_moe_dispatch_contract_and_launcher_reject_invalid_indices(monkeypatch) -> None:
    from tessera import runtime as rt

    bad_output = _moe_dispatch_module()
    bad_output.functions[0].result_types[0] = IRType("tensor<9x14xf32>", ("9", "14"), "fp32")
    assert not supports_moe_dispatch(bad_output)
    monkeypatch.setattr(
        "tessera.compiler.rocm_native._compile_moe_dispatch_tile_ir",
        _fake_moe_dispatch_compile,
    )
    package = package_moe_dispatch(_moe_dispatch_module(), pipeline_name="tessera-lower-to-rocm")
    with pytest.raises(RuntimeError, match="arrays disagree"):
        rt._submit_rocm_gfx1151_native(
            package.image,
            package.descriptor,
            {
                "x": np.zeros((7, 13), dtype=np.float32),
                "token": np.array([0, 1, 2, 3, 4, 5, 6, 0, 7], dtype=np.int32),
                "o": np.zeros((9, 13), dtype=np.float32),
            },
            {"T": 7, "S": 9, "H": 13},
            None,
        )


@pytest.mark.parametrize(
    "table,start,tokens,output_shape,error",
    [
        ([0, 1, 2, 4], 3, 7, (7, 3, 8), "arrays disagree"),
        ([0, 1, 2, 3], -1, 7, (7, 3, 8), "arrays disagree"),
        ([0, 1, 2, 3], 15, 2, (2, 3, 8), "arrays disagree"),
        ([0, 1, 2, 3], 3, 7, (7, 3, 9), "arrays disagree"),
    ],
)
def test_rocm_paged_kv_rejects_invalid_launch_before_hip(
    monkeypatch, table, start, tokens, output_shape, error
) -> None:
    from tessera import runtime as rt

    monkeypatch.setattr(
        "tessera.compiler.rocm_native._compile_paged_kv_tile_ir",
        _fake_paged_kv_compile,
    )
    package = package_paged_kv_read(_paged_kv_module(), pipeline_name="tessera-lower-to-rocm")
    with pytest.raises(RuntimeError, match=error):
        rt._submit_rocm_gfx1151_native(
            package.image,
            package.descriptor,
            {
                "pages": np.zeros((4, 4, 3, 8), dtype=np.float32),
                "page_table": np.array(table, dtype=np.int32),
                "slice": np.zeros(output_shape, dtype=np.float32),
            },
            {
                "P": 4,
                "LP": 4,
                "PageSize": 4,
                "H": 3,
                "D": 8,
                "Start": start,
                "Tokens": tokens,
            },
            None,
        )


def test_builtin_gfx1151_launcher_registers_typed_rocm_abis(monkeypatch) -> None:
    from tessera import runtime as rt

    rt.unregister_native_launcher("rocm_gfx1151")
    rt._ensure_builtin_native_launcher("rocm_gfx1151", GFX1151_SOFTMAX_F32_ABI)
    try:
        registration = rt._native_launchers["rocm_gfx1151"]
        assert registration.binary_formats == ("hsaco",)
        assert registration.submit is rt._submit_rocm_gfx1151_native
    finally:
        rt.unregister_native_launcher("rocm_gfx1151")


def test_builtin_gfx1151_launcher_registers_moe_abi_in_isolation() -> None:
    from tessera import runtime as rt

    rt.unregister_native_launcher("rocm_gfx1151")
    rt._ensure_builtin_native_launcher("rocm_gfx1151", GFX1151_MOE_DISPATCH_F32_ABI)
    try:
        assert "rocm_gfx1151" in rt._native_launchers
        assert rt._native_launchers["rocm_gfx1151"].submit is rt._submit_rocm_gfx1151_native
    finally:
        rt.unregister_native_launcher("rocm_gfx1151")


@pytest.mark.skipif(
    os.environ.get("TESSERA_ROCM_E2E_DEVICE_TEST") != "1",
    reason="set TESSERA_ROCM_E2E_DEVICE_TEST=1 on the exact gfx1151 host",
)
@pytest.mark.parametrize(
    "dtype,kind,axis,keepdims",
    [
        ("fp16", "sum", 0, False),
        ("bf16", "mean", 1, True),
        ("fp32", "max", 2, False),
    ],
)
def test_exact_gfx1151_reduction_descriptor_matches_arbitrary_axis_oracle(dtype, kind, axis, keepdims) -> None:
    from tessera import runtime as rt

    module = _reduction_module(dtype=dtype, kind=kind, axis=axis, keepdims=keepdims)
    package = package_reduction(module, pipeline_name="tessera-lower-to-rocm")
    artifact = rt.RuntimeArtifact(
        graph_ir="graph",
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
        metadata={"target": "rocm_gfx1151"},
        native_image=package.image,
        launch_descriptor=package.descriptor,
    )
    rng = np.random.default_rng(2202)
    if dtype == "bf16":
        ml_dtypes = pytest.importorskip("ml_dtypes")
        numpy_dtype = ml_dtypes.bfloat16
    else:
        numpy_dtype = np.float16 if dtype == "fp16" else np.float32
    x = np.ascontiguousarray(rng.standard_normal((2, 3, 5)), dtype=numpy_dtype)
    oracle = {"sum": np.sum, "mean": np.mean, "max": np.max}[kind]
    expected = oracle(x.astype(np.float32), axis=axis, keepdims=keepdims).astype(np.float32)
    output = np.zeros(expected.shape, dtype=np.float32)
    result = rt.launch(
        artifact,
        {
            "x": x,
            "o": output,
            "Outer": package.descriptor.provenance["outer"],
            "AxisExtent": package.descriptor.provenance["axis_extent"],
            "Inner": package.descriptor.provenance["inner"],
        },
    )
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_allclose(result["output"], expected, atol=2e-5, rtol=0)


@pytest.mark.skipif(
    os.environ.get("TESSERA_ROCM_E2E_DEVICE_TEST") != "1",
    reason="set TESSERA_ROCM_E2E_DEVICE_TEST=1 on the exact gfx1151 host",
)
@pytest.mark.parametrize("start,end", [(0, 1), (3, 10), (7, 9), (0, 16)])
def test_exact_gfx1151_paged_kv_descriptor_matches_permuted_page_oracle(start, end) -> None:
    from tessera import runtime as rt

    module = _paged_kv_module(start=start, end=end)
    package = package_paged_kv_read(module, pipeline_name="tessera-lower-to-rocm")
    artifact = rt.RuntimeArtifact(
        graph_ir="graph",
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
        metadata={"target": "rocm_gfx1151"},
        native_image=package.image,
        launch_descriptor=package.descriptor,
    )
    rng = np.random.default_rng(2203)
    pages = np.ascontiguousarray(rng.standard_normal((4, 4, 3, 8)), dtype=np.float32)
    table = np.array([2, 0, 3, 1], dtype=np.int32)
    logical = pages[table].reshape(16, 3, 8)
    tokens = end - start
    output = np.zeros((tokens, 3, 8), dtype=np.float32)
    result = rt.launch(
        artifact,
        {
            "pages": pages,
            "page_table": table,
            "slice": output,
            "P": 4,
            "LP": 4,
            "PageSize": 4,
            "H": 3,
            "D": 8,
            "Start": start,
            "Tokens": tokens,
        },
    )
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_array_equal(output, logical[start:end])


@pytest.mark.skipif(
    os.environ.get("TESSERA_ROCM_E2E_DEVICE_TEST") != "1",
    reason="set TESSERA_ROCM_E2E_DEVICE_TEST=1 on the exact gfx1151 host",
)
@pytest.mark.parametrize("tokens,slots,hidden", [(1, 1, 1), (7, 9, 13), (17, 5, 257)])
def test_exact_gfx1151_moe_dispatch_descriptor_matches_gather_oracle(tokens, slots, hidden) -> None:
    from tessera import runtime as rt

    module = _moe_dispatch_module(tokens=tokens, slots=slots, hidden=hidden)
    package = package_moe_dispatch(module, pipeline_name="tessera-lower-to-rocm")
    artifact = rt.RuntimeArtifact(
        graph_ir="graph",
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
        metadata={"target": "rocm_gfx1151"},
        native_image=package.image,
        launch_descriptor=package.descriptor,
    )
    rng = np.random.default_rng(2204)
    x = np.ascontiguousarray(rng.standard_normal((tokens, hidden)), dtype=np.float32)
    token = np.ascontiguousarray(rng.integers(0, tokens, size=slots), dtype=np.int32)
    output = np.zeros((slots, hidden), dtype=np.float32)
    result = rt.launch(
        artifact,
        {"x": x, "token": token, "o": output, "T": tokens, "S": slots, "H": hidden},
    )
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_array_equal(output, x[token])


@pytest.mark.skipif(
    os.environ.get("TESSERA_ROCM_E2E_DEVICE_TEST") != "1",
    reason="set TESSERA_ROCM_E2E_DEVICE_TEST=1 on the exact gfx1151 host",
)
@pytest.mark.parametrize(
    "dtype,numpy_dtype,tolerance",
    [("fp32", np.float32, 1e-5), ("fp16", np.float16, 3e-3)],
)
@pytest.mark.parametrize("shape", [(1, 1), (4, 256), (3, 17), (2, 257)])
def test_exact_gfx1151_descriptor_launch_matches_oracle(dtype, numpy_dtype, tolerance, shape) -> None:
    from tessera import runtime as rt

    package = package_softmax(_softmax_module(dtype, shape), pipeline_name="tessera-lower-to-rocm")
    artifact = rt.RuntimeArtifact(
        graph_ir="graph",
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
        metadata={
            "target": "rocm_gfx1151",
            "compiler_path": "rocm_gfx1151_native_descriptor",
        },
        native_image=package.image,
        launch_descriptor=package.descriptor,
    )
    rng = np.random.default_rng(1151)
    x = np.ascontiguousarray(rng.standard_normal(shape), dtype=numpy_dtype)
    output = np.zeros_like(x)
    result = rt.launch(
        artifact,
        {"x": x, "o": output, "Rows": shape[0], "K": shape[-1]},
    )
    assert result["ok"] is True, result.get("reason")
    xf = x.astype(np.float32)
    shifted = xf - xf.max(axis=-1, keepdims=True)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(result["output"].astype(np.float32), expected, atol=tolerance, rtol=0)


@pytest.mark.skipif(
    os.environ.get("TESSERA_ROCM_E2E_DEVICE_TEST") != "1",
    reason="set TESSERA_ROCM_E2E_DEVICE_TEST=1 on the exact gfx1151 host",
)
def test_exact_gfx1151_cold_warm_identity_and_driver_library_provenance() -> None:
    from tessera.compiler import rocm_native

    rocm_native._cache.clear()
    cold = package_softmax(_softmax_module(), pipeline_name="tessera-lower-to-rocm")
    warm = package_softmax(_softmax_module(), pipeline_name="tessera-lower-to-rocm")
    assert cold.image.compile_state == "cold"
    assert warm.image.compile_state == "warm_cache"
    assert cold.image.image_digest == warm.image.image_digest
    assert cold.image.payload == warm.image.payload
    assert cold.image.device_libraries == warm.image.device_libraries
    names = {item.logical_name for item in cold.image.device_libraries}
    assert {"rocm.ocml", "rocm.ockl"}.issubset(names)
    assert {
        "rocm.oclc_unsafe_math_off",
        "rocm.oclc_finite_only_off",
        "rocm.oclc_wavefrontsize64_off",
        "rocm.oclc_isa_version_1151",
        "rocm.oclc_abi_version_600",
    }.issubset(names)
