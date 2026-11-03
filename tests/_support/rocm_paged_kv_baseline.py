"""Frozen pre-E2E-REAL-6 ROCm paged-KV packaging baseline (differential tests only).

This is ``rocm_native.package_paged_kv_read`` and its ``_paged_kv_contract``
as they stood on main ``b498bc6f0``. The kernel already came from the native
Graph -> Schedule -> Tile replay, but every launch field -- buffers, ABI,
scalar list, shape guards, geometry and provenance -- was authored here from
the Python Graph object, and the page layout was read off the Graph argument.
Production now projects those fields from the replayed Tile contract
(``rocm_native.package_scheduled_paged_kv``); this module is the **declared
oracle** Decision #31(a) allows. Its only consumers are the differential tests
in ``tests/unit/test_rocm_paged_kv_migration.py``.
"""
from __future__ import annotations

import hashlib

from tessera.compiler.graph_ir import GraphIRModule
from tessera.compiler.native_artifact import (
    BufferBinding,
    LaunchDescriptor,
    LaunchGeometry,
    NativeEntryPoint,
    NativeImageArtifact,
    OrderingSemantics,
    ScalarArgument,
    ShapeGuard,
)
from tessera.compiler import rocm_native
from tessera.compiler.rocm_native import (
    GFX_PAGED_KV_F32_ABI,
    GFX_PAGED_KV_STRIDED_F32_ABI,
    ROCMNativePackage,
    _SHAPE_FREE_DIRECTIVES,
    _directive_symbol,
    _shape,
)


def _paged_kv_contract(
    module: GraphIRModule,
) -> tuple[str, str, str, tuple[int, int, int, int, int, int, int]] | None:
    if not (len(module.functions) == 1 and len(module.functions[0].body) == 1
            and module.functions[0].body[0].op_name == "tessera.kv_cache.read"):
        return None
    function = module.functions[0]
    op = function.body[0]
    if len(op.operands) != 2 or len(function.result_types) != 1:
        return None
    pages_name, table_name = (value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in function.args}
    pages = args.get(pages_name)
    table = args.get(table_name)
    pages_shape = _shape(module, pages_name)
    table_shape = _shape(module, table_name)
    if (
        pages is None
        or table is None
        or pages.ir_type.dtype != "fp32"
        or table.ir_type.dtype != "int32"
        or pages_shape is None
        or len(pages_shape) != 4
        or table_shape is None
        or len(table_shape) != 1
    ):
        return None
    physical_pages, page_size, heads, dim = pages_shape
    logical_pages = table_shape[0]
    start = op.kwargs.get("start")
    end = op.kwargs.get("end")
    if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool):
        return None
    tokens = end - start
    result = function.result_types[0]
    try:
        result_shape = tuple(int(value) for value in result.shape)
    except (TypeError, ValueError):
        return None
    if (
        start < 0
        or tokens <= 0
        or end > logical_pages * page_size
        or result.dtype != "fp32"
        or result_shape != (tokens, heads, dim)
    ):
        return None
    return (
        pages_name,
        table_name,
        op.result or "output",
        (physical_pages, logical_pages, page_size, heads, dim, start, tokens),
    )


def _graph_layout(module: GraphIRModule, pages_name: str) -> str:
    arg = next(arg for arg in module.functions[0].args if arg.name == pages_name)
    layout = arg.layout or arg.ir_type.layout or "row_major"
    if layout not in {"row_major", "strided"}:
        raise ValueError("paged storage requires row_major or strided pages")
    return layout


def baseline_paged_kv_read(
    module: GraphIRModule, *, pipeline_name: str, architecture: str = "gfx1151",
) -> ROCMNativePackage:
    """The retired Graph-authored descriptor over the native replayed Tile."""
    from tessera.compiler.scheduled_paged_kv import lower_scheduled_paged_kv_graph

    contract = _paged_kv_contract(module)
    if contract is None:
        raise ValueError(
            "gfx1151 paged-KV packaging requires static f32 [P,PS,H,D] pages, "
            "rank-1 int32 page table, explicit valid start/end, and f32 output"
        )
    pages_name, table_name, output_name, dims = contract
    physical_pages, logical_pages, page_size, heads, dim, start, tokens = dims
    artifact = lower_scheduled_paged_kv_graph(module, target=f"rocm_{architecture}")
    layout = _graph_layout(module, pages_name)
    abi = GFX_PAGED_KV_STRIDED_F32_ABI if layout == "strided" else GFX_PAGED_KV_F32_ABI
    scalar_names: tuple[str, ...] = ("P", "LP", "PageSize", "H", "D", "Start", "Tokens")
    if layout == "strided":
        scalar_names += ("StrideP", "StridePage", "StrideH", "StrideD")
    tile_ir = artifact.tile_ir
    (
        target_ir,
        backend_ir,
        payload,
        compiler_fp,
        toolchain_fp,
        device_libraries,
        compile_state,
    ) = rocm_native._compile_paged_kv_tile_ir(tile_ir, architecture=architecture)
    entry = _directive_symbol(target_ir, _SHAPE_FREE_DIRECTIVES["paged_kv"])
    image = NativeImageArtifact(
        target=f"rocm_{architecture}",
        architecture=architecture,
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(),
        binary_format="hsaco",
        payload=payload,
        entry_points=(NativeEntryPoint(entry, abi),),
        compile_state=compile_state,
        device_libraries=device_libraries,
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=abi,
        buffers=(
            BufferBinding(0, pages_name, "input", "fp32", 4, layout, 4),
            BufferBinding(1, table_name, "input", "int32", 1, "row_major", 4),
            BufferBinding(2, output_name, "output", "fp32", 3, "row_major", 4),
        ),
        scalars=tuple(
            ScalarArgument(3 + index, name, "int64")
            for index, name in enumerate(scalar_names)
        ),
        shape_guards=(
            ShapeGuard(pages_name, 0, "eq", physical_pages),
            ShapeGuard(pages_name, 1, "eq", page_size),
            ShapeGuard(pages_name, 2, "eq", heads),
            ShapeGuard(pages_name, 3, "eq", dim),
            ShapeGuard(table_name, 0, "eq", logical_pages),
            ShapeGuard(output_name, 0, "eq", tokens),
            ShapeGuard(output_name, 1, "eq", heads),
            ShapeGuard(output_name, 2, "eq", dim),
        ),
        geometry=LaunchGeometry(policy=f"{architecture}_paged_kv_{'strided' if layout == 'strided' else 'direct'}_256"),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "E2E-REAL-6" if layout == "strided" else "ROCM-E2E-2",
            "sync_key": "ROCM-STRIDED-PAGED-KV-20261008" if layout == "strided" else "E2E-SPINE-2026-07-18",
            "route": "direct",
            "shape": list(dims),
            "storage": "f32",
            "table_storage": "i32",
            **({"page_layout": layout, "page_stride_policy": "positive_runtime_element_strides",
                "source_extent_policy": "checked_physical_span"} if layout == "strided" else {}),
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
            "schedule_digest": artifact.schedule_digest,
        },
    )
    return ROCMNativePackage(tile_ir, target_ir, backend_ir, image, descriptor)
