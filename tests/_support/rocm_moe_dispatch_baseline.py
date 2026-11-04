"""Frozen pre-E2E-REAL-6 ROCm MoE-dispatch packaging baseline (differential tests only).

This is ``rocm_native.package_moe_dispatch`` and its ``_moe_dispatch_contract``
as they stood on main ``07bf5eb38``. The kernel already came from the native
Graph -> Schedule -> Tile replay (``NativeMoeDispatch.h``), but every launch
field -- buffers, ABI, scalar list, shape guards, geometry and provenance --
was authored here from the Python Graph object. Production now projects those
fields from the replayed Tile contract
(``rocm_native.package_scheduled_moe_dispatch``); this module is the
**declared oracle** Decision #31(a) allows. Its only consumers are the
differential tests in ``tests/unit/test_rocm_moe_dispatch_migration.py``.
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
    GFX_MOE_DISPATCH_F32_ABI,
    ROCMNativePackage,
    _SHAPE_FREE_DIRECTIVES,
    _directive_symbol,
    _shape,
)


def _moe_dispatch_contract(
    module: GraphIRModule,
) -> tuple[str, str, str, tuple[int, int, int]] | None:
    if not (len(module.functions) == 1 and len(module.functions[0].body) == 1
            and module.functions[0].body[0].op_name == "tessera.moe_dispatch"):
        return None
    function = module.functions[0]
    op = function.body[0]
    if len(op.operands) != 2 or len(function.result_types) != 1:
        return None
    if op.kwargs.get("transport") is not None:
        return None
    x_name, token_name = (value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in function.args}
    x, token = args.get(x_name), args.get(token_name)
    x_shape, token_shape = _shape(module, x_name), _shape(module, token_name)
    if (
        x is None
        or token is None
        or x.ir_type.dtype != "fp32"
        or token.ir_type.dtype != "int32"
        or x_shape is None
        or len(x_shape) != 2
        or token_shape is None
        or len(token_shape) != 1
    ):
        return None
    tokens, hidden = x_shape
    slots = token_shape[0]
    result = function.result_types[0]
    try:
        result_shape = tuple(int(value) for value in result.shape)
    except (TypeError, ValueError):
        return None
    if result.dtype != "fp32" or result_shape != (slots, hidden):
        return None
    return x_name, token_name, op.result or "output", (tokens, slots, hidden)


def baseline_moe_dispatch(
    module: GraphIRModule, *, pipeline_name: str, architecture: str = "gfx1151",
) -> ROCMNativePackage:
    """The retired Graph-authored descriptor over the native replayed Tile."""
    from tessera.compiler.scheduled_moe_dispatch import lower_scheduled_moe_dispatch

    contract = _moe_dispatch_contract(module)
    if contract is None:
        raise ValueError(
            "gfx1151 MoE dispatch packaging requires static f32 [T,H] input, "
            "rank-1 int32 token indices, and f32 [S,H] output"
        )
    x_name, token_name, output_name, dims = contract
    tokens, slots, hidden = dims
    artifact = lower_scheduled_moe_dispatch(module, target=f"rocm_{architecture}")
    tile_ir = artifact.tile_ir
    (
        target_ir,
        backend_ir,
        payload,
        compiler_fp,
        toolchain_fp,
        device_libraries,
        compile_state,
    ) = rocm_native._compile_moe_dispatch_tile_ir(tile_ir, architecture=architecture)
    entry = _directive_symbol(target_ir, _SHAPE_FREE_DIRECTIVES["moe_dispatch"])
    image = NativeImageArtifact(
        target=f"rocm_{architecture}",
        architecture=architecture,
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(),
        binary_format="hsaco",
        payload=payload,
        entry_points=(NativeEntryPoint(entry, GFX_MOE_DISPATCH_F32_ABI),),
        compile_state=compile_state,
        device_libraries=device_libraries,
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=GFX_MOE_DISPATCH_F32_ABI,
        buffers=(
            BufferBinding(0, x_name, "input", "fp32", 2, "row_major", 4),
            BufferBinding(1, token_name, "input", "int32", 1, "row_major", 4),
            BufferBinding(2, output_name, "output", "fp32", 2, "row_major", 4),
        ),
        scalars=tuple(ScalarArgument(3 + index, name, "int64") for index, name in enumerate(("T", "S", "H"))),
        shape_guards=(
            ShapeGuard(x_name, 0, "eq", tokens),
            ShapeGuard(x_name, 1, "eq", hidden),
            ShapeGuard(token_name, 0, "eq", slots),
            ShapeGuard(output_name, 0, "eq", slots),
            ShapeGuard(output_name, 1, "eq", hidden),
        ),
        # Frozen text was the literal gfx1151 policy; parameterised on the
        # arch (2026-11-04) so the oracle can stand for gfx1201 too. At
        # architecture="gfx1151" it is byte-identical to the frozen packager.
        geometry=LaunchGeometry(policy=f"{architecture}_moe_dispatch_direct_256"),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "ROCM-E2E-2",
            "sync_key": "E2E-SPINE-2026-07-18",
            "route": "direct_gather",
            "schedule_digest": artifact.schedule_digest,
            "shape": list(dims),
            "storage": "f32",
            "index_storage": "i32",
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return ROCMNativePackage(tile_ir, target_ir, backend_ir, image, descriptor)
