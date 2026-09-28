"""Frozen pre-E2E-REAL-6 gfx1151 unary packaging baselines (differential tests only).

These are the Graph-owned ``rocm_native.package_softmax`` /
``package_reduction`` constructors as they stood on main ``d8da67f7``: they
read the Python Graph object and author Tile IR text directly, beside the
compiled Graph -> Schedule -> Tile route. Production callers now consume a
``ScheduledKernelArtifact`` (``rocm_native.package_scheduled_kernel``); this
module is the **declared oracle** Decision #31(a) allows -- its only consumers
are the differential tests in ``tests/unit/test_rocm_unary_migration.py``.

Two known divergences are kept verbatim, because a baseline that is quietly
corrected stops being evidence of what the retired route did:

* ``tessera.reduce`` derived its combiner from the op *name*, so
  ``tessera.reduce {kind = "max"}`` was packaged as a **sum** (the compiled
  route reads ``kind`` and computes the max; ``kind = "min"`` is refused).
* ``numeric_policy`` keyword arguments were ignored (unchanged by the
  migration; recorded as an open Decision #32 item across all targets).
"""
from __future__ import annotations

import hashlib
import math

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
    GFX_REDUCE_BF16_ABI,
    GFX_REDUCE_F16_ABI,
    GFX_REDUCE_F32_ABI,
    GFX_SOFTMAX_F16_ABI,
    GFX_SOFTMAX_F32_ABI,
    ROCMNativePackage,
    _shape,
    requests_reduction,
    requests_softmax,
)


def emit_softmax_tile_ir(*, entry: str, storage: str) -> str:
    """Emit the shared semantic softmax envelope with ROCm-owned math intent."""
    if storage not in {"f16", "f32"}:
        raise ValueError(f"unsupported gfx1151 softmax storage {storage!r}")
    return f'''module {{
  llvm.func @{entry}(%x: !llvm.ptr, %o: !llvm.ptr,
                     %rows: i64, %columns: i64) {{
    tile.softmax_kernel %x, %o, %rows, %columns {{
      storage = "{storage}", accum = "f32", axis = -1 : i64,
      exp_mode = "accurate", ftz = false
    }} : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }}
}}
'''

def emit_reduce_tile_ir(
    *, entry: str, storage: str, kind: str, axis: int, keepdims: bool, inner_is_one: bool = False
) -> str:
    """Emit the shared arbitrary-axis mixed-precision reduction envelope."""
    if storage not in {"f16", "bf16", "f32"}:
        raise ValueError(f"unsupported gfx1151 reduction storage {storage!r}")
    if kind not in {"sum", "mean", "max"}:
        raise ValueError(f"unsupported gfx1151 reduction kind {kind!r}")
    if axis < 0:
        raise ValueError("gfx1151 reduction requires a normalized axis")
    return f'''module {{
  llvm.func @{entry}(%x: !llvm.ptr, %o: !llvm.ptr,
                     %outer: i64, %axis_extent: i64, %inner: i64) {{
    tile.reduce_kernel %x, %o, %outer, %axis_extent, %inner {{
      storage = "{storage}", accum = "f32", kind = "{kind}",
      axis = {axis} : i64, keepdims = {str(keepdims).lower()},
      schedule = "serial", nan_mode = "propagate",
      inner_is_one = {str(inner_is_one).lower()}
    }} : !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }}
}}
'''

def _softmax_contract(
    module: GraphIRModule,
) -> tuple[str, str, str, tuple[int, ...]] | None:
    if not requests_softmax(module):
        return None
    function = module.functions[0]
    op = function.body[0]
    if len(op.operands) != 1 or op.kwargs.get("axis", -1) != -1:
        return None
    input_name = op.operands[0].removeprefix("%")
    arg = next((item for item in function.args if item.name == input_name), None)
    shape = _shape(module, input_name)
    if (
        arg is None
        or shape is None
        or arg.ir_type.dtype not in {"fp16", "fp32"}
        or not function.result_types
        or function.result_types[0].dtype != arg.ir_type.dtype
    ):
        return None
    return input_name, op.result or "output", arg.ir_type.dtype, shape

def _reduction_contract(
    module: GraphIRModule,
) -> tuple[str, str, str, str, tuple[int, ...], tuple[int, ...], int, bool] | None:
    if not requests_reduction(module):
        return None
    function = module.functions[0]
    op = function.body[0]
    if len(op.operands) != 1 or len(function.result_types) != 1:
        return None
    input_name = op.operands[0].removeprefix("%")
    arg = next((item for item in function.args if item.name == input_name), None)
    shape = _shape(module, input_name)
    if arg is None or arg.ir_type.dtype not in {"fp16", "bf16", "fp32"} or shape is None:
        return None
    raw_axis = op.kwargs.get("axis", -1)
    if not isinstance(raw_axis, int) or isinstance(raw_axis, bool):
        return None
    axis = raw_axis + len(shape) if raw_axis < 0 else raw_axis
    if axis < 0 or axis >= len(shape):
        return None
    keepdims = bool(op.kwargs.get("keepdims", False))
    output_shape = shape[:axis] + ((1,) if keepdims else ()) + shape[axis + 1 :]
    result = function.result_types[0]
    try:
        declared_output_shape = tuple(int(dim) for dim in result.shape)
    except (TypeError, ValueError):
        return None
    if result.dtype != "fp32" or declared_output_shape != output_shape:
        return None
    kind = "max" if op.op_name in {"tessera.max", "tessera.amax"} else "mean" if op.op_name == "tessera.mean" else "sum"
    return (
        input_name,
        op.result or "output",
        arg.ir_type.dtype,
        kind,
        shape,
        output_shape,
        axis,
        keepdims,
    )

def baseline_softmax(module: GraphIRModule, *, pipeline_name: str) -> ROCMNativePackage:
    contract = _softmax_contract(module)
    if contract is None:
        raise ValueError("gfx1151 native packaging requires one static f16/f32 last-axis softmax")
    input_name, output_name, dtype, shape = contract
    storage = "f16" if dtype == "fp16" else "f32"
    entry = f"tessera_tile_softmax_{storage}"
    abi_id = GFX_SOFTMAX_F16_ABI if dtype == "fp16" else GFX_SOFTMAX_F32_ABI
    alignment = 2 if dtype == "fp16" else 4
    tile_ir = emit_softmax_tile_ir(entry=entry, storage=storage)
    (
        target_ir,
        backend_ir,
        payload,
        compiler_fp,
        toolchain_fp,
        device_libraries,
        compile_state,
    ) = rocm_native._compile_tile_ir(tile_ir)
    image = NativeImageArtifact(
        target="rocm_gfx1151",
        architecture="gfx1151",
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(),
        binary_format="hsaco",
        payload=payload,
        entry_points=(NativeEntryPoint(entry, abi_id),),
        compile_state=compile_state,
        device_libraries=device_libraries,
    )
    rows = math.prod(shape[:-1]) if len(shape) > 1 else 1
    columns = shape[-1]
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=abi_id,
        buffers=(
            BufferBinding(0, input_name, "input", dtype, len(shape), "row_major", alignment),
            BufferBinding(1, output_name, "output", dtype, len(shape), "row_major", alignment),
        ),
        scalars=(
            ScalarArgument(2, "Rows", "int64"),
            ScalarArgument(3, "K", "int64"),
        ),
        shape_guards=tuple(
            ShapeGuard(name, axis, "eq", extent)
            for name in (input_name, output_name)
            for axis, extent in enumerate(shape)
        ),
        geometry=LaunchGeometry(policy="gfx1151_softmax_workgroup_per_row_256"),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "ROCM-E2E-1",
            "sync_key": "E2E-SPINE-2026-07-18",
            "schedule": "workgroup_per_row_256",
            "shape": list(shape),
            "storage": storage,
            "accum": "f32",
            "axis": -1,
            "exp_mode": "accurate",
            "ftz": False,
            "rows": rows,
            "columns": columns,
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return ROCMNativePackage(tile_ir, target_ir, backend_ir, image, descriptor)

def baseline_reduction(module: GraphIRModule, *, pipeline_name: str) -> ROCMNativePackage:
    contract = _reduction_contract(module)
    if contract is None:
        raise ValueError(
            "gfx1151 reduction packaging requires one static f16/bf16/f32 "
            "sum/mean/max with f32 output and one normalized axis"
        )
    input_name, output_name, dtype, kind, shape, output_shape, axis, keepdims = contract
    storage = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}[dtype]
    entry = f"tessera_tile_reduce_{kind}_{storage}"
    abi_id = {
        "fp16": GFX_REDUCE_F16_ABI,
        "bf16": GFX_REDUCE_BF16_ABI,
        "fp32": GFX_REDUCE_F32_ABI,
    }[dtype]
    outer = math.prod(shape[:axis]) if axis else 1
    axis_extent = shape[axis]
    inner = math.prod(shape[axis + 1 :]) if axis + 1 < len(shape) else 1
    tile_ir = emit_reduce_tile_ir(
        entry=entry,
        storage=storage,
        kind=kind,
        axis=axis,
        keepdims=keepdims,
        inner_is_one=inner == 1,
    )
    (
        target_ir,
        backend_ir,
        payload,
        compiler_fp,
        toolchain_fp,
        device_libraries,
        compile_state,
    ) = rocm_native._compile_reduction_tile_ir(tile_ir)
    image = NativeImageArtifact(
        target="rocm_gfx1151",
        architecture="gfx1151",
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(),
        binary_format="hsaco",
        payload=payload,
        entry_points=(NativeEntryPoint(entry, abi_id),),
        compile_state=compile_state,
        device_libraries=device_libraries,
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=abi_id,
        buffers=(
            BufferBinding(
                0,
                input_name,
                "input",
                dtype,
                len(shape),
                "row_major",
                2 if dtype in {"fp16", "bf16"} else 4,
            ),
            BufferBinding(1, output_name, "output", "fp32", len(output_shape), "row_major", 4),
        ),
        scalars=(
            ScalarArgument(2, "Outer", "int64"),
            ScalarArgument(3, "AxisExtent", "int64"),
            ScalarArgument(4, "Inner", "int64"),
        ),
        shape_guards=tuple(
            [ShapeGuard(input_name, index, "eq", extent) for index, extent in enumerate(shape)]
            + [ShapeGuard(output_name, index, "eq", extent) for index, extent in enumerate(output_shape)]
        ),
        geometry=LaunchGeometry(policy="gfx1151_reduce_workgroup_per_output_256"),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "ROCM-E2E-2",
            "sync_key": "E2E-SPINE-2026-07-18",
            "schedule": "workgroup_per_output_256",
            "shape": list(shape),
            "storage": storage,
            "accum": "f32",
            "kind": kind,
            "axis": axis,
            "keepdims": keepdims,
            "nan_mode": "propagate",
            "outer": outer,
            "axis_extent": axis_extent,
            "inner": inner,
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return ROCMNativePackage(tile_ir, target_ir, backend_ir, image, descriptor)
