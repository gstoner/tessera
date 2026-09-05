"""Frozen pre-F2 unary packaging baselines, used only by differential tests.

These reconstruct Tile IR deliberately to compare against native scheduling.
Production callers must consume ScheduledKernelArtifact instead.
"""
import hashlib
import math

from tessera.compiler.graph_ir import GraphIRModule
from tessera.compiler.native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint,
    NativeImageArtifact, OrderingSemantics, ResourceRecord, ScalarArgument, ShapeGuard,
)
from tessera.compiler.nvidia_native import (
    NVIDIANativePackage, SM120_REDUCE_F16_ABI, SM120_REDUCE_BF16_ABI,
    SM120_REDUCE_F32_ABI, SM120_SOFTMAX_F16_ABI, SM120_SOFTMAX_BF16_ABI,
    SM120_SOFTMAX_F32_ABI, _compile_tile_ir, _reduction_contract, _shape,
    _softmax_storage, emit_reduce_tile_ir, emit_softmax_tile_ir,
)


def baseline_reduction(
    module: GraphIRModule,
    *,
    pipeline_name: str,
    schedule: str = "serial",
) -> NVIDIANativePackage:
    contract = _reduction_contract(module)
    if contract is None:
        raise ValueError(
            "SM120 reduction packaging requires static f16/bf16/f32 input, f32 output, "
            "one normalized axis and sum/mean/max/min semantics"
        )
    storage, kind, axis, keepdims = contract
    if schedule not in {"serial", "cooperative_128"}:
        raise ValueError("SM120 reduction schedule must be serial or cooperative_128")
    storage_ir = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}[storage]
    entry = f"tessera_tile_reduce_{kind}_{storage_ir}_{schedule}"
    abi_id = {
        "fp16": SM120_REDUCE_F16_ABI,
        "bf16": SM120_REDUCE_BF16_ABI,
        "fp32": SM120_REDUCE_F32_ABI,
    }[storage]
    tile_ir = emit_reduce_tile_ir(
        entry=entry, storage=storage_ir, kind=kind, axis=axis,
        keepdims=keepdims, schedule=schedule,
    )
    (lowered, ptx, metrics, compiler_fp, toolchain_fp, device_libraries, compile_state) = _compile_tile_ir(
        tile_ir, entry
    )
    image = NativeImageArtifact(
        target="nvidia_sm120",
        architecture="sm_120a",
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(lowered.encode()).hexdigest(),
        binary_format="ptx",
        payload=ptx.encode("ascii"),
        entry_points=(NativeEntryPoint(entry, abi_id),),
        compile_state=compile_state,
        device_libraries=device_libraries,
        resource_record=ResourceRecord(provenance="ptxas --arch=sm_120a -v", metrics=metrics),
    )
    fn = module.functions[0]
    op = fn.body[0]
    input_name = op.operands[0].removeprefix("%")
    output_name = op.result or "output"
    shape = _shape(module, input_name)
    assert shape is not None
    outer = math.prod(shape[:axis]) if axis else 1
    axis_extent = shape[axis]
    inner = math.prod(shape[axis + 1:]) if axis + 1 < len(shape) else 1
    output_shape = shape[:axis] + ((1,) if keepdims else ()) + shape[axis + 1:]
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=abi_id,
        buffers=(
            BufferBinding(0, input_name, "input", storage, len(shape), "row_major", 2 if storage in {"fp16", "bf16"} else 4),
            BufferBinding(1, output_name, "output", "fp32", len(output_shape), "row_major", 4),
        ),
        scalars=(ScalarArgument(2, "Outer", "int64"),
                 ScalarArgument(3, "AxisExtent", "int64"),
                 ScalarArgument(4, "Inner", "int64")),
        shape_guards=tuple(
            [ShapeGuard(input_name, axis, "eq", extent) for axis, extent in enumerate(shape)]
            + [ShapeGuard(output_name, axis, "eq", extent) for axis, extent in enumerate(output_shape)]
        ),
        geometry=LaunchGeometry(policy=f"sm120_reduce_{schedule}"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("completion",)),
        provenance={
            "work_item": "NVIDIA-E2E-2",
            "sync_key": "E2E-SPINE-2026-07-18",
            "schedule": schedule,
            "shape": list(shape),
            "storage": storage_ir,
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
    return NVIDIANativePackage(tile_ir, lowered, ptx, image, descriptor)


def baseline_softmax(
    module: GraphIRModule,
    *,
    pipeline_name: str,
) -> NVIDIANativePackage:
    """Compile and package one static f16/bf16/f32 last-axis softmax request."""
    storage = _softmax_storage(module)
    if storage is None:
        raise ValueError("SM120 native softmax packaging requires one static f16/bf16/f32 last-axis softmax")
    storage_ir = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}[storage]
    entry = f"tessera_tile_softmax_{storage_ir}"
    abi_id = {
        "fp16": SM120_SOFTMAX_F16_ABI,
        "bf16": SM120_SOFTMAX_BF16_ABI,
        "fp32": SM120_SOFTMAX_F32_ABI,
    }[storage]
    alignment = 2 if storage in {"fp16", "bf16"} else 4
    tile_ir = emit_softmax_tile_ir(entry=entry, storage=storage_ir)
    (lowered, ptx, metrics, compiler_fp, toolchain_fp, device_libraries, compile_state) = _compile_tile_ir(
        tile_ir, entry
    )
    image = NativeImageArtifact(
        target="nvidia_sm120",
        architecture="sm_120a",
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(lowered.encode()).hexdigest(),
        binary_format="ptx",
        payload=ptx.encode("ascii"),
        entry_points=(NativeEntryPoint(entry, abi_id),),
        compile_state=compile_state,
        device_libraries=device_libraries,
        resource_record=ResourceRecord(
            provenance="ptxas --arch=sm_120a -v",
            metrics=metrics,
        ),
    )
    fn = module.functions[0]
    op = fn.body[0]
    input_name = op.operands[0].removeprefix("%")
    output_name = op.result or "output"
    shape = _shape(module, input_name)
    assert shape is not None
    rows = math.prod(shape[:-1]) if len(shape) > 1 else 1
    columns = shape[-1]
    guards = tuple(
        ShapeGuard(name, axis, "eq", extent) for name in (input_name, output_name) for axis, extent in enumerate(shape)
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=abi_id,
        buffers=(
            BufferBinding(0, input_name, "input", storage, len(shape), "row_major", alignment),
            BufferBinding(1, output_name, "output", storage, len(shape), "row_major", alignment),
        ),
        scalars=(
            ScalarArgument(2, "Rows", "int64"),
            ScalarArgument(3, "K", "int64"),
        ),
        shape_guards=guards,
        geometry=LaunchGeometry(policy="sm120_softmax_thread_per_row_128"),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "NVIDIA-E2E-2",
            "sync_key": "E2E-SPINE-2026-07-18",
            "schedule": "thread_per_row_128",
            "shape": list(shape),
            "storage": storage_ir,
            "accum": "f32",
            "axis": -1,
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return NVIDIANativePackage(tile_ir, lowered, ptx, image, descriptor)
