"""Pre-native norm packaging retained only for differential device tests."""
import hashlib
import math
from tessera.compiler.graph_ir import GraphIRModule
from tessera.compiler.native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint, NativeImageArtifact,
    OrderingSemantics, ResourceRecord, ScalarArgument, ShapeGuard,
)
from tessera.compiler.nvidia_native import (
    NVIDIANativePackage, SM120_NORM_F16_ABI, SM120_NORM_BF16_ABI, SM120_NORM_F32_ABI,
    _norm_contract, emit_norm_tile_ir, _compile_tile_ir,
)

def baseline_norm(
    module: GraphIRModule,
    *,
    pipeline_name: str,
) -> NVIDIANativePackage:
    contract = _norm_contract(module)
    if contract is None:
        raise ValueError(
            "SM120 norm packaging requires static f16/bf16/f32 input and "
            "same-storage output, last-axis unweighted RMSNorm/LayerNorm, and "
            "finite nonnegative epsilon"
        )
    storage, kind, epsilon, shape = contract
    storage_ir = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}[storage]
    epsilon_key = hashlib.sha256(f"{epsilon:.17g}".encode()).hexdigest()[:10]
    entry = f"tessera_tile_norm_{kind}_{storage_ir}_{epsilon_key}"
    abi_id = {
        "fp16": SM120_NORM_F16_ABI,
        "bf16": SM120_NORM_BF16_ABI,
        "fp32": SM120_NORM_F32_ABI,
    }[storage]
    tile_ir = emit_norm_tile_ir(
        entry=entry,
        storage=storage_ir,
        kind=kind,
        epsilon=epsilon,
    )
    (
        lowered,
        ptx,
        metrics,
        compiler_fp,
        toolchain_fp,
        device_libraries,
        compile_state,
    ) = _compile_tile_ir(tile_ir, entry)
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
            provenance="ptxas --arch=sm_120a -v", metrics=metrics
        ),
    )
    fn = module.functions[0]
    op = fn.body[0]
    input_name = op.operands[0].removeprefix("%")
    output_name = op.result or "output"
    rows = math.prod(shape[:-1]) if len(shape) > 1 else 1
    columns = shape[-1]
    alignment = 2 if storage in {"fp16", "bf16"} else 4
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=abi_id,
        buffers=(
            BufferBinding(
                0, input_name, "input", storage, len(shape), "row_major", alignment
            ),
            BufferBinding(
                1,
                output_name,
                "output",
                storage,
                len(shape),
                "row_major",
                alignment,
            ),
        ),
        scalars=(
            ScalarArgument(2, "Rows", "int64"),
            ScalarArgument(3, "Columns", "int64"),
        ),
        shape_guards=tuple(
            [ShapeGuard(input_name, axis, "eq", extent) for axis, extent in enumerate(shape)]
            + [ShapeGuard(output_name, axis, "eq", extent) for axis, extent in enumerate(shape)]
        ),
        geometry=LaunchGeometry(policy="sm120_norm_serial_rows"),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "NVIDIA-BF16-CANONICAL-BREADTH",
            "schedule": "serial_rows",
            "shape": list(shape),
            "storage": storage_ir,
            "accum": "f32",
            "kind": kind,
            "epsilon": epsilon,
            "rows": rows,
            "columns": columns,
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return NVIDIANativePackage(tile_ir, lowered, ptx, image, descriptor)


