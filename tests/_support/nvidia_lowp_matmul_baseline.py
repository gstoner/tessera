"""Retired Graph-authored SM120 low-precision matmul packaging, kept only as a #31(a) oracle.

E2E-REAL-6 (sync ``E2E-REAL-6-sm120-lowp-matmul-2026-11-04``). Production
``nvidia_native.package_scheduled_nvfp4_matmul`` and
``package_scheduled_int4_matmul`` read every launch field from a replayed
``ScheduledMatmulArtifact`` (``scheduled_matmul.verify_nvfp4_projection`` /
``verify_matmul_projection``). Before that cut:

* NVFP4 lowered Graph -> Schedule -> Tile but took its scale operand names,
  logical shapes, batch axes and transpose flags from the caller's Graph
  object. :func:`baseline_nvfp4_matmul` is that packager, as at f1c64009a.
* INT4 packaged its artifact without projecting it. Its Graph-side launch
  contract is reconstructed by :func:`graph_authored_int4_launch` from the
  Graph signature alone, independent of any native contract.
* FP6/MXFP4 authored its Tile IR in Python with no Schedule at all.
  :func:`baseline_mx_matmul` is that packager (sync
  ``E2E-REAL-6-sm120-mx-matmul-2026-11-04``).

Nothing in ``python/tessera`` may import this module.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re

from tessera.compiler import nvidia_native as native
from tessera.compiler.graph_ir import GraphIRModule
from tessera.compiler.native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint, NativeImageArtifact,
    OrderingSemantics, ResourceRecord, ScalarArgument, ShapeGuard,
)
from tessera.compiler.scheduled_matmul import ScheduledMatmulArtifact


def baseline_nvfp4_matmul(
    module: GraphIRModule,
    *,
    pipeline_name: str,
    scheduled_artifact: ScheduledMatmulArtifact | None = None,
) -> native.NVIDIANativePackage:
    """The retired packager: Schedule/Tile lowered, launch fields read from the Graph."""
    if not native.supports_nvfp4_matmul(module):
        raise ValueError("SM120 NVFP4 packaging requires one static rank-2 matmul with logical scale_a/scale_b views")
    from tessera.compiler.scheduled_matmul import (find_tessera_opt, lower_scheduled_matmul, run_tessera_opt,
                                                   _canonical_matmul_graph_ir, _graph_contract)

    # Give the scale operands first-class Graph semantics. Keep the caller-owned
    # module intact because callers commonly reuse the source for oracle runs.
    scheduled_module = copy.deepcopy(module)
    fn = scheduled_module.functions[0]
    op = fn.body[0]
    expected_contract = {
        "scale_layout": {"granularity": "block", "block": [1, 16], "format": "ue4m3"},
        "numeric_policy": {"accum": "fp32", "execution_mode": "exact_per_block"},
        "physical_contract": "nvidia_sm120_nvfp4_blockscale_v1",
    }
    if op.op_name == "tessera.scaled_matmul":
        for name, expected in expected_contract.items():
            if op.kwargs.get(name) != expected:
                raise ValueError(
                    f"NVFP4 scaled_matmul requires {name}={expected!r}"
                )
    elif op.op_name in {"tessera.matmul", "tessera.gemm"}:
        op.op_name = "tessera.scaled_matmul"
        op.kwargs.update(expected_contract)
    else:
        raise ValueError("NVFP4 packaging requires matmul, gemm, or scaled_matmul")
    if scheduled_artifact is None:
        artifact = lower_scheduled_matmul(scheduled_module, target="nvidia_sm120")
    else:
        artifact = scheduled_artifact
        contract = _graph_contract(scheduled_module, "nvidia_sm120")
        if artifact.graph_ir != _canonical_matmul_graph_ir(scheduled_module, "nvidia_sm120", contract):
            raise ValueError("NVFP4 shared Schedule belongs to a different logical Graph")
        artifact.validate()
        if (artifact.a_name, artifact.b_name, artifact.output_name) != (
                contract[3], contract[4], contract[5]):
            raise ValueError("NVFP4 shared Schedule buffer roles differ from its logical Graph")
    tool = find_tessera_opt()
    if tool is None or run_tessera_opt(tool, artifact.schedule_ir, "--tessera-schedule-to-tile") != artifact.tile_ir:
        raise ValueError("NVFP4 Tile does not replay from its canonical Schedule")
    if (artifact.target != "nvidia_sm120" or artifact.storage != "nvfp4"
            or artifact.accum != "f32" or artifact.output_dtype != "fp32"
            or artifact.dynamic_m or artifact.dynamic_n or artifact.dynamic_k
            or artifact.bias_name or artifact.residual_name or artifact.activation != "none"):
        raise ValueError("unsupported scheduled SM120 NVFP4 package contract")
    tile_ir = artifact.tile_ir
    if ('physical_contract = "nvidia_sm120_nvfp4_blockscale_v1"' not in tile_ir
            or "tessera.scale_vector_size = 16 : i64" not in tile_ir
            or "tessera.storage_pack" not in tile_ir):
        raise ValueError("NVFP4 Tile lost its packed K16 scale contract")
    independent_rhs = op.kwargs.get("batching") == "independent_rhs"
    shared_lhs = op.kwargs.get("batching") == "shared_lhs"
    rhs_batched = independent_rhs or shared_lhs
    batch_aware = rhs_batched or (op.kwargs.get("batching") == "shared_rhs_rows" and op.kwargs.get("transposeA", False))
    abi_id = native.SM120_NVFP4_BATCH_ABI if batch_aware else native.SM120_NVFP4_ABI
    expected_entry = "tessera_tile_matmul_nvfp4_batched" if batch_aware else "tessera_tile_matmul_nvfp4"
    entry_match = re.search(r"llvm.func @([A-Za-z0-9_]+)\(", tile_ir)
    if entry_match is None or entry_match.group(1) != expected_entry:
        raise ValueError("NVFP4 Schedule/Tile emitted an unexpected runtime entry")
    entry = entry_match.group(1)
    (lowered, ptx, metrics, compiler_fp, toolchain_fp, device_libraries, compile_state) = native._compile_tile_ir(
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
    a_name, b_name = artifact.a_name, artifact.b_name
    scale_a_name, scale_b_name = (value.removeprefix("%") for value in op.operands[2:4])
    a_shape, b_shape = native._shape(scheduled_module, a_name), native._shape(scheduled_module, b_name)
    assert a_shape is not None and b_shape is not None
    ta, tb = (op.kwargs.get(name, False) for name in ("transposeA", "transposeB"))
    m, n, k = artifact.m, artifact.n, artifact.k
    batch_prefix = (b_shape[:-2] if shared_lhs else a_shape[:-2]) if op.kwargs.get("batching") in {"shared_rhs_rows", "independent_rhs", "shared_lhs"} else ()
    batch_count = math.prod(batch_prefix) if batch_prefix else 1
    batch_shape = (*batch_prefix, m // batch_count) if batch_prefix else None
    output_name = artifact.output_name
    row_shape = batch_shape if batch_shape else (m,)
    physical_a, physical_b = list(a_shape), list(b_shape)
    physical_a[-2 if ta else -1] = (k + 1) // 2
    physical_b[-1 if tb else -2] = (k + 1) // 2
    scale_a_shape = native._shape(scheduled_module, scale_a_name)
    scale_b_shape = native._shape(scheduled_module, scale_b_name)
    assert scale_a_shape is not None and scale_b_shape is not None
    physical_shapes = {
        a_name: tuple(physical_a), b_name: tuple(physical_b),
        scale_a_name: scale_a_shape,
        scale_b_name: scale_b_shape,
        output_name: (*row_shape, n),
    }
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=abi_id,
        buffers=(
            BufferBinding(0, a_name, "input", "uint8", len(a_shape), "row_major", 1),
            BufferBinding(1, b_name, "input", "uint8", len(b_shape), "row_major", 1),
            BufferBinding(2, scale_a_name, "input", "uint8", len(a_shape), "row_major", 1),
            BufferBinding(3, scale_b_name, "input", "uint8", len(b_shape), "row_major", 1),
            BufferBinding(4, output_name, "output", "fp32", len(row_shape)+1, "row_major", 4),
        ),
        scalars=(
            ScalarArgument(5, "M", "int64"),
            ScalarArgument(6, "N", "int64"),
            ScalarArgument(7, "K", "int64"),
        ) + ((ScalarArgument(8, "BatchRows", "int64"), ScalarArgument(9, "BatchCount", "int64")) if batch_aware else ()),
        shape_guards=tuple(
            ShapeGuard(name, dim, "eq", extent)
            for name, shape in physical_shapes.items()
            for dim, extent in enumerate(shape)
        ),
        geometry=LaunchGeometry(policy="sm120_nvfp4_batched_m16n8k64" if batch_aware else "sm120_nvfp4_m16n8k64"),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "NVIDIA-E2E-1",
            "sync_key": "E2E-SPINE-2026-07-18",
            "schedule": "graph_schedule_tile_sm120_nvfp4_k16_m16n8k64",
            "shape": [m, n, k],
            "batch_rows": [batch_count,batch_shape[-1]] if batch_shape else None,
            "logical_batch_shape": list(batch_prefix),
            "batching": "shared_lhs" if shared_lhs else ("independent_rhs" if independent_rhs else ("shared_rhs_rows" if batch_shape else None)),
            "transposeA": ta, "transposeB": tb,
            "scale_vector_size": 16,
            "storage_pack": {
                "logical": "nvfp4",
                "container": "int8",
                "factor": 2,
                "signedness": "format_defined",
            },
            "schedule_digest": artifact.schedule_digest,
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return native.NVIDIANativePackage(tile_ir, lowered, ptx, image, descriptor)


def graph_authored_int4_launch(module: GraphIRModule) -> dict[str, object]:
    """Buffers, guards and scalar names a Graph-side INT4 packager would author.

    Read only from the Graph signature: A and B in operand order, then the
    result, each as a packed int8 container with K halved.
    """
    fn = module.functions[0]
    op = fn.body[0]
    args = {arg.name: arg for arg in fn.args}
    a_name, b_name = (value.removeprefix("%") for value in op.operands)
    m, k = (int(x) for x in args[a_name].ir_type.shape)
    n = int(args[b_name].ir_type.shape[1])
    packed_k = (k + 1) // 2
    output = op.result
    return {
        "buffers": (
            BufferBinding(0, a_name, "input", "uint8", 2, "row_major", 1),
            BufferBinding(1, b_name, "input", "uint8", 2, "row_major", 1),
            BufferBinding(2, output, "output", "int32", 2, "row_major", 4),
        ),
        "guards": frozenset({
            ShapeGuard(a_name, 0, "eq", m), ShapeGuard(a_name, 1, "eq", packed_k),
            ShapeGuard(b_name, 0, "eq", packed_k), ShapeGuard(b_name, 1, "eq", n),
            ShapeGuard(output, 0, "eq", m), ShapeGuard(output, 1, "eq", n),
        }),
        "scalars": ("M", "N", "K"),
        "shape": [m, n, k],
    }


# ---------------------------------------------------------------------------
# OCP MX FP6/MXFP4 (sync ``E2E-REAL-6-sm120-mx-matmul-2026-11-04``). Before the
# cut, ``nvidia_native.package_mx_matmul`` authored its Tile IR in Python and
# read every launch field from the caller's Graph; no Schedule existed for
# FP6/MXFP4. :func:`emit_mx_matmul_tile_ir` and :func:`baseline_mx_matmul`
# are that packager as at 4606fc710, with one change: the Tile launch op now
# states the UE8M0 K32 scale contract, which the NVIDIA materializer requires
# (Decision #21a) instead of assuming it.
# ---------------------------------------------------------------------------


def emit_mx_matmul_tile_ir(*, entry: str, storage: str) -> str:
    """Emit a typed FP6/MXFP4 launch kernel with explicit UE8M0 scales."""
    if storage not in {"e2m3", "e3m2", "fp4_e2m1"}:
        raise ValueError(f"unsupported SM120 MX matmul storage {storage!r}")
    fragment_k = 64 if storage == "fp4_e2m1" else 32
    logical_storage = storage if storage == "fp4_e2m1" else f"fp6_{storage}"
    packed_format = native._packed_format_attr(logical_storage)
    return f'''module {{
  llvm.func @{entry}(%a: !llvm.ptr, %b: !llvm.ptr,
                     %scale_a: !llvm.ptr, %scale_b: !llvm.ptr,
                     %d: !llvm.ptr, %m: i64, %n: i64, %k: i64)
      attributes {{nvvm.kernel}} {{
    tile.matmul_kernel %a, %b, %scale_a, %scale_b, %d, %m, %n, %k {{
      mma = #tile.mma_desc<family = "mma_sync", m = 16, n = 8, k = {fragment_k}, a = "{storage}", b = "{storage}", acc = "f32", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "f32">,
      warps = 1 : i64, staging = "global",
      physical_contract = "nvidia_sm120_mx_blockscale_v1",
      tessera.scale_vector_size = 32 : i64,
      tessera.scale_format = "ue8m0",
      tessera.storage_packed = true,
      tessera.storage_container = "int8",
      tessera.storage_pack = {packed_format}
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }}
}}
'''


def baseline_mx_matmul(module: GraphIRModule, *, pipeline_name: str) -> native.NVIDIANativePackage:
    """The retired packager: Python-authored Tile, launch fields read from the Graph."""
    if not native.supports_mx_matmul(module):
        raise ValueError(
            "SM120 MX packaging requires one static rank-2 FP6/MXFP4 matmul with logical UE8M0 scale_a/scale_b views"
        )
    fn = module.functions[0]
    op = fn.body[0]
    a_name, b_name = native._matrix_names(op)
    args = {arg.name: arg for arg in fn.args}
    storage = args[a_name].ir_type.dtype
    physical = {"fp6_e2m3": "e2m3", "fp6_e3m2": "e3m2", "fp4_e2m1": "fp4_e2m1"}[storage]
    abi_id = {
        "fp6_e2m3": native.SM120_FP6_E2M3_ABI,
        "fp6_e3m2": native.SM120_FP6_E3M2_ABI,
        "fp4_e2m1": native.SM120_MXFP4_ABI,
    }[storage]
    entry = f"tessera_tile_matmul_mx_{physical}"
    tile_ir = emit_mx_matmul_tile_ir(entry=entry, storage=physical)
    (lowered, ptx, metrics, compiler_fp, toolchain_fp, device_libraries, compile_state) = native._compile_tile_ir(
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
    scale_names = native._scale_names(module)
    assert scale_names is not None
    scale_a_name, scale_b_name = scale_names
    a_shape = native._static_shape(module, a_name)
    b_shape = native._static_shape(module, b_name)
    assert a_shape is not None and b_shape is not None
    m, k = a_shape
    n = b_shape[1]
    physical_k = k if storage.startswith("fp6") else (k + 1) // 2
    scale_k = (k + 31) // 32
    output_name = op.result or "output"
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=abi_id,
        buffers=(
            BufferBinding(0, a_name, "input", "uint8", 2, "row_major", 1),
            BufferBinding(1, b_name, "input", "uint8", 2, "row_major", 1),
            BufferBinding(2, scale_a_name, "input", "uint8", 2, "row_major", 1),
            BufferBinding(3, scale_b_name, "input", "uint8", 2, "row_major", 1),
            BufferBinding(4, output_name, "output", "fp32", 2, "row_major", 4),
        ),
        scalars=(ScalarArgument(5, "M", "int64"), ScalarArgument(6, "N", "int64"), ScalarArgument(7, "K", "int64")),
        shape_guards=(
            ShapeGuard(a_name, 0, "eq", m), ShapeGuard(a_name, 1, "eq", physical_k),
            ShapeGuard(b_name, 0, "eq", physical_k), ShapeGuard(b_name, 1, "eq", n),
            ShapeGuard(scale_a_name, 0, "eq", m), ShapeGuard(scale_a_name, 1, "eq", scale_k),
            ShapeGuard(scale_b_name, 0, "eq", scale_k), ShapeGuard(scale_b_name, 1, "eq", n),
            ShapeGuard(output_name, 0, "eq", m), ShapeGuard(output_name, 1, "eq", n),
        ),
        geometry=LaunchGeometry(policy="sm120_mx_m16n8"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("completion",)),
        provenance={
            "work_item": "NVIDIA-E2E-2",
            "sync_key": "E2E-SPINE-2026-07-18",
            "schedule": "warp_m16n8_mx",
            "shape": [m, n, k],
            "storage": storage,
            "scale_dtype": "ue8m0",
            "scale_vector_size": 32,
            "storage_pack": {
                "logical": storage,
                "container": "int8",
                "factor": 2 if storage == "fp4_e2m1" else 1,
                "signedness": "format_defined",
            },
            "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        },
    )
    return native.NVIDIANativePackage(tile_ir, lowered, ptx, image, descriptor)
