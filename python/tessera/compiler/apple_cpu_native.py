"""Compiler-owned Apple CPU descriptor packages for Apple native E2E work.

Static f32 C-ABI calls live here.  LU/QR/SVD use the same descriptor schema as
the single-result families, with one explicit output binding per SSA result.
That preserves the C ABI's result ordering without inventing a packed tuple
buffer or routing a descriptor package back through a legacy artifact.  E2E-2
extends the same static rank-2 contract to BNNS f16/bf16 matmul.

E2E-REAL-6 Apple CPU (2026-11-03): the static f32 ``matmul`` / ``gemm`` and
rank-2 ``softmax`` envelopes no longer build their descriptor from the Python
Graph object. They lower through the native Graph -> Schedule -> Tile
boundaries (``scheduled_matmul`` / ``scheduled_kernel``) and
``package_scheduled_{matmul,kernel}`` binds the replayed artifact to the
Accelerate delegate through a verified ``tessera_apple.cpu.accelerate_gemm`` /
``vector_reduce`` Target IR op (Decision #28 Tier 3 behind a declared op, one
lowering path per Decision #31). The retired Graph-owned constructor is the
test-only oracle ``tests/_support/apple_cpu_baseline.py``.

E2E-REAL-6 Apple CPU factorizations (2026-11-04): ``batched_gemm``,
``tri_solve``, ``cholesky``, ``cholesky_solve``, ``lu``, ``qr`` and ``svd``
follow. Their native owner is ``scheduled_apple_cpu_kernel`` (C++
``NativeAppleCPUKernel.h``); ``package_scheduled_apple_cpu_kernel`` binds the
replayed artifact to its Accelerate/LAPACK entry through the verified
``tessera_apple.cpu.call`` the Apple CPU value lowering produced. Only the
BNNS f16/bf16 matmul still packages here from Graph IR; it is owed.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .graph_ir import GraphIRModule
from .native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint,
    NativeImageArtifact, OrderingSemantics, ShapeGuard,
)

if TYPE_CHECKING:
    from .scheduled_apple_cpu_kernel import ScheduledAppleCPUKernelArtifact
    from .scheduled_kernel import ScheduledKernelArtifact
    from .scheduled_matmul import ScheduledMatmulArtifact


APPLE_CPU_DESCRIPTOR_STATES: dict[str, str] = {
    "tessera.softmax": "descriptor_ready",
    "tessera.matmul": "descriptor_ready",
    "tessera.gemm": "descriptor_ready",
    "tessera.batched_gemm": "descriptor_ready",
    "tessera.cholesky": "descriptor_ready",
    "tessera.tri_solve": "descriptor_ready",
    "tessera.cholesky_solve": "descriptor_ready",
    "tessera.lu": "descriptor_ready",
    "tessera.qr": "descriptor_ready",
    "tessera.svd": "descriptor_ready",
}

_VALUE_SYMBOLS = {
    "tessera.softmax": ("softmax", "tessera_apple_cpu_softmax_f32"),
    "tessera.matmul": ("matmul", "tessera_apple_cpu_gemm_f32"),
    "tessera.gemm": ("gemm", "tessera_apple_cpu_gemm_f32"),
    "tessera.batched_gemm": ("batched_gemm", "tessera_apple_cpu_gemm_f32_batched"),
    "tessera.cholesky": ("cholesky", "tessera_apple_cpu_cholesky_f32"),
    "tessera.tri_solve": ("tri_solve", "tessera_apple_cpu_tri_solve_f32"),
    "tessera.cholesky_solve": ("cholesky_solve", "tessera_apple_cpu_cholesky_solve_f32"),
    "tessera.lu": ("lu", "tessera_apple_cpu_lu_f32"),
    "tessera.qr": ("qr", "tessera_apple_cpu_qr_f32"),
    "tessera.svd": ("svd", "tessera_apple_cpu_svd_f32"),
}

_LOW_PRECISION_MATMUL_SYMBOLS = {
    ("tessera.matmul", "fp16"): ("matmul", "tessera_apple_cpu_gemm_f16"),
    ("tessera.gemm", "fp16"): ("gemm", "tessera_apple_cpu_gemm_f16"),
    ("tessera.matmul", "bf16"): ("matmul", "tessera_apple_cpu_gemm_bf16"),
    ("tessera.gemm", "bf16"): ("gemm", "tessera_apple_cpu_gemm_bf16"),
}

_TUPLE_RESULT_DTYPES: dict[str, tuple[str, ...]] = {
    "tessera.lu": ("fp32", "int32"),
    "tessera.qr": ("fp32", "fp32"),
    "tessera.svd": ("fp32", "fp32", "fp32"),
}


@dataclass(frozen=True)
class AppleCPUNativePackage:
    tile_ir: str
    target_ir: str
    backend_ir: str
    image: NativeImageArtifact
    descriptor: LaunchDescriptor


def _runtime_library_path() -> Path | None:
    configured = os.environ.get("TESSERA_APPLE_CPU_RUNTIME_LIB")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    try:
        # The runtime owns atomic local compilation/cache publication; package
        # production reads the resulting immutable dylib into its image.
        from tessera.runtime import _build_apple_cpu_runtime_shared

        return _build_apple_cpu_runtime_shared(Path(__file__).resolve().parents[3])
    except Exception:
        return None


def tools_available() -> bool:
    return _runtime_library_path() is not None


def value_descriptor_state(op_name: str) -> str:
    return APPLE_CPU_DESCRIPTOR_STATES.get(op_name, "unsupported")


def _entry_for(op_name: str, dtype: str) -> tuple[str, str] | None:
    if dtype == "fp32":
        return _VALUE_SYMBOLS.get(op_name)
    return _LOW_PRECISION_MATMUL_SYMBOLS.get((op_name, dtype))


def native_package_kind(module: GraphIRModule) -> str | None:
    """Classify one canonical Apple CPU descriptor request.

    Recognition belongs with the Apple package producer.  In particular, the
    shared driver must not grow a second list of CPU value-ABI families merely
    to decide whether canonical compilation may promote a package.
    """
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return None
    fn, op = module.functions[0], module.functions[0].body[0]
    from .scheduled_apple_cpu_kernel import APPLE_CPU_KERNEL_OPS, supports_scheduled_apple_cpu_kernel
    if op.op_name in APPLE_CPU_KERNEL_OPS and not supports_scheduled_apple_cpu_kernel(module):
        # E2E-REAL-6: these families have one authority, the native contract;
        # a request it refuses is not an Apple CPU descriptor request at all.
        # What it admits also passes every check below (it is stricter).
        return None
    names = tuple(value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in fn.args}
    if not names or any(name not in args for name in names):
        return None
    input_dtypes = {args[name].ir_type.dtype for name in names}
    if len(input_dtypes) != 1:
        return None
    dtype = input_dtypes.pop()
    if dtype is None or _entry_for(op.op_name, dtype) is None:
        return None
    expected_result_dtypes = _TUPLE_RESULT_DTYPES.get(op.op_name, (dtype,))
    if (len(fn.result_types) != len(expected_result_dtypes)
            or tuple(result.dtype for result in fn.result_types) != expected_result_dtypes
            or len(fn.return_values) != len(expected_result_dtypes)):
        return None
    try:
        input_shapes = [tuple(int(v) for v in args[name].ir_type.shape) for name in names]
        output_shapes = [tuple(int(v) for v in result.shape) for result in fn.result_types]
    except (TypeError, ValueError):
        return None
    if any(dimension <= 0 for shape in input_shapes + output_shapes for dimension in shape):
        return None
    if dtype in {"fp16", "bf16"}:
        if (len(names) != 2 or any(len(shape) != 2 for shape in input_shapes)
                or len(output_shapes) != 1 or output_shapes[0] != (input_shapes[0][0], input_shapes[1][1])
                or input_shapes[0][1] != input_shapes[1][0]):
            return None
    if op.op_name == "tessera.softmax":
        try:
            axis = int(op.kwargs.get("axis", -1))
        except (TypeError, ValueError):
            return None
        if (len(names) != 1 or len(input_shapes[0]) != 2 or output_shapes != [input_shapes[0]]
                or axis not in (-1, 1)):
            return None
    return op.op_name.removeprefix("tessera.")


def supports_native_package(module: GraphIRModule) -> bool:
    """Whether the canonical Apple CPU descriptor contract accepts ``module``."""

    return native_package_kind(module) is not None


def scheduled_package_kind(module: GraphIRModule) -> str | None:
    """The family whose request lowers through native Schedule/Tile, if any.

    The static f32 ``matmul``/``gemm`` and rank-2 ``softmax`` envelopes and the
    native batched-GEMM/LAPACK families; only the BNNS f16/bf16 matmul still
    packages from Graph IR (owed).
    """
    from .scheduled_apple_cpu_kernel import APPLE_CPU_KERNEL_OPS

    kind = native_package_kind(module)
    if kind == "softmax" or (kind is not None and f"tessera.{kind}" in APPLE_CPU_KERNEL_OPS):
        return kind
    if kind in {"matmul", "gemm"}:
        fn = module.functions[0]
        if all(arg.ir_type.dtype == "fp32" for arg in fn.args):
            return kind
    return None


def native_package_available(module: GraphIRModule) -> bool:
    """Whether ``package_native`` can produce ``module``'s package on this host.

    A scheduled family needs the production ``tessera-opt`` (with the Apple
    dialect) as well as the runtime dylib; the Graph-owned families need only
    the dylib.
    """
    if native_package_kind(module) is None or not tools_available():
        return False
    if scheduled_package_kind(module) is None:
        return True
    from .scheduled_matmul import find_tessera_opt
    return find_tessera_opt() is not None


def _verified_target_ir(op_text: str) -> str:
    """Parse and verify one Apple CPU Target IR op with the production compiler.

    The op must be a declared ``tessera_apple`` op the MLIR verifier accepts;
    an unregistered spelling or a missing required attribute fails closed
    before any image is built.
    """
    import re
    from .scheduled_matmul import find_tessera_opt, run_tessera_opt

    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("E2E-REAL-6 Apple CPU Target IR requires production tessera-opt")
    module = 'module attributes {tessera.target = "apple_cpu"} {\n  ' + op_text + "\n}\n"
    printed = run_tessera_opt(tool, module, "--verify-each")
    name = op_text.split(" ", 1)[0]
    if len(re.findall(r"(?m)^\s*" + re.escape(name) + r" \{", printed)) != 1:
        raise ValueError(f"Apple CPU Target IR did not round-trip one {name}")
    return op_text


def _runtime_image(target_ir: str, symbol: str, abi: str, *, pipeline_name: str) -> NativeImageArtifact:
    library = _runtime_library_path()
    if library is None:
        raise RuntimeError("E2E-REAL-6 Apple CPU scheduled packaging requires an Apple CPU runtime dylib")
    payload = library.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    return NativeImageArtifact(
        target="apple_cpu", architecture="apple_silicon_cpu", pipeline_name=pipeline_name,
        compiler_fingerprint="apple-cpu-runtime-abi-v1",
        toolchain_fingerprint=hashlib.sha256(("apple_cpu|" + digest).encode()).hexdigest(),
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(), binary_format="shared_object",
        payload=payload, entry_points=(NativeEntryPoint(symbol, abi),), compile_state="prepackaged",
    )


def package_scheduled_matmul(
    artifact: ScheduledMatmulArtifact,
    *,
    pipeline_name: str,
    op_kind: str = "matmul",
) -> AppleCPUNativePackage:
    """Bind a replayed Apple CPU Schedule/Tile matmul to Accelerate's cblas_sgemm.

    E2E-REAL-6 Apple CPU (2026-11-03). Every descriptor field is projected from
    the artifact ``verify_matmul_projection`` just proved equal to the native
    Schedule replay; nothing is read from a Graph object. The consumer is the
    hand-written ``tessera_apple_cpu_gemm_f32`` wrapper over cblas_sgemm, a
    declared Decision #28 Tier-3 delegate reached through the verified
    ``tessera_apple.cpu.accelerate_gemm`` Target IR op. Accelerate owns its
    blocking, so the Schedule's macro tile is a named dropped decision.
    """
    artifact.validate()
    from .scheduled_matmul import verify_matmul_projection
    verify_matmul_projection(artifact)
    if (
        artifact.target != "apple_cpu"
        or artifact.architecture != "apple_silicon_cpu"
        or (artifact.a_dtype, artifact.b_dtype, artifact.output_dtype) != ("fp32", "fp32", "fp32")
        or (artifact.storage, artifact.accum) != ("f32", "f32")
        or artifact.bias_name is not None or artifact.residual_name is not None
        or artifact.activation != "none" or artifact.split_k != 1
        or artifact.b_layout != "col_major" or artifact.b_codes_dtype or artifact.scale_name
        or artifact.dynamic_m or artifact.dynamic_n or artifact.dynamic_k
        or op_kind not in {"matmul", "gemm"}
    ):
        raise ValueError("Apple CPU scheduled matmul requires the static plain f32 cblas_sgemm contract")
    symbol = "tessera_apple_cpu_gemm_f32"
    abi = f"tessera.apple.cpu.value.{op_kind}.fp32.v1"
    target_ir = _verified_target_ir(
        f'tessera_apple.cpu.accelerate_gemm {{source = "tile.matmul_kernel", '
        f'result = "{artifact.output_name}", ordinal = 0, framework = "Accelerate", '
        f'abi = "cblas_sgemm", dtype = "f32", tessera.accum = "f32", '
        f'tessera.symbol = "{symbol}", tessera.schedule_hash = "{artifact.schedule_digest}"}}'
    )
    image = _runtime_image(target_ir, symbol, abi, pipeline_name=pipeline_name)
    a, b, out = artifact.a_name, artifact.b_name, artifact.output_name
    shapes = ((a, (artifact.m, artifact.k)), (b, (artifact.k, artifact.n)), (out, (artifact.m, artifact.n)))
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=symbol, abi_id=abi,
        buffers=(
            BufferBinding(0, a, "input", "fp32", 2, "row_major", 4),
            BufferBinding(1, b, "input", "fp32", 2, "row_major", 4),
            BufferBinding(2, out, "output", "fp32", 2, "row_major", 4),
        ),
        shape_guards=tuple(ShapeGuard(name, axis, "eq", extent)
                           for name, shape in shapes for axis, extent in enumerate(shape)),
        geometry=LaunchGeometry(policy="apple_cpu_value_executor"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("return",)),
        provenance={
            "work_item": "E2E-REAL-6", "route": "apple_cpu_value_executor", "op_kind": op_kind,
            "value_call": {"op": "tessera_apple.cpu.call", "op_kind": op_kind, "symbol": symbol,
                           "status": "executable"},
            "result_count": 1, "shape": [artifact.m, artifact.n, artifact.k],
            "storage": "f32", "accum": "f32", "target_ir_op": "tessera_apple.cpu.accelerate_gemm",
            # Decisions the delegated Accelerate route does not materialize,
            # named rather than lost (Decision #32).
            "dropped_macro_tile": [artifact.macro_tile_m, artifact.macro_tile_n],
            "dropped_reason": "delegated_to_accelerate_cblas_sgemm",
            "schedule_digest": artifact.schedule_digest, "tile_ir_digest": artifact.tile_digest,
        },
    )
    return AppleCPUNativePackage(artifact.tile_ir, target_ir, target_ir, image, descriptor)


def package_scheduled_kernel(
    artifact: ScheduledKernelArtifact,
    *,
    pipeline_name: str,
) -> AppleCPUNativePackage:
    """Bind a replayed Apple CPU Schedule/Tile softmax to its Accelerate ABI.

    E2E-REAL-6 Apple CPU (2026-11-03). ``verify_unary_ancestry`` replays the
    Schedule and projects every field below from native-printed IR. The
    consumer is the hand-written ``tessera_apple_cpu_softmax_f32`` row softmax,
    a Tier-3 delegate reached through the verified
    ``tessera_apple.cpu.vector_reduce`` Target IR op.
    """
    artifact.validate()
    from .native_unary_contract import verify_unary_ancestry
    verify_unary_ancestry(artifact, target="apple_cpu", architecture="apple_silicon_cpu")
    if (
        artifact.family != "softmax" or artifact.kind != "softmax" or artifact.record
        or artifact.dtype != "fp32" or (artifact.storage, artifact.accum) != ("f32", "f32")
        or len(artifact.input_shape) != 2 or artifact.output_shape != artifact.input_shape
        or artifact.axis != -1 or artifact.schedule != "serial" or artifact.workgroup_size != 1
        or max(artifact.input_shape) > 2**31 - 1
    ):
        raise ValueError("Apple CPU scheduled kernel requires the static rank-two f32 softmax contract")
    symbol = "tessera_apple_cpu_softmax_f32"
    abi = "tessera.apple.cpu.value.softmax.fp32.v1"
    target_ir = _verified_target_ir(
        f'tessera_apple.cpu.vector_reduce {{source = "tile.softmax_kernel", '
        f'result = "{artifact.output_name}", ordinal = 0, framework = "Accelerate", '
        f'abi = "row_softmax_f32", dtype = "f32", tessera.accum = "f32", '
        f'tessera.symbol = "{symbol}", tessera.schedule_hash = "{artifact.schedule_digest}"}}'
    )
    image = _runtime_image(target_ir, symbol, abi, pipeline_name=pipeline_name)
    x, out = artifact.input_name, artifact.output_name
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=symbol, abi_id=abi,
        buffers=(
            BufferBinding(0, x, "input", "fp32", 2, "row_major", 4),
            BufferBinding(1, out, "output", "fp32", 2, "row_major", 4),
        ),
        shape_guards=tuple(ShapeGuard(name, axis, "eq", extent)
                           for name, shape in ((x, artifact.input_shape), (out, artifact.output_shape))
                           for axis, extent in enumerate(shape)),
        geometry=LaunchGeometry(policy="apple_cpu_value_executor"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("return",)),
        provenance={
            "work_item": "E2E-REAL-6", "route": "apple_cpu_value_executor", "op_kind": "softmax",
            "value_call": {"op": "tessera_apple.cpu.call", "op_kind": "softmax", "symbol": symbol,
                           "status": "executable", "axis": -1},
            "result_count": 1, "shape": list(artifact.input_shape), "storage": "f32", "accum": "f32",
            "exp_mode": "accurate", "target_ir_op": "tessera_apple.cpu.vector_reduce",
            "schedule_digest": artifact.schedule_digest, "tile_ir_digest": artifact.tile_digest,
        },
    )
    return AppleCPUNativePackage(artifact.tile_ir, target_ir, target_ir, image, descriptor)


def package_scheduled_apple_cpu_kernel(
    artifact: ScheduledAppleCPUKernelArtifact,
    *,
    pipeline_name: str,
) -> AppleCPUNativePackage:
    """Bind a replayed Apple CPU batched-GEMM/LAPACK kernel to its C ABI entry.

    E2E-REAL-6 Apple CPU factorizations (2026-11-04). Every descriptor field --
    buffers, dtypes, shape guards, layout and the semantic flags the delegate
    reads -- is projected from the contract
    ``verify_apple_cpu_kernel_projection`` just proved equal to the native
    replay of all three boundaries; nothing is read from a Graph object. The
    consumer is a hand-written Accelerate/LAPACK wrapper in
    ``apple_cpu_runtime.cpp``, a declared Decision #28 Tier-3 delegate reached
    through the verified ``tessera_apple.cpu.call`` Target IR op.
    """
    from .scheduled_apple_cpu_kernel import verify_apple_cpu_kernel_projection

    projection = verify_apple_cpu_kernel_projection(artifact)
    kind, symbol = projection.kind, projection.symbol
    abi = f"tessera.apple.cpu.value.{kind}.fp32.v1"
    image = _runtime_image(artifact.target_ir, symbol, abi, pipeline_name=pipeline_name)
    buffers = tuple(
        BufferBinding(ordinal, name, "input" if ordinal < projection.inputs else "output",
                      dtype, len(shape), "row_major", 4)
        for ordinal, (name, shape, dtype) in enumerate(
            zip(projection.bindings, projection.shapes, projection.dtypes))
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=symbol, abi_id=abi, buffers=buffers,
        shape_guards=tuple(ShapeGuard(name, axis, "eq", extent)
                           for name, shape in zip(projection.bindings, projection.shapes)
                           for axis, extent in enumerate(shape)),
        geometry=LaunchGeometry(policy="apple_cpu_value_executor"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("return",)),
        provenance={
            "work_item": "E2E-REAL-6", "route": "apple_cpu_value_executor", "op_kind": kind,
            "value_call": {"op": "tessera_apple.cpu.call", "op_kind": kind, "symbol": symbol,
                           "status": "executable", **projection.semantics},
            "result_count": len(projection.bindings) - projection.inputs,
            "storage": "f32", "target_ir_op": "tessera_apple.cpu.call",
            "delegate_abi": projection.abi, "abi_scalars": dict(projection.scalars),
            "schedule_digest": projection.schedule_digest,
            "tile_ir_digest": hashlib.sha256(artifact.tile_ir.encode()).hexdigest(),
        },
    )
    return AppleCPUNativePackage(artifact.tile_ir, artifact.target_ir, artifact.target_ir, image, descriptor)


def package_native(module: GraphIRModule, *, pipeline_name: str) -> AppleCPUNativePackage:
    kind = native_package_kind(module)
    if kind is None:
        raise ValueError("Apple CPU native packaging requires one static supported descriptor contract")
    scheduled = scheduled_package_kind(module)
    if scheduled == "softmax":
        from .scheduled_kernel import lower_scheduled_kernel
        return package_scheduled_kernel(
            lower_scheduled_kernel(module, target="apple_cpu"), pipeline_name=pipeline_name)
    if scheduled in {"matmul", "gemm"}:
        # ``tessera.gemm`` is matmul's legacy alias; it survives only as the
        # descriptor's ABI label over the same ``tessera_apple_cpu_gemm_f32``.
        from .scheduled_matmul import lower_scheduled_matmul
        return package_scheduled_matmul(
            lower_scheduled_matmul(module, target="apple_cpu"), pipeline_name=pipeline_name,
            op_kind=scheduled)
    if scheduled is not None:
        from .scheduled_apple_cpu_kernel import lower_scheduled_apple_cpu_kernel
        return package_scheduled_apple_cpu_kernel(
            lower_scheduled_apple_cpu_kernel(module), pipeline_name=pipeline_name)
    # Owed family (BNNS f16/bf16 matmul): still Graph-owned until it gains a
    # native Schedule/Tile contract.
    fn, op = module.functions[0], module.functions[0].body[0]
    names = tuple(value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in fn.args}
    dtype = args[names[0]].ir_type.dtype
    assert dtype is not None
    entry = _entry_for(op.op_name, dtype)
    assert entry is not None
    kind, symbol = entry
    shapes = {name: tuple(int(v) for v in args[name].ir_type.shape) for name in names}
    output_names = tuple(name.removeprefix("%") for name in fn.return_values)
    output_shapes = tuple(tuple(int(v) for v in result.shape) for result in fn.result_types)
    abi = f"tessera.apple.cpu.value.{kind}.{dtype}.v1"
    target_ir = f'tessera_apple.cpu.call @{symbol} {{abi = "{abi}", status = "executable"}}'
    library = _runtime_library_path()
    if library is None:
        raise RuntimeError("APPLE-CPU-E2E-1 requires an Apple CPU runtime dylib")
    payload = library.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    image = NativeImageArtifact(
        target="apple_cpu", architecture="apple_silicon_cpu", pipeline_name=pipeline_name,
        compiler_fingerprint="apple-cpu-runtime-abi-v1",
        toolchain_fingerprint=hashlib.sha256(("apple_cpu|" + digest).encode()).hexdigest(),
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(), binary_format="shared_object",
        payload=payload, entry_points=(NativeEntryPoint(symbol, abi),), compile_state="prepackaged",
    )
    input_buffers = tuple(
        BufferBinding(i, name, "input", dtype, len(shapes[name]), "row_major", 2 if dtype != "fp32" else 4)
        for i, name in enumerate(names)
    )
    output_buffers = tuple(
        BufferBinding(len(names) + index, name, "output", cast(str, result.dtype),
                      len(output_shapes[index]), "row_major", 2 if result.dtype != "fp32" else 4)
        for index, (name, result) in enumerate(zip(output_names, fn.result_types))
    )
    buffers = input_buffers + output_buffers
    guards = tuple(
        ShapeGuard(name, axis, "eq", extent)
        for name, shape in list(shapes.items()) + list(zip(output_names, output_shapes))
        for axis, extent in enumerate(shape)
    )
    call = {
        "op": "tessera_apple.cpu.call", "op_kind": kind, "symbol": symbol,
        "status": "executable", **dict(op.kwargs),
    }
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=symbol, abi_id=abi,
        buffers=buffers, shape_guards=guards,
        geometry=LaunchGeometry(policy="apple_cpu_value_executor"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("return",)),
        provenance={"work_item": ("APPLE-NATIVE-E2E-2" if dtype != "fp32" else "APPLE-CPU-E2E-1"),
                    "route": "apple_cpu_value_executor", "op_kind": kind,
                    "value_call": call, "result_count": len(output_names)},
    )
    return AppleCPUNativePackage("apple.cpu.value", target_ir, target_ir, image, descriptor)
