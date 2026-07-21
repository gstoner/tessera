"""Compiler-owned Apple CPU descriptor packages for APPLE-CPU-E2E-1.

Static f32 C-ABI calls live here.  LU/QR/SVD use the same descriptor schema as
the single-result families, with one explicit output binding per SSA result.
That preserves the C ABI's result ordering without inventing a packed tuple
buffer or routing a descriptor package back through a legacy artifact.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .graph_ir import GraphIRModule
from .native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint,
    NativeImageArtifact, OrderingSemantics, ShapeGuard,
)


APPLE_CPU_DESCRIPTOR_STATES: dict[str, str] = {
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


def supports_native_package(module: GraphIRModule) -> bool:
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return False
    fn, op = module.functions[0], module.functions[0].body[0]
    if op.op_name not in _VALUE_SYMBOLS:
        return False
    names = tuple(value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in fn.args}
    if not names or any(name not in args or args[name].ir_type.dtype != "fp32" for name in names):
        return False
    expected_result_dtypes = _TUPLE_RESULT_DTYPES.get(op.op_name, ("fp32",))
    if (len(fn.result_types) != len(expected_result_dtypes)
            or tuple(result.dtype for result in fn.result_types) != expected_result_dtypes
            or len(fn.return_values) != len(expected_result_dtypes)):
        return False
    try:
        [tuple(int(v) for v in args[name].ir_type.shape) for name in names]
        [tuple(int(v) for v in result.shape) for result in fn.result_types]
    except (TypeError, ValueError):
        return False
    return True


def package_native(module: GraphIRModule, *, pipeline_name: str) -> AppleCPUNativePackage:
    if not supports_native_package(module):
        raise ValueError("APPLE-CPU-E2E-1 requires one static f32 descriptor contract with declared result bindings")
    fn, op = module.functions[0], module.functions[0].body[0]
    kind, symbol = _VALUE_SYMBOLS[op.op_name]
    names = tuple(value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in fn.args}
    shapes = {name: tuple(int(v) for v in args[name].ir_type.shape) for name in names}
    output_names = tuple(name.removeprefix("%") for name in fn.return_values)
    output_shapes = tuple(tuple(int(v) for v in result.shape) for result in fn.result_types)
    abi = f"tessera.apple.cpu.value.{kind}.f32.v1"
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
        BufferBinding(i, name, "input", "fp32", len(shapes[name]), "row_major", 4)
        for i, name in enumerate(names)
    )
    output_buffers = tuple(
        BufferBinding(len(names) + index, name, "output", cast(str, result.dtype),
                      len(output_shapes[index]), "row_major", 4)
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
        provenance={"work_item": "APPLE-CPU-E2E-1", "route": "apple_cpu_value_executor", "op_kind": kind,
                    "value_call": call, "result_count": len(output_names)},
    )
    return AppleCPUNativePackage("apple.cpu.value", target_ir, target_ir, image, descriptor)
