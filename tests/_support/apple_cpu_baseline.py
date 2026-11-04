"""Frozen Graph-owned Apple CPU descriptor constructor (Decision #31(a) oracle).

E2E-REAL-6 Apple CPU (2026-11-03). Before this slice,
``apple_cpu_native.package_native`` read the Python Graph object directly to
build the launch descriptor for the static f32 ``matmul`` / ``gemm`` and
rank-2 ``softmax`` envelopes. Those families now lower through the native
Graph -> Schedule -> Tile boundaries and package the replayed artifact
(``package_scheduled_{matmul,kernel}``). This module freezes the retired
constructor, for exactly those families, so
``tests/unit/test_apple_cpu_scheduled_migration.py`` can prove the compiled
route reproduces it. It is test-only: nothing under ``python/`` imports it.
"""

from __future__ import annotations

import hashlib

from tessera.compiler import apple_cpu_native
from tessera.compiler.graph_ir import GraphIRModule
from tessera.compiler.native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint,
    NativeImageArtifact, OrderingSemantics, ShapeGuard,
)

_SYMBOLS = {
    "tessera.softmax": ("softmax", "tessera_apple_cpu_softmax_f32"),
    "tessera.matmul": ("matmul", "tessera_apple_cpu_gemm_f32"),
    "tessera.gemm": ("gemm", "tessera_apple_cpu_gemm_f32"),
}


def retired_contract(module: GraphIRModule) -> bool:
    """The envelope the retired constructor accepted for these families.

    ``native_package_kind`` is the shared classifier; at the retirement point
    it admitted softmax for ``axis % 2 == 1`` (so also 3 and -3, out of range
    for rank two), which the frozen predicate keeps so a refusal difference is
    visible rather than silently narrowed.
    """
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return False
    fn, op = module.functions[0], module.functions[0].body[0]
    if op.op_name not in _SYMBOLS:
        return False
    names = tuple(v.removeprefix("%") for v in op.operands)
    args = {a.name: a for a in fn.args}
    if not names or any(n not in args for n in names):
        return False
    if {args[n].ir_type.dtype for n in names} != {"fp32"}:
        return False
    if (len(fn.result_types) != 1 or fn.result_types[0].dtype != "fp32"
            or len(fn.return_values) != 1):
        return False
    try:
        shapes = [tuple(int(v) for v in args[n].ir_type.shape) for n in names]
        out = tuple(int(v) for v in fn.result_types[0].shape)
    except (TypeError, ValueError):
        return False
    if any(d <= 0 for s in [*shapes, out] for d in s):
        return False
    if op.op_name == "tessera.softmax":
        try:
            axis = int(op.kwargs.get("axis", -1))
        except (TypeError, ValueError):
            return False
        return len(names) == 1 and len(shapes[0]) == 2 and out == shapes[0] and axis % 2 == 1
    return True


def package(module: GraphIRModule, *, pipeline_name: str):
    """The retired constructor body, verbatim for the f32 families."""
    if not retired_contract(module):
        raise ValueError("Apple CPU native packaging requires one static supported descriptor contract")
    fn, op = module.functions[0], module.functions[0].body[0]
    names = tuple(value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in fn.args}
    dtype = "fp32"
    kind, symbol = _SYMBOLS[op.op_name]
    shapes = {name: tuple(int(v) for v in args[name].ir_type.shape) for name in names}
    output_names = tuple(name.removeprefix("%") for name in fn.return_values)
    output_shapes = tuple(tuple(int(v) for v in result.shape) for result in fn.result_types)
    abi = f"tessera.apple.cpu.value.{kind}.{dtype}.v1"
    target_ir = f'tessera_apple.cpu.call @{symbol} {{abi = "{abi}", status = "executable"}}'
    # Looked up through the module so a test's runtime fixture reaches both routes.
    library = apple_cpu_native._runtime_library_path()
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
        BufferBinding(i, name, "input", dtype, len(shapes[name]), "row_major", 4)
        for i, name in enumerate(names)
    )
    output_buffers = tuple(
        BufferBinding(len(names) + index, name, "output", "fp32", len(output_shapes[index]), "row_major", 4)
        for index, name in enumerate(output_names)
    )
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
        buffers=input_buffers + output_buffers, shape_guards=guards,
        geometry=LaunchGeometry(policy="apple_cpu_value_executor"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("return",)),
        provenance={"work_item": ("APPLE-NATIVE-E2E-2" if op.op_name == "tessera.softmax"
                                  else "APPLE-CPU-E2E-1"),
                    "route": "apple_cpu_value_executor", "op_kind": kind,
                    "value_call": call, "result_count": len(output_names)},
    )
    return apple_cpu_native.AppleCPUNativePackage("apple.cpu.value", target_ir, target_ir, image, descriptor)
