"""Frozen pre-E2E-REAL-6 x86 unary packaging baselines (differential tests only).

These are the Graph-owned x86 softmax / reduction constructors as they stood
before E2E-REAL-6's x86 unary cut (2026-09-28): ``_softmax_contract`` /
``_reduction_contract`` read the Python Graph object to decide admission, and
``emit_softmax_tile_ir`` / ``emit_reduce_tile_ir`` author Tile IR text beside
the compiled Graph -> Schedule -> Tile route. ``package_softmax`` /
``package_reduction`` are the Graph-owned packagers verbatim from
``bad0b66d^`` (the last commit before production switched their bodies to the
scheduled lowering on 2026-09-08); the contracts and emitters are byte-for-byte
what production still carried until this cut.

Production callers now consume a ``ScheduledKernelArtifact``
(``x86_native.package_scheduled_kernel``) and admit through
``scheduled_kernel.supports_scheduled_kernel(target="x86")``. This module is
the **declared oracle** Decision #31(a) allows; its only consumers are the
differential tests (``tests/unit/test_x86_unary_migration.py``) and the
constructor-shape pins in ``tests/unit/test_x86_e2e_spine.py``.

Known divergences are kept verbatim, because a baseline that is quietly
corrected stops being evidence of what the retired route did:

* ``keepdims`` was coerced with ``bool(...)``, so ``keepdims=1`` was admitted.
* reduction ``schedule`` hints (``cooperative_128``) were ignored rather than
  refused; the compiled route refuses a non-serial x86 reduction.
* ``numeric_policy`` keyword arguments were ignored (unchanged by the
  migration; an open Decision #32 item across all targets).
"""
from __future__ import annotations

import math

from tessera.compiler.graph_ir import GraphIRModule
from tessera.compiler.native_artifact import (
    BufferBinding,
    LaunchDescriptor,
    LaunchGeometry,
    OrderingSemantics,
    ScalarArgument,
    ShapeGuard,
)
from tessera.compiler import x86_native as _x86_native
from tessera.compiler.x86_native import (
    X86_AVX512_ARCHITECTURE,
    X86_BASE_ARCHITECTURE,
    X86_REDUCE_F32_ABI,
    X86_SOFTMAX_F32_ABI,
    X86NativePackage,
    _image,
    _shape,
    requests_reduction,
    requests_softmax,
)


def emit_softmax_tile_ir(*, entry: str) -> str:
    return f'''module {{
  llvm.func @{entry}(%x: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %k: i64) {{
    tile.softmax_kernel %x, %o, %rows, %k {{
      storage = "f32", accum = "f32", axis = -1 : i64,
      exp_mode = "accurate", ftz = false
    }} : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }}
}}
'''


def emit_reduce_tile_ir(*, entry: str, kind: str, axis: int, keepdims: bool) -> str:
    if kind not in {"sum", "mean", "max"}:
        raise ValueError(f"unsupported x86 reduction kind {kind!r}")
    return f'''module {{
  llvm.func @{entry}(%x: !llvm.ptr, %o: !llvm.ptr,
                     %outer: i64, %axis_extent: i64, %inner: i64) {{
    tile.reduce_kernel %x, %o, %outer, %axis_extent, %inner {{
      storage = "f32", accum = "f32", kind = "{kind}", axis = {axis} : i64,
      keepdims = {str(keepdims).lower()}, schedule = "serial",
      nan_mode = "propagate", inner_is_one = true
    }} : !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }}
}}
'''


def _softmax_contract(module: GraphIRModule) -> tuple[str, str, tuple[int, ...]] | None:
    if not requests_softmax(module):
        return None
    function, op = module.functions[0], module.functions[0].body[0]
    if len(op.operands) != 1 or len(function.result_types) != 1 or op.kwargs.get("axis", -1) != -1:
        return None
    input_name = op.operands[0].removeprefix("%")
    arg = next((value for value in function.args if value.name == input_name), None)
    shape = _shape(module, input_name)
    result = function.result_types[0]
    if arg is None or arg.ir_type.dtype != "fp32" or shape is None or result.dtype != "fp32":
        return None
    try:
        if tuple(int(value) for value in result.shape) != shape:
            return None
    except (TypeError, ValueError):
        return None
    return input_name, op.result or "output", shape


def _reduction_contract(
    module: GraphIRModule,
) -> tuple[str, str, str, tuple[int, ...], tuple[int, ...], int, bool] | None:
    if not requests_reduction(module):
        return None
    function, op = module.functions[0], module.functions[0].body[0]
    if len(op.operands) != 1 or len(function.result_types) != 1:
        return None
    input_name = op.operands[0].removeprefix("%")
    arg = next((value for value in function.args if value.name == input_name), None)
    shape = _shape(module, input_name)
    if arg is None or arg.ir_type.dtype != "fp32" or shape is None:
        return None
    raw_axis = op.kwargs.get("axis", -1)
    if not isinstance(raw_axis, int) or isinstance(raw_axis, bool):
        return None
    axis = raw_axis + len(shape) if raw_axis < 0 else raw_axis
    if axis != len(shape) - 1:
        return None
    keepdims = bool(op.kwargs.get("keepdims", False))
    output_shape = shape[:-1] + ((1,) if keepdims else ())
    result = function.result_types[0]
    try:
        declared = tuple(int(value) for value in result.shape)
    except (TypeError, ValueError):
        return None
    if result.dtype != "fp32" or declared != output_shape:
        return None
    kind = "max" if op.op_name in {"tessera.max", "tessera.amax"} else "mean" if op.op_name == "tessera.mean" else "sum"
    return input_name, op.result or "output", kind, shape, output_shape, axis, keepdims


def package_softmax(
    module: GraphIRModule, *, pipeline_name: str,
    architecture: str = X86_AVX512_ARCHITECTURE,
) -> X86NativePackage:
    contract = _softmax_contract(module)
    if contract is None:
        raise ValueError("x86 native softmax requires one static f32 last-axis operation")
    input_name, output_name, shape = contract
    symbol = (
        "tessera_x86_base_softmax_f32"
        if architecture == X86_BASE_ARCHITECTURE
        else "tessera_x86_avx512_softmax_f32"
    )
    tile_ir = emit_softmax_tile_ir(entry="tessera_tile_x86_softmax_f32")
    target_ir, payload, compiler, toolchain = (
        _x86_native._lower(tile_ir, symbol, "softmax", architecture)
        if architecture == X86_BASE_ARCHITECTURE
        else _x86_native._lower(tile_ir, symbol, "softmax")
    )
    image = _image(target_ir=target_ir, payload=payload, compiler=compiler, toolchain=toolchain,
                   pipeline_name=pipeline_name, symbol=symbol, abi=X86_SOFTMAX_F32_ABI,
                   architecture=architecture)
    rows, columns = (math.prod(shape[:-1]) if len(shape) > 1 else 1), shape[-1]
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=symbol, abi_id=X86_SOFTMAX_F32_ABI,
        buffers=(
            BufferBinding(0, input_name, "input", "fp32", len(shape), "row_major", 4),
            BufferBinding(1, output_name, "output", "fp32", len(shape), "row_major", 4),
        ),
        scalars=(ScalarArgument(2, "Rows", "int64"), ScalarArgument(3, "K", "int64")),
        shape_guards=tuple(
            ShapeGuard(name, axis, "eq", extent)
            for name in (input_name, output_name) for axis, extent in enumerate(shape)
        ),
        geometry=LaunchGeometry(policy=f"{architecture}_rows"),
        ordering=OrderingSemantics(ordered_submission=True, residency="all", synchronization=("return",)),
        provenance={
            "work_item": (
                "E2E-SPINE-3" if architecture == X86_BASE_ARCHITECTURE
                else "X86-E2E-1"
            ),
            "route": f"{architecture}_c_abi",
            "shape": list(shape), "rows": rows, "columns": columns,
            "storage": "f32", "accum": "f32",
        },
    )
    return X86NativePackage(tile_ir, target_ir, target_ir, image, descriptor)


def package_reduction(
    module: GraphIRModule, *, pipeline_name: str,
    architecture: str = X86_AVX512_ARCHITECTURE,
) -> X86NativePackage:
    contract = _reduction_contract(module)
    if contract is None:
        raise ValueError("x86 native reduction requires one static f32 last-axis sum/mean/max")
    input_name, output_name, kind, shape, output_shape, axis, keepdims = contract
    symbol = (
        "tessera_x86_base_reduce_f32"
        if architecture == X86_BASE_ARCHITECTURE
        else "tessera_x86_avx512_reduce_f32"
    )
    tile_ir = emit_reduce_tile_ir(entry=f"tessera_tile_x86_reduce_{kind}_f32", kind=kind, axis=axis, keepdims=keepdims)
    target_ir, payload, compiler, toolchain = (
        _x86_native._lower(tile_ir, symbol, "reduction", architecture)
        if architecture == X86_BASE_ARCHITECTURE
        else _x86_native._lower(tile_ir, symbol, "reduction")
    )
    image = _image(target_ir=target_ir, payload=payload, compiler=compiler, toolchain=toolchain,
                   pipeline_name=pipeline_name, symbol=symbol, abi=X86_REDUCE_F32_ABI,
                   architecture=architecture)
    outer, extent = (math.prod(shape[:-1]) if len(shape) > 1 else 1), shape[-1]
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=symbol, abi_id=X86_REDUCE_F32_ABI,
        buffers=(
            BufferBinding(0, input_name, "input", "fp32", len(shape), "row_major", 4),
            BufferBinding(1, output_name, "output", "fp32", len(output_shape), "row_major", 4),
        ),
        scalars=(
            ScalarArgument(2, "Outer", "int64"), ScalarArgument(3, "AxisExtent", "int64"),
            ScalarArgument(4, "Inner", "int64"),
        ),
        shape_guards=tuple(
            [ShapeGuard(input_name, index, "eq", value) for index, value in enumerate(shape)]
            + [ShapeGuard(output_name, index, "eq", value) for index, value in enumerate(output_shape)]
        ),
        geometry=LaunchGeometry(policy=f"{architecture}_rows"),
        ordering=OrderingSemantics(ordered_submission=True, residency="all", synchronization=("return",)),
        provenance={
            "work_item": (
                "E2E-SPINE-3" if architecture == X86_BASE_ARCHITECTURE
                else "X86-E2E-1"
            ),
            "route": f"{architecture}_c_abi", "kind": kind,
            "shape": list(shape), "axis": axis, "keepdims": keepdims,
            "outer": outer, "axis_extent": extent, "inner": 1,
            "storage": "f32", "accum": "f32",
        },
    )
    return X86NativePackage(tile_ir, target_ir, target_ir, image, descriptor)


def supports_softmax(module: GraphIRModule) -> bool:
    return _softmax_contract(module) is not None


def supports_reduction(module: GraphIRModule) -> bool:
    return _reduction_contract(module) is not None
