"""Frozen pre-E2E-REAL-6 x86 elementwise / cohort-2 / breadth packagers (differential tests only).

These are the Graph-owned x86 constructors as they stood before E2E-REAL-6's
x86 elementwise / cohort-2 / breadth cut (2026-09-28), byte-for-byte from
``x86_native.py`` and ``x86_breadth.py`` at ``origin/claude/foundation-batch-3``
(2a70e08a): ``_elementwise_contract`` / ``_cohort2_contract`` /
``graph_breadth_contract`` read the Python Graph object to decide admission,
and ``emit_elementwise_tile_ir`` / ``emit_cohort2_tile_ir`` / ``package_abi``
author Tile IR text beside the compiled Graph -> Schedule -> Tile route.

Production now admits through ``scheduled_kernel.supports_scheduled_kernel(
target="x86")`` and packages the replayed ``ScheduledKernelArtifact`` with
``x86_native.package_scheduled_kernel`` (the native owner is
``src/compiler/programming_model/lib/NativeX86Kernel.h``; normalization rides
``schedule.norm``). ``tessera.alibi`` alone keeps its retired constructor in
production (its Graph operand list is not decodable by position), so
``x86_native`` still carries the narrowed ``_alibi_contract`` /
``_emit_alibi_tile_ir``.

This module is the **declared oracle** Decision #31(a) allows; its only
consumers are the differential tests (``tests/unit/test_x86_kernel_differential.py``)
and the constructor-shape pins in the pre-existing x86 spine tests. The
``abs``/``floor``/``ceil``/``cumsum`` branches still call
``scheduled_absolute`` exactly as the retired packagers did.

Known divergences are kept verbatim, because a baseline that is quietly
corrected stops being evidence of what the retired route did:

* keyword attributes the ABI does not implement were ignored rather than
  refused (``numeric_policy`` on a norm, ``lower=False`` on a Cholesky,
  ``trans``/``unit_diag`` on a triangular solve, a norm ``axis`` other than
  the last, a comparison ``signedness``, any unknown keyword);
* ``keepdims`` was coerced with ``bool(...)``;
* a flattened (``axis=None``) scan over a rank >= 2 operand was served, with
  the descriptor claiming a rank-1 input;
* a pointwise-loss ``delta``/``beta`` of ``0.0`` was admitted.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, cast

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
    X86_ALIBI_F32_ABI,
    X86_ARGREDUCE_F32_ABI,
    X86_ARGREDUCE_KINDS,
    X86_BINARY_F32_ABI,
    X86_BINARY_KINDS,
    X86_BINARY_MATH_F32_ABI,
    X86_BINARY_MATH_KINDS,
    X86_BITWISE_I32_ABI,
    X86_BITWISE_KINDS,
    X86_COMPARE_F32_ABI,
    X86_COMPARE_KINDS,
    X86_LOGICAL_I8_ABI,
    X86_LOGICAL_KINDS,
    X86_NORM_F32_ABI,
    X86_NORM_KINDS,
    X86_PREDICATE_F32_ABI,
    X86_PREDICATE_KINDS,
    X86_ROPE_F32_ABI,
    X86_SCAN_F32_ABI,
    X86_SCAN_KINDS,
    X86_TRANSCENDENTAL_F32_ABI,
    X86_TRANSCENDENTAL_KINDS,
    X86_UNARY_F32_ABI,
    X86_UNARY_KINDS,
    X86_WHERE_F32_ABI,
    X86_WHERE_KINDS,
    X86NativePackage,
    _image,
    _shape,
    requests_cohort2,
    requests_elementwise,
)
from tessera.compiler.x86_breadth import package_abi


def emit_elementwise_tile_ir(*, entry: str, family: str, kind: str) -> str:
    if family not in {
        "unary", "binary", "predicate", "compare", "logical", "bitwise",
        "where", "transcendental", "binary_math",
    }:
        raise ValueError(f"unsupported x86 elementwise family {family!r}")
    storage = "i8" if family == "logical" else "i32" if family == "bitwise" else "f32"
    output_storage = "i8" if family in {"predicate", "compare", "logical"} else storage
    binary_arity = (
        family in {"binary", "compare"}
        or (family == "logical" and kind != "not")
        or (family == "bitwise" and kind not in {"not", "popcount"})
    )
    if family == "where":
        arguments = "%c: !llvm.ptr, %a: !llvm.ptr, %b: !llvm.ptr, %o: !llvm.ptr, %n: i64"
        operands = "%c, %a, %b, %o, %n"
        types = "!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64"
    elif binary_arity or family == "binary_math":
        arguments = "%a: !llvm.ptr, %b: !llvm.ptr, %o: !llvm.ptr, %n: i64"
        operands = "%a, %b, %o, %n"
        types = "!llvm.ptr, !llvm.ptr, !llvm.ptr, i64"
    else:
        arguments = "%x: !llvm.ptr, %o: !llvm.ptr, %n: i64"
        operands = "%x, %o, %n"
        types = "!llvm.ptr, !llvm.ptr, i64"
    condition = ', condition_storage = "i8"' if family == "where" else ""
    return f'''module {{
  llvm.func @{entry}({arguments}) {{
    tile.elementwise_kernel {operands} {{
      family = "{family}", kind = "{kind}", storage = "{storage}",
      output_storage = "{output_storage}"{condition}
    }} : {types}
    llvm.return
  }}
}}
'''


def emit_cohort2_tile_ir(*, entry: str, family: str, kind: str = "", eps: float = 0.0) -> str:
    if family in {"argreduce", "scan"}:
        op = "argreduce_kernel" if family == "argreduce" else "scan_kernel"
        attrs = (
            f'kind = "{kind}", storage = "f32", output_storage = "i32", tie_break = "first"'
            if family == "argreduce"
            else f'kind = "{kind}", storage = "f32", inclusive = true'
        )
        return f'''module {{
  llvm.func @{entry}(%x: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %cols: i64) {{
    tile.{op} %x, %o, %rows, %cols {{ {attrs} }} : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }}
}}
'''
    if family == "norm":
        return f'''module {{
  llvm.func @{entry}(%x: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %cols: i64) {{
    %eps = arith.constant {float(eps):.9e} : f32
    tile.norm_kernel %x, %o, %rows, %cols, %eps {{
      kind = "{kind}", storage = "f32", accum = "f32", axis = -1 : i64, affine = false
    }} : !llvm.ptr, !llvm.ptr, i64, i64, f32
    llvm.return
  }}
}}
'''
    if family == "rope":
        return f'''module {{
  llvm.func @{entry}(%x: !llvm.ptr, %theta: !llvm.ptr, %o: !llvm.ptr, %rows: i64, %cols: i64) {{
    tile.rope_kernel %x, %theta, %o, %rows, %cols {{ storage = "f32", layout = "interleaved_pairs" }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }}
}}
'''
    if family == "alibi":
        return f'''module {{
  llvm.func @{entry}(%slopes: !llvm.ptr, %o: !llvm.ptr, %h: i64, %s: i64) {{
    tile.alibi_kernel %slopes, %o, %h, %s {{ storage = "f32", formula = "slope_times_j_minus_i" }} : !llvm.ptr, !llvm.ptr, i64, i64
    llvm.return
  }}
}}
'''
    raise ValueError(f"unsupported X86-E2E-2 cohort-2 family {family!r}")


def _elementwise_contract(
    module: GraphIRModule,
) -> tuple[str, str, tuple[str, ...], str, tuple[int, ...], tuple[str, ...], str] | None:
    if not requests_elementwise(module):
        return None
    function, op = module.functions[0], module.functions[0].body[0]
    if len(function.result_types) != 1:
        return None
    if op.op_name in X86_UNARY_KINDS:
        family, kind, expected_operands = "unary", X86_UNARY_KINDS[op.op_name], 1
    elif op.op_name in X86_BINARY_KINDS:
        family, kind, expected_operands = "binary", X86_BINARY_KINDS[op.op_name], 2
    elif op.op_name in X86_PREDICATE_KINDS:
        family, kind, expected_operands = "predicate", X86_PREDICATE_KINDS[op.op_name], 1
    elif op.op_name in X86_COMPARE_KINDS:
        family, kind, expected_operands = "compare", X86_COMPARE_KINDS[op.op_name], 2
    elif op.op_name in X86_LOGICAL_KINDS:
        family, kind = "logical", X86_LOGICAL_KINDS[op.op_name]
        expected_operands = 1 if kind == "not" else 2
    elif op.op_name in X86_BITWISE_KINDS:
        family, kind = "bitwise", X86_BITWISE_KINDS[op.op_name]
        expected_operands = 1 if kind in {"not", "popcount"} else 2
    elif op.op_name in X86_WHERE_KINDS:
        family, kind, expected_operands = "where", X86_WHERE_KINDS[op.op_name], 3
    elif op.op_name in X86_TRANSCENDENTAL_KINDS:
        family, kind, expected_operands = (
            "transcendental", X86_TRANSCENDENTAL_KINDS[op.op_name], 1
        )
    else:
        family, kind, expected_operands = (
            "binary_math", X86_BINARY_MATH_KINDS[op.op_name], 2
        )
    if len(op.operands) != expected_operands:
        return None
    names = tuple(value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in function.args}
    shapes = tuple(_shape(module, name) for name in names)
    input_dtypes = (
        ("bool", "fp32", "fp32") if family == "where"
        else ("bool",) * expected_operands if family == "logical"
        else ("int32",) * expected_operands if family == "bitwise"
        else ("fp32",) * expected_operands
    )
    if (
        any(
            name not in args or args[name].ir_type.dtype != dtype
            for name, dtype in zip(names, input_dtypes)
        )
        or any(shape is None for shape in shapes)
        or len(set(shapes)) != 1
    ):
        return None
    shape = shapes[0]
    assert shape is not None
    result = function.result_types[0]
    expected_dtype = (
        "bool" if family in {"predicate", "compare", "logical"}
        else "int32" if family == "bitwise" else "fp32"
    )
    try:
        result_shape = tuple(int(value) for value in result.shape)
    except (TypeError, ValueError):
        return None
    if result.dtype != expected_dtype or result_shape != shape:
        return None
    return family, kind, names, op.result or "output", shape, input_dtypes, expected_dtype


def _cohort2_contract(module: GraphIRModule) -> dict[str, object] | None:
    if not requests_cohort2(module):
        return None
    function, op = module.functions[0], module.functions[0].body[0]
    args = {arg.name: arg for arg in function.args}
    names = tuple(value.removeprefix("%") for value in op.operands)
    if len(function.result_types) != 1:
        return None
    result = function.result_types[0]
    try:
        output_shape = tuple(int(value) for value in result.shape)
    except (TypeError, ValueError):
        return None
    output_name = op.result or "output"
    if op.op_name in X86_ARGREDUCE_KINDS or op.op_name in X86_SCAN_KINDS:
        if len(names) != 1 or names[0] not in args:
            return None
        shape = _shape(module, names[0])
        if shape is None or args[names[0]].ir_type.dtype != "fp32":
            return None
        raw_axis = op.kwargs.get("axis", -1)
        if raw_axis is None:
            axis = 0
            shape = (math.prod(shape),)
        elif isinstance(raw_axis, int) and not isinstance(raw_axis, bool):
            axis = raw_axis + len(shape) if raw_axis < 0 else raw_axis
        else:
            return None
        if axis != len(shape) - 1:
            return None
        rows, cols = (math.prod(shape[:-1]) if len(shape) > 1 else 1), shape[-1]
        if op.op_name in X86_SCAN_KINDS:
            if result.dtype != "fp32" or output_shape != shape:
                return None
            return {"family": "scan", "kind": X86_SCAN_KINDS[op.op_name],
                    "inputs": names, "output": output_name, "shape": shape,
                    "output_shape": output_shape, "rows": rows, "cols": cols}
        keepdims = bool(op.kwargs.get("keepdims", False))
        expected = shape[:-1] + ((1,) if keepdims else ())
        if result.dtype != "int32" or output_shape != expected:
            return None
        return {"family": "argreduce", "kind": X86_ARGREDUCE_KINDS[op.op_name],
                "inputs": names, "output": output_name, "shape": shape,
                "output_shape": output_shape, "rows": rows, "cols": cols,
                "keepdims": keepdims}
    if op.op_name in X86_NORM_KINDS:
        if len(names) != 1 or names[0] not in args:
            return None
        shape = _shape(module, names[0])
        if (shape is None or args[names[0]].ir_type.dtype != "fp32" or
                result.dtype != "fp32" or output_shape != shape):
            return None
        eps_default = 1e-6 if op.op_name == "tessera.rmsnorm_safe" else 1e-5
        eps = float(op.kwargs.get("eps", eps_default))
        if not math.isfinite(eps) or eps <= 0.0:
            return None
        return {"family": "norm", "kind": X86_NORM_KINDS[op.op_name],
                "inputs": names, "output": output_name, "shape": shape,
                "output_shape": output_shape,
                "rows": math.prod(shape[:-1]) if len(shape) > 1 else 1,
                "cols": shape[-1], "eps": eps}
    if op.op_name == "tessera.rope":
        if len(names) != 2 or any(name not in args for name in names):
            return None
        shape, theta_shape = _shape(module, names[0]), _shape(module, names[1])
        if (shape is None or theta_shape != shape or shape[-1] % 2 or
                any(args[name].ir_type.dtype != "fp32" for name in names) or
                result.dtype != "fp32" or output_shape != shape):
            return None
        return {"family": "rope", "kind": "rope", "inputs": names,
                "output": output_name, "shape": shape, "output_shape": output_shape,
                "rows": math.prod(shape[:-1]) if len(shape) > 1 else 1,
                "cols": shape[-1]}
    if len(names) != 1 or names[0] not in args:
        return None
    slopes_shape = _shape(module, names[0])
    h, s = op.kwargs.get("num_heads"), op.kwargs.get("seq_len")
    if (not isinstance(h, int) or isinstance(h, bool) or not isinstance(s, int) or
            isinstance(s, bool) or h <= 0 or s <= 0 or slopes_shape != (h,) or
            args[names[0]].ir_type.dtype != "fp32" or result.dtype != "fp32" or
            output_shape != (h, s, s)):
        return None
    return {"family": "alibi", "kind": "alibi", "inputs": names,
            "output": output_name, "shape": slopes_shape, "output_shape": output_shape,
            "rows": h, "cols": s}


def supports_cohort2(module: GraphIRModule) -> bool:
    return _cohort2_contract(module) is not None


def supports_elementwise(module: GraphIRModule) -> bool:
    return _elementwise_contract(module) is not None


def supports_promoted_elementwise(module: GraphIRModule) -> bool:
    """Measured automatic-selection policy for the X86-E2E-2 first cohort.

    Unary and predicate descriptors meet the retained-route bound at every
    measured size.  Binary descriptors retain a small fixed validation cost,
    so the canonical selector promotes them only from the measured 16K-element
    crossover; explicit packaging remains available for every valid shape.
    """

    contract = _elementwise_contract(module)
    if contract is None:
        return False
    family, _, _, _, shape, _, _ = contract
    elements = math.prod(shape)
    if family in {"unary", "predicate", "logical"}:
        return True
    if family == "compare":
        return elements >= 32_768
    if family in {"binary", "bitwise"}:
        return elements >= 16_384 if family == "binary" else elements >= 32_768
    if family == "transcendental":
        return True
    if family == "where":
        return elements >= 1_048_576
    if family == "binary_math":
        return elements >= 8_224
    return False


def package_cohort2(module: GraphIRModule, *, pipeline_name: str) -> X86NativePackage:
    contract = _cohort2_contract(module)
    if contract is None:
        raise ValueError("x86 native cohort 2 requires one supported static f32 operation")
    if contract["family"] == "scan" and contract["kind"] == "sum":
        from tessera.compiler.scheduled_absolute import lower_cumsum, package_cumsum
        return package_cumsum(lower_cumsum(module),pipeline_name=pipeline_name)
    family, kind = str(contract["family"]), str(contract["kind"])
    variants = {
        "argreduce": ("tessera_x86_avx512_argreduce_f32", X86_ARGREDUCE_F32_ABI),
        "scan": ("tessera_x86_avx512_scan_f32", X86_SCAN_F32_ABI),
        "norm": (
            "tessera_x86_avx512_rmsnorm_f32" if kind == "rmsnorm"
            else "tessera_x86_avx512_layernorm_f32",
            X86_NORM_F32_ABI,
        ),
        "rope": ("tessera_x86_avx512_rope_f32", X86_ROPE_F32_ABI),
        "alibi": ("tessera_x86_avx512_alibi_f32", X86_ALIBI_F32_ABI),
    }
    symbol, abi = variants[family]
    tile_ir = emit_cohort2_tile_ir(
        entry=f"tessera_tile_x86_{family}_{kind}", family=family, kind=kind,
        eps=float(cast(Any, contract.get("eps", 0.0))),
    )
    target_ir, payload, compiler, toolchain = _x86_native._lower(tile_ir, symbol, family)
    image = _image(
        target_ir=target_ir, payload=payload, compiler=compiler,
        toolchain=toolchain, pipeline_name=pipeline_name, symbol=symbol, abi=abi,
    )
    input_names = cast(tuple[str, ...], contract["inputs"])
    output_name = str(contract["output"])
    shape = cast(tuple[int, ...], contract["shape"])
    output_shape = cast(tuple[int, ...], contract["output_shape"])
    output_dtype = "int32" if family == "argreduce" else "fp32"
    bindings = [
        BufferBinding(index, name, "input", "fp32", len(shape), "row_major", 4)
        for index, name in enumerate(input_names)
    ]
    bindings.append(BufferBinding(
        len(bindings), output_name, "output", output_dtype, len(output_shape),
        "row_major", 4,
    ))
    scalar_names = ("H", "S") if family == "alibi" else ("Rows", "Cols")
    scalars = [
        ScalarArgument(len(bindings) + index, name, "int64")
        for index, name in enumerate(scalar_names)
    ]
    if family == "norm":
        scalars.append(ScalarArgument(len(bindings) + 2, "Epsilon", "float32"))
    shapes = {name: shape for name in input_names}
    if family == "alibi":
        shapes[input_names[0]] = (int(cast(Any, contract["rows"])),)
    shapes[output_name] = output_shape
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=symbol, abi_id=abi,
        buffers=tuple(bindings), scalars=tuple(scalars),
        shape_guards=tuple(
            ShapeGuard(name, axis, "eq", extent)
            for name, value_shape in shapes.items()
            for axis, extent in enumerate(value_shape)
        ),
        geometry=LaunchGeometry(policy=f"x86_avx512_{family}"),
        ordering=OrderingSemantics(
            ordered_submission=True, residency="all", synchronization=("return",),
        ),
        provenance={
            "work_item": "X86-E2E-2", "route": "avx512_c_abi",
            "family": family, "kind": kind, "shape": list(shape),
            "output_shape": list(output_shape), "rows": int(cast(Any, contract["rows"])),
            "cols": int(cast(Any, contract["cols"])), "storage": "f32",
            **({"eps": float(cast(Any, contract["eps"]))} if family == "norm" else {}),
            **({"tie_break": "first"} if family == "argreduce" else {}),
            **({"inclusive": True} if family == "scan" else {}),
        },
    )
    return X86NativePackage(tile_ir, target_ir, target_ir, image, descriptor)


def package_elementwise(module: GraphIRModule, *, pipeline_name: str) -> X86NativePackage:
    contract = _elementwise_contract(module)
    if contract is None:
        raise ValueError(
            "x86 native elementwise requires one static same-shape f32 unary/binary "
            "operation or f32-to-bool predicate"
        )
    if contract[:2] == ("unary", "abs"):
        from tessera.compiler.scheduled_absolute import lower_absolute, package_absolute
        return package_absolute(lower_absolute(module), pipeline_name=pipeline_name)
    if contract[:2] in (("unary", "floor"), ("unary", "ceil")):
        from tessera.compiler.scheduled_absolute import lower_floor, lower_ceil, package_unary
        lower = lower_floor if contract[1] == "floor" else lower_ceil
        return package_unary(lower(module), pipeline_name=pipeline_name)
    family, kind, input_names, output_name, shape, input_dtypes, output_dtype = contract
    if family == "unary":
        symbol, abi = "tessera_x86_avx512_unary_f32", X86_UNARY_F32_ABI
    elif family == "binary":
        symbol, abi = "tessera_x86_avx512_binary_f32", X86_BINARY_F32_ABI
    elif family == "predicate":
        symbol, abi = "tessera_x86_avx512_predicate_f32", X86_PREDICATE_F32_ABI
    elif family == "compare":
        symbol, abi = "tessera_x86_avx512_compare_f32", X86_COMPARE_F32_ABI
    elif family == "logical":
        symbol, abi = "tessera_x86_avx512_logical_i8", X86_LOGICAL_I8_ABI
    elif family == "bitwise":
        symbol, abi = "tessera_x86_avx512_bitwise_i32", X86_BITWISE_I32_ABI
    elif family == "where":
        symbol, abi = "tessera_x86_avx512_where_f32", X86_WHERE_F32_ABI
    elif family == "transcendental":
        symbol, abi = (
            "tessera_x86_avx512_transcendental_f32", X86_TRANSCENDENTAL_F32_ABI
        )
    else:
        symbol = (
            "tessera_x86_avx512_pow_f32" if kind == "pow"
            else "tessera_x86_avx512_silu_mul_f32"
        )
        abi = X86_BINARY_MATH_F32_ABI
    tile_ir = emit_elementwise_tile_ir(
        entry=f"tessera_tile_x86_{family}_{kind}", family=family, kind=kind,
    )
    target_ir, payload, compiler, toolchain = _x86_native._lower(
        tile_ir, symbol, "elementwise"
    )
    image = _image(
        target_ir=target_ir, payload=payload, compiler=compiler,
        toolchain=toolchain, pipeline_name=pipeline_name, symbol=symbol, abi=abi,
    )
    bindings = [
        BufferBinding(index, name, "input", dtype, len(shape), "row_major",
                      1 if dtype == "bool" else 4)
        for index, (name, dtype) in enumerate(zip(input_names, input_dtypes))
    ]
    bindings.append(BufferBinding(
        len(bindings), output_name, "output", output_dtype, len(shape),
        "row_major", 1 if output_dtype == "bool" else 4,
    ))
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=symbol, abi_id=abi,
        buffers=tuple(bindings),
        scalars=(ScalarArgument(len(bindings), "N", "int64"),),
        shape_guards=tuple(
            ShapeGuard(name, axis, "eq", extent)
            for name in (*input_names, output_name)
            for axis, extent in enumerate(shape)
        ),
        geometry=LaunchGeometry(policy="x86_avx512_flat"),
        ordering=OrderingSemantics(
            ordered_submission=True, residency="all", synchronization=("return",),
        ),
        provenance={
            "work_item": "X86-E2E-2", "route": "avx512_c_abi",
            "family": family, "kind": kind, "shape": list(shape),
            "elements": math.prod(shape),
            "storage": (
                "mixed_i8_f32" if family == "where"
                else "i8" if input_dtypes[0] == "bool"
                else "i32" if input_dtypes[0] == "int32" else "f32"
            ),
            "output_storage": "i8" if output_dtype == "bool" else "i32" if output_dtype == "int32" else "f32",
        },
    )
    return X86NativePackage(tile_ir, target_ir, target_ir, image, descriptor)


GRAPH_PROMOTION_THRESHOLDS: Mapping[str, int | None] = {
    "gather": 1_048_576,
    "pointwise_loss": 16_384,
    "cholesky": 2_048,
    "tri_solve": 512,
}


_POINTWISE_LOSSES: Mapping[str, tuple[int, str | None]] = {
    "tessera.mse_loss": (0, None), "tessera.loss.mse": (0, None),
    "tessera.mae_loss": (1, None), "tessera.loss.mae": (1, None),
    "tessera.huber_loss": (2, "delta"), "tessera.loss.huber": (2, "delta"),
    "tessera.smooth_l1_loss": (3, "beta"),
    "tessera.loss.smooth_l1": (3, "beta"),
    "tessera.log_cosh_loss": (4, None), "tessera.loss.log_cosh": (4, None),
}


def _static_shape(module: GraphIRModule, name: str) -> tuple[int, ...] | None:
    argument = next(
        (item for item in module.functions[0].args if item.name == name), None
    )
    if argument is None or argument.ir_type.rank is None:
        return None
    try:
        shape = tuple(int(value) for value in argument.ir_type.shape)
    except (TypeError, ValueError):
        return None
    return shape if shape and all(value > 0 for value in shape) else None


def _result_shape(module: GraphIRModule) -> tuple[int, ...] | None:
    if len(module.functions[0].result_types) != 1:
        return None
    try:
        shape = tuple(int(value) for value in module.functions[0].result_types[0].shape)
    except (TypeError, ValueError):
        return None
    return shape if all(value > 0 for value in shape) else None


def graph_breadth_contract(module: GraphIRModule) -> dict[str, Any] | None:
    """Return an isomorphic public-Graph/direct-ABI contract, if one exists."""
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return None
    function, op = module.functions[0], module.functions[0].body[0]
    args = {argument.name: argument for argument in function.args}
    names = tuple(value.removeprefix("%") for value in op.operands)
    output_name = op.result or "output"
    output_shape = _result_shape(module)
    if output_shape is None or function.result_types[0].dtype != "fp32":
        return None
    if op.op_name == "tessera.gather":
        if len(names) != 2 or any(name not in args for name in names):
            return None
        source_shape, index_shape = (_static_shape(module, name) for name in names)
        axis = op.kwargs.get("axis", 0)
        if (
            source_shape is None or index_shape is None
            or len(source_shape) != 1 or len(index_shape) != 1
            or output_shape != index_shape or axis not in {0, -1}
            or args[names[0]].ir_type.dtype != "fp32"
            or args[names[1]].ir_type.dtype != "int64"
        ):
            return None
        return {
            "key": "gather_f32", "family": "gather", "inputs": names,
            "output": output_name,
            "shapes": {"source": source_shape, "indices": index_shape,
                       "output": output_shape},
            "names": {"source": names[0], "indices": names[1],
                      "output": output_name},
            "scalars": {"SourceN": source_shape[0], "N": index_shape[0]},
        }
    if op.op_name in _POINTWISE_LOSSES:
        if len(names) != 2 or any(name not in args for name in names):
            return None
        shapes = tuple(_static_shape(module, name) for name in names)
        if (
            any(shape is None for shape in shapes) or len(set(shapes)) != 1
            or output_shape != shapes[0]
            or any(args[name].ir_type.dtype != "fp32" for name in names)
            or str(op.kwargs.get("reduction", "mean")) != "none"
        ):
            return None
        shape = shapes[0]
        kind, parameter_name = _POINTWISE_LOSSES[op.op_name]
        parameter = float(op.kwargs.get(parameter_name, 1.0)) if parameter_name else 0.0
        if not math.isfinite(parameter) or parameter < 0.0:
            return None
        return {
            "key": "pointwise_loss_f32", "family": "pointwise_loss",
            "inputs": names, "output": output_name,
            "shapes": {"prediction": shape, "target": shape, "output": output_shape},
            "names": {"prediction": names[0], "target": names[1],
                      "output": output_name},
            "scalars": {"N": math.prod(shape), "Kind": kind,
                        "Parameter": parameter},
            "kind": kind, "parameter": parameter,
        }
    if op.op_name in {"tessera.cholesky", "tessera.tri_solve"}:
        expected = 1 if op.op_name == "tessera.cholesky" else 2
        if len(names) != expected or any(name not in args for name in names):
            return None
        shapes = tuple(_static_shape(module, name) for name in names)
        matrix_shape = shapes[0]
        if (
            matrix_shape is None or len(matrix_shape) not in {2, 3}
            or matrix_shape[-1] != matrix_shape[-2]
            or any(args[name].ir_type.dtype != "fp32" for name in names)
        ):
            return None
        batch, n = (1, matrix_shape[0]) if len(matrix_shape) == 2 else (
            matrix_shape[0], matrix_shape[1]
        )
        if op.op_name == "tessera.cholesky":
            if output_shape != matrix_shape:
                return None
            return {
                "key": "cholesky_f32", "family": "cholesky",
                "inputs": names, "output": output_name,
                "shapes": {"matrix": matrix_shape, "lower": output_shape},
                "names": {"matrix": names[0], "lower": output_name},
                "scalars": {"Batch": batch, "N": n},
            }
        rhs_shape = shapes[1]
        if rhs_shape is None or output_shape != rhs_shape:
            return None
        if len(matrix_shape) == 2:
            valid_rhs = len(rhs_shape) == 2 and rhs_shape[0] == n
            columns = rhs_shape[1] if valid_rhs else 0
        else:
            valid_rhs = len(rhs_shape) == 3 and rhs_shape[:2] == (batch, n)
            columns = rhs_shape[2] if valid_rhs else 0
        if not valid_rhs:
            return None
        return {
            "key": "tri_solve_f32", "family": "tri_solve",
            "inputs": names, "output": output_name,
            "shapes": {"matrix": matrix_shape, "rhs": rhs_shape,
                       "output": output_shape},
            "names": {"matrix": names[0], "rhs": names[1],
                      "output": output_name},
            "scalars": {"Batch": batch, "N": n, "M": columns,
                        "Lower": int(bool(op.kwargs.get("lower", True)))},
        }
    return None


def requests_graph_breadth(module: GraphIRModule) -> bool:
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return False
    return module.functions[0].body[0].op_name in {
        "tessera.gather", "tessera.cholesky", "tessera.tri_solve",
        *_POINTWISE_LOSSES,
    }


def supports_graph_breadth(module: GraphIRModule) -> bool:
    return graph_breadth_contract(module) is not None


def supports_promoted_graph_breadth(module: GraphIRModule) -> bool:
    """Measured Level-C selector policy; thresholds are benchmark-owned."""
    contract = graph_breadth_contract(module)
    if contract is None:
        return False
    output_shape = _result_shape(module)
    assert output_shape is not None
    elements = math.prod(output_shape)
    family = str(contract["family"])
    threshold = GRAPH_PROMOTION_THRESHOLDS[family]
    return threshold is not None and elements >= threshold


def package_graph_breadth(
    module: GraphIRModule, *, pipeline_name: str,
) -> X86NativePackage:
    contract = graph_breadth_contract(module)
    if contract is None:
        raise ValueError("x86 breadth packaging requires one isomorphic static Graph operation")
    package = package_abi(
        str(contract["key"]), pipeline_name=pipeline_name,
        buffer_shapes=cast(Mapping[str, tuple[int, ...]], contract["shapes"]),
        buffer_names=cast(Mapping[str, str], contract["names"]),
    )
    descriptor = replace(package.descriptor, provenance={
        **package.descriptor.provenance,
        "graph_level": True, "selector_family": str(contract["family"]),
        "graph_scalars": cast(Mapping[str, object], contract["scalars"]),
        **({"kind": int(contract["kind"]),
            "parameter": float(contract["parameter"])}
           if "kind" in contract else {}),
    })
    return replace(package, descriptor=descriptor)
