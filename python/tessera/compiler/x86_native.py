"""Typed native-image packaging for the X86-E2E-1 AVX-512 pilot."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .graph_ir import GraphIRModule
from .native_artifact import (
    BufferBinding,
    LaunchDescriptor,
    LaunchGeometry,
    NativeEntryPoint,
    NativeImageArtifact,
    OrderingSemantics,
    ScalarArgument,
    ShapeGuard,
)


X86_SOFTMAX_F32_ABI = "tessera.x86.softmax.x_o_rows_k.f32.v1"
X86_REDUCE_F32_ABI = "tessera.x86.reduce.x_o_outer_axis_inner.f32.v1"
X86_MATMUL_F32_ABI = "tessera.x86.matmul.a_b_o_m_n_k.f32.v1"
X86_MATMUL_BF16_F32_ABI = "tessera.x86.matmul.a_b_o_m_n_k.bf16_f32.v1"
X86_MATMUL_U8S8_S32_ABI = "tessera.x86.matmul.a_b_o_m_n_k.u8s8_s32.v1"
X86_MATMUL_F64_ABI = "tessera.x86.matmul.a_b_o_m_n_k.f64.v1"
X86_ATTENTION_F32_ABI = "tessera.x86.attention.q_k_v_o_dims.f32.v1"
X86_ATTENTION_EXT_F32_ABI = "tessera.x86.attention_ext.q_k_v_bias_o_dims.f32.v1"
X86_UNARY_F32_ABI = "tessera.x86.elementwise.unary.x_o_n_kind.f32.v1"
X86_BINARY_F32_ABI = "tessera.x86.elementwise.binary.a_b_o_n_kind.f32.v1"
X86_PREDICATE_F32_ABI = "tessera.x86.elementwise.predicate.x_o_n_kind.f32_i8.v1"
X86_COMPARE_F32_ABI = "tessera.x86.elementwise.compare.a_b_o_n_kind.f32_i8.v1"
X86_LOGICAL_I8_ABI = "tessera.x86.elementwise.logical.a_b_o_n_kind.i8.v1"
X86_BITWISE_I32_ABI = "tessera.x86.elementwise.bitwise.a_b_o_n_kind.i32.v1"
X86_WHERE_F32_ABI = "tessera.x86.elementwise.where.c_a_b_o_n.i8_f32.v1"
X86_TRANSCENDENTAL_F32_ABI = "tessera.x86.elementwise.transcendental.x_o_n_kind.f32.v1"
X86_BINARY_MATH_F32_ABI = "tessera.x86.elementwise.binary_math.a_b_o_n.f32.v1"
X86_ARGREDUCE_F32_ABI = "tessera.x86.argreduce.x_o_rows_cols.f32_i32.v1"
X86_SCAN_F32_ABI = "tessera.x86.scan.x_o_rows_cols.f32.v1"
X86_NORM_F32_ABI = "tessera.x86.norm.x_o_rows_cols_eps.f32.v1"
X86_ROPE_F32_ABI = "tessera.x86.rope.x_theta_o_rows_cols.f32.v1"
X86_ALIBI_F32_ABI = "tessera.x86.alibi.slopes_o_h_s.f32.v1"

X86_ARGREDUCE_KINDS = {"tessera.argmax": "argmax", "tessera.argmin": "argmin"}
X86_SCAN_KINDS = {
    "tessera.cumsum": "sum", "tessera.cumprod": "product",
    "tessera.cummax": "max", "tessera.cummin": "min",
}
X86_NORM_KINDS = {
    "tessera.rmsnorm": "rmsnorm", "tessera.rmsnorm_safe": "rmsnorm",
    "tessera.layer_norm": "layernorm",
}

X86_UNARY_KINDS = {
    "tessera.sqrt": "sqrt", "tessera.rsqrt": "rsqrt",
    "tessera.reciprocal": "reciprocal", "tessera.absolute": "abs",
    "tessera.abs": "abs", "tessera.sign": "sign",
    "tessera.floor": "floor", "tessera.ceil": "ceil",
    "tessera.trunc": "trunc", "tessera.round": "round",
}
X86_BINARY_KINDS = {
    "tessera.sub": "sub", "tessera.subtract": "sub",
    "tessera.div": "div", "tessera.divide": "div",
    "tessera.maximum": "maximum", "tessera.minimum": "minimum",
    "tessera.add": "add", "tessera.mul": "mul",
    "tessera.multiply": "mul", "tessera.mod": "mod",
    "tessera.floor_div": "floor_div", "tessera.floor_divide": "floor_div",
}
X86_PREDICATE_KINDS = {
    "tessera.isnan": "isnan", "tessera.isinf": "isinf",
    "tessera.isfinite": "isfinite",
}
X86_COMPARE_KINDS = {
    "tessera.eq": "eq", "tessera.equal": "eq",
    "tessera.ne": "ne", "tessera.not_equal": "ne",
    "tessera.lt": "lt", "tessera.less": "lt",
    "tessera.le": "le", "tessera.less_equal": "le",
    "tessera.gt": "gt", "tessera.greater": "gt",
    "tessera.ge": "ge", "tessera.greater_equal": "ge",
}
X86_LOGICAL_KINDS = {
    "tessera.logical_and": "and", "tessera.logical_or": "or",
    "tessera.logical_xor": "xor", "tessera.logical_not": "not",
}
X86_BITWISE_KINDS = {
    "tessera.bitwise_and": "and", "tessera.bitwise_or": "or",
    "tessera.bitwise_xor": "xor", "tessera.bitwise_not": "not",
    "tessera.popcount": "popcount",
}
X86_WHERE_KINDS = {"tessera.where": "where"}
X86_TRANSCENDENTAL_KINDS = {
    "tessera.exp": "exp", "tessera.log": "log", "tessera.tanh": "tanh",
    "tessera.sigmoid": "sigmoid", "tessera.silu": "silu",
    "tessera.gelu": "gelu", "tessera.erf": "erf",
    "tessera.softplus": "softplus", "tessera.expm1": "expm1",
    "tessera.log1p": "log1p", "tessera.cos": "cos", "tessera.tan": "tan",
    "tessera.sinh": "sinh", "tessera.cosh": "cosh", "tessera.asin": "asin",
    "tessera.acos": "acos", "tessera.atan": "atan", "tessera.erfc": "erfc",
    "tessera.sin": "sin", "tessera.lgamma": "lgamma",
    "tessera.digamma": "digamma",
}
X86_BINARY_MATH_KINDS = {
    "tessera.pow": "pow", "tessera.power": "pow",
    "tessera.silu_mul": "silu_mul", "tessera.swiglu": "silu_mul",
}


@dataclass(frozen=True)
class X86NativePackage:
    tile_ir: str
    target_ir: str
    backend_ir: str
    image: NativeImageArtifact
    descriptor: LaunchDescriptor


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _tessera_opt() -> Path | None:
    if configured := os.environ.get("TESSERA_OPT"):
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    root = _repo_root()
    for path in (
        root / "build-rocm-7.14-llvm23-clean/tools/tessera-opt/tessera-opt",
        root / "build/tools/tessera-opt/tessera-opt",
    ):
        if path.is_file():
            return path
    found = shutil.which("tessera-opt")
    return Path(found) if found else None


def _library_path() -> Path | None:
    if configured := os.environ.get("TESSERA_X86_ELEMENTWISE_LIB"):
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    root = _repo_root()
    for path in (
        root / "build-rocm-7.14-llvm23-clean/src/compiler/codegen/tessera_x86_backend/libtessera_x86_elementwise.so",
        root / "build/src/compiler/codegen/tessera_x86_backend/libtessera_x86_elementwise.so",
    ):
        if path.is_file():
            return path
    return None


def tools_available() -> bool:
    return _tessera_opt() is not None and _library_path() is not None


def supports_native_package(module: GraphIRModule) -> bool:
    """Whether *module* has one complete X86-E2E-1 stable-ABI contract.

    This is intentionally stricter than the operation-name predicates used by
    the explicit packaging path: automatic canonical selection must not turn a
    dynamic shape, unsupported dtype, or unsupported attention variant into a
    packaging error.  Those cases remain on their retained x86 route.
    """

    from .x86_breadth import supports_promoted_graph_breadth

    return any((
        supports_softmax(module),
        supports_reduction(module),
        supports_promoted_matmul(module),
        supports_attention(module),
        supports_promoted_elementwise(module),
        supports_promoted_graph_breadth(module),
    ))


def _version_fingerprint(tool: Path) -> str:
    result = subprocess.run([str(tool), "--version"], capture_output=True, text=True, check=False)
    text = "\n".join(value.strip() for value in (result.stdout, result.stderr) if value.strip())
    return hashlib.sha256((text or str(tool)).encode()).hexdigest()


def _lower(tile_ir: str, symbol: str) -> tuple[str, bytes, str, str]:
    tool, library = _tessera_opt(), _library_path()
    if tool is None or library is None:
        raise RuntimeError("X86-E2E-1 requires tessera-opt and libtessera_x86_elementwise.so")
    result = subprocess.run(
        [str(tool), "-", "--tessera-tile-to-x86=prefer-amx=false"],
        input=tile_ir, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError("x86 typed lowering failed: " + (result.stderr.strip() or str(result.returncode)))
    target_ir = result.stdout
    if f"call @{symbol}" not in target_ir:
        raise RuntimeError(f"x86 typed lowering did not emit {symbol}")
    payload = library.read_bytes()
    compiler = _version_fingerprint(tool)
    toolchain = hashlib.sha256(f"{compiler}|{hashlib.sha256(payload).hexdigest()}|x86_64_avx512".encode()).hexdigest()
    return target_ir, payload, compiler, toolchain


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


def emit_matmul_tile_ir(
    *, entry: str, a_storage: str = "f32", b_storage: str = "f32",
    accum: str = "f32", output: str = "f32",
) -> str:
    return f'''module {{
  llvm.func @{entry}(%a: !llvm.ptr, %b: !llvm.ptr, %o: !llvm.ptr,
                     %m: i64, %n: i64, %k: i64) {{
    tile.matmul_kernel %a, %b, %o, %m, %n, %k {{
      mma = #tile.mma_desc<family = "auto", m = 16, n = 16, k = 16, a = "{a_storage}", b = "{b_storage}", acc = "{accum}", a_layout = "row_major", b_layout = "col_major", k_blocks = 1>,
      epilogue = #tile.epilogue<bias = false, activation = "none", output = "{output}">,
      warps = 1 : i64, staging = "global"
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }}
}}
'''


def emit_attention_tile_ir(
    *, entry: str, scale: float, causal: bool, bias: bool,
    window: int, softcap: float,
) -> str:
    optional_arg = "%bias: !llvm.ptr, " if bias else ""
    optional_operand = "%bias, " if bias else ""
    return f'''module {{
  llvm.func @{entry}(%q: !llvm.ptr, %key: !llvm.ptr, %v: !llvm.ptr,
                     {optional_arg}%o: !llvm.ptr, %b: i64, %hq: i64, %hkv: i64,
                     %sq: i64, %sk: i64, %d: i64, %dv: i64) {{
    tile.attention_kernel %q, %key, %v, {optional_operand}%o,
        %b, %hq, %hkv, %sq, %sk, %d, %dv {{
      storage = "f32", accum = "f32", scale = {float(scale)!r} : f32,
      causal = {str(causal).lower()}, bias = {str(bias).lower()},
      window_left = {window} : i64, window_right = {window} : i64,
      softcap = {float(softcap)!r} : f32, dropout_p = 0.0 : f32,
      dropout_seed = 0 : i64
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, {"!llvm.ptr, " * (1 + int(bias))}i64, i64, i64, i64, i64, i64, i64
    llvm.return
  }}
}}
'''


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


def requests_softmax(module: GraphIRModule) -> bool:
    return (
        len(module.functions) == 1 and len(module.functions[0].body) == 1
        and module.functions[0].body[0].op_name in {"tessera.softmax", "tessera.softmax_safe"}
    )


def requests_reduction(module: GraphIRModule) -> bool:
    return (
        len(module.functions) == 1 and len(module.functions[0].body) == 1
        and module.functions[0].body[0].op_name
        in {"tessera.sum", "tessera.mean", "tessera.max", "tessera.amax"}
    )


def requests_matmul(module: GraphIRModule) -> bool:
    return (
        len(module.functions) == 1 and len(module.functions[0].body) == 1
        and module.functions[0].body[0].op_name in {"tessera.matmul", "tessera.gemm"}
    )


def requests_attention(module: GraphIRModule) -> bool:
    return (
        len(module.functions) == 1 and len(module.functions[0].body) == 1
        and module.functions[0].body[0].op_name == "tessera.flash_attn"
    )


def requests_elementwise(module: GraphIRModule) -> bool:
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return False
    name = module.functions[0].body[0].op_name
    return any(name in kinds for kinds in (
        X86_UNARY_KINDS, X86_BINARY_KINDS, X86_PREDICATE_KINDS,
        X86_COMPARE_KINDS, X86_LOGICAL_KINDS, X86_BITWISE_KINDS,
        X86_WHERE_KINDS, X86_TRANSCENDENTAL_KINDS, X86_BINARY_MATH_KINDS,
    ))


def requests_cohort2(module: GraphIRModule) -> bool:
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return False
    return module.functions[0].body[0].op_name in {
        *X86_ARGREDUCE_KINDS, *X86_SCAN_KINDS, *X86_NORM_KINDS,
        "tessera.rope", "tessera.alibi",
    }


def _shape(module: GraphIRModule, name: str) -> tuple[int, ...] | None:
    arg = next((value for value in module.functions[0].args if value.name == name), None)
    if arg is None or arg.ir_type.rank is None or arg.ir_type.rank < 1:
        return None
    try:
        shape = tuple(int(value) for value in arg.ir_type.shape)
    except (TypeError, ValueError):
        return None
    return shape if all(value > 0 for value in shape) else None


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


def _matmul_contract(
    module: GraphIRModule,
) -> tuple[str, str, str, tuple[int, int, int], tuple[str, str, str]] | None:
    if not requests_matmul(module):
        return None
    function, op = module.functions[0], module.functions[0].body[0]
    if len(op.operands) != 2 or len(function.result_types) != 1:
        return None
    a_name, b_name = (value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in function.args}
    a_shape, b_shape = _shape(module, a_name), _shape(module, b_name)
    if (
        a_name not in args or b_name not in args
        or a_shape is None or b_shape is None or len(a_shape) != 2 or len(b_shape) != 2
    ):
        return None
    m, k = a_shape
    kb, n = b_shape
    result = function.result_types[0]
    try:
        output_shape = tuple(int(value) for value in result.shape)
    except (TypeError, ValueError):
        return None
    dtype_contracts: dict[tuple[str, str], tuple[str, str, str, str]] = {
        ("fp32", "fp32"): ("fp32", "f32", "f32", "f32"),
        ("bf16", "bf16"): ("fp32", "bf16", "bf16", "f32"),
        ("uint8", "int8"): ("int32", "u8", "i8", "i32"),
        ("fp64", "fp64"): ("fp64", "f64", "f64", "f64"),
    }
    a_dtype = args[a_name].ir_type.dtype
    b_dtype = args[b_name].ir_type.dtype
    if not isinstance(a_dtype, str) or not isinstance(b_dtype, str):
        return None
    key = (a_dtype, b_dtype)
    dtype_contract = dtype_contracts.get(key)
    if dtype_contract is None:
        return None
    result_dtype, _, _, _ = dtype_contract
    if k != kb or result.dtype != result_dtype or output_shape != (m, n):
        return None
    return a_name, b_name, op.result or "output", (m, n, k), (
        a_dtype, b_dtype, result_dtype,
    )


def _attention_contract(
    module: GraphIRModule,
) -> tuple[tuple[str, str, str], str | None, str, tuple[int, ...], float, bool, int, float] | None:
    if not requests_attention(module):
        return None
    function, op = module.functions[0], module.functions[0].body[0]
    if len(op.operands) not in {3, 4} or len(function.result_types) != 1:
        return None
    names = cast(
        tuple[str, str, str],
        tuple(value.removeprefix("%") for value in op.operands[:3]),
    )
    args = {arg.name: arg for arg in function.args}
    shapes = tuple(_shape(module, name) for name in names)
    if (
        any(name not in args or args[name].ir_type.dtype != "fp32" for name in names)
        or any(shape is None or len(shape) != 4 for shape in shapes)
    ):
        return None
    q_shape, k_shape, v_shape = shapes
    assert q_shape is not None and k_shape is not None and v_shape is not None
    b, hq, sq, d = q_shape
    bk, hkv, sk, dk = k_shape
    bv, hv, sv, dv = v_shape
    if b != bk or b != bv or hq != hkv or hkv != hv or sk != sv or d != dk:
        return None
    output_name = op.result or "output"
    result = function.result_types[0]
    try:
        output_shape = tuple(int(value) for value in result.shape)
    except (TypeError, ValueError):
        return None
    if result.dtype != "fp32" or output_shape != (b, hq, sq, dv):
        return None
    bias_name = op.operands[3].removeprefix("%") if len(op.operands) == 4 else None
    if bias_name is not None:
        bias = args.get(bias_name)
        if bias is None or bias.ir_type.dtype != "fp32" or _shape(module, bias_name) != (b, hq, sq, sk):
            return None
    raw_window = op.kwargs.get("window", -1)
    if isinstance(raw_window, (tuple, list)):
        if len(raw_window) != 2 or raw_window[0] != raw_window[1]:
            return None
        raw_window = raw_window[0]
    if not isinstance(raw_window, int) or isinstance(raw_window, bool) or raw_window < -1:
        return None
    softcap = float(op.kwargs.get("softcap", op.kwargs.get("logit_softcap", 0.0)) or 0.0)
    dropout = float(op.kwargs.get("dropout_p", op.kwargs.get("dropout", 0.0)) or 0.0)
    scale = float(op.kwargs.get("scale", 1.0 / math.sqrt(float(d))))
    if not math.isfinite(scale) or scale <= 0.0 or not math.isfinite(softcap) or softcap < 0.0 or dropout != 0.0:
        return None
    return names, bias_name, output_name, (b, hq, hkv, sq, sk, d, dv), scale, bool(op.kwargs.get("causal", False)), raw_window, softcap


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


def supports_softmax(module: GraphIRModule) -> bool:
    return _softmax_contract(module) is not None


def supports_reduction(module: GraphIRModule) -> bool:
    return _reduction_contract(module) is not None


def supports_matmul(module: GraphIRModule) -> bool:
    return _matmul_contract(module) is not None


def supports_promoted_matmul(module: GraphIRModule) -> bool:
    contract = _matmul_contract(module)
    return contract is not None and contract[-1] == ("fp32", "fp32", "fp32")


def supports_attention(module: GraphIRModule) -> bool:
    return _attention_contract(module) is not None


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


def _image(
    *, target_ir: str, payload: bytes, compiler: str, toolchain: str,
    pipeline_name: str, symbol: str, abi: str,
) -> NativeImageArtifact:
    return NativeImageArtifact(
        target="x86", architecture="x86_64_avx512", pipeline_name=pipeline_name,
        compiler_fingerprint=compiler, toolchain_fingerprint=toolchain,
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(),
        binary_format="shared_object", payload=payload,
        entry_points=(NativeEntryPoint(symbol, abi),), compile_state="prepackaged",
    )


def package_cohort2(module: GraphIRModule, *, pipeline_name: str) -> X86NativePackage:
    contract = _cohort2_contract(module)
    if contract is None:
        raise ValueError("x86 native cohort 2 requires one supported static f32 operation")
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
    target_ir, payload, compiler, toolchain = _lower(tile_ir, symbol)
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


def package_softmax(module: GraphIRModule, *, pipeline_name: str) -> X86NativePackage:
    contract = _softmax_contract(module)
    if contract is None:
        raise ValueError("x86 native softmax requires one static f32 last-axis operation")
    input_name, output_name, shape = contract
    symbol = "tessera_x86_avx512_softmax_f32"
    tile_ir = emit_softmax_tile_ir(entry="tessera_tile_x86_softmax_f32")
    target_ir, payload, compiler, toolchain = _lower(tile_ir, symbol)
    image = _image(target_ir=target_ir, payload=payload, compiler=compiler, toolchain=toolchain,
                   pipeline_name=pipeline_name, symbol=symbol, abi=X86_SOFTMAX_F32_ABI)
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
        geometry=LaunchGeometry(policy="x86_avx512_rows"),
        ordering=OrderingSemantics(ordered_submission=True, residency="all", synchronization=("return",)),
        provenance={
            "work_item": "X86-E2E-1", "route": "avx512_c_abi",
            "shape": list(shape), "rows": rows, "columns": columns,
            "storage": "f32", "accum": "f32",
        },
    )
    return X86NativePackage(tile_ir, target_ir, target_ir, image, descriptor)


def package_reduction(module: GraphIRModule, *, pipeline_name: str) -> X86NativePackage:
    contract = _reduction_contract(module)
    if contract is None:
        raise ValueError("x86 native reduction requires one static f32 last-axis sum/mean/max")
    input_name, output_name, kind, shape, output_shape, axis, keepdims = contract
    symbol = "tessera_x86_avx512_reduce_f32"
    tile_ir = emit_reduce_tile_ir(entry=f"tessera_tile_x86_reduce_{kind}_f32", kind=kind, axis=axis, keepdims=keepdims)
    target_ir, payload, compiler, toolchain = _lower(tile_ir, symbol)
    image = _image(target_ir=target_ir, payload=payload, compiler=compiler, toolchain=toolchain,
                   pipeline_name=pipeline_name, symbol=symbol, abi=X86_REDUCE_F32_ABI)
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
        geometry=LaunchGeometry(policy="x86_avx512_rows"),
        ordering=OrderingSemantics(ordered_submission=True, residency="all", synchronization=("return",)),
        provenance={
            "work_item": "X86-E2E-1", "route": "avx512_c_abi", "kind": kind,
            "shape": list(shape), "axis": axis, "keepdims": keepdims,
            "outer": outer, "axis_extent": extent, "inner": 1,
            "storage": "f32", "accum": "f32",
        },
    )
    return X86NativePackage(tile_ir, target_ir, target_ir, image, descriptor)


def package_matmul(module: GraphIRModule, *, pipeline_name: str) -> X86NativePackage:
    contract = _matmul_contract(module)
    if contract is None:
        raise ValueError("x86 native matmul requires one static rank-2 f32 matmul")
    a_name, b_name, output_name, (m, n, k), dtypes = contract
    a_dtype, b_dtype, output_dtype = dtypes
    variants = {
        ("fp32", "fp32", "fp32"): (
            "tessera_x86_avx512_gemm_f32", X86_MATMUL_F32_ABI,
            "f32", "f32", "f32", "f32", (4, 4, 4), ("avx512f", "fma"),
        ),
        ("bf16", "bf16", "fp32"): (
            "tessera_x86_avx512_gemm_bf16", X86_MATMUL_BF16_F32_ABI,
            "bf16", "bf16", "f32", "f32", (2, 2, 4), ("avx512_bf16",),
        ),
        ("uint8", "int8", "int32"): (
            "tessera_x86_avx512_vnni_gemm_u8s8_s32", X86_MATMUL_U8S8_S32_ABI,
            "u8", "i8", "i32", "i32", (1, 1, 4), ("avx512bw", "avx512_vnni"),
        ),
        ("fp64", "fp64", "fp64"): (
            "tessera_x86_avx512_gemm_f64", X86_MATMUL_F64_ABI,
            "f64", "f64", "f64", "f64", (8, 8, 8), ("avx512f", "fma"),
        ),
    }
    symbol, abi, a_storage, b_storage, accum, output_storage, byte_sizes, features = variants[dtypes]
    tile_ir = emit_matmul_tile_ir(
        entry=f"tessera_tile_x86_matmul_{a_storage}_{b_storage}_{output_storage}",
        a_storage=a_storage, b_storage=b_storage, accum=accum, output=output_storage,
    )
    target_ir, payload, compiler, toolchain = _lower(tile_ir, symbol)
    image = _image(
        target_ir=target_ir, payload=payload, compiler=compiler, toolchain=toolchain,
        pipeline_name=pipeline_name, symbol=symbol, abi=abi,
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=symbol, abi_id=abi,
        buffers=(
            BufferBinding(0, a_name, "input", a_dtype, 2, "row_major", byte_sizes[0]),
            BufferBinding(1, b_name, "input", b_dtype, 2, "row_major", byte_sizes[1]),
            BufferBinding(2, output_name, "output", output_dtype, 2, "row_major", byte_sizes[2]),
        ),
        scalars=(
            ScalarArgument(3, "M", "int64"), ScalarArgument(4, "N", "int64"),
            ScalarArgument(5, "K", "int64"),
        ),
        shape_guards=(
            ShapeGuard(a_name, 0, "eq", m), ShapeGuard(a_name, 1, "eq", k),
            ShapeGuard(b_name, 0, "eq", k), ShapeGuard(b_name, 1, "eq", n),
            ShapeGuard(output_name, 0, "eq", m), ShapeGuard(output_name, 1, "eq", n),
        ),
        geometry=LaunchGeometry(policy="x86_avx512_gemm_rows"),
        ordering=OrderingSemantics(ordered_submission=True, residency="all", synchronization=("return",)),
        provenance={
            "work_item": "X86-E2E-1" if abi == X86_MATMUL_F32_ABI else "X86-E2E-2",
            "route": "avx512_c_abi", "shape": [m, n, k],
            "a_storage": a_storage, "b_storage": b_storage,
            "output_storage": output_storage, "accum": accum,
            "required_features": list(features),
        },
    )
    return X86NativePackage(tile_ir, target_ir, target_ir, image, descriptor)


def package_attention(module: GraphIRModule, *, pipeline_name: str) -> X86NativePackage:
    contract = _attention_contract(module)
    if contract is None:
        raise ValueError(
            "x86 native attention requires static rank-4 f32 MHA, optional exact-shape "
            "f32 bias, symmetric window semantics, and dropout=0"
        )
    names, bias_name, output_name, dims, scale, causal, window, softcap = contract
    extended = bias_name is not None or window >= 0 or softcap > 0.0
    symbol = "tessera_x86_flash_attn_ext_f32" if extended else "tessera_x86_flash_attn_f32"
    abi = X86_ATTENTION_EXT_F32_ABI if extended else X86_ATTENTION_F32_ABI
    semantic = hashlib.sha256(
        f"{scale:.17g}:{causal}:{bool(bias_name)}:{window}:{softcap:.17g}".encode()
    ).hexdigest()[:10]
    tile_ir = emit_attention_tile_ir(
        entry=f"tessera_tile_x86_attention_{semantic}", scale=scale, causal=causal,
        bias=bias_name is not None, window=window, softcap=softcap,
    )
    target_ir, payload, compiler, toolchain = _lower(tile_ir, symbol)
    image = _image(
        target_ir=target_ir, payload=payload, compiler=compiler, toolchain=toolchain,
        pipeline_name=pipeline_name, symbol=symbol, abi=abi,
    )
    q_name, k_name, v_name = names
    b, hq, hkv, sq, sk, d, dv = dims
    bindings = [
        BufferBinding(0, q_name, "input", "fp32", 4, "row_major", 4),
        BufferBinding(1, k_name, "input", "fp32", 4, "row_major", 4),
        BufferBinding(2, v_name, "input", "fp32", 4, "row_major", 4),
    ]
    if bias_name is not None:
        bindings.append(BufferBinding(3, bias_name, "input", "fp32", 4, "row_major", 4))
    bindings.append(BufferBinding(len(bindings), output_name, "output", "fp32", 4, "row_major", 4))
    shapes = {
        q_name: (b, hq, sq, d), k_name: (b, hkv, sk, d),
        v_name: (b, hkv, sk, dv), output_name: (b, hq, sq, dv),
    }
    if bias_name is not None:
        shapes[bias_name] = (b, hq, sq, sk)
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=symbol, abi_id=abi,
        buffers=tuple(bindings),
        scalars=tuple(
            ScalarArgument(len(bindings) + index, name, "int64")
            for index, name in enumerate(("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv"))
        ),
        shape_guards=tuple(
            ShapeGuard(name, axis, "eq", extent)
            for name, shape in shapes.items() for axis, extent in enumerate(shape)
        ),
        geometry=LaunchGeometry(policy="x86_avx512_attention_rows"),
        ordering=OrderingSemantics(ordered_submission=True, residency="all", synchronization=("return",)),
        provenance={
            "work_item": "X86-E2E-1", "route": "avx512_c_abi",
            "shape": list(dims), "storage": "f32", "accum": "f32",
            "scale": scale, "causal": causal, "bias": bias_name is not None,
            "window": window, "softcap": softcap, "extended": extended,
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
    target_ir, payload, compiler, toolchain = _lower(tile_ir, symbol)
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


__all__ = [
    "X86NativePackage", "X86_ALIBI_F32_ABI", "X86_ARGREDUCE_F32_ABI",
    "X86_ATTENTION_EXT_F32_ABI", "X86_ATTENTION_F32_ABI",
    "X86_BINARY_F32_ABI", "X86_BINARY_MATH_F32_ABI", "X86_BITWISE_I32_ABI", "X86_COMPARE_F32_ABI",
    "X86_LOGICAL_I8_ABI", "X86_MATMUL_BF16_F32_ABI", "X86_MATMUL_F32_ABI",
    "X86_MATMUL_F64_ABI", "X86_MATMUL_U8S8_S32_ABI", "X86_PREDICATE_F32_ABI",
    "X86_NORM_F32_ABI", "X86_REDUCE_F32_ABI", "X86_ROPE_F32_ABI",
    "X86_SCAN_F32_ABI", "X86_SOFTMAX_F32_ABI", "X86_TRANSCENDENTAL_F32_ABI",
    "X86_UNARY_F32_ABI", "X86_WHERE_F32_ABI",
    "X86_BINARY_MATH_KINDS", "X86_BITWISE_KINDS", "X86_COMPARE_KINDS",
    "X86_LOGICAL_KINDS", "X86_TRANSCENDENTAL_KINDS", "X86_WHERE_KINDS",
    "emit_attention_tile_ir", "emit_cohort2_tile_ir", "emit_elementwise_tile_ir", "emit_matmul_tile_ir", "emit_reduce_tile_ir",
    "emit_softmax_tile_ir", "package_attention", "package_matmul",
    "package_cohort2", "package_elementwise", "package_reduction", "package_softmax", "requests_attention",
    "requests_cohort2",
    "requests_matmul", "requests_reduction", "requests_softmax",
    "supports_attention", "supports_cohort2", "supports_elementwise", "supports_promoted_elementwise",
    "supports_matmul", "supports_promoted_matmul", "supports_reduction",
    "supports_softmax", "tools_available",
]
