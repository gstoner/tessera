"""Native x86 elementwise / cohort-2 / breadth contracts (E2E-REAL-6, 2026-09-28).

One isolated static x86 Graph operation lowers through the native compiler --
``--tessera-graph-to-schedule`` records a content-addressed contract
(``src/compiler/programming_model/lib/NativeX86Kernel.h``, or
``NativeAbsolute.h`` for ``absolute``/``floor``/``ceil``/``cumsum``) and
``--tessera-schedule-to-tile`` emits the launch envelope the stable x86 C ABI
consumes. The package is projected from the *replayed* Tile IR contract, never
from the Python Graph object.

This module holds three things:

* :func:`admit` -- the host-free admission the scheduled contract uses
  (``scheduled_kernel.supports_scheduled_kernel(target="x86")``). It mirrors
  what the native contract accepts and normalizes only spelling: catalog
  aliases to the ODS op name, a missing axis to its explicit default. Every
  semantic keyword the ABI does not implement is refused (Decision #21a).
* :func:`lower` -- Graph -> Schedule -> Tile through ``tessera-opt``.
* :func:`project` -- replay both boundaries and parse the serialized contract.

The retired Graph-owned constructors are the declared differential oracle in
``tests/_support/x86_kernel_baseline.py`` (Decision #31(a)).
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .graph_ir import GraphIRModule

#: Spellings the retired x86 tables accepted that are not ODS op names.
X86_KERNEL_GRAPH_ALIASES: dict[str, str] = {
    "tessera.abs": "tessera.absolute",
    "tessera.subtract": "tessera.sub",
    "tessera.divide": "tessera.div",
    "tessera.multiply": "tessera.mul",
    "tessera.floor_divide": "tessera.floor_div",
    "tessera.equal": "tessera.eq",
    "tessera.not_equal": "tessera.ne",
    "tessera.less": "tessera.lt",
    "tessera.less_equal": "tessera.le",
    "tessera.greater": "tessera.gt",
    "tessera.greater_equal": "tessera.ge",
    "tessera.power": "tessera.pow",
    "tessera.swiglu": "tessera.silu_mul",
    "tessera.mse_loss": "tessera.loss.mse",
    "tessera.mae_loss": "tessera.loss.mae",
    "tessera.huber_loss": "tessera.loss.huber",
    "tessera.smooth_l1_loss": "tessera.loss.smooth_l1",
    "tessera.log_cosh_loss": "tessera.loss.log_cosh",
}


def _elementwise(subfamily: str, kinds: dict[str, str]) -> dict[str, tuple[str, str, str]]:
    return {op: ("elementwise", subfamily, kind) for op, kind in kinds.items()}


#: ODS op name -> (family, subfamily, kind). Mirrors ``kX86KernelSpecs``.
X86_KERNEL_OPS: dict[str, tuple[str, str, str]] = {
    **_elementwise("unary", {
        "tessera.sqrt": "sqrt", "tessera.rsqrt": "rsqrt",
        "tessera.reciprocal": "reciprocal", "tessera.sign": "sign",
        "tessera.round": "round",
    }),
    **_elementwise("binary", {
        "tessera.sub": "sub", "tessera.div": "div", "tessera.maximum": "maximum",
        "tessera.minimum": "minimum", "tessera.add": "add", "tessera.mul": "mul",
        "tessera.mod": "mod", "tessera.floor_div": "floor_div",
    }),
    **_elementwise("predicate", {
        "tessera.isnan": "isnan", "tessera.isinf": "isinf",
        "tessera.isfinite": "isfinite",
    }),
    **_elementwise("compare", {
        "tessera.eq": "eq", "tessera.ne": "ne", "tessera.lt": "lt",
        "tessera.le": "le", "tessera.gt": "gt", "tessera.ge": "ge",
    }),
    **_elementwise("logical", {
        "tessera.logical_and": "and", "tessera.logical_or": "or",
        "tessera.logical_xor": "xor", "tessera.logical_not": "not",
    }),
    **_elementwise("bitwise", {
        "tessera.bitwise_and": "and", "tessera.bitwise_or": "or",
        "tessera.bitwise_xor": "xor", "tessera.bitwise_not": "not",
        "tessera.popcount": "popcount",
    }),
    **_elementwise("where", {"tessera.where": "where"}),
    **_elementwise("transcendental", {
        name: name.removeprefix("tessera.") for name in (
            "tessera.exp", "tessera.log", "tessera.tanh", "tessera.sigmoid",
            "tessera.silu", "tessera.gelu", "tessera.erf", "tessera.softplus",
            "tessera.expm1", "tessera.log1p", "tessera.cos", "tessera.tan",
            "tessera.sinh", "tessera.cosh", "tessera.asin", "tessera.acos",
            "tessera.atan", "tessera.erfc", "tessera.sin", "tessera.lgamma",
            "tessera.digamma",
        )
    }),
    **_elementwise("binary_math", {"tessera.pow": "pow", "tessera.silu_mul": "silu_mul"}),
    "tessera.argmax": ("argreduce", "argreduce", "argmax"),
    "tessera.argmin": ("argreduce", "argreduce", "argmin"),
    "tessera.cumprod": ("scan", "scan", "product"),
    "tessera.cummax": ("scan", "scan", "max"),
    "tessera.cummin": ("scan", "scan", "min"),
    "tessera.rope": ("rope", "rope", "rope"),
    "tessera.gather": ("x86_abi", "gather_f32", "gather"),
    "tessera.loss.mse": ("x86_abi", "pointwise_loss_f32", "pointwise_loss"),
    "tessera.loss.mae": ("x86_abi", "pointwise_loss_f32", "pointwise_loss"),
    "tessera.loss.huber": ("x86_abi", "pointwise_loss_f32", "pointwise_loss"),
    "tessera.loss.smooth_l1": ("x86_abi", "pointwise_loss_f32", "pointwise_loss"),
    "tessera.loss.log_cosh": ("x86_abi", "pointwise_loss_f32", "pointwise_loss"),
    "tessera.cholesky": ("x86_abi", "cholesky_f32", "cholesky"),
    "tessera.tri_solve": ("x86_abi", "tri_solve_f32", "tri_solve"),
}

#: ODS ops the older NativeAbsolute contract owns: op -> (family, subfamily,
#: kind, serialized contract attribute stem).
X86_ABSOLUTE_OPS: dict[str, tuple[str, str, str, str]] = {
    "tessera.absolute": ("elementwise", "unary", "abs", "absolute"),
    "tessera.floor": ("elementwise", "unary", "floor", "floor"),
    "tessera.ceil": ("elementwise", "unary", "ceil", "ceil"),
    "tessera.trunc": ("elementwise", "unary", "trunc", "trunc"),
    "tessera.cumsum": ("scan", "scan", "sum", "cumsum"),
}

#: Tile launch op per artifact family.
TILE_OPS: dict[str, str] = {
    "elementwise": "tile.elementwise_kernel",
    "argreduce": "tile.argreduce_kernel",
    "scan": "tile.scan_kernel",
    "rope": "tile.rope_kernel",
    "x86_abi": "tile.x86_abi_kernel",
}

_STORAGE = {"fp32": "f32", "bool": "i8", "int32": "i32", "int64": "i64"}
_ELEMENTWISE_STORAGE = {
    "logical": ("bool", "bool"), "bitwise": ("int32", "int32"),
    "predicate": ("fp32", "bool"), "compare": ("fp32", "bool"),
}
_LOSS_PARAMETER = {"tessera.loss.huber": "delta", "tessera.loss.smooth_l1": "beta"}


def canonical_op_name(name: str) -> str:
    return X86_KERNEL_GRAPH_ALIASES.get(name, name)


def owns(module: GraphIRModule) -> bool:
    """Whether *module* names an op this contract (or NativeAbsolute) owns."""
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        return False
    name = canonical_op_name(module.functions[0].body[0].op_name)
    return name in X86_KERNEL_OPS or name in X86_ABSOLUTE_OPS


@dataclass(frozen=True)
class X86KernelRequest:
    """The normalized Graph request one x86 native kernel is lowered from."""

    graph_op: str
    family: str
    subfamily: str
    kind: str
    record: str
    bindings: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)


def _static(shape: Any, *, allow_scalar: bool = False) -> tuple[int, ...]:
    try:
        dims = tuple(int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("x86 native kernel requires static shapes") from exc
    if (not dims and not allow_scalar) or any(value <= 0 for value in dims):
        raise ValueError("x86 native kernel requires positive static extents")
    return dims


def _bool(kwargs: dict[str, Any], name: str, default: bool) -> bool:
    value = kwargs.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"x86 native kernel {name} must be a boolean")
    return value


def _int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"x86 native kernel {name} must be an integer")
    return value


def admit(module: GraphIRModule) -> X86KernelRequest:
    """Admit one x86 native kernel request or raise ``ValueError``."""
    if len(module.functions) != 1:
        raise ValueError("x86 native kernel requires one Graph function")
    function = module.functions[0]
    if len(function.body) != 1 or len(function.result_types) != 1:
        raise ValueError("x86 native kernel requires one Graph operation and result")
    op = function.body[0]
    graph_op = canonical_op_name(op.op_name)
    if graph_op in X86_KERNEL_OPS:
        family, subfamily, kind = X86_KERNEL_OPS[graph_op]
        record = "x86_kernel"
    elif graph_op in X86_ABSOLUTE_OPS:
        family, subfamily, kind, record = X86_ABSOLUTE_OPS[graph_op]
    else:
        raise ValueError(f"{op.op_name} has no x86 native kernel contract")
    args = {arg.name: arg for arg in function.args}
    inputs = tuple(value.removeprefix("%") for value in op.operands)
    if not inputs or any(name not in args for name in inputs):
        raise ValueError("x86 native kernel operands must be function arguments")
    if set(inputs) != set(args):
        raise ValueError("x86 native kernel entry has an unused argument")
    if len(set(inputs)) != len(inputs):
        # A launch descriptor names each buffer once (E_LAUNCH_DESCRIPTOR_SCHEMA);
        # the retired constructor admitted `add(x, x)` and then failed to package.
        raise ValueError("x86 native kernel binds each operand to a distinct argument")
    output = op.result or function.return_values[0].removeprefix("%")
    if not output or output in inputs:
        raise ValueError("x86 native kernel output binding must be distinct")
    shapes = [_static(args[name].ir_type.shape) for name in inputs]
    dtypes = [str(args[name].ir_type.dtype) for name in inputs]
    result = function.result_types[0]
    result_shape = _static(result.shape, allow_scalar=family == "argreduce")
    result_dtype = str(result.dtype)
    # None-valued keywords mean "not given" (the frontend spells an absent
    # policy as None); `axis=None` on argmax/argmin is the flatten request.
    raw = {key: value for key, value in op.kwargs.items()
           if value is not None or (key == "axis" and family in {"argreduce", "scan"})}
    allowed: set[str] = set()
    kwargs: dict[str, Any] = {}

    if family == "elementwise":
        stored_in, stored_out = _ELEMENTWISE_STORAGE.get(subfamily, ("fp32", "fp32"))
        arity = (3 if subfamily == "where" else
                 2 if subfamily in {"binary", "compare", "binary_math"} else
                 (1 if kind in {"not", "popcount"} else 2)
                 if subfamily in {"logical", "bitwise"} else 1)
        expected = (("bool", "fp32", "fp32") if subfamily == "where"
                    else (stored_in,) * arity)
        if len(inputs) != arity or tuple(dtypes) != expected:
            raise ValueError(f"x86 native {subfamily} {kind} requires {arity} {expected} operands")
        if any(shape != result_shape for shape in shapes) or result_dtype != stored_out:
            raise ValueError("x86 native elementwise requires same-shape operands and its result dtype")
        if record != "x86_kernel" and dtypes != ["fp32"]:
            raise ValueError("x86 native absolute/floor/ceil requires f32")
    elif family in {"argreduce", "scan"}:
        if len(inputs) != 1 or dtypes != ["fp32"]:
            raise ValueError("x86 native argreduce/scan requires one f32 operand")
        shape = shapes[0]
        allowed = {"axis"}
        raw_axis = raw.get("axis", -1)
        flatten = raw_axis is None
        if flatten and (family == "scan" and len(shape) != 1):
            raise ValueError("x86 native scan cannot flatten a rank >= 2 operand")
        if not flatten:
            axis = _int(raw_axis, "axis")
            if axis not in {-1, len(shape) - 1}:
                raise ValueError("x86 native argreduce/scan requires the last axis")
            kwargs["axis"] = axis
        elif family == "scan":
            kwargs["axis"] = -1  # rank-1: flattening is the last axis
        if flatten and _bool(raw, "keepdims", False) and len(shape) != 1:
            raise ValueError("x86 native argreduce flatten-with-keepdims keeps every axis (NumPy); "
                             "the ABI result is rank-1")
        logical = (math.prod(shape),) if flatten else shape
        if family == "scan":
            if result_dtype != "fp32" or result_shape != logical:
                raise ValueError("x86 native scan result must match its operand")
        else:
            allowed.add("keepdims")
            keepdims = _bool(raw, "keepdims", False)
            if keepdims:
                kwargs["keepdims"] = True
            expected_shape = logical[:-1] + ((1,) if keepdims else ())
            if result_dtype != "int32" or result_shape != expected_shape:
                raise ValueError("x86 native argreduce result must be int32 with its axis removed")
    elif family == "rope":
        if len(shapes[0]) < 2:
            raise ValueError("x86 native rope requires rank >= 2 (LEGALITY_ROPE_RANK)")
        if (len(inputs) != 2 or dtypes != ["fp32", "fp32"] or shapes[0] != shapes[1]
                or shapes[0] != result_shape or result_dtype != "fp32" or result_shape[-1] % 2):
            raise ValueError("x86 native rope requires same-shape f32 x/theta with an even last dimension")
    else:
        if result_dtype != "fp32":
            raise ValueError("x86 native breadth requires an f32 result")
        if kind == "gather":
            allowed = {"axis"}
            if (len(inputs) != 2 or dtypes != ["fp32", "int64"] or len(shapes[0]) != 1
                    or len(shapes[1]) != 1 or result_shape != shapes[1]):
                raise ValueError("x86 native gather requires a rank-1 f32 source and rank-1 int64 indices")
            if "axis" in raw:
                axis = _int(raw["axis"], "axis")
                if axis not in {0, -1}:
                    raise ValueError("x86 native gather requires axis 0")
                kwargs["axis"] = axis
        elif kind == "pointwise_loss":
            allowed = {"reduction"}
            if (len(inputs) != 2 or dtypes != ["fp32", "fp32"] or shapes[0] != shapes[1]
                    or result_shape != shapes[0]):
                raise ValueError("x86 native pointwise loss requires same-shape f32 operands")
            if raw.get("reduction", "mean") != "none":
                raise ValueError('x86 native pointwise loss requires reduction = "none"')
            kwargs["reduction"] = "none"
            parameter = _LOSS_PARAMETER.get(graph_op)
            if parameter is not None:
                allowed.add(parameter)
                value = raw.get(parameter, 1.0)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(f"x86 native loss {parameter} must be a number")
                value = float(value)
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"x86 native loss {parameter} must be positive finite")
                if parameter in raw:
                    kwargs[parameter] = value
        else:
            cholesky = kind == "cholesky"
            allowed = {"lower"} if cholesky else {"lower", "trans", "unit_diag"}
            if len(inputs) != (1 if cholesky else 2) or any(d != "fp32" for d in dtypes):
                raise ValueError("x86 native linalg requires f32 operands")
            matrix = shapes[0]
            # The Graph ODS cholesky/tri_solve are rank-2 ("batched rank-3 is a
            # follow-on", TesseraOps.cpp; pinned by
            # apple_cholesky_graph_ir_invalid.mlir). The batched envelope stays
            # on x86_breadth's retained constructor.
            if len(matrix) != 2 or matrix[-1] != matrix[-2]:
                raise ValueError("x86 native linalg requires a square rank-2 matrix")
            lower = _bool(raw, "lower", True)
            if "lower" in raw:
                kwargs["lower"] = lower
            if cholesky:
                if not lower or result_shape != matrix:
                    raise ValueError("x86 native cholesky implements lower = True only")
            else:
                for flag in ("trans", "unit_diag"):
                    if _bool(raw, flag, False):
                        raise ValueError("x86 native triangular solve implements trans/unit_diag = False only")
                rhs = shapes[1]
                n = matrix[-1]
                valid = len(rhs) == 2 and rhs[0] == n
                if not valid or result_shape != rhs:
                    raise ValueError("x86 native triangular solve rhs/result disagree with the matrix")
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"x86 native kernel has unsupported keyword(s) {unknown}")
    return X86KernelRequest(
        graph_op=graph_op, family=family, subfamily=subfamily, kind=kind, record=record,
        bindings=inputs + (output,), shapes=tuple(shapes) + (result_shape,),
        dtypes=tuple(dtypes) + (result_dtype,), kwargs=kwargs,
    )


def lower(module: GraphIRModule, request: X86KernelRequest, tool: Any) -> tuple[str, str, str]:
    """Graph -> Schedule -> Tile for one admitted request: (graph, schedule, tile)."""
    from .x86_compile_cache import run as run_tessera_opt

    targeted = copy.deepcopy(module)
    op = targeted.functions[0].body[0]
    op.op_name = request.graph_op
    op.kwargs = dict(request.kwargs)
    targeted.module_attrs["tessera.target"] = '"x86"'
    targeted.module_attrs["tessera.arch"] = '"zen5-avx512"'
    targeted.module_attrs["tessera.launch_bindings"] = json.dumps(list(request.bindings))
    graph_ir = targeted.to_mlir(target="x86", canonical=True)
    schedule_ir = run_tessera_opt(tool, graph_ir, "--tessera-graph-to-schedule")
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    return graph_ir, schedule_ir, tile_ir


# ---------------------------------------------------------------------------
# Projection: read the serialized contract back out of native-printed IR.
# ---------------------------------------------------------------------------

class _AttrParser:
    """Parse the printed MLIR attribute subset the contracts use."""

    _TOKEN = re.compile(
        r'\s*(?:(?P<str>"(?:[^"\\]|\\.)*")|(?P<arr>array<i64(?::\s*(?P<dims>[-0-9, ]*))?>)'
        r'|(?P<num>[-+]?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?|0x[0-9A-Fa-f]+))'
        r'(?:\s*:\s*(?P<ty>[if]\d+))?|(?P<bool>true|false)|(?P<key>[A-Za-z_][\w.]*)'
        r'|(?P<punct>[{}\[\],=]))'
    )

    def __init__(self, text: str) -> None:
        self.text, self.pos = text, 0

    def _next(self) -> re.Match[str]:
        match = self._TOKEN.match(self.text, self.pos)
        if match is None:
            raise ValueError(f"x86 kernel contract is malformed near {self.text[self.pos:self.pos + 40]!r}")
        self.pos = match.end()
        return match

    def _peek(self) -> re.Match[str]:
        saved = self.pos
        try:
            return self._next()
        finally:
            self.pos = saved

    def value(self) -> Any:
        token = self._next()
        if token["str"] is not None:
            return json.loads(token["str"])
        if token["arr"] is not None:
            dims = token["dims"] or ""
            return tuple(int(d) for d in dims.split(",") if d.strip())
        if token["num"] is not None:
            text, ty = token["num"], token["ty"] or ""
            if ty.startswith("f") or ("." in text or "e" in text.lower()) and not text.startswith("0x"):
                if text.startswith("0x"):
                    raise ValueError("x86 kernel contract hex floats are unsupported")
                return float(text)
            return int(text, 0)
        if token["bool"] is not None:
            return token["bool"] == "true"
        punct = token["punct"]
        if punct == "[":
            items: list[Any] = []
            if self._peek()["punct"] == "]":
                self._next()
                return items
            while True:
                items.append(self.value())
                close = self._next()["punct"]
                if close == "]":
                    return items
                if close != ",":
                    raise ValueError("x86 kernel contract array is malformed")
        if punct == "{":
            return self.mapping()
        raise ValueError("x86 kernel contract value is malformed")

    def mapping(self) -> dict[str, Any]:
        entries: dict[str, Any] = {}
        if self._peek()["punct"] == "}":
            self._next()
            return entries
        while True:
            key = self._next()["key"]
            if key is None or self._next()["punct"] != "=":
                raise ValueError("x86 kernel contract entry is malformed")
            if key in entries:
                raise ValueError(f"x86 kernel contract repeats {key!r}")
            entries[key] = self.value()
            close = self._next()["punct"]
            if close == "}":
                return entries
            if close != ",":
                raise ValueError("x86 kernel contract dictionary is malformed")


def parse_contract(tile_ir: str, record: str) -> dict[str, Any]:
    """The one ``tessera.<record>_contract`` dictionary in *tile_ir*."""
    marker = f"tessera.{record}_contract = {{"
    if tile_ir.count(marker) != 1:
        raise ValueError(f"x86 kernel Tile artifact requires one tessera.{record}_contract")
    parser = _AttrParser(tile_ir)
    parser.pos = tile_ir.index(marker) + len(marker)
    return parser.mapping()


@dataclass(frozen=True)
class X86KernelProjection:
    family: str
    subfamily: str
    kind: str
    record: str
    bindings: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    storage: tuple[str, ...]
    scalars: dict[str, Any]
    extras: dict[str, Any]
    numeric_policy: str
    schedule_digest: str


def project(artifact: Any) -> X86KernelProjection:
    """Replay both native boundaries and project the serialized contract."""
    from .scheduled_matmul import find_tessera_opt
    from .x86_compile_cache import run as run_tessera_opt

    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("x86 native kernel packaging requires native Schedule replay")
    if run_tessera_opt(tool, artifact.graph_ir, "--tessera-graph-to-schedule") != artifact.schedule_ir:
        raise ValueError("x86 native kernel Schedule disagrees with Graph replay")
    if run_tessera_opt(tool, artifact.schedule_ir, "--tessera-schedule-to-tile") != artifact.tile_ir:
        raise ValueError("x86 native kernel Tile disagrees with Schedule replay")
    header = re.match(r'\s*module attributes \{([^{}]*)\}', artifact.tile_ir)
    if header is None or any(
        re.search(r'(?:^|,)\s*' + re.escape(key) + r' = "' + value + r'"(?:,|$)', header[1]) is None
        for key, value in (("tessera.target", "x86"), ("tessera.arch", "zen5-avx512"))
    ):
        raise ValueError("x86 native kernel parent target disagrees")
    record = artifact.record
    contract = parse_contract(artifact.tile_ir, record)
    digests = re.findall(r'tessera\.schedule_hash = "([0-9a-f]{64})"', artifact.tile_ir)
    if digests != [artifact.schedule_digest]:
        raise ValueError("x86 native kernel Tile artifact has a stale schedule digest")
    bindings = contract.get("bindings")
    if (not isinstance(bindings, list) or len(bindings) < 2
            or any(type(name) is not str or not name for name in bindings)
            or bindings[-1] in bindings[:-1]):
        raise ValueError("x86 native kernel bindings must be names with a distinct output")
    if record == "x86_kernel":
        family, kind = contract.get("family"), contract.get("kind")
        subfamily = (contract.get("elementwise_family") if family == "elementwise"
                     else contract.get("abi_key") if family == "abi" else family)
        family = "x86_abi" if family == "abi" else family
        shapes = tuple(tuple(shape) for shape in contract.get("shapes", ()))
        storage = tuple(contract.get("storage", ()))
        scalars = contract.get("scalars", {})
        policy = "c_abi_" + str(subfamily)
        extras = {key: value for key, value in contract.items() if key not in {
            "family", "kind", "graph_op", "bindings", "shapes", "storage", "layout",
            "scalars", "elementwise_family", "abi_key"}}
        if contract.get("layout") != "row_major":
            raise ValueError("x86 native kernel requires row_major layout")
    else:
        family, subfamily, kind, _ = next(
            spec for spec in X86_ABSOLUTE_OPS.values() if spec[3] == record)
        shape = contract.get("shape")
        if not isinstance(shape, tuple):
            raise ValueError("x86 native absolute contract is missing its shape")
        shapes = (shape, shape)
        storage = ("f32", "f32")
        if contract.get("storage") != "f32" or contract.get("layout") != "row_major":
            raise ValueError("x86 native absolute contract has an unsupported policy")
        policy = str(contract.get("numeric_policy", ""))
        if contract.get("kind") != kind:
            raise ValueError("x86 native absolute contract kind disagrees")
        scalars = ({"N": math.prod(shape)} if family == "elementwise"
                   else {"Rows": math.prod(shape[:-1]), "Cols": shape[-1]})
        extras = {"inclusive": True} if family == "scan" else {}
    if (len(shapes) != len(bindings) or len(storage) != len(bindings)
            or not isinstance(scalars, dict)):
        raise ValueError("x86 native kernel contract bindings/shapes/storage disagree")
    if family not in TILE_OPS or artifact.family != family or artifact.kind != kind:
        raise ValueError("x86 native kernel artifact family/kind disagrees with its contract")
    tile_ops = re.findall(r"(?m)^\s*(tile\.[A-Za-z_0-9]+)", artifact.tile_ir)
    if tile_ops != [TILE_OPS[family]]:
        raise ValueError("x86 native kernel Tile artifact requires exactly its one launch op")
    if (tuple(bindings) != (*artifact.input_names, artifact.output_name)
            or shapes[0] != artifact.input_shape or shapes[-1] != artifact.output_shape):
        raise ValueError("x86 native kernel descriptor fields disagree with native IR")
    return X86KernelProjection(
        family=str(family), subfamily=str(subfamily), kind=str(kind), record=record,
        bindings=tuple(bindings), shapes=shapes, storage=storage, scalars=dict(scalars),
        extras=extras, numeric_policy=policy, schedule_digest=artifact.schedule_digest,
    )


__all__ = [
    "TILE_OPS", "X86_ABSOLUTE_OPS", "X86_KERNEL_GRAPH_ALIASES", "X86_KERNEL_OPS",
    "X86KernelProjection", "X86KernelRequest", "admit", "canonical_op_name", "lower",
    "owns", "parse_contract", "project",
]
