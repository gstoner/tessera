"""
tessera.compiler.graph_ir — Python → structured Graph IR lowering.

The canonical compiler product is a verified :class:`GraphIRModule` object.
MLIR text is an inspection and interop serialization produced from that object,
not the source of truth. Lightweight environments do not need MLIR Python
bindings; when bindings are installed, ``construct_mlir_module`` can parse the
serialized text as an optional native validation step.

Inspection format (MLIR text):
    func.func @step(%W: tensor<*xbf16> {tessera.effect = "read"},
                    %X: tensor<*xbf16> {tessera.effect = "read"},
                    %Y: tensor<*xbf16> {tessera.effect = "write"}) {
      %0 = tessera.matmul(%X, %W) : (tensor<*xbf16>, tensor<*xbf16>) -> tensor<*xbf16>
      tessera.copy %0, %Y : tensor<*xbf16>
      return
    }

Reference: CLAUDE.md §Four-Layer IR Stack — Graph IR
           src/ir/TesseraOps.td
"""

from __future__ import annotations
import ast
import inspect
import json
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..diagnostics import DiagnosticLevel, DiagnosticWhere, SourceLocation, TesseraDiagnostic, TesseraErrorCode
from .legality import TensorContract, check_op_legality
from .op_catalog import GRAPH_OP_MAP, graph_name_for


# ─────────────────────────────────────────────────────────────────────────────
# IR value types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IRType:
    """A simplified type representation for Graph IR emission."""
    mlir_str: str   # e.g. "tensor<*xbf16>", "f32", "index"
    shape: Tuple[str, ...] = ()
    dtype: Optional[str] = None
    layout: Optional[str] = None

    def __str__(self) -> str:
        return self.mlir_str

    @property
    def rank(self) -> Optional[int]:
        return len(self.shape) if self.shape and "*" not in self.shape else None


def _normalize_dtype(dtype: Optional[str]) -> Optional[str]:
    if dtype is None:
        return None
    aliases = {
        "f64": "fp64",
        "f32": "fp32",
        "f16": "fp16",
        "float64": "fp64",
        "float32": "fp32",
        "float16": "fp16",
    }
    return aliases.get(str(dtype), str(dtype))


def _mlir_dtype(dtype: Optional[str]) -> str:
    dtype = _normalize_dtype(dtype)
    mapping = {
        "fp64": "f64",
        "fp32": "f32",
        "fp16": "f16",
        "bf16": "bf16",
        "fp8_e4m3": "xf8E4M3FN",
        "fp8_e5m2": "xf8E5M2",
        "fp6_e2m3": "!tessera.fp6_e2m3",
        "fp6_e3m2": "!tessera.fp6_e3m2",
        "fp4_e2m1": "!tessera.fp4_e2m1",
        "nvfp4": "!tessera.nvfp4",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "bool": "i1",
    }
    return mapping.get(dtype or "", dtype or "?")


def tensor_ir_type(
    shape: Tuple[Any, ...] = ("*",),
    dtype: Optional[str] = None,
    *,
    layout: Optional[str] = None,
) -> IRType:
    """Create a tensor IR type while preserving source shape/dtype/layout metadata."""

    normalized_shape = tuple("?" if dim is None else str(dim) for dim in shape)
    normalized_dtype = _normalize_dtype(dtype)
    shape_text = "*"
    if normalized_shape and normalized_shape != ("*",):
        shape_text = "x".join(normalized_shape)
    dtype_text = _mlir_dtype(normalized_dtype) if normalized_dtype else "?"
    return IRType(f"tensor<{shape_text}x{dtype_text}>", normalized_shape, normalized_dtype, layout)


@dataclass(frozen=True)
class NumericPolicy:
    """Canonical numerics contract carried through Graph/Schedule/Tile IR."""

    storage: str = "bf16"
    accum: str = "f32"
    rounding: str = "nearest_even"
    scale: float = 1.0
    quant_axis: str = "none"
    deterministic: bool = False

    def to_mlir_attr(self) -> str:
        det = "true" if self.deterministic else "false"
        return (
            "#tessera.numeric_policy<"
            f"storage = \"{self.storage}\", accum = \"{self.accum}\", "
            f"rounding = \"{self.rounding}\", scale = {self.scale}, "
            f"quant_axis = \"{self.quant_axis}\", deterministic = {det}>"
        )


@dataclass(frozen=True)
class KVCacheSpec:
    """Graph-level KV cache state object."""

    max_seq: int
    head_dim: int
    dtype_policy: NumericPolicy = field(default_factory=NumericPolicy)
    eviction: str = "rolling_window"
    page_size: int = 256

    def create_attrs(self) -> str:
        return (
            f"max_seq = {self.max_seq}, head_dim = {self.head_dim}, "
            f"eviction = \"{self.eviction}\", page_size = {self.page_size}, "
            f"numeric_policy = {self.dtype_policy.to_mlir_attr()}"
        )


@dataclass(frozen=True)
class GraphIRMesh:
    """Structured module-level mesh declaration from the Tessera DSL."""

    name: str
    axes: Tuple[str, ...] = ()
    shape: Tuple[int, ...] = ()
    attrs: Dict[str, Any] = field(default_factory=dict)
    source_span: Optional["SourceSpan"] = None

    @classmethod
    def from_attrs(
        cls,
        name: str,
        attrs: Dict[str, Any],
        *,
        source_span: Optional["SourceSpan"] = None,
    ) -> "GraphIRMesh":
        axes = tuple(str(axis) for axis in attrs.get("axes", ()))
        shape = tuple(int(dim) for dim in attrs.get("shape", ()))
        extra = {key: value for key, value in attrs.items() if key not in {"axes", "shape"}}
        return cls(name=name, axes=axes, shape=shape, attrs=extra, source_span=source_span)

    def to_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = dict(self.attrs)
        if self.axes:
            metadata["axes"] = list(self.axes)
        if self.shape:
            metadata["shape"] = list(self.shape)
        return metadata


@dataclass(frozen=True)
class GraphIRTypeAlias:
    """Structured module-level type alias preserved by the frontend."""

    name: str
    type_expr: str
    source_span: Optional["SourceSpan"] = None


@dataclass(frozen=True)
class GraphIRConstant:
    """Structured module-level constant declaration preserved by the frontend."""

    name: str
    type_expr: str
    value: Any
    source_span: Optional["SourceSpan"] = None

    def to_metadata(self) -> Dict[str, Any]:
        return {"type": self.type_expr, "value": self.value}


# Common types used in Phase 1
TENSOR_BF16   = tensor_ir_type(("*",), "bf16")
TENSOR_FP16   = tensor_ir_type(("*",), "fp16")
TENSOR_FP32   = tensor_ir_type(("*",), "fp32")
TENSOR_OPAQUE = tensor_ir_type(("*",), None)   # unknown dtype
KV_CACHE      = IRType("!tessera.kv_cache")
INDEX         = IRType("index")
BOOL          = IRType("i1")


@dataclass(frozen=True)
class SourceSpan:
    """Source location for frontend diagnostics."""

    line: int
    col: int
    end_line: Optional[int] = None
    end_col: Optional[int] = None
    source_name: Optional[str] = None

    def format(self) -> str:
        origin = f"{self.source_name}:" if self.source_name else ""
        end = ""
        if self.end_line is not None and self.end_col is not None:
            end = f"-{self.end_line}:{self.end_col}"
        return f"{origin}{self.line}:{self.col}{end}"


@dataclass(frozen=True)
class GraphIRDiagnostic:
    severity: str
    message: str
    span: Optional[SourceSpan] = None
    code: str = "GRAPH_IR"

    def format(self) -> str:
        structured = self.to_tessera_diagnostic()
        legacy_code = f" [{self.code}]" if self.code else ""
        loc = f" at {self.span.format()}" if self.span else ""
        return f"{structured.level.value.upper()} [{structured.code.value}]{legacy_code}{loc}: {self.message}\n  where: {structured.where}"

    def to_tessera_diagnostic(self) -> TesseraDiagnostic:
        return TesseraDiagnostic(
            level=_diagnostic_level(self.severity),
            message=self.message,
            location=_source_location(self.span),
            code=_error_code_for_ir_diagnostic(self.code),
            where=DiagnosticWhere(
                ir_level=_ir_level_for_diagnostic(self.code),
                pass_name="verifier",
            ),
            hints=_hints_for_ir_diagnostic(self.code),
        )


@dataclass(frozen=True)
class GraphIRVerificationResult:
    diagnostics: Tuple[GraphIRDiagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)

    def format(self) -> str:
        return "\n".join(d.format() for d in self.diagnostics)

    def structured_diagnostics(self) -> Tuple[TesseraDiagnostic, ...]:
        return tuple(d.to_tessera_diagnostic() for d in self.diagnostics)


class GraphIRVerificationError(ValueError):
    pass


class MLIRObjectUnavailable(RuntimeError):
    pass


def _diagnostic_level(severity: str) -> DiagnosticLevel:
    return {
        "fatal": DiagnosticLevel.FATAL,
        "error": DiagnosticLevel.ERROR,
        "warning": DiagnosticLevel.WARNING,
        "info": DiagnosticLevel.INFO,
        "note": DiagnosticLevel.NOTE,
    }.get(severity.lower(), DiagnosticLevel.ERROR)


def _source_location(span: Optional[SourceSpan]) -> Optional[SourceLocation]:
    if span is None:
        return None
    return SourceLocation(file=span.source_name or "<tessera-ir>", line=span.line, column=span.col)


def _ir_level_for_diagnostic(code: str) -> str:
    if code.startswith("SCHEDULE_IR"):
        return "schedule-ir"
    if code.startswith("TILE_IR"):
        return "tile-ir"
    if code.startswith("TARGET_IR"):
        return "target-ir"
    return "graph-ir"


def _error_code_for_ir_diagnostic(code: str) -> TesseraErrorCode:
    if "SHAPE" in code or "RANK" in code or "TYPE" in code or "DTYPE" in code:
        return TesseraErrorCode.SHAPE_MISMATCH
    if code.startswith("SCHEDULE_IR"):
        return TesseraErrorCode.SCHEDULE_FUSE_FAIL
    if code.startswith("TILE_IR"):
        return TesseraErrorCode.TILE_LOWERING
    if code.startswith("TARGET_IR"):
        return TesseraErrorCode.TARGET_CODEGEN
    return TesseraErrorCode.GRAPH_INVALID


def _hints_for_ir_diagnostic(code: str) -> List[str]:
    if "SHAPE" in code or "RANK" in code:
        return ["dump Graph IR and check tensor shape/layout metadata", "add explicit shape constraints or op.assert guards"]
    if code.startswith("SCHEDULE_IR"):
        return ["inspect Schedule IR artifact and movement/fusion attributes"]
    if code.startswith("TILE_IR"):
        return ["inspect Tile IR async copy, queue, and barrier ordering"]
    if code.startswith("TARGET_IR"):
        return ["inspect Target IR and selected target profile"]
    return ["dump the failing IR level with TESSERA_DEBUG_IR=1"]


def _dtype_to_ir_type(dtype: str) -> IRType:
    """Map a Tessera dtype string to an MLIR type."""
    return tensor_ir_type(("*",), dtype)


# ─────────────────────────────────────────────────────────────────────────────
# IR argument descriptor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IRArg:
    """Represents a function argument in the emitted Graph IR."""
    name: str
    ir_type: IRType
    effect: Optional[str] = None        # "read", "write", "reduce_sum", etc.
    shard_spec: Optional[str] = None    # tessera.shard attribute (Phase 2+)
    dim_names: Tuple[str, ...] = ()
    layout: Optional[str] = None

    def to_mlir(self) -> str:
        attrs = []
        if self.effect:
            attrs.append(f'tessera.effect = "{self.effect}"')
        if self.shard_spec:
            attrs.append(f"tessera.shard = {self.shard_spec}")
        if self.dim_names:
            dims = "[" + ", ".join(f'"{name}"' for name in self.dim_names) + "]"
            attrs.append(f"tessera.dim_names = {dims}")
        layout = self.layout or self.ir_type.layout
        if layout:
            attrs.append(f'tessera.layout = "{layout}"')
        attr_str = (" {" + ", ".join(attrs) + "}") if attrs else ""
        return f"%{self.name}: {self.ir_type}{attr_str}"


# ─────────────────────────────────────────────────────────────────────────────
# Graph IR op emission helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IROp:
    """A single emitted op in the Graph IR function body.

    Optional metadata fields (G3, 2026-05-19) — populated by the
    ``propagate_numeric_policy`` pass and consumed by lowering /
    audit / ``.explain()``.  Default ``None`` keeps existing
    producers backward-compatible: an op with no policy attached is
    treated as having an implicit policy matching its
    ``result_type`` storage.

    Attributes
    ----------
    result, op_name, operands, operand_types, result_type, attrs,
    kwargs, source_span, inferred_type:
        Core op fields.
    numeric_policy:
        Optional :class:`tessera.compiler.primitive_coverage.NumericPolicy`
        record.  When populated, downstream passes can read storage /
        accumulator / scale / math_mode without re-deriving from op
        name.  See :func:`tessera.compiler.numeric_policy_pass.propagate_numeric_policy`.
    """
    result: Optional[str]     # None for void ops (e.g. copy/store)
    op_name: str              # e.g. "tessera.matmul"
    operands: List[str]       # e.g. ["%X", "%W"]
    operand_types: List[str]  # e.g. ["tensor<*xbf16>", "tensor<*xbf16>"]
    result_type: Optional[str] = None
    attrs: Optional[str] = None   # additional op attributes
    kwargs: Dict[str, Any] = field(default_factory=dict)
    source_span: Optional["SourceSpan"] = None
    inferred_type: Optional[IRType] = None
    numeric_policy: Optional[Any] = None  # NumericPolicy when populated
    # Phase A (2026-05-20) — optional metadata fields.  Producers
    # fill what they know; consumers must tolerate ``None`` /
    # ``frozenset()`` defaults.  See
    # docs/architecture/frontend_substrate_plan.md for the contract.
    value_kind: Optional[str] = None
    """Optional value-kind tag — ``"tensor"`` / ``"multivector"`` /
    ``"complex"`` / ``"energy"`` / ``"mixed"``.  ``None`` means the
    producer didn't set it (distinct from "definitely tensor")."""
    verification_facts: frozenset[str] = field(default_factory=frozenset)
    """Lane-invariant facts about this op — e.g., ``{"holomorphic"}``
    for ops projected from a ``@complex_jit`` view, ``{"ga_only"}``
    for Clifford views.  Empty when no invariants are claimed."""

    @property
    def result_names(self) -> List[str]:
        """SSA result names for this op (without the ``%`` sigil).

        Single-result ops carry one name in ``result``; multi-result ops
        (``qr`` → ``Q``/``R``, ``svd`` → ``U``/``S``/``V``, ``lu`` →
        ``LU``/``pivots``) carry a comma-separated list (e.g. ``"q,r"``).
        """
        if self.result is None:
            return []
        return [n.strip() for n in self.result.split(",") if n.strip()]

    def to_mlir(self, indent: str = "  ") -> str:
        ops_str = ", ".join(self.operands)
        types_in = ", ".join(self.operand_types)
        names = self.result_names
        if names and self.result_type:
            lhs = ", ".join(f"%{n}" for n in names) + " = "
            type_str = f" : ({types_in}) -> {self.result_type}"
        elif names:
            lhs = ", ".join(f"%{n}" for n in names) + " = "
            type_str = ""
        else:
            lhs = ""
            type_str = f" : {types_in}" if types_in else ""
        attr_parts = []
        if self.attrs:
            attr_parts.append(self.attrs)
        attr_parts.extend(f"{k} = {_format_attr_value(v)}" for k, v in self.kwargs.items())
        if (
            self.op_name == "tessera.rl.ppo_policy_loss"
            and "operandSegmentSizes" not in (self.attrs or "")
            and "operand_segment_sizes" not in (self.attrs or "")
            and "operandSegmentSizes" not in self.kwargs
            and "operand_segment_sizes" not in self.kwargs
        ):
            side_count = max(0, min(3, len(self.operands) - 3))
            segments = [1, 1, 1] + [1 if i < side_count else 0 for i in range(3)]
            attr_parts.append(
                "operandSegmentSizes = array<i32: "
                + ", ".join(str(v) for v in segments)
                + ">"
            )
        attr_str = f" {{{', '.join(attr_parts)}}}" if attr_parts else ""
        return f"{indent}{lhs}{self.op_name}({ops_str}){attr_str}{type_str}"


def _format_attr_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if value is None:
        return "none"
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_format_attr_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return json.dumps(json.dumps(value, sort_keys=True))
    return repr(value)


# ─────────────────────────────────────────────────────────────────────────────
# Graph IR module
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphIRModule:
    """
    A collection of Graph IR function definitions — the top-level unit of
    emission. Corresponds to an MLIR `builtin.module`.

    Attributes:
        functions   : list of emitted function definitions
        module_attrs: top-level module attributes (e.g., tessera.ir.version)
    """
    functions: List["GraphIRFunction"] = field(default_factory=list)
    module_attrs: Dict[str, str] = field(default_factory=lambda: {
        "tessera.ir.version": '"1.0"',
    })
    meshes: List[GraphIRMesh] = field(default_factory=list)
    type_aliases: List[GraphIRTypeAlias] = field(default_factory=list)
    constants: List[GraphIRConstant] = field(default_factory=list)

    def verify(self) -> "GraphIRVerificationResult":
        return GraphIRVerifier().verify_module(self)

    def to_mlir(self, *, verify: bool = True) -> str:
        if verify:
            result = self.verify()
            if not result.ok:
                raise GraphIRVerificationError(result.format())
        attrs = self._emitted_module_attrs()
        attr_lines = ", ".join(f"{k} = {v}" for k, v in attrs.items())
        lines = [f"module attributes {{{attr_lines}}} {{"]
        for fn in self.functions:
            for line in fn.to_mlir().splitlines():
                lines.append("  " + line)
        lines.append("}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"GraphIRModule({len(self.functions)} functions)"

    def _emitted_module_attrs(self) -> Dict[str, str]:
        attrs = dict(self.module_attrs)
        if self.meshes:
            attrs["tessera.meshes"] = _format_attr_value({
                mesh.name: mesh.to_metadata() for mesh in self.meshes
            })
        if self.type_aliases:
            attrs["tessera.type_aliases"] = _format_attr_value({
                alias.name: alias.type_expr for alias in self.type_aliases
            })
        if self.constants:
            attrs["tessera.constants"] = _format_attr_value({
                const.name: const.to_metadata() for const in self.constants
            })
        return attrs


@dataclass
class GraphIRFunction:
    """
    A single `func.func @name(...)` block in the Graph IR.

    Built by GraphIRBuilder when lowering a Python function.

    The optional ``lane`` field records which frontend lane produced
    the function — one of ``"tessera_jit"`` / ``"textual_dsl"`` /
    ``"clifford_jit"`` / ``"complex_jit"`` / ``"energy_jit"``.  This
    lets downstream passes apply lane-specific optimizations without
    re-deriving lane membership (e.g., a ``@complex_jit`` function is
    provably holomorphic, so anti-holomorphic-branch elimination is
    safe).  Default is ``"tessera_jit"`` to match the most common
    decoration path.
    """
    name: str
    args: List[IRArg] = field(default_factory=list)
    result_types: List[IRType] = field(default_factory=list)
    body: List[IROp] = field(default_factory=list)
    fn_attrs: Dict[str, str] = field(default_factory=dict)
    return_values: List[str] = field(default_factory=list)
    lane: str = "tessera_jit"
    # Phase A (2026-05-20) — function-level optional metadata.
    verification_facts: frozenset[str] = field(default_factory=frozenset)
    """Function-level lane invariants — e.g.,
    ``{"holomorphic", "complex_jit_verified"}`` for a function
    projected from a ``@complex_jit`` view.  Empty when no
    invariants are claimed."""
    source_hash: Optional[str] = None
    """Optional SHA-256 of the function's source text.  Useful for
    cache invalidation, drift detection, and reproducibility
    proofs.  ``None`` when the producer doesn't have the source
    (e.g., textual-DSL parse without a backing file)."""

    def to_mlir(self, *, verify: bool = False) -> str:
        if verify:
            result = GraphIRVerifier().verify_function(self)
            if not result.ok:
                raise GraphIRVerificationError(result.format())
        args_str = ", ".join(a.to_mlir() for a in self.args)
        if self.result_types:
            ret_str = " -> (" + ", ".join(str(t) for t in self.result_types) + ")"
        else:
            ret_str = ""
        attr_str = ""
        if self.fn_attrs:
            pairs = ", ".join(f"{k} = {v}" for k, v in self.fn_attrs.items())
            attr_str = f" attributes {{{pairs}}}"
        lines = [f"func.func @{self.name}({args_str}){ret_str}{attr_str} {{"]
        for op in self.body:
            lines.append(op.to_mlir())
        if self.return_values:
            values = ", ".join(self.return_values)
            types = ", ".join(str(t) for t in self.result_types)
            lines.append(f"  return {values} : {types}")
        else:
            lines.append("  return")
        lines.append("}")
        return "\n".join(lines)


class GraphIRVerifier:
    """Lightweight verifier for the Python Graph IR object model.

    This is intentionally conservative: it catches frontend construction errors
    before MLIR text is emitted. If native MLIR bindings are present, callers can
    use `construct_mlir_module` for a parse-backed object as well.
    """

    def verify_module(self, module: GraphIRModule) -> GraphIRVerificationResult:
        diagnostics: List[GraphIRDiagnostic] = []
        diagnostics.extend(self._verify_named_declarations(
            "mesh",
            [(mesh.name, mesh.source_span) for mesh in module.meshes],
            "GRAPH_IR_DUP_MESH",
        ))
        diagnostics.extend(self._verify_named_declarations(
            "type alias",
            [(alias.name, alias.source_span) for alias in module.type_aliases],
            "GRAPH_IR_DUP_TYPE_ALIAS",
        ))
        diagnostics.extend(self._verify_named_declarations(
            "constant",
            [(const.name, const.source_span) for const in module.constants],
            "GRAPH_IR_DUP_CONSTANT",
        ))
        for mesh in module.meshes:
            if mesh.axes and mesh.shape and len(mesh.axes) != len(mesh.shape):
                diagnostics.append(GraphIRDiagnostic(
                    "error",
                    f"mesh {mesh.name!r} has {len(mesh.axes)} axes but {len(mesh.shape)} shape dimensions",
                    span=mesh.source_span,
                    code="GRAPH_IR_MESH_RANK",
                ))
        seen = set()
        for fn in module.functions:
            if fn.name in seen:
                diagnostics.append(GraphIRDiagnostic("error", f"duplicate function {fn.name!r}", code="GRAPH_IR_DUP_FUNC"))
            seen.add(fn.name)
            diagnostics.extend(self.verify_function(fn).diagnostics)
        return GraphIRVerificationResult(tuple(diagnostics))

    def verify_function(self, fn: GraphIRFunction) -> GraphIRVerificationResult:
        diagnostics: List[GraphIRDiagnostic] = []
        arg_names = set()
        for arg in fn.args:
            if arg.name in arg_names:
                diagnostics.append(GraphIRDiagnostic(
                    "error",
                    f"duplicate argument %{arg.name}",
                    code="GRAPH_IR_DUP_ARG",
                ))
            arg_names.add(arg.name)
        defined = {f"%{arg.name}" for arg in fn.args}
        results = set()
        value_types: Dict[str, IRType] = {f"%{arg.name}": arg.ir_type for arg in fn.args}
        control_stack: List[str] = []
        for op in fn.body:
            if op.op_name.startswith("tessera.scf."):
                self._verify_control_marker(op, control_stack, diagnostics)
            op_result_names = op.result_names
            if op_result_names:
                # Multi-result ops (lu/qr/svd) carry several SSA names; each is
                # a distinct definition for SSA-use checking. The aggregate
                # result_type drives type inference only for the single-result
                # case — per-result types are validated downstream.
                parsed_type = op.inferred_type or _parse_mlir_tensor_type(op.result_type or "")
                for name in op_result_names:
                    result_name = f"%{name}"
                    if result_name in defined or result_name in results:
                        diagnostics.append(GraphIRDiagnostic(
                            "error",
                            f"duplicate SSA result {result_name}",
                            span=op.source_span,
                            code="GRAPH_IR_DUP_VALUE",
                        ))
                    results.add(result_name)
                    if len(op_result_names) == 1:
                        value_types[result_name] = parsed_type
            if len(op.operands) != len(op.operand_types):
                diagnostics.append(GraphIRDiagnostic(
                    "error",
                    f"op {op.op_name!r} has {len(op.operands)} operands but {len(op.operand_types)} operand types",
                    span=op.source_span,
                    code="GRAPH_IR_OPERAND_TYPE_MISMATCH",
                ))
            for operand in op.operands:
                if operand == "%?":
                    diagnostics.append(GraphIRDiagnostic(
                        "error",
                        f"op {op.op_name!r} has unresolved operand",
                        span=op.source_span,
                        code="GRAPH_IR_UNRESOLVED_OPERAND",
                    ))
                elif operand.startswith("%") and operand not in defined and operand not in results:
                    diagnostics.append(GraphIRDiagnostic(
                        "error",
                        f"op {op.op_name!r} uses undefined operand {operand}",
                        span=op.source_span,
                        code="GRAPH_IR_UNDEFINED_OPERAND",
                    ))
            self._verify_op_types(op, value_types, diagnostics)
        if control_stack:
            diagnostics.append(GraphIRDiagnostic(
                "error",
                f"unterminated control region {control_stack[-1]!r}",
                code="GRAPH_IR_CONTROL_UNBALANCED",
            ))
        if fn.result_types:
            if not fn.return_values:
                diagnostics.append(GraphIRDiagnostic(
                    "error",
                    f"function {fn.name!r} declares results but has no return values",
                    code="GRAPH_IR_RETURN_MISSING",
                ))
            elif len(fn.return_values) != len(fn.result_types):
                diagnostics.append(GraphIRDiagnostic(
                    "error",
                    f"function {fn.name!r} returns {len(fn.return_values)} values but declares {len(fn.result_types)} results",
                    code="GRAPH_IR_RETURN_ARITY",
                ))
            for value in fn.return_values:
                if value.startswith("%") and value not in defined and value not in results:
                    diagnostics.append(GraphIRDiagnostic(
                        "error",
                        f"function {fn.name!r} returns undefined value {value}",
                        code="GRAPH_IR_RETURN_UNDEFINED",
                    ))
        return GraphIRVerificationResult(tuple(diagnostics))

    def _verify_named_declarations(
        self,
        noun: str,
        declarations: List[Tuple[str, Optional[SourceSpan]]],
        code: str,
    ) -> List[GraphIRDiagnostic]:
        diagnostics: List[GraphIRDiagnostic] = []
        seen = set()
        for name, span in declarations:
            if name in seen:
                diagnostics.append(GraphIRDiagnostic("error", f"duplicate {noun} {name!r}", span=span, code=code))
            seen.add(name)
        return diagnostics

    def _verify_control_marker(
        self,
        op: IROp,
        stack: List[str],
        diagnostics: List[GraphIRDiagnostic],
    ) -> None:
        suffix = op.op_name.removeprefix("tessera.scf.")
        if suffix.endswith(".begin"):
            stack.append(suffix.removesuffix(".begin"))
            return
        if suffix == "else":
            if not stack or stack[-1] != "if":
                diagnostics.append(GraphIRDiagnostic(
                    "error",
                    "else marker without an open if region",
                    span=op.source_span,
                    code="GRAPH_IR_CONTROL_ELSE",
                ))
            return
        if suffix.endswith(".end"):
            kind = suffix.removesuffix(".end")
            if not stack or stack[-1] != kind:
                diagnostics.append(GraphIRDiagnostic(
                    "error",
                    f"control end for {kind!r} does not match open region",
                    span=op.source_span,
                    code="GRAPH_IR_CONTROL_UNBALANCED",
                ))
                return
            stack.pop()

    def _verify_op_types(
        self,
        op: IROp,
        value_types: Dict[str, IRType],
        diagnostics: List[GraphIRDiagnostic],
    ) -> None:
        operand_contracts = [
            _tensor_contract_for(value_types.get(operand))
            for operand in op.operands
            if value_types.get(operand) is not None
        ]
        legality = check_op_legality(op.op_name, operand_contracts)
        for diag in legality.diagnostics:
            if diag.code in {
                "LEGALITY_COLLECTIVE_EFFECT",
                "LEGALITY_FLASH_ATTN_RANK",
            }:
                continue
            code = "GRAPH_IR_MATMUL_SHAPE" if diag.code == "LEGALITY_MATMUL_K_MISMATCH" else diag.code
            diagnostics.append(GraphIRDiagnostic(
                diag.severity,
                diag.message,
                span=op.source_span,
                code=code,
            ))


def _tensor_contract_for(ir_type: Optional[IRType]) -> TensorContract:
    if ir_type is None:
        return TensorContract()
    return TensorContract(
        shape=tuple(ir_type.shape),
        dtype=ir_type.dtype,
        layout=ir_type.layout,
        memory_space="global",
    )


class MLIRObjectModule:
    """Parse-backed MLIR object wrapper when native bindings are unavailable."""

    def __init__(self, graph_module: GraphIRModule, native_module: Any = None) -> None:
        self.graph_module = graph_module
        self.native_module = native_module

    @property
    def is_native(self) -> bool:
        return self.native_module is not None

    def to_mlir(self) -> str:
        return str(self.native_module) if self.native_module is not None else self.graph_module.to_mlir()


@dataclass
class GraphIRConstructionContext:
    """Own the structured Graph IR construction and serialization boundary.

    Builders mutate the object model through this context. Verification always
    runs before text serialization or optional native MLIR parsing so callers do
    not accidentally treat a string renderer as the compiler source of truth.
    """

    module: GraphIRModule = field(default_factory=GraphIRModule)
    validate_native_mlir: bool = True
    _verified: Optional[GraphIRVerificationResult] = field(default=None, init=False, repr=False)
    serialization_count: int = field(default=0, init=False)

    def add_function(self, fn: GraphIRFunction) -> GraphIRFunction:
        self.module.functions.append(fn)
        self._verified = None
        return fn

    def verify(self) -> GraphIRVerificationResult:
        self._verified = self.module.verify()
        return self._verified

    def require_verified(self) -> GraphIRVerificationResult:
        result = self.verify()
        if not result.ok:
            raise GraphIRVerificationError(result.format())
        return result

    def to_mlir(self) -> str:
        self.require_verified()
        self.serialization_count += 1
        return self.module.to_mlir(verify=False)

    def construct_mlir_module(self) -> MLIRObjectModule:
        self.require_verified()
        text = self.to_mlir()
        if not self.validate_native_mlir:
            return MLIRObjectModule(self.module)
        return _parse_optional_native_mlir(self.module, text)


def _parse_optional_native_mlir(module: GraphIRModule, text: str) -> MLIRObjectModule:
    try:
        from mlir import ir as mlir_ir  # type: ignore
    except Exception:
        return MLIRObjectModule(module)
    with mlir_ir.Context() as context:
        native_module = mlir_ir.Module.parse(text, context)
    return MLIRObjectModule(module, native_module=native_module)


def construct_mlir_module(module: GraphIRModule) -> MLIRObjectModule:
    """Construct a verified MLIR module object.

    The Python package does not require MLIR bindings in its lightweight test
    environment. When bindings are installed, this parses through `mlir.ir`;
    otherwise it returns a verified object wrapper around the Graph IR model.
    """

    return GraphIRConstructionContext(module=module).construct_mlir_module()


# ─────────────────────────────────────────────────────────────────────────────
# AST-based op extractor (Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

class _OpExtractor(ast.NodeVisitor):
    """
    Walks a function AST and extracts tessera.ops calls as IROp instances.

    Phase 1 scope: recognises simple dataflow built from `ops.gemm(A, B)`,
    `tessera.ops.gemm(A, B)`, nested calls, keyword literals, and
    `Y[:] = ops.gemm(A, B)` stores.
    """

    # Map from op bare name to Graph IR op name
    _OP_MAP = GRAPH_OP_MAP

    def __init__(self, arg_names: List[str], arg_types: Optional[Dict[str, IRType]] = None) -> None:
        self.ops: List[IROp] = []
        self.diagnostics: List[GraphIRDiagnostic] = []
        self._counter = 0
        self._arg_names = set(arg_names)
        self._value_types: Dict[str, IRType] = {f"%{name}": typ for name, typ in (arg_types or {}).items()}
        # SSA-renaming for reassigned locals (audit-fix 2026-05-31).
        # The Graph IR verifier rejects ``%c = ... %c = ...``; Python's
        # ``c = ...; c = ...`` (and the desugared ``c += b``) used to
        # produce exactly that pattern. We now mint versioned SSA names
        # (``c`` → ``c__1`` → ``c__2`` ...) on each reassignment and
        # track the *current* SSA name in ``_name_alias`` so subsequent
        # reads of ``c`` resolve to the latest version. Function args
        # seed ``_taken_ssa_names`` so reassigning an arg name (``a = a +
        # 1`` for arg ``a``) immediately versions.
        self._taken_ssa_names: set[str] = set(arg_names)
        self._name_alias: Dict[str, str] = {name: name for name in arg_names}

    def _fresh(self) -> str:
        name = f"v{self._counter}"
        self._counter += 1
        return name

    def _reserve_ssa_for_assign(self, user_name: str) -> str:
        """Reserve (but do not yet expose) a unique SSA name for an
        upcoming assignment to ``user_name``.

        Critical ordering: ``c = c + 1`` must see the OLD ``c`` when
        evaluating ``c + 1`` and only adopt the new SSA name *after*
        the op emits. So this function reserves the new SSA name in
        ``_taken_ssa_names`` (to avoid collisions) but does NOT update
        ``_name_alias`` — the caller in ``visit_Assign`` updates the
        alias on emission success.

        First assignment to a never-bound name returns the name itself.
        Reassignment mints ``user_name__1``, ``user_name__2``, ....
        """
        if user_name not in self._taken_ssa_names:
            self._taken_ssa_names.add(user_name)
            return user_name
        n = 1
        while f"{user_name}__{n}" in self._taken_ssa_names:
            n += 1
        ssa = f"{user_name}__{n}"
        self._taken_ssa_names.add(ssa)
        return ssa

    def _resolve_ssa_name(self, user_name: str) -> str:
        """Look up the current SSA name for ``user_name``. Returns the
        user name itself if no alias has been recorded (so unbound names
        still produce ``%<name>``, which downstream sees as a symbolic
        free-variable reference)."""
        return self._name_alias.get(user_name, user_name)

    def _resolve_name(self, node: ast.expr) -> Optional[str]:
        if isinstance(node, ast.Name):
            return self._resolve_ssa_name(node.id)
        if isinstance(node, ast.Attribute):
            parent = self._resolve_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return None

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handle `Y[:] = expr` or `result = expr`."""
        if not node.targets:
            return
        tgt = node.targets[0]
        if isinstance(tgt, ast.Name):
            # SSA-renaming (audit-fix 2026-05-31): reserve a unique SSA
            # name for this assignment BEFORE evaluating the RHS, but
            # only adopt it as the current alias *after* the RHS emits.
            # ``c = c + 1`` must see the OLD ``c`` while evaluating the
            # right-hand side; only afterward does ``c`` resolve to the
            # new SSA.
            ssa_name = self._reserve_ssa_for_assign(tgt.id)
            if self._emit_expr(node.value, result_name=ssa_name) is not None:
                # Adopt the new SSA name as the current alias.
                self._name_alias[tgt.id] = ssa_name
                return
            # ``_emit_expr`` returned None — release the reserved name
            # so a later assignment can reuse it.
            self._taken_ssa_names.discard(ssa_name)
            # Alias-by-rhs-name shortcut: ``c = a`` (no op needed —
            # ``c`` becomes an alias for the current SSA of ``a``).
            if isinstance(node.value, ast.Name):
                rhs_ssa = self._resolve_ssa_name(node.value.id)
                self._name_alias[tgt.id] = rhs_ssa
                # ``c`` is now an alias for ``a``'s current SSA name.
                # Don't bind ``c`` itself as a separate SSA (it'd
                # collide with itself on a future reassignment); the
                # alias map suffices for reads.
                return
        elif isinstance(tgt, ast.Subscript):
            value = self._emit_expr(node.value)
            if value is not None:
                dest = self._resolve_name(tgt.value) or "?"
                self.ops.append(IROp(
                    result=None,
                    op_name="tessera.copy",
                    operands=[value, f"%{dest}"],
                    operand_types=[
                        str(self._value_types.get(value, TENSOR_OPAQUE)),
                        str(self._value_types.get(f"%{dest}", TENSOR_OPAQUE)),
                    ],
                    source_span=_span_from_ast(node),
                ))
                return
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value and self._emit_expr(node.value) is not None:
            return
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        """Lower Python `if/else` to `tessera.scf.if.*` markers.

        Three cases:

        1. **Static condition** — ``ast.literal_eval`` resolves the test
           to a bool. The markers carry ``condition=True/False``.
        2. **Dynamic condition, lowerable** — ``_emit_expr`` produces an
           SSA value for the test expression. The markers carry that
           value as a single operand AND ``kind="dynamic"``.
        3. **Dynamic condition, not lowerable** (the audit gap before
           D.1) — the test is a comparison / boolean op the frontend
           doesn't yet emit. We still record the if/else structurally
           with ``condition_text="..."`` carrying the unparsed Python
           source. Diagnostic is demoted from "unsupported warning" to
           an informational note naming the unlowered axis. Downstream
           consumers always see the structural shape; the specific
           lowering quality is then visible per-op.
        """
        condition = self._static_condition(node.test)
        if condition is not None:
            # Case 1 — static.
            self.ops.append(self._marker(
                "tessera.scf.if.begin", node, condition=condition))
            for stmt in node.body:
                self.visit(stmt)
            if node.orelse:
                self.ops.append(self._marker(
                    "tessera.scf.else", node, condition=condition))
                for stmt in node.orelse:
                    self.visit(stmt)
            self.ops.append(self._marker(
                "tessera.scf.if.end", node, condition=condition))
            return
        # Cases 2 / 3 — dynamic.
        cond_value = self._emit_expr(node.test)
        if cond_value is not None:
            # Case 2 — dynamic, SSA-bound.
            self._note(
                node, "Python if/else lowered with dynamic SSA condition",
                code="PY_FRONTEND_DYNAMIC_IF_LOWERED")
            self.ops.append(IROp(
                result=None,
                op_name="tessera.scf.if.begin",
                operands=[cond_value],
                operand_types=[str(self._value_types.get(
                    cond_value, TENSOR_OPAQUE))],
                kwargs={"kind": "dynamic"},
                source_span=_span_from_ast(node),
            ))
            for stmt in node.body:
                self.visit(stmt)
            if node.orelse:
                self.ops.append(IROp(
                    result=None,
                    op_name="tessera.scf.else",
                    operands=[cond_value],
                    operand_types=[str(self._value_types.get(
                        cond_value, TENSOR_OPAQUE))],
                    kwargs={"kind": "dynamic"},
                    source_span=_span_from_ast(node),
                ))
                for stmt in node.orelse:
                    self.visit(stmt)
            self.ops.append(IROp(
                result=None,
                op_name="tessera.scf.if.end",
                operands=[cond_value],
                operand_types=[str(self._value_types.get(
                    cond_value, TENSOR_OPAQUE))],
                kwargs={"kind": "dynamic"},
                source_span=_span_from_ast(node),
            ))
            return
        # Case 3 — dynamic, condition expression not (yet) lowerable.
        cond_text = _safe_unparse(node.test)
        self._note(
            node, ("Python if/else lowered structurally but the condition "
                   "expression isn't (yet) emitted as SSA"),
            code="PY_FRONTEND_DYNAMIC_IF_UNLOWERED_CONDITION")
        self.ops.append(self._marker(
            "tessera.scf.if.begin", node,
            kind="dynamic", condition_text=cond_text))
        for stmt in node.body:
            self.visit(stmt)
        if node.orelse:
            self.ops.append(self._marker(
                "tessera.scf.else", node,
                kind="dynamic", condition_text=cond_text))
            for stmt in node.orelse:
                self.visit(stmt)
        self.ops.append(self._marker(
            "tessera.scf.if.end", node,
            kind="dynamic", condition_text=cond_text))

    def visit_For(self, node: ast.For) -> None:
        """Lower Python `for` to `tessera.scf.for.*` markers.

        Like ``visit_If``: static trip count (``range(N)``) keeps the
        ``trip_count=N`` attribute; a lowerable runtime trip count emits
        it as an SSA operand; an unlowerable iterable records its source
        text so the downstream sees the structural for-loop boundary.
        """
        range_trip_count = self._static_range_trip_count(node.iter)
        induction = node.target.id if isinstance(node.target, ast.Name) else "_"
        if range_trip_count is not None:
            # Static trip count.
            self.ops.append(self._marker(
                "tessera.scf.for.begin", node,
                induction=induction, trip_count=range_trip_count))
            for stmt in node.body:
                self.visit(stmt)
            self.ops.append(self._marker(
                "tessera.scf.for.end", node,
                induction=induction, trip_count=range_trip_count))
            return
        # Dynamic — try to emit the iterable's count as SSA.
        trip_value = self._dynamic_range_trip_value(node.iter)
        if trip_value is not None:
            self._note(
                node, "Python for-loop lowered with dynamic SSA trip count",
                code="PY_FRONTEND_DYNAMIC_FOR_LOWERED")
            self.ops.append(IROp(
                result=None,
                op_name="tessera.scf.for.begin",
                operands=[trip_value],
                operand_types=[str(self._value_types.get(
                    trip_value, TENSOR_OPAQUE))],
                kwargs={"kind": "dynamic", "induction": induction},
                source_span=_span_from_ast(node),
            ))
            for stmt in node.body:
                self.visit(stmt)
            self.ops.append(IROp(
                result=None,
                op_name="tessera.scf.for.end",
                operands=[trip_value],
                operand_types=[str(self._value_types.get(
                    trip_value, TENSOR_OPAQUE))],
                kwargs={"kind": "dynamic", "induction": induction},
                source_span=_span_from_ast(node),
            ))
            return
        # Iterable not (yet) lowerable — record structurally with source text.
        iter_text = _safe_unparse(node.iter)
        self._note(
            node, ("Python for-loop lowered structurally but the iterable "
                   "expression isn't (yet) emitted as SSA"),
            code="PY_FRONTEND_DYNAMIC_FOR_UNLOWERED_ITERABLE")
        self.ops.append(self._marker(
            "tessera.scf.for.begin", node,
            kind="dynamic", induction=induction, iter_text=iter_text))
        for stmt in node.body:
            self.visit(stmt)
        self.ops.append(self._marker(
            "tessera.scf.for.end", node,
            kind="dynamic", induction=induction, iter_text=iter_text))

    def visit_While(self, node: ast.While) -> None:
        """Lower Python `while` to `tessera.scf.while.*` markers.

        While loops always have a dynamic condition (no static form
        makes sense — a literal True/False is either an infinite loop
        or dead code). The shape is symmetric with dynamic ``if``.
        """
        cond_value = self._emit_expr(node.test)
        if cond_value is not None:
            self._note(
                node, "Python while-loop lowered with dynamic SSA condition",
                code="PY_FRONTEND_WHILE_LOWERED")
            self.ops.append(IROp(
                result=None,
                op_name="tessera.scf.while.begin",
                operands=[cond_value],
                operand_types=[str(self._value_types.get(
                    cond_value, TENSOR_OPAQUE))],
                kwargs={"kind": "dynamic"},
                source_span=_span_from_ast(node),
            ))
            for stmt in node.body:
                self.visit(stmt)
            self.ops.append(IROp(
                result=None,
                op_name="tessera.scf.while.end",
                operands=[cond_value],
                operand_types=[str(self._value_types.get(
                    cond_value, TENSOR_OPAQUE))],
                kwargs={"kind": "dynamic"},
                source_span=_span_from_ast(node),
            ))
            return
        cond_text = _safe_unparse(node.test)
        self._note(
            node, ("Python while-loop lowered structurally but the condition "
                   "expression isn't (yet) emitted as SSA"),
            code="PY_FRONTEND_WHILE_UNLOWERED_CONDITION")
        self.ops.append(self._marker(
            "tessera.scf.while.begin", node,
            kind="dynamic", condition_text=cond_text))
        for stmt in node.body:
            self.visit(stmt)
        self.ops.append(self._marker(
            "tessera.scf.while.end", node,
            kind="dynamic", condition_text=cond_text))

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Lower ``x += y`` (and friends) by desugaring to ``x = x op y``.

        Only desugars when the target is a plain ``Name`` and the op is
        one of ``+ * - /`` (the four BinOps the frontend knows). The
        rewritten Assign is visited through the normal ``visit_Assign``
        path, so the resulting IR matches what the user would have
        written explicitly.
        """
        if not isinstance(node.target, ast.Name):
            self._unsupported(
                node,
                "augmented assignment with non-Name target is not lowered")
            return
        if not isinstance(node.op, (ast.Add, ast.Mult, ast.Sub, ast.Div)):
            self._unsupported(
                node,
                f"augmented assignment with {type(node.op).__name__} is not "
                "lowered")
            return
        desugared = ast.Assign(
            targets=[ast.Name(id=node.target.id, ctx=ast.Store())],
            value=ast.BinOp(
                left=ast.Name(id=node.target.id, ctx=ast.Load()),
                op=node.op,
                right=node.value,
            ),
        )
        ast.copy_location(desugared, node)
        ast.copy_location(desugared.targets[0], node.target)
        ast.copy_location(desugared.value, node)
        self._note(
            node, f"augmented assignment desugared to `{node.target.id} = "
                  f"{node.target.id} {type(node.op).__name__.lower()} ...`",
            code="PY_FRONTEND_AUGASSIGN_DESUGARED")
        self.visit_Assign(desugared)

    def _unsupported(self, node: ast.AST, message: str) -> None:
        self.diagnostics.append(GraphIRDiagnostic(
            "warning",
            message,
            span=_span_from_ast(node),
            code="PY_FRONTEND_UNSUPPORTED",
        ))

    def _note(self, node: ast.AST, message: str, *, code: str) -> None:
        """D.1 — informational diagnostic for dynamic-control-flow lowering
        decisions. Distinct from ``_unsupported`` (which is a warning that
        the construct wasn't lowered at all). A ``note`` records *how* a
        dynamic construct was lowered so a reader knows what to expect
        from downstream passes."""
        self.diagnostics.append(GraphIRDiagnostic(
            "info",
            message,
            span=_span_from_ast(node),
            code=code,
        ))

    def _dynamic_range_trip_value(self, node: ast.expr) -> Optional[str]:
        """When ``for i in range(...)`` has a runtime argument, try to
        lower the trip-count expression to an SSA value. v1 supports a
        single-arg ``range(n)`` with an SSA-emittable ``n``.
        """
        if not isinstance(node, ast.Call):
            return None
        name = self._resolve_name(node.func)
        if name != "range":
            return None
        if len(node.args) != 1 or node.keywords:
            return None  # range(start, stop, step) — TODO
        return self._emit_expr(node.args[0])

    def _marker(self, op_name: str, node: ast.AST, **kwargs: Any) -> IROp:
        return IROp(
            result=None,
            op_name=op_name,
            operands=[],
            operand_types=[],
            kwargs=kwargs,
            source_span=_span_from_ast(node),
        )

    def _static_range_trip_count(self, node: ast.expr) -> Optional[int]:
        if not isinstance(node, ast.Call):
            return None
        name = self._resolve_name(node.func)
        if name != "range":
            return None
        try:
            args = [int(ast.literal_eval(arg)) for arg in node.args]
        except Exception:
            return None
        if len(args) == 1:
            start, stop, step = 0, args[0], 1
        elif len(args) == 2:
            start, stop, step = args[0], args[1], 1
        elif len(args) == 3:
            start, stop, step = args
        else:
            return None
        if step == 0:
            return None
        return max(0, len(range(start, stop, step)))

    def _static_condition(self, node: ast.expr) -> Optional[bool]:
        try:
            value = ast.literal_eval(node)
        except Exception:
            value = None
        if isinstance(value, bool):
            return value
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
            try:
                lhs = ast.literal_eval(node.left)
                rhs = ast.literal_eval(node.comparators[0])
            except Exception:
                return None
            op = node.ops[0]
            if isinstance(op, ast.Eq):
                return lhs == rhs
            if isinstance(op, ast.NotEq):
                return lhs != rhs
            if isinstance(op, ast.Lt):
                return lhs < rhs
            if isinstance(op, ast.LtE):
                return lhs <= rhs
            if isinstance(op, ast.Gt):
                return lhs > rhs
            if isinstance(op, ast.GtE):
                return lhs >= rhs
        return None

    def _emit_expr(self, node: ast.expr, result_name: Optional[str] = None) -> Optional[str]:
        if isinstance(node, ast.Call):
            op = self._try_map_call(node)
            if op is None:
                return None
            op.result = result_name or self._fresh()
            _ir_type = op.inferred_type if hasattr(op, "inferred_type") else _parse_mlir_tensor_type(op.result_type or "tensor<*x?>")
            if _ir_type is not None:
                self._value_types[f"%{op.result}"] = _ir_type
            self.ops.append(op)
            return f"%{op.result}"
        if isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Mult, ast.Sub, ast.Div)):
            op = self._try_map_binop(node)
            if op is None:
                return None
            op.result = result_name or self._fresh()
            _ir_type = op.inferred_type if hasattr(op, "inferred_type") else _parse_mlir_tensor_type(op.result_type or "tensor<*x?>")
            if _ir_type is not None:
                self._value_types[f"%{op.result}"] = _ir_type
            self.ops.append(op)
            return f"%{op.result}"
        if isinstance(node, ast.Name):
            return f"%{self._resolve_ssa_name(node.id)}"
        if isinstance(node, ast.Attribute) and node.attr == "T":
            value = self._emit_expr(node.value)
            if value is None:
                return None
            result = result_name or self._fresh()
            self.ops.append(IROp(
                result=result,
                op_name="tessera.transpose",
                operands=[value],
                operand_types=[str(self._value_types.get(value, TENSOR_OPAQUE))],
                result_type=str(self._value_types.get(value, TENSOR_OPAQUE)),
                source_span=_span_from_ast(node),
            ))
            self._value_types[f"%{result}"] = self._value_types.get(value, TENSOR_OPAQUE)
            return f"%{result}"
        return None

    def _try_map_binop(self, node: ast.BinOp) -> Optional[IROp]:
        # D.5 fix (2026-05-31): visit_AugAssign accepts Sub/Div, but pre-fix
        # this mapper only knew Add/Mult. Result: `c -= b` desugared to
        # `c = c - b`, _emit_expr returned None on the Sub BinOp, and the
        # subtract was silently dropped. Now lowered symmetrically.
        if isinstance(node.op, ast.Add):
            op_name = "tessera.add"
        elif isinstance(node.op, ast.Mult):
            op_name = "tessera.mul"
        elif isinstance(node.op, ast.Sub):
            op_name = "tessera.sub"
        elif isinstance(node.op, ast.Div):
            op_name = "tessera.div"
        else:
            return None
        operands: list[str] = []
        operand_types: list[str] = []
        kwargs: dict[str, Any] = {}

        for side, expr in (("left", node.left), ("right", node.right)):
            try:
                literal = ast.literal_eval(expr)
            except (ValueError, TypeError):
                literal = None
            if isinstance(literal, (int, float)):
                kwargs["scalar"] = float(literal)
                kwargs["scalar_side"] = side
                continue
            operand = self._emit_expr(expr)
            if operand is None:
                return None
            operands.append(operand)
            operand_types.append(str(self._value_types.get(operand, TENSOR_OPAQUE)))

        if not operands:
            return None
        result_type = _infer_result_type(op_name, [self._value_types.get(operand, TENSOR_OPAQUE) for operand in operands])
        return IROp(
            result=None,
            op_name=op_name,
            operands=operands,
            operand_types=operand_types,
            result_type=str(result_type),
            kwargs=kwargs,
            source_span=_span_from_ast(node),
            inferred_type=result_type,
        )

    def _try_map_call(self, call: ast.Call) -> Optional[IROp]:
        name = self._resolve_name(call.func)
        if not name:
            return None
        mlir_name = graph_name_for(name)
        if not mlir_name:
            return None

        operands = []
        operand_types = []
        for arg in call.args:
            operand = self._emit_expr(arg)
            operands.append(operand if operand else "%?")
            operand_types.append(str(self._value_types.get(operand or "%?", TENSOR_OPAQUE)))

        result_type = _infer_result_type(mlir_name, [self._value_types.get(operand, TENSOR_OPAQUE) for operand in operands])

        kwargs = {}
        for kw in call.keywords:
            if kw.arg is None:
                continue
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, TypeError):
                value_name = self._emit_expr(kw.value)
                kwargs[kw.arg] = value_name or "?"

        return IROp(
            result=None,  # filled in by caller
            op_name=mlir_name,
            operands=operands,
            operand_types=operand_types,
            result_type=str(result_type),
            kwargs=kwargs,
            source_span=_span_from_ast(call),
            inferred_type=result_type,
        )


def _safe_unparse(node: ast.AST) -> str:
    """``ast.unparse`` for D.1 / D.2 / D.3 — record an unlowered
    dynamic-control-flow expression as the original Python source text
    so downstream consumers (and humans reading IR dumps) know what the
    construct was about, even if the expression itself isn't yet
    emitted as SSA. Truncated to a sensible length; defensive against
    ``unparse`` raising on edge AST nodes."""
    try:
        text = ast.unparse(node)
    except Exception:
        text = "<unparseable>"
    if len(text) > 120:
        text = text[:117] + "..."
    return text


def _span_from_ast(node: ast.AST) -> SourceSpan:
    end_col_offset = getattr(node, "end_col_offset", None)
    end_col = end_col_offset + 1 if end_col_offset is not None else None
    return SourceSpan(
        line=int(getattr(node, "lineno", 0) or 0),
        col=int(getattr(node, "col_offset", 0) or 0) + 1,
        end_line=getattr(node, "end_lineno", None),
        end_col=end_col,
    )


def _infer_result_type(op_name: str, operand_types: List[IRType]) -> IRType:
    if not operand_types:
        return TENSOR_OPAQUE
    if op_name == "tessera.matmul" and len(operand_types) >= 2:
        lhs, rhs = operand_types[0], operand_types[1]
        dtype = lhs.dtype or rhs.dtype
        if lhs.rank == 2 and rhs.rank == 2:
            return tensor_ir_type((lhs.shape[0], rhs.shape[1]), dtype, layout=lhs.layout)
        return tensor_ir_type(("*",), dtype, layout=lhs.layout)
    if op_name == "tessera.batched_gemm" and len(operand_types) >= 2:
        # Rank-3 C[b] = A[b] @ B[b]: result is B×M×N from A's (B,M) + B's N.
        lhs, rhs = operand_types[0], operand_types[1]
        dtype = lhs.dtype or rhs.dtype
        if lhs.rank == 3 and rhs.rank == 3:
            return tensor_ir_type((lhs.shape[0], lhs.shape[1], rhs.shape[2]),
                                  dtype, layout=lhs.layout)
        return tensor_ir_type(("*",), dtype, layout=lhs.layout)
    if op_name == "tessera.ebm.energy_quadratic" and operand_types:
        x = operand_types[0]
        if x.rank == 2:
            return tensor_ir_type((x.shape[0],), x.dtype, layout=x.layout)
        return tensor_ir_type(("*",), x.dtype, layout=x.layout)
    if op_name == "tessera.ebm.langevin_step" and operand_types:
        return operand_types[0]
    if op_name in {"tessera.transpose"} and operand_types[0].rank is not None:
        first = operand_types[0]
        return tensor_ir_type(tuple(reversed(first.shape)), first.dtype, layout=first.layout)
    return operand_types[0]


def _parse_mlir_tensor_type(text: str) -> IRType:
    match = re.fullmatch(r"tensor<(.+)x([^x<>]+)>", text or "")
    if not match:
        return IRType(text or "tensor<*x?>")
    shape_text, dtype_text = match.groups()
    reverse_dtype = {
        "f64": "fp64",
        "f32": "fp32",
        "f16": "fp16",
        "bf16": "bf16",
        "xf8E4M3FN": "fp8_e4m3",
        "xf8E5M2": "fp8_e5m2",
        "!tessera.fp6_e2m3": "fp6_e2m3",
        "!tessera.fp6_e3m2": "fp6_e3m2",
        "!tessera.fp4_e2m1": "fp4_e2m1",
        "!tessera.nvfp4": "nvfp4",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "i1": "bool",
        "?": None,
    }
    # Extended MLIR float spellings such as `xf8E4M3FN` start with the same
    # `x` separator used between shape dimensions, so the simple regex above
    # sees `tensor<2x3xxf8E4M3FN>` as shape `2x3x` plus dtype `f8E4M3FN`.
    # Reattach that leading marker before splitting the shape.
    if shape_text.endswith("x") and f"x{dtype_text}" in reverse_dtype:
        shape_text = shape_text[:-1]
        dtype_text = f"x{dtype_text}"
    shape = ("*",) if shape_text == "*" else tuple(shape_text.split("x"))
    return tensor_ir_type(shape, reverse_dtype.get(dtype_text, dtype_text))


# ─────────────────────────────────────────────────────────────────────────────
# GraphIRBuilder
# ─────────────────────────────────────────────────────────────────────────────

class GraphIRBuilder:
    """
    Lowers a Python function (decorated with @jit) into a GraphIRFunction.

    Structured construction pipeline:
      1. Extract parameter annotations → IRArg list (with effect attrs)
      2. Walk function AST → IROp list (tessera ops only)
      3. Assemble GraphIRFunction inside GraphIRConstructionContext

    Usage:
        builder = GraphIRBuilder()
        fn_ir = builder.lower(step)
        module = builder.module()
        print(module.to_mlir())
    """

    def __init__(self) -> None:
        self.context = GraphIRConstructionContext()
        self.diagnostics: List[GraphIRDiagnostic] = []

    def lower(
        self,
        fn: Callable,
        effect_tag: Optional[str] = None,
        target_attr: Optional[str] = None,
        source_text: Optional[str] = None,
    ) -> "GraphIRFunction":
        """
        Lower fn to a GraphIRFunction and add it to the module.

        Args:
            fn          : Python function to lower
            effect_tag  : optional effect annotation for the function (e.g., "pure")
            target_attr : optional GPU target attribute dict string emitted as
                          tessera.target on the module (Phase 3+)
            source_text : optional Python source text for functions whose
                          source cannot be retrieved with inspect.getsource()

        Returns:
            GraphIRFunction — the emitted function IR
        """
        if target_attr is not None:
            self.context.module.module_attrs["tessera.target"] = target_attr
        import typing

        sig = inspect.signature(fn)
        hints = {}
        try:
            hints = typing.get_type_hints(fn)
        except Exception:
            pass

        args = []
        for param_name, param in sig.parameters.items():
            ann = hints.get(param_name) or param.annotation

            # Detect RegionType, Tensor[...] and dtype annotations.
            effect = None
            ir_type = _annotation_to_ir_type(ann)
            dim_names: Tuple[str, ...] = ()
            layout = ir_type.layout
            if ann is not inspect.Parameter.empty:
                # RegionType from tessera.distributed.region
                if hasattr(ann, "mode"):
                    effect = ann.mode
                if hasattr(ann, "__dims__"):
                    dim_names = tuple(str(dim) for dim in getattr(ann, "__dims__"))

            args.append(IRArg(name=param_name, ir_type=ir_type, effect=effect, dim_names=dim_names, layout=layout))

        # Extract ops from AST
        arg_names = [a.name for a in args]
        ops = self._extract_ops(fn, arg_names, {a.name: a.ir_type for a in args}, source_text=source_text)

        # Build function attrs
        fn_attrs = {}
        if effect_tag:
            fn_attrs["tessera.effect"] = f'"{effect_tag}"'

        fn_ir = GraphIRFunction(
            name=fn.__name__,
            args=args,
            body=ops,
            fn_attrs=fn_attrs,
        )
        self.context.add_function(fn_ir)
        return fn_ir

    def _extract_ops(
        self,
        fn: Callable,
        arg_names: List[str],
        arg_types: Optional[Dict[str, IRType]] = None,
        source_text: Optional[str] = None,
    ) -> List[IROp]:
        """Walk the function AST and extract recognized tessera op calls."""
        try:
            source = source_text if source_text is not None else inspect.getsource(fn)
            source = textwrap.dedent(source)
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            return []

        extractor = _OpExtractor(arg_names, arg_types)
        # Only visit the function body (skip decorator lines)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == fn.__name__:
                    for stmt in node.body:
                        extractor.visit(stmt)
                    break

        self.diagnostics.extend(extractor.diagnostics)
        return extractor.ops

    def module(self) -> GraphIRModule:
        """Return the assembled GraphIRModule."""
        return self.context.module

    def reset(self) -> None:
        """Clear all accumulated functions."""
        self.context = GraphIRConstructionContext()
        self.diagnostics = []


def _annotation_to_ir_type(ann: Any) -> IRType:
    if ann is inspect.Parameter.empty:
        return TENSOR_OPAQUE
    # String annotation that is itself an MLIR tensor type
    # (e.g. "tensor<2x4x8xf32>") — parse it so the AST front door produces
    # statically-shaped args. Without this, an annotated Python function lowers
    # to opaque args and shape inference (matmul/batched_gemm) falls back to a
    # dynamic result, which then fails the static-shape value lane.
    if isinstance(ann, str):
        # `from __future__ import annotations` (PEP 563) stringifies a
        # string-literal annotation to include its surrounding quotes
        # (`'"tensor<...>"'`); a non-future module yields the bare string. Strip
        # one layer of matching quotes so both forms parse.
        text = ann.strip()
        if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
            text = text[1:-1].strip()
        if text.startswith("tensor<"):
            parsed = _parse_mlir_tensor_type(text)
            if parsed is not None:
                return parsed
    dims = getattr(ann, "__dims__", None)
    dtype = getattr(ann, "dtype", None)
    shape = getattr(ann, "shape", None)
    layout = getattr(ann, "layout", None)
    if dims is not None:
        return tensor_ir_type(tuple("?" for _ in dims), dtype, layout=layout)
    if dtype is not None:
        normalized_shape = _shape_from_annotation(shape)
        return tensor_ir_type(normalized_shape, dtype, layout=layout)
    return TENSOR_OPAQUE


def _shape_from_annotation(shape: Any) -> Tuple[Any, ...]:
    if shape is None:
        return ("*",)
    if not isinstance(shape, tuple):
        shape = (shape,)
    if any(item is Ellipsis for item in shape):
        return ("*",)
    return tuple("?" if item is None else item for item in shape)
