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
import copy
import inspect
import json
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, cast

from ..diagnostics import DiagnosticLevel, DiagnosticWhere, SourceLocation, TesseraDiagnostic, TesseraErrorCode
from .legality import TensorContract, check_op_legality
from .op_catalog import GRAPH_OP_MAP, graph_name_for


# Ops that take trailing *scalar* parameters positionally (a tensor operand plus
# scalars like k / axis).  When such an arg is a literal, the AST frontend binds
# it to the named attribute here instead of emitting a bogus operand — so e.g.
# ``top_k(x, 5)`` lowers to ``tessera.top_k(%x) {k = 5}`` exactly like the kwarg
# form ``top_k(x, k=5)``.  Order = the positional scalar params after the
# operands.  Tensor positional args are unaffected (they still emit as operands).
_POSITIONAL_ATTR_PARAMS: Dict[str, tuple[str, ...]] = {
    # W1.2 (2026-08-03) — trailing POSITIONAL parameters that are attributes,
    # not operands. Without these the frontend counts them as tensor operands,
    # which is why 41 ops failed an arity probe and why structural view ops
    # could not be emitted at all.
    #
    # Each was checked against the reference body rather than assumed: a
    # parameter consumed via `len()` / `zip()` / `int()` is static, while
    # `selective_ssm`'s `delta` is documented `(B, S, D)` and is a real
    # operand — that one is catalog arity drift, fixed in op_catalog instead.
    "tessera.arange": ("start",),
    "tessera.cast": ("dtype",),
    "tessera.chunk": ("chunks",),
    "tessera.dynamic_slice": ("start_indices", "slice_sizes"),
    "tessera.dynamic_update_slice": ("start_indices",),
    "tessera.factorized_matmul": ("rank",),
    "tessera.istft": ("hop", "axis", "n_fft", "center", "length", "onesided"),
    "tessera.stft": ("hop", "axis", "n_fft", "center", "pad_mode", "onesided"),
    "tessera.pack": ("layout",),
    "tessera.pad": ("pad_width",),
    "tessera.rearrange": ("layout",),
    "tessera.repeat": ("repeats",),
    "tessera.roll": ("shift",),
    "tessera.select": ("index",),
    "tessera.split": ("indices_or_sections",),
    "tessera.tile": ("reps",),
    "tessera.tile_view": ("BM", "BN"),
    "tessera.rng_normal": ("shape",),
    "tessera.rng_uniform": ("shape",),
    "tessera.rng_philox_normal": ("shape",),
    "tessera.rng_philox_uniform": ("shape",),

    "tessera.top_k": ("k", "axis"),
    # slice(x, start_indices, slice_sizes): the two trailing positional args are
    # index/size *lists* (StableHLO dynamic-slice form), not tensors — bind them
    # as attributes so the op lowers to ``tessera.slice(%x) {start_indices=…,
    # slice_sizes=…}`` instead of dropping them as "%?" operands. (The list-of-
    # ints is literal-eval'd; contrast cat/stack, whose list-of-tensors flattens
    # into operands.)
    "tessera.slice": ("start_indices", "slice_sizes"),
    # P1 frontend emission — the 6 structural view ops (P1a ODS) carry their
    # axes/perm/shape as a *discardable* attribute. Those ODS ops take ONLY
    # `TensorType:$x`; without binding the trailing positional arg as an
    # attribute it drops to a bogus "%?" operand and the op fails the 1-operand
    # verifier. Attribute NAMES match the canonical lit-fixture vocabulary
    # (tests/tessera-ir/phase2/structural_view_ops.mlir) so the §6.B view/copy
    # analysis reads one spelling: squeeze/unsqueeze=`axes`, permute=`perm`
    # (the C++ PermuteOp::fold reads getAttrOfType<ArrayAttr>("perm")),
    # expand/broadcast=`shape`, flatten=`start`/`end`. Tuple/list values
    # serialize to `[…]` array attrs via _format_attr_value.
    "tessera.squeeze": ("axes",),
    "tessera.unsqueeze": ("axes",),
    "tessera.permute": ("perm",),
    "tessera.expand": ("shape",),
    "tessera.broadcast": ("shape",),
    "tessera.flatten": ("start", "end"),
    # reshape/view — their own ODS ops (same SameOperandsAndResultElementType
    # contract); the target shape binds as the `shape` attr, not an operand.
    # `kv_cache.read(cache, start, end)` -- the bounds are scalars. Unlike
    # `cache.commit`/`rollback`/`page_lookup`, which the ODS declares with
    # `Index` OPERANDS, `kv_cache.read` has no Graph IR ODS op at all, so
    # there is no operand contract to contradict and binding them as
    # attributes is what lets the op emit and verify.
    "tessera.kv_cache.read": ("start", "end"),
    "tessera.reshape": ("shape",),
    "tessera.view": ("shape",),
}


_SPECTRAL_GRAPH_OPS = frozenset({
    "tessera.fft",
    "tessera.ifft",
    "tessera.rfft",
    "tessera.irfft",
    "tessera.dct",
    "tessera.stft",
    "tessera.istft",
    "tessera.spectral_filter",
    "tessera.spectral_conv",
})


def _canonicalize_spectral_attrs(
    op_name: str, operand_types: List[IRType], attrs: Dict[str, Any]
) -> None:
    """Stamp the canonical Graph spectral contract at its Python producer.

    The public API retains NumPy-compatible ``norm``/``n``/``n_fft`` names,
    while Graph IR has one target-independent vocabulary.  Leaving both
    spellings in emitted IR made artifact identity depend on consumer-side
    defaults and allowed the Python and C++ lowering lanes to disagree.
    """
    if op_name not in _SPECTRAL_GRAPH_OPS:
        return

    normalization = attrs.pop("norm", attrs.get("normalization", "backward"))
    attrs["normalization"] = normalization
    if "output_length" not in attrs and "length" in attrs:
        attrs["output_length"] = attrs.pop("length")
    if "logical_length" not in attrs:
        legacy_length = attrs.pop("n", attrs.pop("n_fft", None))
        if legacy_length is not None:
            attrs["logical_length"] = legacy_length
    else:
        attrs.pop("n", None)
        attrs.pop("n_fft", None)

    half_spectrum = op_name in {
        "tessera.rfft",
        "tessera.irfft",
        "tessera.spectral_conv",
    } or (
        op_name in {"tessera.stft", "tessera.istft"}
        and attrs.get("onesided", True)
    )
    default_layout = (
        "full_real"
        if op_name == "tessera.dct"
        else "half_spectrum_nyquist_explicit"
        if half_spectrum
        else "full_complex"
    )
    attrs.setdefault("spectrum_layout", default_layout)
    if op_name in {"tessera.stft", "tessera.istft"}:
        attrs.setdefault("center", False)
        attrs.setdefault("onesided", True)
        attrs.setdefault("pad_mode", "constant")
    if op_name in {"tessera.fft", "tessera.ifft", "tessera.rfft", "tessera.irfft"}:
        attrs.setdefault("hermitian_weight", "none")

    if (
        "logical_length" not in attrs
        and op_name == "tessera.spectral_conv"
        and len(operand_types) >= 2
    ):
        x, kernel = operand_types[:2]
        axis = _transform_axis(x, attrs)
        if axis is not None and x.shape and kernel.shape:
            x_extent = _dim(x.shape[axis])
            kernel_extent = _dim(kernel.shape[axis])
            if x_extent is not None and kernel_extent is not None:
                full_extent = x_extent + kernel_extent - 1
                attrs["logical_length"] = 1 << (full_extent - 1).bit_length()

    if "logical_length" not in attrs and operand_types:
        first = operand_types[0]
        axis = _transform_axis(first, attrs)
        if axis is not None and first.shape:
            extent = _dim(first.shape[axis])
            if extent is not None:
                if op_name == "tessera.irfft":
                    extent = 2 * (extent - 1)
                elif op_name == "tessera.istft":
                    extent = 2 * (extent - 1)
                attrs["logical_length"] = extent

    if op_name == "tessera.dct":
        attrs.setdefault("type", 2)
        attrs.setdefault("transpose_basis", False)


# Keyword-only TENSOR operands — the other half of the same contract.
#
# W1.3 (2026-08-03). `_POSITIONAL_ATTR_PARAMS` above fixes one direction: a
# positional arg that is really an attribute. The keyword side had the mirror
# defect and it was worse, because it was silent. Every `call.keywords` entry
# became an attribute regardless of what it bound, so a tensor passed by keyword
# emitted as a *string attribute holding an SSA name*:
#
#     tessera.attn_top_k_blocks(%Q, %K, %V) {scores = "%s", top_k = 2}
#
# `%s` is nowhere in the operand list. The dataflow edge is gone: use-def walks,
# liveness, and DCE all see `%s` as unused, and the graph verifier cannot catch
# it because it only checks `op.operands` — an edge hidden in an attribute is
# invisible to the one check that would have found it. The IR *looks* fine in a
# dump, which is why this survived. (Decision #30: the emitter must derive the
# operand/attribute split, not take the call syntax's word for it.)
#
# Declaration is load-bearing for ORDER, not just for documentation. Deriving
# alone -- "a kwarg binding an SSA value is an operand" -- would make the operand
# list depend on the order the caller happened to write the keywords, so
# `f(Q, scores=s, top_k=2)` and `f(Q, top_k=2, scores=s)` would emit different
# IR for the same program. The declared tuple pins the position.
_KEYWORD_OPERANDS: Dict[str, tuple[str, ...]] = {
    # Affine normalization has a fixed ABI: data, scale, bias. Alphabetical
    # fallback used to silently swap beta/gamma in the legacy candidate.
    "tessera.layer_norm": ("gamma", "beta"),
    "tessera.rmsnorm": ("gamma",),
    # NSA branch 3: per-block scores (B, H, S_q, num_blocks), unwrapped and
    # `np.asarray`d by the reference exactly like Q/K/V. A real operand.
    "tessera.attn_top_k_blocks": ("scores",),
    # Packed-sequence cumulative lengths — index tensors, not scalars. The
    # catalog already counted them (min_arity=5); only the frontend dropped them.
    "tessera.varlen_sdpa": ("cu_seqlens_q", "cu_seqlens_k"),
    # MoE routing: `scores` (B*S, num_experts) and `route` (per-token expert
    # index, asarray'd to int64) are OPTIONAL tensor operands. Optional is why
    # they were missed by a survey of *required* keyword-only parameters — the
    # operand/attribute question is about what a parameter BINDS, and has
    # nothing to do with whether it has a default.
    "tessera.moe": ("scores", "route"),
}


# Keyword-only ATTRIBUTES. These already emitted correctly (a scalar keyword
# becomes an attribute, which is right) -- what was missing was a DECLARATION
# SITE, so nothing could tell "this op requires a keyword-only attribute" from
# "the catalog's arity is wrong". `test_op_arity_contract` had to scope itself to
# positional parameters for exactly that reason; with this map it no longer does.
#
# Values are the required keyword-only attribute names, so a rename in the
# reference signature is caught rather than silently emitting a stale key.
_KEYWORD_ATTR_PARAMS: Dict[str, tuple[str, ...]] = {
    "tessera.es_low_rank_correction": ("out_dim", "rank", "sigma"),
    "tessera.attn_sliding_window": ("window_size",),
    "tessera.attn_top_k_blocks": ("top_k", "block_size"),
    "tessera.deepseek_sparse_attention": ("window_size", "block_size", "top_k"),
    "tessera.ebm_inner_step": ("eta",),
    "tessera.ebm_refinement": ("eta", "T"),
    "tessera.lookahead_sparse_attention": ("window_size", "block_size"),
    "tessera.masked_fill": ("value",),
    "tessera.memory_index_select": ("block_size",),
    "tessera.mor_partition": ("step",),
    "tessera.mor_router": ("max_depth",),
    "tessera.msa_index_scores": ("block_size",),
    "tessera.msa_select_blocks": ("top_k", "block_size"),
    "tessera.msa_sparse_attention": ("block_size", "top_k"),
    "tessera.rope_split": ("rope_dim",),
    "tessera.softcap": ("cap",),
    # W2.2 -- newly reachable ops (`nesterov` and the two fused
    # loss+optimizer steps). `kind` selects the LOSS, so it is a semantic key
    # in the Decision #21a sense: it must be stated, never defaulted.
    "tessera.nesterov": ("lr",),
    "tessera.training.loss_sgd": ("kind", "lr"),
    "tessera.training.loss_adamw": ("kind",),
}


def _scalar_const_env(fn: Any) -> Dict[str, Any]:
    """Scalar constants visible to ``fn`` — closure free vars + module globals —
    so a positional scalar param passed by *name* (``top_k(x, k)`` with ``k`` a
    free variable) resolves to its value as an op attribute, like a literal.
    Only int/float/bool/str are captured (the kinds that lower to attributes)."""
    env: Dict[str, Any] = {}
    code = getattr(fn, "__code__", None)
    closure = getattr(fn, "__closure__", None)
    if code is not None and closure:
        for name, cell in zip(code.co_freevars, closure):
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if isinstance(val, (int, float, bool, str)):
                env[name] = val
    g = getattr(fn, "__globals__", None)
    if isinstance(g, dict) and code is not None:
        for name in code.co_names:
            if name not in env and isinstance(g.get(name), (int, float, bool, str)):
                env[name] = g[name]
    return env


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
        "int4": "i4",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "bool": "i1",
        # Canonical Tessera dtype names remain complex64/complex128 in the
        # registry; MLIR's builtin tensor element syntax is complex<f32/f64>.
        "complex64": "complex<f32>",
        "complex128": "complex<f64>",
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


def handle_ir_type(mnemonic: str) -> IRType:
    """A stateful HANDLE type — `!tessera.kv_cache` and friends.

    W1.3. Everything the Python emitter could name was a tensor, so the five
    cache ops were filed as "opaque cache handle rather than a tensor type; its
    result is not describable by a tensor shape rule". True as far as it went,
    and beside the point: the rule does not have to be a *tensor* rule.

    The type was never missing either. `Tessera_KVCacheType` has been in
    `TesseraOps.td` all along, and the ODS states these signatures exactly:

        tessera.kv_cache.append : (!tessera.kv_cache, tensor, tensor)
                                      -> !tessera.kv_cache

    The Python side emitted `tensor<*x?>` for both ends of that, so the handle
    -- a distinct type with its own verifier contract -- silently became an
    untyped tensor at the level that states the contract first. That is
    Decision #32 information loss, and it is invisible precisely because
    `tensor<*x?>` is what an unknown tensor looks like too.

    Deliberately carries no shape or dtype: a handle has neither, and giving it
    `("*",)` would let shape-reasoning code treat it as a rank-unknown tensor
    rather than reject it.
    """
    return IRType(f"!tessera.{mnemonic}")


#: The stateful handle types the Graph IR ODS declares. Names match the ODS
#: `mnemonic` field so the two cannot drift apart silently.
HANDLE_KV_CACHE = handle_ir_type("kv_cache")
HANDLE_CACHE_PAGE = handle_ir_type("cache_page")
HANDLE_RING = handle_ir_type("ring")


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
    model_parameter: bool = False
    model_parameter_bytes_bound: Optional[int] = None

    def to_mlir(self) -> str:
        attrs = []
        if self.effect:
            attrs.append(f'tessera.effect = "{self.effect}"')
        if self.shard_spec:
            attrs.append(f"tessera.shard = {self.shard_spec}")
        if self.dim_names:
            dims = "[" + ", ".join(f'"{name}"' for name in self.dim_names) + "]"
            attrs.append(f"tessera.dim_names = {dims}")
        if self.model_parameter:
            attrs.append("tessera.model.parameter")
        if self.model_parameter_bytes_bound is not None:
            attrs.append(
                "tessera.model.parameter_bytes_bound = "
                f"{self.model_parameter_bytes_bound} : i64"
            )
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
    inferred_types: Tuple[IRType, ...] = ()
    """The FULL result contract for a multi-result op.

    `inferred_type` is the primary result and stays the single-result answer
    every existing consumer reads. Without this field a declared multi-result
    rule stopped at the emitter: `_try_map_call` called the single-result
    `_infer_result_type`, which drops every element after the first, so
    `kv_cache.read`'s `(K, V)` reached Graph IR as one SSA value and no caller
    could destructure V. Declaring the contract in the registry is not the same
    as emitting it."""
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

    def to_mlir(self, indent: str = "  ", *, canonical: bool = False) -> str:
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
        attr_parts.extend(
            f"{key} = {_format_named_attr(key, value)}"
            for key, value in self.kwargs.items()
        )
        # W2.2: carry registered semantics into Graph IR. Emit ``pure`` too:
        # absence means "unregistered/unknown", never "probably pure".
        if "tessera.effect_kind" not in (self.attrs or "") and \
                "tessera.effect_kind" not in self.kwargs:
            from .op_catalog import get_op_spec
            spec = get_op_spec(self.op_name)
            if spec is not None:
                attr_parts.append(
                    f'tessera.effect_kind = "{spec.effect}"'
                )
                if spec.aliasing != "none":
                    attr_parts.append(f'tessera.aliasing = "{spec.aliasing}"')
                if spec.stochastic_identity != "none":
                    attr_parts.append(
                        "tessera.stochastic_identity = "
                        f'"{spec.stochastic_identity}"'
                    )
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
        if (
            self.op_name == "tessera.flash_attn"
            and "operandSegmentSizes" not in (self.attrs or "")
            and "operand_segment_sizes" not in (self.attrs or "")
            and "operandSegmentSizes" not in self.kwargs
            and "operand_segment_sizes" not in self.kwargs
        ):
            bias_count = 1 if len(self.operands) == 4 else 0
            attr_parts.append(
                "operandSegmentSizes = array<i32: 1, 1, 1, "
                f"{bias_count}>"
            )
        attr_str = f" {{{', '.join(attr_parts)}}}" if attr_parts else ""
        if canonical:
            # Parseable custom-assembly form: `op %operands {attrs} : type` — no
            # parens around the operands, so it round-trips through tessera-opt's
            # parser directly. This is the dialect's own printed form and lets the
            # driver feed Graph IR to the Apple value pipeline WITHOUT the old
            # fragile text rewrite (the `tessera.op(...)` -> `tessera.op ...`
            # regex, which also mishandled nested parens / dotted op names). The
            # default (non-canonical) form below keeps the paren rendering used by
            # human-inspection / golden-text consumers unchanged.
            operand_part = f" {ops_str}" if ops_str else ""
            return f"{indent}{lhs}{self.op_name}{operand_part}{attr_str}{type_str}"
        return f"{indent}{lhs}{self.op_name}({ops_str}){attr_str}{type_str}"


def _format_named_attr(key: str, value: Any) -> str:
    if key == "numeric_policy" and isinstance(value, dict):
        return _format_dictionary_attr(value)
    return _format_attr_value(value)


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
    if isinstance(value, float):
        # MLIR requires a decimal point in scientific literals. Python's
        # shortest representation spells small values as ``1e-05``, which is
        # not accepted as a FloatAttr in a generic attribute dictionary.
        rendered = repr(value)
        if "e" in rendered.lower():
            mantissa, exponent = rendered.lower().split("e", 1)
            if "." not in mantissa:
                mantissa += ".0"
            return f"{mantissa}e{exponent}"
        return rendered
    return repr(value)


def _format_dictionary_attr(value: Dict[str, Any]) -> str:
    """Render a typed MLIR dictionary rather than JSON inside a string attr."""
    return "{" + ", ".join(
        f"{key} = {_format_attr_value(item)}"
        for key, item in sorted(value.items())
    ) + "}"


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

    def verify(self, *, target: object = "cpu") -> "GraphIRVerificationResult":
        return GraphIRVerifier().verify_module(self, target=target)

    def to_mlir(
        self,
        *,
        verify: bool = True,
        canonical: bool = False,
        target: object = "cpu",
    ) -> str:
        if verify:
            result = self.verify(target=target)
            if not result.ok:
                raise GraphIRVerificationError(result.format())
        attrs = self._emitted_module_attrs()
        attr_lines = ", ".join(f"{k} = {v}" for k, v in attrs.items())
        lines = [f"module attributes {{{attr_lines}}} {{"]
        for fn in self.functions:
            for line in fn.to_mlir(canonical=canonical).splitlines():
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
    structured_cfg: Any | None = None
    """Tracer-owned multi-block CFG carrier.  This remains separate from the
    flat inspection serialization until the native region dialect consumes the
    complete block/edge ABI."""

    def to_mlir(self, *, verify: bool = False, canonical: bool = False) -> str:
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
            lines.append(op.to_mlir(canonical=canonical))
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

    def verify_module(
        self,
        module: GraphIRModule,
        *,
        target: object = "cpu",
    ) -> GraphIRVerificationResult:
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
            diagnostics.extend(self.verify_function(fn, target=target).diagnostics)
        return GraphIRVerificationResult(tuple(diagnostics))

    def verify_function(
        self,
        fn: GraphIRFunction,
        *,
        target: object = "cpu",
    ) -> GraphIRVerificationResult:
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
            # An attribute may not hide a dataflow EDGE. This is the check whose
            # absence let keyword tensor operands emit as `{scores = "%s"}` for
            # as long as they did: the loop above verifies `op.operands`, so an
            # edge parked in an attribute was invisible to the one place that
            # would have caught it. Verifying operands is not the same as
            # verifying that every edge IS an operand.
            #
            # The test is "names a value that is NOT an operand", not "starts
            # with %". `tessera.graph.debug_value` carries `{name = "%A"}`
            # alongside `%A` in its operand list -- there the attribute is a
            # LABEL for an edge that already exists, which is exactly the case
            # this check must not flag. Flagging the sigil alone would have made
            # the diagnostic mean "an attribute mentions a value", which is not
            # a defect and would have pushed the next person to weaken it.
            operand_set = set(op.operands)
            for attr_name, attr_value in (op.kwargs or {}).items():
                if not isinstance(attr_value, str) or not attr_value.startswith("%"):
                    continue
                if attr_value in operand_set:
                    continue
                diagnostics.append(GraphIRDiagnostic(
                    "error",
                    f"op {op.op_name!r} attribute {attr_name!r} holds SSA "
                    f"value {attr_value}, which is not an operand -- a value "
                    f"edge must be an operand, not an attribute "
                    f"(declare it in _KEYWORD_OPERANDS)",
                    span=op.source_span,
                    code="GRAPH_IR_SSA_VALUE_IN_ATTRIBUTE",
                ))
            self._verify_op_types(op, value_types, diagnostics, target=target)
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
        *,
        target: object = "cpu",
    ) -> None:
        operand_contracts = [
            _tensor_contract_for(value_types.get(operand))
            for operand in op.operands
            if value_types.get(operand) is not None
        ]
        result_type = op.inferred_type or _parse_mlir_tensor_type(op.result_type or "")
        legality = check_op_legality(
            op.op_name,
            operand_contracts,
            target=target,
            result=_tensor_contract_for(result_type),
        )
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

    def __init__(self, arg_names: List[str], arg_types: Optional[Dict[str, IRType]] = None,
                 const_env: Optional[Dict[str, Any]] = None) -> None:
        self.ops: List[IROp] = []
        self.diagnostics: List[GraphIRDiagnostic] = []
        self._counter = 0
        self._arg_names = set(arg_names)
        # Scalar constants captured from the function's closure / module globals
        # so a positional scalar param passed by *name* (``top_k(x, k)`` where
        # ``k`` is a free variable) resolves to its value as an attribute, like a
        # literal does. Only int/float/bool/str are captured.
        self._const_env: Dict[str, Any] = dict(const_env or {})
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
        # The function's returned SSA values + their IR types, captured at the
        # `return` statement so the built GraphIRFunction declares real outputs.
        self.return_values: List[str] = []
        self.return_result_types: List[IRType] = []

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
        elif isinstance(tgt, ast.Tuple) and all(
                isinstance(e, ast.Name) for e in tgt.elts):
            # Destructuring a multi-result op: `v, i = ops.top_k(x, 2)`.
            # Only a bare `ast.Name` target was handled, so a tuple target fell
            # through and the RHS was never emitted at all -- the op vanished
            # from the body rather than failing. Declaring a multi-result
            # contract and emitting both SSA results is only half of it if no
            # Python spelling can bind them.
            first = self._emit_expr(node.value)
            if first is None:
                return
            emitted = self.ops[-1] if self.ops else None
            names = emitted.result_names if emitted is not None else []
            if len(names) != len(tgt.elts):
                # Arity mismatch: bind what we can rather than silently
                # dropping the statement, and leave the rest unbound so a
                # later use reports an undefined value instead of a wrong one.
                names = names[:len(tgt.elts)]
            # `all(isinstance(...))` above does not narrow the element type for
            # the checker, so bind through an explicitly-typed list.
            targets = [e for e in tgt.elts if isinstance(e, ast.Name)]
            for target, ssa in zip(targets, names):
                self._name_alias[target.id] = ssa
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
        if node.value is None:
            return
        # Capture the returned SSA value(s) + type(s) so the function declares
        # real outputs (tuple returns → multiple values). If any element can't
        # be lowered to an SSA value, OR it resolves to a value that isn't
        # actually defined in this function body (e.g. it was produced inside an
        # unlowered control-flow region — the while_loop / dynamic-for eager
        # fallback), leave returns unset so the function emits a value-less
        # `return`, as before. Declaring a return value that the verifier can't
        # find a definition for would raise GRAPH_IR_RETURN_UNDEFINED.
        elts = (node.value.elts if isinstance(node.value, ast.Tuple)
                else [node.value])
        names: List[str] = []
        types: List[IRType] = []
        for elt in elts:
            ssa = self._emit_expr(elt)
            if ssa is None or not self._is_defined_value(ssa):
                self.return_values = []
                self.return_result_types = []
                self.generic_visit(node)
                return
            names.append(ssa)
            types.append(self._value_types.get(ssa, TENSOR_OPAQUE))
        self.return_values = names
        self.return_result_types = types

    def _is_defined_value(self, ssa: str) -> bool:
        """True if ``ssa`` names a value defined in this function body — an
        argument or an emitted op result. Mirrors the verifier's notion of a
        defined value so we never declare a return the verifier would reject."""
        if not ssa.startswith("%"):
            return False
        if ssa in self._value_types:
            return True
        return ssa[1:] in self._arg_names

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
                self._unsupported(
                    node,
                    "call target is not a registered Graph operation",
                )
                return None
            # One SSA name per declared result. A multi-result op given a
            # single name would print `%v0 = ... -> (tK, tV)`, which is not
            # valid MLIR and leaves V unnameable.
            if len(op.inferred_types) > 1:
                names = [result_name or self._fresh()]
                names += [self._fresh() for _ in op.inferred_types[1:]]
                op.result = ",".join(names)
                for name, typ in zip(names, op.inferred_types):
                    self._value_types[f"%{name}"] = typ
                self.ops.append(op)
                return f"%{names[0]}"
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
        kwargs: Dict[str, Any] = {}
        # Positional scalar-literal args become attributes, not operands. Many
        # ops take a tensor + a scalar parameter positionally (top_k(x, k),
        # argmax(x, axis), ...); without this they'd be emitted as a bogus "%?"
        # operand and the op (and the whole graph) would be dropped. The op's
        # trailing scalar param names are listed in _POSITIONAL_ATTR_PARAMS; a
        # scalar positional arg binds to the next unused name. (Tensor args still
        # emit as operands; kwargs below still win for explicitly-named params.)
        attr_params = _POSITIONAL_ATTR_PARAMS.get(mlir_name, ())
        scalar_seen = 0
        for arg in call.args:
            # Variadic tensor-list operand (the cat/stack pattern:
            # ``cat([a, b], axis=…)``). A list/tuple of defined tensor values
            # flattens into individual operands so the op lowers to
            # ``tessera.cat %a, %b {axis = …}`` instead of a dropped "%?".
            if isinstance(arg, (ast.List, ast.Tuple)):
                raw = [self._emit_expr(e) for e in arg.elts]
                if raw and all(
                        v is not None and self._is_defined_value(v) for v in raw):
                    elts: list[str] = [v for v in raw if v is not None]
                    for v in elts:
                        operands.append(v)
                        operand_types.append(
                            str(self._value_types.get(v, TENSOR_OPAQUE)))
                    continue
            operand = self._emit_expr(arg)
            if operand is not None and self._is_defined_value(operand):
                operands.append(operand)
                operand_types.append(str(self._value_types.get(operand, TENSOR_OPAQUE)))
                continue
            literal: Any = None
            try:
                literal = ast.literal_eval(arg)
            except (ValueError, TypeError, SyntaxError):
                # a positional scalar passed by name (closure / global constant)
                if isinstance(arg, ast.Name) and arg.id in self._const_env:
                    literal = self._const_env[arg.id]
            if literal is not None and scalar_seen < len(attr_params):
                kwargs[attr_params[scalar_seen]] = literal
                scalar_seen += 1
            else:
                operands.append(operand if operand else "%?")
                operand_types.append(str(self._value_types.get(operand or "%?", TENSOR_OPAQUE)))

        # Keywords are bound BEFORE the result type is inferred. They used to be
        # bound after, which quietly disabled every attribute-reading shape rule
        # for the normal calling convention: `cast(x, dtype="bf16")` inferred
        # from an empty attribute set and returned `tensor<*x?>` instead of bf16,
        # and `adam(..., state_dtype="fp32")` never saw `state_dtype` at all.
        # The rules were correct; they were being consulted too early.
        kw_operands: Dict[str, str] = {}
        for kw in call.keywords:
            if kw.arg is None:
                continue
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
                continue
            except (ValueError, TypeError):
                pass
            value_name = self._emit_expr(kw.value)
            # A keyword that binds a DEFINED SSA value is a tensor operand, not
            # an attribute. Recording it as `{scores = "%s"}` produced an
            # attribute holding an SSA name -- syntactically fine, semantically
            # nothing, and it erased the edge from the operand list where every
            # dataflow consumer looks. Derive the split from what the value IS
            # (Decision #30) rather than from the call syntax.
            if value_name and self._is_defined_value(value_name):
                kw_operands[kw.arg] = value_name
            else:
                kwargs[kw.arg] = value_name or "?"

        # The public ``sum`` spelling canonicalizes to the parameterized
        # ``tessera.reduce`` ODS operation. Preserve the selected combiner;
        # otherwise the Graph operation is both unverifiable (``kind`` is
        # required) and semantically ambiguous to autodiff/lowering.
        if mlir_name == "tessera.reduce" and "kind" not in kwargs:
            surface = name.rsplit(".", 1)[-1]
            if surface == "sum":
                kwargs["kind"] = "sum"

        # Declared order first, so the operand list does not depend on the order
        # the caller happened to write the keywords. Anything undeclared is
        # appended by sorted name -- deterministic, and flagged by
        # `test_keyword_operand_vocabulary` as vocabulary that needs declaring.
        if kw_operands:
            declared = _KEYWORD_OPERANDS.get(mlir_name, ())
            ordered = [n for n in declared if n in kw_operands]
            ordered += sorted(n for n in kw_operands if n not in declared)
            for name in ordered:
                value = kw_operands[name]
                operands.append(value)
                operand_types.append(
                    str(self._value_types.get(value, TENSOR_OPAQUE)))

        inferred_operands = [
            self._value_types.get(operand, TENSOR_OPAQUE) for operand in operands
        ]
        _canonicalize_spectral_attrs(mlir_name, inferred_operands, kwargs)
        if mlir_name == "tessera.depth_attn":
            # Resolve public defaults at the frontend boundary. The typed Graph
            # op itself keeps these attributes required and therefore never
            # delegates an absent numeric decision to a backend.
            kwargs.setdefault("eps", 1.0e-6)
            kwargs.setdefault(
                "numeric_policy",
                {"storage": "fp32", "softmax": "fp32", "accum": "fp32"},
            )

        result_types = _infer_result_types(
            mlir_name,
            inferred_operands,
            attrs=kwargs)
        result_type = result_types[0]

        # Structural view ops: derive the result type from the shape/axes/perm
        # attr (now fully bound) instead of the input-type fallback, so a static
        # reshape/view/squeeze/… emits its real output shape and is not erased by
        # the identity folder.
        _struct = _struct_result_type(
            mlir_name,
            self._value_types.get(operands[0]) if operands else None,
            kwargs)
        if _struct is not None:
            result_type = _struct
            result_types = (_struct,)

        # MLIR spells a multi-result signature `-> (t0, t1)`; a single result
        # has no parentheses. Emitting the tuple form for one result would be
        # a different (and wrong) type.
        printed = (str(result_type) if len(result_types) == 1
                   else "(" + ", ".join(str(t) for t in result_types) + ")")
        return IROp(
            result=None,  # filled in by caller
            op_name=mlir_name,
            operands=operands,
            operand_types=operand_types,
            result_type=printed,
            kwargs=kwargs,
            source_span=_span_from_ast(call),
            inferred_type=result_type,
            inferred_types=tuple(result_types),
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


# ─────────────────────────────────────────────────────────────────────────────
# W1.2 — shape rules, dispatched by NAME from the one registry
#
# These were a five-case if-chain ending in `return operand_types[0]`. Each is
# now a named rule the catalog can point at, so `primitive_coverage.shape_rule`
# has something real to auto-flip from instead of reporting an axis closed that
# nothing implemented (Decision #29).
#
# Behavior is unchanged by design: `unclassified` still resolves to
# same-as-first. The difference is that it is now a *declared, counted* status
# rather than an implicit default, so it can be driven to zero. See
# `op_catalog.unclassified_shape_ops()` — a shrink-only ratchet.
# ─────────────────────────────────────────────────────────────────────────────

def _shape_same_as_first(operand_types: List[IRType], attrs: Optional[Dict[str, Any]] = None) -> IRType:
    return operand_types[0]


def _shape_matmul_2d(operand_types: List[IRType], attrs: Optional[Dict[str, Any]] = None) -> IRType:
    if len(operand_types) < 2:
        return operand_types[0]
    lhs, rhs = operand_types[0], operand_types[1]
    dtype = lhs.dtype or rhs.dtype
    if lhs.rank == 2 and rhs.rank == 2:
        return tensor_ir_type((lhs.shape[0], rhs.shape[1]), dtype, layout=lhs.layout)
    return tensor_ir_type(("*",), dtype, layout=lhs.layout)


def _shape_es_population_features(
    operand_types: List[IRType], attrs: Optional[Dict[str, Any]] = None
) -> IRType:
    """Preserve population/row axes and replace the trailing feature extent."""
    x = operand_types[0]
    if x.rank is None or x.rank < 2:
        return tensor_ir_type(("*",), x.dtype, layout=x.layout)
    out_dim = _dim((attrs or {}).get("out_dim"))
    dims = list(x.shape)
    dims[-1] = str(out_dim) if out_dim is not None else "*"
    return tensor_ir_type(tuple(dims), x.dtype, layout=x.layout)


def _coalition_players(extent: Any) -> Optional[int]:
    """Derive n from a lattice axis extent 2^n; None when non-static or not a
    power of two (the op verifier fails closed at runtime — the shape rule
    reports what it can prove)."""
    try:
        size = int(extent)
    except (TypeError, ValueError):
        return None
    if size <= 0 or (size & (size - 1)) != 0:
        return None
    return size.bit_length() - 1


def _shape_coalition_marginal(
    operand_types: List[IRType], attrs: Optional[Dict[str, Any]] = None
) -> IRType:
    """[..., 2^n] → [..., n, 2^n]; storage dtype preserved (Decision #15a —
    the lattice family accumulates in fp64 via numeric_policy, it does not
    change the tensor's storage dtype)."""
    x = operand_types[0]
    if x.rank is None or x.rank < 1:
        return tensor_ir_type(("*",), x.dtype, layout=x.layout)
    n = _coalition_players(x.shape[-1])
    dims = list(x.shape)
    dims.insert(len(dims) - 1, str(n) if n is not None else "*")
    return tensor_ir_type(tuple(dims), x.dtype, layout=x.layout)


def _shape_coalition_players_axis(
    operand_types: List[IRType], attrs: Optional[Dict[str, Any]] = None
) -> IRType:
    """[..., 2^n] → [..., n] (semivalue / boltzmann_value); storage dtype of
    the lattice operand preserved."""
    x = operand_types[0]
    if x.rank is None or x.rank < 1:
        return tensor_ir_type(("*",), x.dtype, layout=x.layout)
    n = _coalition_players(x.shape[-1])
    dims = list(x.shape[:-1]) + [str(n) if n is not None else "*"]
    return tensor_ir_type(tuple(dims), x.dtype, layout=x.layout)


def _shape_tridiagonal_rhs(
    operand_types: List[IRType], attrs: Optional[Dict[str, Any]] = None
) -> IRType:
    """Solve result: shape and storage dtype of the right-hand side (the
    fourth operand); the three diagonals parameterize the system."""
    return operand_types[3] if len(operand_types) >= 4 else operand_types[-1]


def _shape_segment_mex(
    operand_types: List[IRType], attrs: Optional[Dict[str, Any]] = None
) -> IRType:
    """Ragged segmented mex: (values, segment_ids) + num_segments attr →
    int64[num_segments]. num_segments is a semantic key: absent means the
    static shape is unknown, never defaulted (Decision #21a)."""
    num = _dim((attrs or {}).get("num_segments"))
    return tensor_ir_type((str(num) if num is not None else "*",), "i64")


def _shape_batched_gemm_3d(operand_types: List[IRType], attrs: Optional[Dict[str, Any]] = None) -> IRType:
    # Rank-3 C[b] = A[b] @ B[b]: result is B×M×N from A's (B,M) + B's N.
    if len(operand_types) < 2:
        return operand_types[0]
    lhs, rhs = operand_types[0], operand_types[1]
    dtype = lhs.dtype or rhs.dtype
    if lhs.rank == 3 and rhs.rank == 3:
        return tensor_ir_type((lhs.shape[0], lhs.shape[1], rhs.shape[2]),
                              dtype, layout=lhs.layout)
    return tensor_ir_type(("*",), dtype, layout=lhs.layout)


def _shape_same_shape_bool(operand_types: List[IRType], attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """First operand's shape, but a BOOLEAN result dtype.

    Comparisons and logical connectives yield a predicate, not a value in the
    operand's dtype. Under the old `return operand_types[0]` fallback,
    `tessera.eq` on two f32 tensors reported `dtype=fp32` -- a silently wrong
    result type that no test caught, because the axis was reported closed.
    Classifying the shape taxonomy is what surfaced it.
    """
    first = operand_types[0]
    return tensor_ir_type(first.shape, "bool", layout=first.layout)


def _shape_reduce_all(operand_types: List[IRType], attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """Full reduction to a scalar, preserving the operand's STORAGE dtype.

    Accumulating internally in higher precision is a `numeric_policy` concern
    (Decision #15a); it must not change the result's storage dtype, or a bf16
    pipeline silently materializes f32/f64 and loses the accelerator's reason
    for using bf16 at all.
    """
    first = operand_types[0]
    return tensor_ir_type((), first.dtype, layout=first.layout)


def _shape_reduce_all_index(operand_types: List[IRType], attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """Full reduction yielding an INDEX (argmax/argmin/count_nonzero)."""
    from .op_catalog import INDEX_DTYPE

    first = operand_types[0]
    return tensor_ir_type((), INDEX_DTYPE, layout=first.layout)


def _shape_same_shape_index(operand_types: List[IRType], attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """Operand shape, declared index dtype (argsort, popcount).

    The width comes from `INDEX_DTYPE` rather than a literal, because it was
    previously spelled out here, again in `_shape_reduce_all_index`, and
    derived a third way by `popcount` from whichever NumPy was installed. Three
    spellings of one convention is how they drift apart.
    """
    from .op_catalog import INDEX_DTYPE

    first = operand_types[0]
    return tensor_ir_type(first.shape, INDEX_DTYPE, layout=first.layout)


def _shape_flatten(operand_types: List[IRType], attrs: Optional[Dict[str, Any]] = None) -> IRType:
    first = operand_types[0]
    if first.rank is None:
        return tensor_ir_type(("*",), first.dtype, layout=first.layout)
    total = 1
    for d in first.shape:
        try:
            total *= int(d)
        except (TypeError, ValueError):
            return tensor_ir_type(("*",), first.dtype, layout=first.layout)
    return tensor_ir_type((str(total),), first.dtype, layout=first.layout)


#: Real storage width -> the complex type whose COMPONENTS have that width.
#: These spellings are `planned_gated` in `tessera.dtype`, which the exemption
#: read as "declaring it conflicts with the dtype capability contract". It does
#: not: the contract has an explicit path for planned/gated dtypes
#: (`allow_planned_gated=True` plus `metadata.dtype_status`), and naming one in
#: a shape rule is not the same as claiming a backend implements it. Promoting
#: complex to CANONICAL would be a capability change; this is not that.
_COMPLEX_FOR_REAL = {"fp64": "complex128", "f64": "complex128"}
_REAL_FOR_COMPLEX = {"complex128": "fp64", "complex64": "fp32"}


def _complex_of(dtype: Optional[str]) -> str:
    """The complex type for this operand, PRESERVING an existing complex width.

    An already-complex operand keeps its own width. Without that branch,
    `_COMPLEX_FOR_REAL` recognised only fp64/f64, so a complex128 operand fell
    through to the complex64 default and `complex_exp(complex128)` was typed
    complex64 -- silently narrowing a double-precision computation, and
    assigning it the wrong ABI at the boundary. The rule has to answer for both
    directions: promote a real, preserve a complex.

    Hard-coding `complex64` (as this did) is wrong for an f64 signal, and
    hard-coding `complex128` -- which is what the numpy-backed reference
    actually returned for every input -- is the `cos(int32) -> float64` defect
    again: the host library's precision choice overriding the compiler's. The
    family's own `_spectral_policy` declares `storage="fp32"`.
    """
    name = dtype or ""
    if name in _REAL_FOR_COMPLEX:      # already complex -- keep its width
        return name
    return _COMPLEX_FOR_REAL.get(name, "complex64")


def _shape_complex_same(operand_types: List[IRType], attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """Same shape, complex result with matching component width (fft / ifft)."""
    first = operand_types[0]
    return tensor_ir_type(first.shape, _complex_of(first.dtype), layout=first.layout)


def _transform_axis(t: IRType, attrs: Optional[Dict[str, Any]]) -> Optional[int]:
    """The axis a spectral op transforms, normalised to a positive index.

    Mirrors the reference's `_axis_from_axes`: an explicit `axes` tuple wins and
    its LAST entry is the transformed axis, otherwise `axis` (default -1).

    Reading this matters only for the rules that RESIZE an axis. `fft`/`ifft`
    preserve shape, so they were correct while ignoring it; `rfft` and `irfft`
    are not -- `rfft` on `8x16` with `axis=0` is `5x16`, and a rule that always
    resizes the trailing dimension would claim `8x9`.
    """
    attrs = attrs or {}
    axes = attrs.get("axes")
    raw = tuple(axes)[-1] if isinstance(axes, (tuple, list)) and axes else attrs.get("axis", -1)
    axis = _dim(raw)
    if axis is None or t.rank is None or not t.shape:
        return None
    if axis < 0:
        axis += len(t.shape)
    return axis if 0 <= axis < len(t.shape) else None


def _shape_rfft(operand_types: List[IRType], attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """Real FFT: the TRANSFORMED axis becomes n//2 + 1, result complex.

    Only the non-redundant half of a real spectrum is kept, since a real
    signal's transform is conjugate-symmetric. Which axis is halved comes from
    `axis`/`axes`, not from an assumption that it is the last one.
    """
    first = operand_types[0]
    complex_dtype = _complex_of(first.dtype)
    axis = _transform_axis(first, attrs)
    if first.rank is None or not first.shape or axis is None:
        return tensor_ir_type(("*",), complex_dtype, layout=first.layout)
    n = _dim(first.shape[axis])
    if n is None:
        return tensor_ir_type(("*",), complex_dtype, layout=first.layout)
    dims = list(first.shape)
    dims[axis] = str(n // 2 + 1)
    return tensor_ir_type(tuple(dims), complex_dtype, layout=first.layout)


def _shape_complex_from_coords(operand_types: List[IRType],
                               attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`stereographic`: `(..., 3)` real coordinates -> `(...)` complex.

    One complex value per POINT, so the trailing coordinate axis is consumed.
    `complex_same` preserves shape and therefore reported
    `tensor<4x3xcomplex64>` for a `tensor<4x3xfp32>` operand -- a wrong result
    shape carried into verification and lowering, not merely a mislabel.
    """
    first = operand_types[0]
    complex_dtype = _complex_of(first.dtype)
    if first.rank is None or not first.shape:
        return tensor_ir_type(("*",), complex_dtype, layout=first.layout)
    return tensor_ir_type(first.shape[:-1], complex_dtype, layout=first.layout)


def _shape_complex_to_real(operand_types: List[IRType],
                           attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`complex_abs` / `complex_arg`: a magnitude or an angle is REAL.

    They sit in the `elementwise` kind, whose default rule is `same_as_first`,
    and that default is not merely imprecise here -- the storage-dtype wrapper
    ENFORCES it, casting `complex_abs`'s float32 magnitude back to complex64.
    A wrong declaration is worse than none once something acts on it.
    """
    first = operand_types[0]
    real = _REAL_FOR_COMPLEX.get(first.dtype or "", first.dtype or "fp32")
    return tensor_ir_type(first.shape, real, layout=first.layout)


def _shape_irfft(operand_types: List[IRType],
                 attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """Inverse real FFT: complex -> REAL, trailing axis 2 * (n - 1).

    `irfft` was exempted with its three siblings under one sentence -- "returns
    complex64, which is planned_gated per Decision #15a" -- which is true of
    them and simply wrong here: it is the member of the family that returns
    real values. The same one-reason-for-N-ops error that hid `kv_cache.read`
    among the cache mutators.

    The inverse of `rfft`'s `n // 2 + 1`. It is the inverse of the EVEN-length
    case; an odd-length original is not recoverable from the spectrum alone,
    which is why `irfft` takes an explicit `n` -- honoured here when given.
    """
    first = operand_types[0]
    real_dtype = _REAL_FOR_COMPLEX.get(first.dtype or "", "fp32")
    axis = _transform_axis(first, attrs)
    if first.rank is None or not first.shape or axis is None:
        return tensor_ir_type(("*",), real_dtype, layout=first.layout)
    dims = list(first.shape)
    explicit = _dim(
        (attrs or {}).get("logical_length", (attrs or {}).get("n"))
    )
    if explicit is not None:
        dims[axis] = str(explicit)
        return tensor_ir_type(tuple(dims), real_dtype, layout=first.layout)
    n = _dim(first.shape[axis])
    if n is None:
        return tensor_ir_type(("*",), real_dtype, layout=first.layout)
    dims[axis] = str(2 * (n - 1))
    return tensor_ir_type(tuple(dims), real_dtype, layout=first.layout)


def _shape_from_shape_attr(operand_types: List[IRType],
                           attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """Result shape comes from an ATTRIBUTE, not from the operands.

    `reshape`, `view`, `broadcast`, `expand`, `tile_view` are all of this form.
    The previous signature -- `(op_name, operand_types)` -- could not express
    them at all: the target shape simply is not in any operand's type. They
    were not "hard to classify", they were *inexpressible*, which is why
    widening the signature was the right fix rather than inventing a rule that
    guesses.
    """
    first = operand_types[0]
    attrs = attrs or {}
    for key in ("shape", "sizes", "new_shape", "target_shape"):
        value = attrs.get(key)
        if value is None:
            continue
        try:
            dims = tuple(str(int(d)) for d in value)
        except (TypeError, ValueError):
            continue
        return tensor_ir_type(dims, first.dtype, layout=first.layout)
    # Attribute absent at trace time (dynamic shape) -- unranked, not a guess.
    return tensor_ir_type(("*",), first.dtype, layout=first.layout)


def _shape_cast(operand_types: List[IRType],
                attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """Same shape; the result DTYPE comes from an attribute.

    This is the storage-dtype half of Decision #15a made explicit: a cast is
    the one elementwise op whose result dtype is not the operand's, and in a
    reduced-precision pipeline getting it wrong is what silently re-inflates
    a bf16 tensor.
    """
    first = operand_types[0]
    attrs = attrs or {}
    target = attrs.get("dtype") or attrs.get("to") or attrs.get("target_dtype")
    if target is None:
        return first
    try:
        from ..dtype import canonicalize_dtype
        target = canonicalize_dtype(str(target).strip("\"'"))
    except Exception:
        target = str(target)
    return tensor_ir_type(first.shape, target, layout=first.layout)


def _shape_quantize_per_tensor(operand_types: List[IRType],
                               attrs: Optional[Dict[str, Any]] = None):
    """(codes, scale) with ONE scale for the whole tensor.

    `quantize_fp8` / `fp6` / `fp4` return a tuple: codes shaped like the input,
    plus a rank-0 scale. Declaring this as `same_as_first` was a false contract
    -- it claims a single tensor for a two-result op.

    The codes come back as f32 today (fake-quant), NOT as native fp8/fp4
    storage. That is deliberate at the reference level and is a BACKEND-PATH
    question: `fp8_e4m3` and `fp4_e2m1` are canonical dtypes and the per-target
    contracts already model them (gfx1151 `unsupported`, x86 `emulated`), so
    the type system can carry real sub-byte storage as soon as a lowering
    produces it. The rule states what the op returns rather than what we wish
    it returned.
    """
    first = operand_types[0]
    codes = tensor_ir_type(first.shape, first.dtype, layout=first.layout)
    scale = tensor_ir_type((), first.dtype, layout=first.layout)
    return (codes, scale)


def _shape_quantize_per_block(operand_types: List[IRType],
                              attrs: Optional[Dict[str, Any]] = None):
    """(codes, per-BLOCK scale) — the micro-scaled形 NVFP4 uses.

    NVFP4 carries one scale per block of `block_size` elements along the last
    axis (16 by default; a non-divisible last dim is a named error), so the
    scale is rank-preserving with the last axis divided rather than rank-0.
    Folding it into the per-tensor rule would misstate the format that
    Blackwell actually implements.
    """
    first = operand_types[0]
    attrs = attrs or {}
    codes = tensor_ir_type(first.shape, first.dtype, layout=first.layout)
    block = attrs.get("block_size", 16)
    if first.rank is None or not first.shape:
        return (codes, tensor_ir_type(("*",), first.dtype, layout=first.layout))
    try:
        blocks = max(1, int(first.shape[-1]) // int(block))
    except (TypeError, ValueError):
        return (codes, tensor_ir_type(("*",), first.dtype, layout=first.layout))
    scale = tensor_ir_type(first.shape[:-1] + (str(blocks),), first.dtype,
                           layout=first.layout)
    return (codes, scale)


def _shape_optimizer_step(operand_types: List[IRType],
                          attrs: Optional[Dict[str, Any]] = None):
    """(updated_param, moment1, moment2) — param dtype and STATE dtype differ.

    This was parked as "deliberately undeclared" on the grounds that an
    optimizer keeping f32 master state with bf16 parameters is correct mixed
    precision rather than a bug. True — but it was never really an exception:
    it was a VOCABULARY GAP. The contract is exactly expressible once rules can
    (a) return a tuple and (b) read an attribute:

        result[0] = param  -> operand 0's shape and storage dtype
        result[1..2] = moments -> operand 0's shape, `state_dtype` attribute

    Declaring it is strictly better than exempting it, because the exemption
    silently also covered the case where an optimizer wrongly rounds its state
    down to the parameter dtype. Now that is a verifiable difference.
    """
    param = operand_types[0]
    attrs = attrs or {}
    state = attrs.get("state_dtype") or "fp32"
    try:
        from ..dtype import canonicalize_dtype

        state = canonicalize_dtype(str(state).strip("\"'"))
    except Exception:
        state = str(state)
    moment = tensor_ir_type(param.shape, state, layout=param.layout)
    return (
        tensor_ir_type(param.shape, param.dtype, layout=param.layout),
        moment,
        moment,
    )


def _shape_select_from_second(operand_types: List[IRType],
                              attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """Select along a candidate axis of operand 1, keyed by operand 0.

    `ebm_self_verify(energies: (B, K), candidates: (B, K, D)) -> (B, D)`:
    operand 0 only *scores* the candidates, operand 1 carries the data. So both
    the shape and the STORAGE DTYPE come from operand 1.

    This is why `dataOperands`-style "operand 0 is the tensor" assumptions have
    to be per-op rather than global -- `reduce_trailing` on operand 0 predicted
    `(B,)` with the energies' dtype, which is wrong on both counts.
    """
    if len(operand_types) < 2:
        return operand_types[0]
    candidates = operand_types[1]
    if candidates.rank is None or len(candidates.shape) < 2:
        return tensor_ir_type(("*",), candidates.dtype, layout=candidates.layout)
    # Drop the candidate axis (axis 1), keep batch and feature dims.
    dims = (candidates.shape[0],) + tuple(candidates.shape[2:])
    return tensor_ir_type(dims, candidates.dtype, layout=candidates.layout)


def _collective_axis(operand_types: List[IRType],
                     attrs: Optional[Dict[str, Any]]) -> tuple[int, Optional[str]]:
    """(tensor axis that scales, mesh axis name) for a collective.

    `all_gather(x, axis="dp")` overloads one parameter: an `int` names a TENSOR
    axis, a `str` names a MESH axis. When it is a mesh axis the tensor axis is
    0, matching the `Block(mesh_axes=("dp",)) -> partition=(0,)` convention
    (CLAUDE.md, "Domain & distribution").
    """
    axis = (attrs or {}).get("axis", "dp")
    if isinstance(axis, int) and not isinstance(axis, bool):
        return axis, None
    return 0, str(axis).strip("\"'")


def _scaled_collective_shape(operand_types: List[IRType],
                             attrs: Optional[Dict[str, Any]],
                             mesh: Optional[Mapping[str, int]],
                             *, multiply: bool) -> IRType:
    """Shared body for all_gather (multiply) and reduce_scatter (divide).

    FAILS CLOSED. When the mesh size is unknown the scaled dimension becomes
    `?` -- explicitly unknown -- rather than the operand's own extent. That
    distinction is the whole point of this rule: returning the operand shape
    is not a neutral fallback, it is the positive claim `world_size == 1`.
    Both ops previously did exactly that, and it went undetected because the
    reference collectives are single-rank no-op stubs, so a probe at
    world_size=1 confirmed the wrong rule. Decision #30: a rule that cannot
    derive a fact must fail closed, not assume the convenient answer.
    """
    first = operand_types[0]
    tensor_axis, mesh_axis = _collective_axis(operand_types, attrs)
    if first.rank is None or not first.shape:
        return first
    dims = list(first.shape)
    if tensor_axis < 0:
        tensor_axis += len(dims)
    if not 0 <= tensor_axis < len(dims):
        return first
    size = (mesh or {}).get(mesh_axis) if mesh_axis else None
    scaled = "?"
    if size:
        try:
            extent = int(dims[tensor_axis])
            scaled = str(extent * int(size)) if multiply else (
                str(extent // int(size)) if extent % int(size) == 0 else "?"
            )
        except (TypeError, ValueError):
            scaled = "?"
    dims[tensor_axis] = scaled
    return tensor_ir_type(tuple(dims), first.dtype, layout=first.layout)


def _shape_all_gather(operand_types: List[IRType],
                      attrs: Optional[Dict[str, Any]] = None,
                      mesh: Optional[Mapping[str, int]] = None) -> IRType:
    """Gathered axis grows by the mesh extent: `d -> d * world_size`."""
    return _scaled_collective_shape(operand_types, attrs, mesh, multiply=True)


def _shape_reduce_scatter(operand_types: List[IRType],
                          attrs: Optional[Dict[str, Any]] = None,
                          mesh: Optional[Mapping[str, int]] = None) -> IRType:
    """Scattered axis shrinks by the mesh extent: `d -> d / world_size`."""
    return _scaled_collective_shape(operand_types, attrs, mesh, multiply=False)


def _dim(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unknown_like(t: IRType, dtype: Optional[str] = None) -> IRType:
    return tensor_ir_type(("*",), dtype or t.dtype, layout=t.layout)


def _shape_matmul_trailing(operand_types: List[IRType],
                           attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """Contract operand 0's LAST axis with operand 1's last, keeping leading dims.

    `linear_general((B,S,D), (D,N)) -> (B,S,N)` and the latent-KV
    compress/expand pair. Distinct from `matmul_2d`, which is rank-2 only and
    would report `(*)` for the rank-3 activations that dominate real models.

    Also covers the grouped/MoE forms whose weight is `(E, D, N)`: the expert
    axis is not a batch axis of the activation, so the result still takes only
    operand 1's LAST dim. Reading operand 1's first dim instead would be right
    for `(D, N)` and wrong for `(E, D, N)` -- which is why this keys on the
    last axis rather than "the other one".
    """
    if len(operand_types) < 2:
        return operand_types[0]
    lhs, rhs = operand_types[0], operand_types[1]
    dtype = lhs.dtype or rhs.dtype
    if lhs.rank is None or rhs.rank is None or not lhs.shape or not rhs.shape:
        return _unknown_like(lhs, dtype)
    return tensor_ir_type(lhs.shape[:-1] + (rhs.shape[-1],), dtype,
                          layout=lhs.layout)


def _shape_same_as_second(operand_types: List[IRType],
                          attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """Result mirrors operand 1 (the right-hand side / the thing being scored).

    `tri_solve(A, b) -> b`-shaped, `cholesky_solve(L, b) -> b`-shaped,
    `target_verify(tokens, logits) -> logits`-shaped. Operand 0 is the
    operator or the key, not the value, so `same_as_first` is wrong on both
    shape and dtype -- the same asymmetry `select_from_second` exists for.
    """
    if len(operand_types) < 2:
        return operand_types[0]
    return operand_types[1]


def _shape_conv_spatial(operand_types: List[IRType],
                        attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """NHWC / NDHWC convolution, valid padding: `out = in - k + 1` per axis.

    One rule for conv2d and conv3d: the spatial rank is read from the operands
    rather than fixed, so `(N, *spatial, Cin)` against `(*kernel, Cin, Cout)`
    works for both. A separate rule per rank would duplicate the arithmetic and
    let the two drift.

    Falls back to unknown when stride/padding are non-default rather than
    guessing -- the caller can pass them, and a rule that ignores stride would
    be confidently wrong by exactly the stride factor.
    """
    if len(operand_types) < 2:
        return operand_types[0]
    x, w = operand_types[0], operand_types[1]
    attrs = attrs or {}
    dtype = x.dtype or w.dtype
    if x.rank is None or w.rank is None or x.rank != w.rank or x.rank < 3:
        return _unknown_like(x, dtype)
    if _dim(attrs.get("stride", 1)) != 1 or _dim(attrs.get("padding", 0)) != 0:
        return _unknown_like(x, dtype)
    spatial = x.rank - 2
    dims: list = [x.shape[0]]
    for axis in range(spatial):
        n, k = _dim(x.shape[1 + axis]), _dim(w.shape[axis])
        dims.append(str(n - k + 1) if n is not None and k is not None else "?")
    dims.append(w.shape[-1])
    return tensor_ir_type(tuple(dims), dtype, layout=x.layout)


def _shape_index_along_axis(operand_types: List[IRType],
                            attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`gather` / `index_select`: the indexed axis becomes len(indices)."""
    if len(operand_types) < 2:
        return operand_types[0]
    x, idx = operand_types[0], operand_types[1]
    axis = _dim((attrs or {}).get("axis", 0)) or 0
    if x.rank is None or idx.rank is None or not idx.shape:
        return _unknown_like(x)
    if axis < 0:
        axis += len(x.shape)
    if not 0 <= axis < len(x.shape):
        return _unknown_like(x)
    dims = list(x.shape)
    dims[axis] = idx.shape[0]
    return tensor_ir_type(tuple(dims), x.dtype, layout=x.layout)


def _shape_drop_axis(operand_types: List[IRType],
                     attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`select(x, i, axis)` removes the axis entirely (unlike a size-1 slice)."""
    x = operand_types[0]
    axis = _dim((attrs or {}).get("axis", 0)) or 0
    if x.rank is None or not x.shape:
        return _unknown_like(x)
    if axis < 0:
        axis += len(x.shape)
    if not 0 <= axis < len(x.shape):
        return _unknown_like(x)
    return tensor_ir_type(x.shape[:axis] + x.shape[axis + 1:], x.dtype,
                          layout=x.layout)


def _shape_insert_axis(operand_types: List[IRType],
                       attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`unsqueeze`: a length-1 axis at `axes`."""
    x = operand_types[0]
    raw = (attrs or {}).get("axes", (attrs or {}).get("axis", 0))
    axis = _dim(raw if not isinstance(raw, (tuple, list)) else
                (raw[0] if raw else 0))
    if x.rank is None or axis is None:
        return _unknown_like(x)
    if axis < 0:
        axis += len(x.shape) + 1
    if not 0 <= axis <= len(x.shape):
        return _unknown_like(x)
    return tensor_ir_type(x.shape[:axis] + ("1",) + x.shape[axis:], x.dtype,
                          layout=x.layout)


def _shape_from_slice_sizes(operand_types: List[IRType],
                            attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`slice` / `dynamic_slice`: the result IS `slice_sizes`."""
    x = operand_types[0]
    sizes = (attrs or {}).get("slice_sizes")
    if not isinstance(sizes, (tuple, list)) or not sizes:
        return _unknown_like(x)
    return tensor_ir_type(tuple(str(s) for s in sizes), x.dtype, layout=x.layout)


def _shape_pad(operand_types: List[IRType],
               attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """Each axis grows by its (before, after) pad."""
    x = operand_types[0]
    widths = (attrs or {}).get("pad_width")
    if x.rank is None or not isinstance(widths, (tuple, list)):
        return _unknown_like(x)
    dims = []
    for axis, extent in enumerate(x.shape):
        n = _dim(extent)
        pad = widths[axis] if axis < len(widths) else 0
        lo, hi = pad if isinstance(pad, (tuple, list)) else (pad, pad)
        lo, hi = _dim(lo), _dim(hi)
        # Explicit `is None` rather than `None not in (...)`: the latter reads
        # fine and does not narrow, so mypy still sees `int | None` in the sum.
        if n is None or lo is None or hi is None:
            dims.append("?")
        else:
            dims.append(str(n + lo + hi))
    return tensor_ir_type(tuple(dims), x.dtype, layout=x.layout)


def _shape_scores_per_block(operand_types: List[IRType],
                            attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """Query x key-BLOCK scores: `(B, H, S_q, num_blocks)`.

    The MSA / memory index branches score each KV block, so the trailing axis
    is a block count -- `S_k // block_size` -- not the head dim. With no
    `block_size` the block count is unknown and says so.
    """
    if len(operand_types) < 2:
        return operand_types[0]
    q, k = operand_types[0], operand_types[1]
    if q.rank is None or k.rank is None or len(q.shape) < 2 or len(k.shape) < 2:
        return _unknown_like(q)
    s_k = _dim(k.shape[-2])
    block = _dim((attrs or {}).get("block_size", 1)) or 1
    blocks = str(s_k // block) if s_k is not None and block else "?"
    return tensor_ir_type(q.shape[:-1] + (blocks,), q.dtype, layout=q.layout)


def _shape_scores_per_block_mask(operand_types: List[IRType],
                                 attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`memory_index_select`: the same block grid as the scores, but a MASK.

    Shares its shape with `scores_per_block` and differs only in dtype, which
    is exactly the case a shape-only check cannot see -- the probe reported
    identical shapes and the two rules were indistinguishable until dtype was
    compared. It selects blocks; it does not score them.
    """
    scores = _shape_scores_per_block(operand_types, attrs)
    return tensor_ir_type(scores.shape, "bool", layout=scores.layout)


def _shape_reduce_trailing_index(operand_types: List[IRType],
                                 attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """Drop the trailing axis, result is an INDEX (argmax-like selection)."""
    from .op_catalog import INDEX_DTYPE

    x = operand_types[0]
    if x.rank is None or not x.shape:
        return _unknown_like(x, INDEX_DTYPE)
    return tensor_ir_type(x.shape[:-1], INDEX_DTYPE, layout=x.layout)


def _shape_reduce_trailing_bool(operand_types: List[IRType],
                                attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """Drop the trailing axis, result is a boolean mask (`mor_partition`)."""
    x = operand_types[0]
    if x.rank is None or not x.shape:
        return _unknown_like(x, "bool")
    return tensor_ir_type(x.shape[:-1], "bool", layout=x.layout)


def _shape_top_k(operand_types: List[IRType],
                 attrs: Optional[Dict[str, Any]] = None):
    """`(values, indices)` — both operand-shaped with the axis cut to k.

    The indices half is why this cannot be `same_shape_index`: the two results
    have the same SHAPE but different dtypes, and a single-tensor rule can
    state only one of them.
    """
    from .op_catalog import INDEX_DTYPE

    x = operand_types[0]
    k = _dim((attrs or {}).get("k"))
    axis = _dim((attrs or {}).get("axis", -1))
    if x.rank is None or not x.shape or k is None or axis is None:
        return (_unknown_like(x), _unknown_like(x, INDEX_DTYPE))
    if axis < 0:
        axis += len(x.shape)
    if not 0 <= axis < len(x.shape):
        return (_unknown_like(x), _unknown_like(x, INDEX_DTYPE))
    dims = list(x.shape)
    dims[axis] = str(k)
    return (tensor_ir_type(tuple(dims), x.dtype, layout=x.layout),
            tensor_ir_type(tuple(dims), INDEX_DTYPE, layout=x.layout))


def _shape_split_equal(operand_types: List[IRType],
                       attrs: Optional[Dict[str, Any]] = None):
    """`chunk` / `split` into N equal pieces along `axis`.

    Returns an N-tuple, so the count must be statically known. When it is not,
    a single unknown piece is returned rather than a guess at N -- an arity the
    op does not have is a worse claim than an unknown shape.
    """
    x = operand_types[0]
    attrs = attrs or {}
    n = _dim(attrs.get("chunks", attrs.get("indices_or_sections")))
    axis = _dim(attrs.get("axis", 0)) or 0
    if x.rank is None or not x.shape or n is None or n <= 0:
        return _unknown_like(x)
    if axis < 0:
        axis += len(x.shape)
    if not 0 <= axis < len(x.shape):
        return _unknown_like(x)
    extent = _dim(x.shape[axis])
    piece = str(extent // n) if extent is not None else "?"
    dims = list(x.shape)
    dims[axis] = piece
    part = tensor_ir_type(tuple(dims), x.dtype, layout=x.layout)
    return tuple(part for _ in range(n))


def _shape_split_halves(operand_types: List[IRType],
                        attrs: Optional[Dict[str, Any]] = None):
    """`rope_split`: the trailing axis is cut into a rope part and the rest.

    Both halves are returned; the split point is the `rope_dim` attribute, so
    they are NOT necessarily equal and must not be assumed so.
    """
    x = operand_types[0]
    rope = _dim((attrs or {}).get("rope_dim"))
    if x.rank is None or not x.shape:
        return (_unknown_like(x), _unknown_like(x))
    total = _dim(x.shape[-1])
    if rope is None or total is None:
        return (_unknown_like(x), _unknown_like(x))
    head = tensor_ir_type(x.shape[:-1] + (str(rope),), x.dtype, layout=x.layout)
    tail = tensor_ir_type(x.shape[:-1] + (str(total - rope),), x.dtype,
                          layout=x.layout)
    return (head, tail)


def _shape_qkv_projection(operand_types: List[IRType],
                          attrs: Optional[Dict[str, Any]] = None):
    """`(Q, K, V)` — one fused projection split three ways.

    `(B, S, D) x (D, 3*D) -> three (B, S, D)`. The fused weight's trailing axis
    is 3x the per-head width, so the result width is `rhs[-1] // 3` rather than
    the input's D; they coincide only in the square case.
    """
    if len(operand_types) < 2:
        return operand_types[0]
    x, w = operand_types[0], operand_types[1]
    if x.rank is None or w.rank is None or not x.shape or not w.shape:
        return tuple(_unknown_like(x) for _ in range(3))
    fused = _dim(w.shape[-1])
    width = str(fused // 3) if fused is not None else "?"
    part = tensor_ir_type(x.shape[:-1] + (width,), x.dtype, layout=x.layout)
    return (part, part, part)


def _shape_state_matrix(operand_types: List[IRType],
                        attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`linear_attn_state`: the recurrent state is `(B, H, D, D)`.

    Linear attention's state is an outer product of key and value features, so
    the sequence axis is contracted away and replaced by a second feature axis
    -- the one shape in this family that is NOT the query's.
    """
    x = operand_types[0]
    if x.rank is None or len(x.shape) < 2:
        return _unknown_like(x)
    return tensor_ir_type(x.shape[:-2] + (x.shape[-1], x.shape[-1]), x.dtype,
                          layout=x.layout)


def _shape_concat_trailing(operand_types: List[IRType],
                           attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`rope_merge`: the two halves rejoin along the trailing axis.

    The inverse of `split_halves`, and like it the parts are not equal -- the
    rope part is `rope_dim` wide and the pass-through is the remainder.
    """
    if len(operand_types) < 2:
        return operand_types[0]
    a, b = operand_types[0], operand_types[1]
    if a.rank is None or b.rank is None or not a.shape or not b.shape:
        return _unknown_like(a)
    lo, hi = _dim(a.shape[-1]), _dim(b.shape[-1])
    total = str(lo + hi) if lo is not None and hi is not None else "?"
    return tensor_ir_type(a.shape[:-1] + (total,), a.dtype, layout=a.layout)


def _shape_tile_trailing(operand_types: List[IRType],
                         attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`tile`: a scalar `reps` repeats the LAST axis (numpy's broadcast rule)."""
    x = operand_types[0]
    reps = (attrs or {}).get("reps")
    if x.rank is None or not x.shape or isinstance(reps, (tuple, list)):
        return _unknown_like(x)
    n, r = _dim(x.shape[-1]), _dim(reps)
    if n is None or r is None:
        return _unknown_like(x)
    return tensor_ir_type(x.shape[:-1] + (str(n * r),), x.dtype, layout=x.layout)


def _shape_flatten_repeat(operand_types: List[IRType],
                          attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`repeat` with no axis FLATTENS -- `(4,6)` repeated twice is `(48,)`.

    Easy to mistake for shape-preserving elementwise repetition; it is not, and
    the rank change is the part a downstream consumer would get wrong.
    """
    x = operand_types[0]
    attrs = attrs or {}
    if attrs.get("axis") is not None:
        return _unknown_like(x)
    reps = _dim(attrs.get("repeats"))
    if x.rank is None or not x.shape or reps is None:
        return _unknown_like(x)
    total = 1
    for extent in x.shape:
        n = _dim(extent)
        if n is None:
            return _unknown_like(x)
        total *= n
    return tensor_ir_type((str(total * reps),), x.dtype, layout=x.layout)


def _shape_select_k_index(operand_types: List[IRType],
                          attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`msa_select_blocks`: trailing axis becomes `top_k`, result is an index."""
    from .op_catalog import INDEX_DTYPE

    x = operand_types[0]
    k = _dim((attrs or {}).get("top_k"))
    if x.rank is None or not x.shape or k is None:
        return _unknown_like(x, INDEX_DTYPE)
    return tensor_ir_type(x.shape[:-1] + (str(k),), INDEX_DTYPE, layout=x.layout)


def _shape_segment_reduce(operand_types: List[IRType],
                          attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """Leading axis becomes the SEGMENT COUNT, which is data-dependent.

    The number of segments is `max(seg_ids) + 1` -- a property of the values in
    operand 1, not of any type. So the rule states `?` there and keeps the
    trailing axes, rather than reusing the input extent (which would be the
    claim "one segment per row", true only when nothing is grouped).
    """
    x = operand_types[0]
    if x.rank is None or not x.shape:
        return _unknown_like(x)
    return tensor_ir_type(("?",) + x.shape[1:], x.dtype, layout=x.layout)


def _shape_lu(operand_types: List[IRType],
              attrs: Optional[Dict[str, Any]] = None):
    """`(LU, pivots)` — the packed factors plus a rank-1 pivot vector."""
    a = operand_types[0]
    if a.rank is None or len(a.shape) < 2:
        return (_unknown_like(a), _unknown_like(a, "int32"))
    return (tensor_ir_type(a.shape, a.dtype, layout=a.layout),
            tensor_ir_type((a.shape[-2],), "int32", layout=a.layout))


def _shape_qr(operand_types: List[IRType],
              attrs: Optional[Dict[str, Any]] = None):
    """Reduced QR: `Q (m, n)` and `R (n, n)` for m >= n."""
    a = operand_types[0]
    if a.rank is None or len(a.shape) < 2:
        return (_unknown_like(a), _unknown_like(a))
    m, n = a.shape[-2], a.shape[-1]
    return (tensor_ir_type(a.shape[:-2] + (m, n), a.dtype, layout=a.layout),
            tensor_ir_type(a.shape[:-2] + (n, n), a.dtype, layout=a.layout))


def _shape_svd(operand_types: List[IRType],
               attrs: Optional[Dict[str, Any]] = None):
    """Reduced SVD: `U (m, n)`, singular values `S (n,)`, `Vt (n, n)`."""
    a = operand_types[0]
    if a.rank is None or len(a.shape) < 2:
        return (_unknown_like(a), _unknown_like(a), _unknown_like(a))
    m, n = a.shape[-2], a.shape[-1]
    return (tensor_ir_type(a.shape[:-2] + (m, n), a.dtype, layout=a.layout),
            tensor_ir_type(a.shape[:-2] + (n,), a.dtype, layout=a.layout),
            tensor_ir_type(a.shape[:-2] + (n, n), a.dtype, layout=a.layout))


def _shape_nonzero(operand_types: List[IRType],
                   attrs: Optional[Dict[str, Any]] = None):
    """One rank-1 index vector per input axis; the LENGTH is data-dependent.

    The arity is the input's rank -- static and worth stating -- while the
    common length depends on how many elements are non-zero and is `?`. Both
    halves matter: an unknown length is not a reason to leave the arity unsaid.
    """
    from .op_catalog import INDEX_DTYPE

    x = operand_types[0]
    if x.rank is None or not x.shape:
        return _unknown_like(x, INDEX_DTYPE)
    part = tensor_ir_type(("?",), INDEX_DTYPE, layout=x.layout)
    return tuple(part for _ in x.shape)


def _shape_spec_accept(operand_types: List[IRType],
                       attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`[path_idx, prefix_length, bonus_token]` — a fixed 3-vector of i32."""
    return tensor_ir_type(("3",), "int32", layout=operand_types[0].layout)


def _shape_spec_accept_tree(operand_types: List[IRType],
                            attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`[path_idx, prefix_length]` — the tree form carries no bonus token."""
    return tensor_ir_type(("2",), "int32", layout=operand_types[0].layout)


def _shape_stft(operand_types: List[IRType],
                attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`(frames, bins)` — `frames = (n - win) // hop + 1`, `bins = win//2 + 1`.

    `bins` is the `rfft` half-spectrum, so the result is complex with the
    input's component width -- the same real-pair model as the rest of the
    transform family.
    """
    x, win = operand_types[0], (operand_types[1] if len(operand_types) > 1 else None)
    dtype = _complex_of(x.dtype)
    attrs = attrs or {}
    hop = _dim(attrs.get("hop"))
    axis = _transform_axis(x, attrs)
    n = _dim(x.shape[axis]) if x.shape and axis is not None else None
    w = _dim(win.shape[-1]) if win is not None and win.shape else None
    fft_n = _dim(attrs.get("logical_length")) or w
    if hop is None or n is None or w is None or fft_n is None or not hop or axis is None:
        return _unknown_like(x, dtype)
    effective = n + (2 * (fft_n // 2) if attrs.get("center", False) else 0)
    effective = max(effective, fft_n)
    frames = str((effective - fft_n) // hop + 1)
    bins = str(fft_n // 2 + 1 if attrs.get("onesided", True) else fft_n)
    result = list(x.shape)
    result[axis : axis + 1] = [frames, bins]
    return tensor_ir_type(tuple(result), dtype, layout=x.layout)


def _shape_conv_full(operand_types: List[IRType],
                     attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`spectral_conv`: FULL convolution, trailing axis `n + m - 1`.

    Not the `n - m + 1` of `conv_spatial`. Spectral convolution keeps the whole
    support, which is why it is a separate rule rather than a padding mode --
    the two differ by `2*(m-1)` and both are plausible-looking numbers.
    """
    if len(operand_types) < 2:
        return operand_types[0]
    x, w = operand_types[0], operand_types[1]
    if x.rank is None or w.rank is None or not x.shape or not w.shape:
        return _unknown_like(x)
    axis = _transform_axis(x, attrs)
    if axis is None or len(x.shape) != len(w.shape):
        return _unknown_like(x)
    n, m = _dim(x.shape[axis]), _dim(w.shape[axis])
    total = str(n + m - 1) if n is not None and m is not None else "?"
    result = []
    for index, (x_extent, w_extent) in enumerate(zip(x.shape, w.shape)):
        if index == axis:
            result.append(total)
        elif x_extent == w_extent or w_extent == "1":
            result.append(x_extent)
        elif x_extent == "1":
            result.append(w_extent)
        else:
            result.append("?")
    return tensor_ir_type(tuple(result), x.dtype, layout=x.layout)


def _shape_arange(operand_types: List[IRType],
                  attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`arange`: length is `ceil((stop - start) / step)`, from attributes only.

    No operand carries it -- `arange` is a SOURCE. The `unclassified` fallback
    returned operand 0's type, which for a source op means echoing whatever
    happened to be passed as `start`.
    """
    attrs = attrs or {}
    start, stop = _dim(attrs.get("start", 0)), _dim(attrs.get("stop"))
    step = _dim(attrs.get("step", 1)) or 1
    dtype = attrs.get("dtype") or "fp32"
    if stop is None or start is None or not step:
        return tensor_ir_type(("?",), dtype)
    length = max(0, -(-(stop - start) // step))  # ceil division
    return tensor_ir_type((str(length),), dtype)


def _shape_einsum(operand_types: List[IRType],
                  attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`einsum`: the result subscripts name the axes, so read them.

    Each output label is looked up in the input labels to find its extent.
    This is the one op whose shape is written down by the caller in full, and
    the `unclassified` fallback (operand 0's shape) ignored it entirely --
    right only for a spec that happens to preserve the first operand.
    """
    spec = (attrs or {}).get("spec") or (attrs or {}).get("equation")
    if not isinstance(spec, str) or "->" not in spec:
        return _unknown_like(operand_types[0])
    lhs, rhs = spec.split("->", 1)
    sizes: Dict[str, str] = {}
    for term, operand in zip(lhs.split(","), operand_types):
        labels = term.strip()
        if operand.rank is None or len(labels) != len(operand.shape):
            return _unknown_like(operand_types[0])
        for label, extent in zip(labels, operand.shape):
            sizes.setdefault(label, extent)
    out = rhs.strip()
    if not out:
        return tensor_ir_type((), operand_types[0].dtype)
    return tensor_ir_type(tuple(sizes.get(label, "?") for label in out),
                          operand_types[0].dtype)


def _shape_istft(operand_types: List[IRType],
                 attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """Inverse STFT: `(frames - 1) * hop + win` samples, and REAL.

    The overlap-add inverse of `_shape_stft`. Like `irfft` it is the member of
    its pair that leaves the complex domain, so its dtype comes from
    `_REAL_FOR_COMPLEX` rather than the operand.
    """
    xf = operand_types[0]
    win = operand_types[1] if len(operand_types) > 1 else None
    dtype = _REAL_FOR_COMPLEX.get(xf.dtype or "", "fp32")
    attrs = attrs or {}
    hop = _dim(attrs.get("hop"))
    axis = _transform_axis(xf, attrs)
    frame_axis = axis - 1 if axis is not None else None
    frames = (
        _dim(xf.shape[frame_axis])
        if xf.rank and frame_axis is not None and frame_axis >= 0
        else None
    )
    window_length = _dim(win.shape[-1]) if win is not None and win.shape else None
    fft_n = _dim(attrs.get("logical_length")) or window_length
    if (
        hop is None or frames is None or fft_n is None
        or frame_axis is None or axis is None
    ):
        return _unknown_like(xf, dtype)
    output_length = _dim(attrs.get("output_length"))
    if output_length is None:
        output_length = (frames - 1) * hop + fft_n
        if attrs.get("center", False):
            output_length -= 2 * (fft_n // 2)
    result = list(xf.shape)
    result[frame_axis : axis + 1] = [str(output_length)]
    return tensor_ir_type(tuple(result), dtype, layout=xf.layout)


def _shape_layout_permute(operand_types: List[IRType],
                          attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """`pack` / `rearrange`: permute by an explicit axis order, else identity.

    Two shapes behind one op, and the rule has to read the attribute to tell
    them apart. A tuple `layout` is an axis permutation; a named layout
    (`"row_major"`, `"identity"`) is shape-preserving.

    The reference used to return the input unchanged for ANY string, so an
    einops-style `"a b -> b a"` was accepted and ignored. That is now a named
    error (Decision #21a -- `layout` selects semantics, so it fails closed),
    which is also what makes this rule expressible: an op that silently means
    two different things cannot have one honest shape rule.
    """
    first = operand_types[0]
    layout = (attrs or {}).get("layout")
    if isinstance(layout, str) and "->" in layout and first.rank is not None:
        symbolic_shape: list[int] = []
        for dim in first.shape:
            value = _dim(dim)
            symbolic_shape.append(-1 if value is None else value)
        from .layout_algebra import rearrange_shape_plan

        plan = rearrange_shape_plan(
            symbolic_shape,
            layout,
            axes_lengths=(attrs or {}).get("axes_lengths"),
        )
        return tensor_ir_type(
            tuple("?" if dim < 0 else str(dim) for dim in plan.output_shape),
            first.dtype,
            layout=first.layout,
        )
    if not isinstance(layout, (tuple, list)):
        return first
    if first.rank is None or len(layout) != len(first.shape):
        return tensor_ir_type(("*",), first.dtype, layout=first.layout)
    try:
        dims = tuple(first.shape[int(axis)] for axis in layout)
    except (TypeError, ValueError, IndexError):
        return tensor_ir_type(("*",), first.dtype, layout=first.layout)
    return tensor_ir_type(dims, first.dtype, layout=first.layout)


def _shape_state_handle(operand_types: List[IRType],
                        attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """A cache mutator threads its handle through: `cache -> updated`.

    `kv_cache.append` / `kv_cache.prune` / `cache.commit` / `cache.rollback`
    all return the handle they were given, and the ODS says so directly --
    `let results = (outs Tessera_KVCacheType:$updated)`. So the result type is
    not derived from operand 0's *inferred* type (which the Python frontend
    reports as `tensor<*x?>`, having no handle vocabulary until now); it is the
    declared handle type. Returning operand 0 verbatim would propagate the
    frontend's wrong guess instead of correcting it.

    These four were previously exempted as "not describable by a tensor shape
    rule". They are not describable by a tensor rule, and they never needed to
    be.
    """
    return HANDLE_KV_CACHE


def _shape_kv_cache_read(operand_types: List[IRType],
                         attrs: Optional[Dict[str, Any]] = None):
    """`kv_cache.read(cache, start, end) -> (K, V)` — TENSORS, not a handle.

    This op was filed with the four mutators above under one shared reason,
    "opaque cache handle rather than a tensor type". That reason is simply
    wrong here: `read` is the one member of the family that does NOT return a
    handle. The reference stacks and returns `(K, V)` arrays.

    Grouping five ops under one sentence is how a wrong classification hides --
    the sentence was true of four of them, so nothing in review pointed at the
    fifth. The shapes stay opaque because they depend on the cache's runtime
    extent, but the ARITY and the tensor-ness are known and now stated.
    """
    return (TENSOR_OPAQUE, TENSOR_OPAQUE)


def _shape_transpose(operand_types: List[IRType], attrs: Optional[Dict[str, Any]] = None) -> IRType:
    first = operand_types[0]
    if first.rank is None:
        return first
    return tensor_ir_type(tuple(reversed(first.shape)), first.dtype, layout=first.layout)


def _shape_reduce_trailing(operand_types: List[IRType], attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """Drop the trailing axis (e.g. a per-row energy over [N, D] -> [N])."""
    x = operand_types[0]
    if x.rank == 2:
        return tensor_ir_type((x.shape[0],), x.dtype, layout=x.layout)
    return tensor_ir_type(("*",), x.dtype, layout=x.layout)


def _shape_depth_attention(operand_types: List[IRType],
                           attrs: Optional[Dict[str, Any]] = None) -> IRType:
    """``query[d], sources[m,...,d] -> [...,d]``."""
    if len(operand_types) != 2:
        return TENSOR_OPAQUE
    query, sources = operand_types
    if query.rank != 1 or sources.rank is None or sources.rank < 2:
        return TENSOR_OPAQUE
    return tensor_ir_type(tuple(sources.shape[1:]), sources.dtype,
                          layout=sources.layout)


_SHAPE_RULES = {
    "same_as_first": _shape_same_as_first,
    "matmul_2d": _shape_matmul_2d,
    "es_population_features": _shape_es_population_features,
    "coalition_marginal": _shape_coalition_marginal,
    "coalition_players_axis": _shape_coalition_players_axis,
    "segment_mex": _shape_segment_mex,
    "tridiagonal_rhs": _shape_tridiagonal_rhs,
    "batched_gemm_3d": _shape_batched_gemm_3d,
    "same_shape_bool": _shape_same_shape_bool,
    "reduce_all": _shape_reduce_all,
    "reduce_all_index": _shape_reduce_all_index,
    "same_shape_index": _shape_same_shape_index,
    "flatten": _shape_flatten,
    "complex_same": _shape_complex_same,
    "rfft": _shape_rfft,
    "irfft": _shape_irfft,
    "select_from_second": _shape_select_from_second,
    "quantize_per_tensor": _shape_quantize_per_tensor,
    "quantize_per_block": _shape_quantize_per_block,
    "optimizer_step": _shape_optimizer_step,
    "from_shape_attr": _shape_from_shape_attr,
    "cast": _shape_cast,
    "transpose": _shape_transpose,
    "reduce_trailing": _shape_reduce_trailing,
    "depth_attention": _shape_depth_attention,
    "matmul_trailing": _shape_matmul_trailing,
    "same_as_second": _shape_same_as_second,
    "conv_spatial": _shape_conv_spatial,
    "index_along_axis": _shape_index_along_axis,
    "drop_axis": _shape_drop_axis,
    "insert_axis": _shape_insert_axis,
    "from_slice_sizes": _shape_from_slice_sizes,
    "pad": _shape_pad,
    "scores_per_block": _shape_scores_per_block,
    "scores_per_block_mask": _shape_scores_per_block_mask,
    "reduce_trailing_index": _shape_reduce_trailing_index,
    "reduce_trailing_bool": _shape_reduce_trailing_bool,
    "top_k": _shape_top_k,
    "split_equal": _shape_split_equal,
    "split_halves": _shape_split_halves,
    "qkv_projection": _shape_qkv_projection,
    "state_matrix": _shape_state_matrix,
    "concat_trailing": _shape_concat_trailing,
    "tile_trailing": _shape_tile_trailing,
    "flatten_repeat": _shape_flatten_repeat,
    "select_k_index": _shape_select_k_index,
    "segment_reduce": _shape_segment_reduce,
    "lu": _shape_lu,
    "qr": _shape_qr,
    "svd": _shape_svd,
    "nonzero": _shape_nonzero,
    "spec_accept": _shape_spec_accept,
    "spec_accept_tree": _shape_spec_accept_tree,
    "stft": _shape_stft,
    "conv_full": _shape_conv_full,
    "arange": _shape_arange,
    "einsum": _shape_einsum,
    "istft": _shape_istft,
    "complex_to_real": _shape_complex_to_real,
    "complex_from_coords": _shape_complex_from_coords,
    "layout_permute": _shape_layout_permute,
    "state_handle": _shape_state_handle,
    "kv_cache_read": _shape_kv_cache_read,
    "all_gather": _shape_all_gather,
    "reduce_scatter": _shape_reduce_scatter,
    # Not yet classified. Same result as the historical fallback, but reached
    # through a NAME, so it is countable and cannot masquerade as a rule.
    "unclassified": _shape_same_as_first,
}


#: Rules that take a third `mesh` argument -- a `{axis_name: size}` mapping.
#: DECLARED rather than discovered by signature introspection, so adding a
#: mesh parameter to a rule is a visible edit here and not an implicit
#: behaviour change in the dispatcher.
_MESH_AWARE_RULES = frozenset({"all_gather", "reduce_scatter"})


def _infer_result_types(op_name: str, operand_types: List[IRType],
                        attrs: Optional[Dict[str, Any]] = None,
                        mesh: Optional[Mapping[str, int]] = None):
    """All result types for `op_name`, as a tuple.

    Most ops have exactly one result and this returns a 1-tuple. Multi-result
    ops (the quantize family, which yields `(codes, scale)`) return the full
    contract here; `_infer_result_type` below returns only the FIRST, which is
    the primary tensor. Callers that emit multiple SSA results should use this
    function -- the single-result one cannot express them and pretending
    otherwise is how `same_as_first` came to be declared for a tuple op.
    """
    result = _infer_result_type(op_name, operand_types, attrs, mesh=mesh, _raw=True)
    return result if isinstance(result, tuple) else (result,)


def _infer_result_type(op_name: str, operand_types: List[IRType],
                       attrs: Optional[Dict[str, Any]] = None,
                       *, mesh: Optional[Mapping[str, int]] = None,
                       _raw: bool = False) -> IRType:
    """Infer an op's result type from its operands, attributes AND mesh.

    `attrs` was added in W1.2. Without it a whole class of ops -- `reshape`,
    `expand`, `cast`, `arange`, and the strided stencils -- could not be
    expressed at all, because their result shape or dtype lives in an
    attribute rather than in any operand's type. Classifying them was blocked
    on the signature, not on the analysis.

    `mesh` is the same story one level out (W1.3). A collective's result shape
    is a function of the MESH -- `all_gather` multiplies its axis by the mesh
    extent, `reduce_scatter` divides -- and no operand type or attribute
    carries that. With no way to say so, both ops returned the operand shape,
    which is not a neutral fallback but the positive claim `world_size == 1`.
    The reference collectives are single-rank no-op stubs, so a probe agreed
    and the wrong rule looked confirmed.
    """
    if not operand_types:
        return TENSOR_OPAQUE
    from .op_catalog import shape_rule_for
    rule = shape_rule_for(op_name)
    fn = _SHAPE_RULES.get(rule)
    if fn is None:
        # A catalog rule name with no implementation here. Drift-gated by
        # test_shape_rule_registry.py; fail loudly rather than silently
        # reverting to the old fallback.
        raise KeyError(
            f"shape rule {rule!r} is declared for {op_name!r} in op_catalog "
            f"but has no implementation in graph_ir._SHAPE_RULES"
        )
    if rule in _MESH_AWARE_RULES:
        # `_SHAPE_RULES` is heterogeneous: most rules take (operands, attrs)
        # and the mesh-aware ones take a third argument. The dict's inferred
        # value type is the 2-arg shape, so the extra argument is invisible to
        # the checker -- `_MESH_AWARE_RULES` is the declaration that makes it
        # safe, and this cast records that rather than widening every rule's
        # signature to carry a parameter almost none of them read.
        mesh_fn = cast(Callable[..., Any], fn)
        result = mesh_fn(operand_types, attrs, mesh)
    else:
        result = fn(operand_types, attrs)
    if isinstance(result, tuple) and not _raw:
        # Single-result callers get the primary tensor; the full contract is
        # available via `_infer_result_types`.
        return result[0]
    return result


def _struct_result_type(op_name: str, in_type: Optional[IRType],
                        kwargs: Dict[str, Any]) -> Optional[IRType]:
    """Result type for the structural view ops from their shape-determining
    attr. Without this, `_infer_result_type`'s fallback returns the INPUT type,
    so a static `reshape(x, (6,4))` would emit `<in> -> <in>` and the identity
    folder could erase a real reshape (silent miscompile). Returns ``None`` when
    the output shape isn't statically derivable (keeps the opaque fallback,
    which the static-shape-guarded folder never touches)."""
    if in_type is None:
        return None
    dtype, layout = in_type.dtype, in_type.layout
    # Ops whose `shape` attr IS the output shape.
    if op_name in ("tessera.reshape", "tessera.view", "tessera.expand",
                   "tessera.broadcast"):
        shape = kwargs.get("shape")
        if isinstance(shape, (list, tuple)) and shape and all(
                isinstance(d, int) for d in shape):
            return tensor_ir_type(tuple(shape), dtype, layout=layout)
        return None
    # The rest derive the output shape from a static input shape.
    ishape = in_type.shape
    if not ishape or "*" in ishape or "?" in ishape:
        return None
    try:
        dims = [int(d) for d in ishape]
    except (ValueError, TypeError):
        return None
    if op_name == "tessera.squeeze":
        axes = kwargs.get("axes")
        if axes is None:
            out = [d for d in dims if d != 1]
        else:
            al = _normalize_axes(axes, len(dims), opname="squeeze")
            for a in al:
                if dims[a] != 1:
                    raise ValueError(
                        "squeeze axes must select size-1 dimensions; "
                        f"axis {a} has size {dims[a]}")
            drop = set(al)
            out = [d for i, d in enumerate(dims) if i not in drop]
        return tensor_ir_type(tuple(out), dtype, layout=layout)
    if op_name == "tessera.unsqueeze":
        axes = kwargs.get("axes")
        if axes is None:
            return None
        raw = [axes] if isinstance(axes, int) else list(axes)
        out_rank = len(dims) + len(raw)
        al = sorted(_normalize_axes(raw, out_rank, opname="unsqueeze"))
        out = list(dims)
        for a in al:
            out.insert(a, 1)
        return tensor_ir_type(tuple(out), dtype, layout=layout)
    if op_name == "tessera.permute":
        perm = kwargs.get("perm")
        if not isinstance(perm, (list, tuple)):
            return None
        if len(perm) != len(dims):
            raise ValueError(
                f"permute perm length must match rank {len(dims)}, "
                f"got {len(perm)}")
        if not all(isinstance(p, int) for p in perm):
            raise ValueError("permute perm must contain integer axes")
        if sorted(perm) != list(range(len(dims))):
            raise ValueError(
                f"permute perm must be a full permutation of rank {len(dims)}, "
                f"got {list(perm)}")
        return tensor_ir_type(tuple(dims[p] for p in perm), dtype, layout=layout)
    if op_name == "tessera.flatten":
        start = int(kwargs.get("start", 0))
        end = int(kwargs.get("end", len(dims) - 1))
        start = start if start >= 0 else len(dims) + start
        end = end if end >= 0 else len(dims) + end
        if not (0 <= start <= end < len(dims)):
            return None
        prod = 1
        for d in dims[start:end + 1]:
            prod *= d
        return tensor_ir_type(tuple(dims[:start] + [prod] + dims[end + 1:]),
                              dtype, layout=layout)
    return None


def _normalize_axes(value: Any, rank: int, *, opname: str) -> list[int]:
    if isinstance(value, int):
        axes = [value]
    elif isinstance(value, (list, tuple)) and all(
            isinstance(a, int) for a in value):
        axes = list(value)
    else:
        raise ValueError(f"{opname} axes must be an int or list of ints")
    out: list[int] = []
    for axis in axes:
        a = axis if axis >= 0 else rank + axis
        if not (0 <= a < rank):
            raise ValueError(
                f"{opname} axis {axis} out of range for rank {rank}")
        if a in out:
            raise ValueError(f"{opname} axes must be unique, got {axes}")
        out.append(a)
    return out


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
        "i4": "int4",
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
            return_values=getattr(self, "_last_return_values", []),
            result_types=getattr(self, "_last_return_types", []),
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

        extractor = _OpExtractor(arg_names, arg_types, _scalar_const_env(fn))
        # Only visit the function body (skip decorator lines)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == fn.__name__:
                    for stmt in node.body:
                        extractor.visit(stmt)
                    break

        self.diagnostics.extend(extractor.diagnostics)
        self._last_return_values = list(extractor.return_values)
        self._last_return_types = list(extractor.return_result_types)
        return extractor.ops

    def module(self) -> GraphIRModule:
        """Return the assembled GraphIRModule."""
        return self.context.module

    def reset(self) -> None:
        """Clear all accumulated functions."""
        self.context = GraphIRConstructionContext()
        self.diagnostics = []


#: Python handle classes -> the ODS type mnemonic they lower to. Without this
#: an annotated `cache: KVCacheHandle` argument lowered to `tensor<*x?>`, so
#: `tessera.kv_cache.append` emitted `(tensor<*x?>, ...) -> !tessera.kv_cache`
#: -- the RESULT typed from the ODS and the OPERAND still an untyped tensor,
#: which is a worse contract than being consistently wrong. The handle
#: subclasses (latent/SSM/delta/memory) share `!tessera.kv_cache` because that
#: is the single stateful-handle type the Graph IR ODS declares; if the dialect
#: ever splits them, this map is the one place that changes.
_HANDLE_ANNOTATIONS: Dict[str, str] = {
    "KVCacheHandle": "kv_cache",
    "LatentKVCacheHandle": "kv_cache",
    "SSMStateHandle": "kv_cache",
    "DeltaNetStateHandle": "kv_cache",
    "MemoryStateHandle": "kv_cache",
}


def _handle_type_for_annotation(ann: Any) -> Optional[IRType]:
    """The handle IR type for a cache-handle annotation, else None.

    Matches on the class NAME rather than by importing the handle classes:
    `graph_ir` is imported during `tessera` package init, and importing
    `tessera.cache` from here reintroduces the import cycle those modules are
    lazy-imported to avoid.
    """
    name = getattr(ann, "__name__", None) if isinstance(ann, type) else None
    if name is None and isinstance(ann, str):
        name = ann.strip().strip("\"'")
    mnemonic = _HANDLE_ANNOTATIONS.get(name or "")
    return handle_ir_type(mnemonic) if mnemonic else None


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
        # A handle type written directly, e.g. `"!tessera.kv_cache"`.
        if text.startswith("!tessera."):
            return IRType(text)
    handle = _handle_type_for_annotation(ann)
    if handle is not None:
        return handle
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


def specialize_module_from_values(
    module: GraphIRModule,
    values: Mapping[str, Any],
) -> GraphIRModule:
    """Return a copy of ``module`` specialized to concrete call values.

    This is the first-call specialization seam for unannotated ``@jit``
    functions. Argument tensor types come from concrete shapes/dtypes, then
    result types are re-inferred in SSA order. The decoration-time symbolic
    module is never mutated, so different shape signatures can coexist.
    """
    import numpy as np

    out = copy.deepcopy(module)
    for fn in out.functions:
        env: Dict[str, IRType] = {}
        for arg in fn.args:
            value = values.get(arg.name)
            if value is None:
                env[f"%{arg.name}"] = arg.ir_type
                continue
            arr = np.asarray(value)
            dtype = {
                np.dtype(np.float16): "f16",
                np.dtype(np.float32): "f32",
                    np.dtype(np.float64): "f64",
                    np.dtype(np.complex64): "complex64",
                    np.dtype(np.complex128): "complex128",
                    np.dtype(np.int32): "i32",
                np.dtype(np.int64): "i64",
                np.dtype(np.bool_): "i1",
            }.get(arr.dtype)
            if dtype is None:
                raise TypeError(f"unsupported specialization dtype {arr.dtype}")
            arg.ir_type = tensor_ir_type(tuple(int(d) for d in arr.shape), dtype)
            arg.dim_names = tuple(str(int(d)) for d in arr.shape)
            env[f"%{arg.name}"] = arg.ir_type

        for op in fn.body:
            operands = [
                name if str(name).startswith("%") else f"%{name}"
                for name in op.operands
            ]
            inferred = _infer_result_type(
                op.op_name, [env.get(name, TENSOR_OPAQUE) for name in operands])
            op.operand_types = [str(env.get(name, TENSOR_OPAQUE)) for name in operands]
            op.inferred_type = inferred
            op.result_type = str(inferred)
            if op.result is not None:
                result = op.result if op.result.startswith("%") else f"%{op.result}"
                env[result] = inferred

        fn.result_types = [
            env.get(v if str(v).startswith("%") else f"%{v}", typ)
            for v, typ in zip(fn.return_values, fn.result_types)
        ]
    return out
