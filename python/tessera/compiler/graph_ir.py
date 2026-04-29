"""
tessera.compiler.graph_ir — Python → Graph IR lowering.

Emits MLIR text for the Tessera Graph IR dialect (tessera.*) from a
Python function's structure. In Phase 1 this is a simplified textual
emitter — it captures function signatures, region annotations, and op
calls and produces valid-looking MLIR that can be round-tripped.

Phase 2 will replace this with a proper MLIR Python bindings emitter that
constructs in-memory IR objects rather than text.

Output format (MLIR text):
    func.func @step(%W: tensor<*xbf16> {tessera.effect = "read"},
                    %X: tensor<*xbf16> {tessera.effect = "read"},
                    %Y: tensor<*xbf16> {tessera.effect = "write"}) {
      %0 = tessera.gemm(%X, %W) : (tensor<*xbf16>, tensor<*xbf16>) -> tensor<*xbf16>
      tessera.copy %0, %Y : tensor<*xbf16>
      return
    }

Reference: CLAUDE.md §Four-Layer IR Stack — Graph IR
           src/ir/TesseraOps.td
"""

from __future__ import annotations
import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# IR value types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IRType:
    """A simplified type representation for Graph IR emission."""
    mlir_str: str   # e.g. "tensor<*xbf16>", "f32", "index"

    def __str__(self) -> str:
        return self.mlir_str


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


# Common types used in Phase 1
TENSOR_BF16   = IRType("tensor<*xbf16>")
TENSOR_FP16   = IRType("tensor<*xf16>")
TENSOR_FP32   = IRType("tensor<*xf32>")
TENSOR_OPAQUE = IRType("tensor<*x?>")   # unknown dtype
KV_CACHE      = IRType("!tessera.kv_cache")
INDEX         = IRType("index")
BOOL          = IRType("i1")


def _dtype_to_ir_type(dtype: str) -> IRType:
    """Map a Tessera dtype string to an MLIR type."""
    _map = {
        "bf16": TENSOR_BF16,
        "fp16": TENSOR_FP16,
        "fp32": TENSOR_FP32,
        "fp64": IRType("tensor<*xf64>"),
        "fp8_e4m3": IRType("tensor<*xf8E4M3FN>"),
        "fp8_e5m2": IRType("tensor<*xf8E5M2>"),
        "fp6_e2m3": IRType("tensor<*x!tessera.fp6_e2m3>"),
        "fp6_e3m2": IRType("tensor<*x!tessera.fp6_e3m2>"),
        "fp4_e2m1": IRType("tensor<*x!tessera.fp4_e2m1>"),
        "nvfp4": IRType("tensor<*x!tessera.nvfp4>"),
        "int8": IRType("tensor<*xi8>"),
        "int16": IRType("tensor<*xi16>"),
        "int32": IRType("tensor<*xi32>"),
        "int64": IRType("tensor<*xi64>"),
        "bool": IRType("tensor<*xi1>"),
    }
    return _map.get(dtype, TENSOR_OPAQUE)


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

    def to_mlir(self) -> str:
        attrs = []
        if self.effect:
            attrs.append(f'tessera.effect = "{self.effect}"')
        if self.shard_spec:
            attrs.append(f"tessera.shard = {self.shard_spec}")
        attr_str = (" {" + ", ".join(attrs) + "}") if attrs else ""
        return f"%{self.name}: {self.ir_type}{attr_str}"


# ─────────────────────────────────────────────────────────────────────────────
# Graph IR op emission helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IROp:
    """A single emitted op in the Graph IR function body."""
    result: Optional[str]     # None for void ops (e.g. copy/store)
    op_name: str              # e.g. "tessera.gemm"
    operands: List[str]       # e.g. ["%X", "%W"]
    operand_types: List[str]  # e.g. ["tensor<*xbf16>", "tensor<*xbf16>"]
    result_type: Optional[str] = None
    attrs: Optional[str] = None   # additional op attributes

    def to_mlir(self, indent: str = "  ") -> str:
        ops_str = ", ".join(self.operands)
        types_in = ", ".join(self.operand_types)
        if self.result is not None and self.result_type:
            lhs = f"%{self.result} = "
            type_str = f" : ({types_in}) -> {self.result_type}"
        elif self.result is not None:
            lhs = f"%{self.result} = "
            type_str = ""
        else:
            lhs = ""
            type_str = f" : {types_in}" if types_in else ""
        attr_str = f" {{{self.attrs}}}" if self.attrs else ""
        return f"{indent}{lhs}{self.op_name}({ops_str}){attr_str}{type_str}"


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

    def to_mlir(self) -> str:
        attr_lines = ", ".join(f"{k} = {v}" for k, v in self.module_attrs.items())
        lines = [f"module attributes {{{attr_lines}}} {{"]
        for fn in self.functions:
            for line in fn.to_mlir().splitlines():
                lines.append("  " + line)
        lines.append("}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"GraphIRModule({len(self.functions)} functions)"


@dataclass
class GraphIRFunction:
    """
    A single `func.func @name(...)` block in the Graph IR.

    Built by GraphIRBuilder when lowering a Python function.
    """
    name: str
    args: List[IRArg] = field(default_factory=list)
    result_types: List[IRType] = field(default_factory=list)
    body: List[IROp] = field(default_factory=list)
    fn_attrs: Dict[str, str] = field(default_factory=dict)

    def to_mlir(self) -> str:
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
        lines.append("  return")
        lines.append("}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# AST-based op extractor (Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

class _OpExtractor(ast.NodeVisitor):
    """
    Walks a function AST and extracts tessera.ops calls as IROp instances.

    Phase 1 scope: recognises `ops.gemm(A, B)`, `tessera.ops.gemm(A, B)`,
    `Y[:] = ops.gemm(A, B)` patterns. Everything else becomes a comment op.
    """

    # Map from op bare name to Graph IR op name
    _OP_MAP = {
        "gemm":       "tessera.gemm",
        "matmul":     "tessera.matmul",
        "conv2d":     "tessera.conv2d",
        "layer_norm": "tessera.layer_norm",
        "softmax":    "tessera.softmax",
        "gelu":       "tessera.gelu",
        "relu":       "tessera.relu",
        "sigmoid":    "tessera.sigmoid",
        "sin":        "tessera.sin",
        "adam":       "tessera.adam",
        "transpose":  "tessera.transpose",
        "cast":       "tessera.cast",
        "flash_attn": "tessera.flash_attn",
        "dropout":    "tessera.dropout",
        "rmsnorm_safe": "tessera.rmsnorm_safe",
        "softmax_safe": "tessera.softmax_safe",
        "kv_cache_append": "tessera.kv_cache.append",
        "kv_cache_prune": "tessera.kv_cache.prune",
    }

    def __init__(self, arg_names: List[str]) -> None:
        self.ops: List[IROp] = []
        self._counter = 0
        self._arg_names = set(arg_names)

    def _fresh(self) -> str:
        name = f"v{self._counter}"
        self._counter += 1
        return name

    def _resolve_name(self, node: ast.expr) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = self._resolve_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return None

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handle `Y[:] = ops.gemm(A, B)` or `result = ops.gemm(A, B)`."""
        if isinstance(node.value, ast.Call):
            call = node.value
            op_mlir = self._try_map_call(call)
            if op_mlir:
                # Determine result name
                tgt = node.targets[0]
                if isinstance(tgt, ast.Name):
                    result_name = tgt.id
                elif isinstance(tgt, ast.Subscript):
                    # Y[:] = ... → void copy into Y
                    result_name = None
                    outer_name = self._resolve_name(tgt.value)
                    op_mlir.result = None
                    op_mlir.op_name = op_mlir.op_name  # keep as-is; copy is implicit
                    # Add a copy op
                    fresh = self._fresh()
                    # First emit the compute op with a temp result
                    op_mlir.result = fresh
                    self.ops.append(op_mlir)
                    # Then emit copy/store
                    dest = outer_name or "?"
                    self.ops.append(IROp(
                        result=None,
                        op_name="tessera.copy",
                        operands=[f"%{fresh}", f"%{dest}"],
                        operand_types=["tensor<*x?>", "tensor<*x?>"],
                    ))
                    return
                else:
                    result_name = self._fresh()
                op_mlir.result = result_name
                self.ops.append(op_mlir)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value and isinstance(node.value, ast.Call):
            op = self._try_map_call(node.value)
            if op:
                fresh = self._fresh()
                op.result = fresh
                self.ops.append(op)

    def _try_map_call(self, call: ast.Call) -> Optional[IROp]:
        name = self._resolve_name(call.func)
        if not name:
            return None
        bare = name.split(".")[-1]
        mlir_name = self._OP_MAP.get(bare)
        if not mlir_name:
            return None

        operands = []
        for arg in call.args:
            n = self._resolve_name(arg)
            operands.append(f"%{n}" if n else "%?")

        return IROp(
            result=None,  # filled in by caller
            op_name=mlir_name,
            operands=operands,
            operand_types=["tensor<*x?>"] * len(operands),
            result_type="tensor<*x?>",
        )


# ─────────────────────────────────────────────────────────────────────────────
# GraphIRBuilder
# ─────────────────────────────────────────────────────────────────────────────

class GraphIRBuilder:
    """
    Lowers a Python function (decorated with @jit) into a GraphIRFunction.

    Phase 1 pipeline:
      1. Extract parameter annotations → IRArg list (with effect attrs)
      2. Walk function AST → IROp list (tessera ops only)
      3. Assemble GraphIRFunction

    Phase 2:
      - Replace textual emission with MLIR Python bindings IR construction
      - Add ShardSpec → tessera.shard attribute lowering
      - Add EffectLattice results as tessera.effect function attrs

    Usage:
        builder = GraphIRBuilder()
        fn_ir = builder.lower(step)
        module = builder.module()
        print(module.to_mlir())
    """

    def __init__(self) -> None:
        self._module = GraphIRModule()

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
            self._module.module_attrs["tessera.target"] = target_attr
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

            # Detect RegionType annotations
            effect = None
            ir_type = TENSOR_OPAQUE
            if ann is not inspect.Parameter.empty:
                ann_name = getattr(ann, "__name__", "") or repr(ann)
                # RegionType from tessera.distributed.region
                if hasattr(ann, "mode"):
                    effect = ann.mode
                # Simple dtype annotations like f16[...], bf16 etc.
                elif hasattr(ann, "__origin__"):
                    pass  # generic alias — leave opaque for now

            args.append(IRArg(name=param_name, ir_type=ir_type, effect=effect))

        # Extract ops from AST
        arg_names = [a.name for a in args]
        ops = self._extract_ops(fn, arg_names, source_text=source_text)

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
        self._module.functions.append(fn_ir)
        return fn_ir

    def _extract_ops(
        self,
        fn: Callable,
        arg_names: List[str],
        source_text: Optional[str] = None,
    ) -> List[IROp]:
        """Walk the function AST and extract recognized tessera op calls."""
        try:
            source = source_text if source_text is not None else inspect.getsource(fn)
            source = textwrap.dedent(source)
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            return []

        extractor = _OpExtractor(arg_names)
        # Only visit the function body (skip decorator lines)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == fn.__name__:
                    for stmt in node.body:
                        extractor.visit(stmt)
                    break

        return extractor.ops

    def module(self) -> GraphIRModule:
        """Return the assembled GraphIRModule."""
        return self._module

    def reset(self) -> None:
        """Clear all accumulated functions."""
        self._module = GraphIRModule()
