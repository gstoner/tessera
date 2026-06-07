"""GraphFn op-translation helpers (shared by the Phase-F tracer).

Historically (Phase-G close-out A/C) this module hosted the **AST bridge** —
``detect_loop_fn`` / ``detect_cond_fn`` / ``run_bridged_*`` — which pattern-matched
a raw Python ``for``/``if`` from the decoration-time graph_ir and mapped the narrow
"construct IS the whole function" shape to GraphFn control ops. **That bridge was
retired in Phase-F F5** once the abstract-interp tracer (``compiler/trace.py``)
became the apple_gpu control-flow front-end.

What remains is the small, reused translation core: the graph_ir-op → GraphFn-
builder table (``_OP_TABLE``) and the op applier (``_apply_op``), which the tracer
uses to lower a (straight-line) graph_ir region into an executable ``GraphFn``,
plus the ``_strip`` SSA helper.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


def _strip(ssa: str) -> str:
    """Drop a leading ``%`` from an SSA name (``"%x"`` → ``"x"``)."""
    return ssa[1:] if ssa.startswith("%") else ssa


# graph_ir op_name -> (GraphFn method name, "binary"|"unary"|"matmul"|"norm"|"softmax")
_OP_TABLE: Dict[str, Tuple[str, str]] = {
    "tessera.matmul": ("matmul", "matmul"),
    "tessera.add": ("add", "binary"),
    "tessera.sub": ("sub", "binary"),
    "tessera.mul": ("mul", "binary"),
    "tessera.div": ("div", "binary"),
    "tessera.relu": ("relu", "unary"),
    "tessera.sigmoid": ("sigmoid", "unary"),
    "tessera.tanh": ("tanh", "unary"),
    "tessera.silu": ("silu", "unary"),
    "tessera.gelu": ("gelu", "unary"),
    "tessera.transpose": ("transpose", "unary"),
    "tessera.softmax": ("softmax", "softmax"),
    "tessera.rmsnorm": ("rmsnorm", "norm"),
    "tessera.layer_norm": ("layer_norm", "norm"),
}


def _apply_op(g: Any, op: Any, env: Dict[str, Any], error_cls) -> Any:
    """Replay one graph_ir op through the ``GraphFn`` builder ``g``, resolving its
    operands from ``env`` (SSA name → ``_Val``). Raises ``error_cls`` for an op
    outside the GraphFn-executable subset (``_OP_TABLE``) or an out-of-scope
    operand."""
    entry = _OP_TABLE.get(op.op_name)
    if entry is None:
        raise error_cls(
            f"apple_gpu trace: op {op.op_name!r} cannot be lowered to a GraphFn "
            f"builder (executable subset: {', '.join(sorted(_OP_TABLE))})"
        )
    method, kind = entry
    fn = getattr(g, method)
    try:
        ins = [env[_strip(o)] for o in op.operands]
    except KeyError as e:
        raise error_cls(
            f"apple_gpu trace: op {op.op_name!r} references a value not in scope: "
            f"{e}"
        )
    kw = op.kwargs or {}
    if kind == "matmul":
        return fn(ins[0], ins[1],
                  transpose_a=bool(kw.get("transpose_a", False)),
                  transpose_b=bool(kw.get("transpose_b", False)))
    if kind == "binary":
        return fn(ins[0], ins[1])
    if kind == "unary":
        return fn(ins[0])
    if kind == "softmax":
        return fn(ins[0], axis=int(kw.get("axis", -1)))
    if kind == "norm":
        return fn(ins[0], eps=float(kw.get("eps", 1e-5)))
    raise error_cls(f"internal: unhandled op kind {kind!r}")  # pragma: no cover
