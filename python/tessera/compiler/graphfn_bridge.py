"""Phase-G close-out, Phase A — AST ``@tessera.jit`` → ``tessera.control_for`` bridge.

The ``@jit`` decorator already lowers a function's AST to a flat ``IROp`` list with
the loop body inline between ``tessera.scf.for.{begin,end}`` markers (see
``graph_ir.py``). This module performs an **IR-to-IR translation** of that list
into a ``_jit_boundary.GraphFn``, then reuses the entire G-A/G-B/G-C execution
machinery: ``tessera.control_for`` → ``tessera-opt --tessera-control-for-to-apple_gpu``
→ ``tessera_apple.gpu.control_loop`` → ``run_graph_loop_f32``. No polymorphic-ops /
abstract-interpretation lift is needed.

**Supported v1 shape** (single-carry bounded loop *as the whole function*):

    @jit(target="apple_gpu")
    def f(x, w):                      # x is the carry; w (and any others) are
        for _ in range(N):            # loop-invariant consts
            x = silu(matmul(x, w))    # body re-binds the carry arg
        return x

**Dispatch policy** (Decision #21): if the structural shape matches but a body op
can't be translated to a GraphFn builder method, raise a stable diagnostic naming
the op + target — never silently fall back to host Python. Non-matching functions
(multi-carry, dynamic trip, ops before/after the loop, nested) return ``None`` from
``detect_loop_fn`` and keep the existing ``@jit`` path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_BEGIN = "tessera.scf.for.begin"
_END = "tessera.scf.for.end"


@dataclass(frozen=True)
class LoopShape:
    """A detected single-carry bounded loop ready for GraphFn translation."""

    arg_names: Tuple[str, ...]      # function args, in order
    carry_arg_index: int           # which arg is the loop carry
    carry_base: str                # the carry arg's name (== arg_names[index])
    next_carry_ssa: str            # body result that becomes the next carry
    trip: int                      # static trip count
    body_ops: Tuple[Any, ...]      # IROps strictly between the begin/end markers


def _strip(ssa: str) -> str:
    return ssa[1:] if ssa.startswith("%") else ssa


def detect_loop_fn(graph_ir_module: Any, target: Any) -> Optional[LoopShape]:
    """Return a :class:`LoopShape` if ``graph_ir_module``'s single function is a
    supported single-carry bounded loop on ``apple_gpu``; else ``None``.

    Conservative by construction: anything outside the v1 shape returns ``None``
    so the existing ``@jit`` path is used unchanged.
    """
    if target != "apple_gpu":
        return None
    fns = getattr(graph_ir_module, "functions", None)
    if not fns or len(fns) != 1:
        return None
    fn = fns[0]
    body = list(getattr(fn, "body", []))
    arg_names = tuple(a.name for a in getattr(fn, "args", []))
    if not arg_names:
        return None

    # Exactly one begin/end pair, and the loop must BE the function body (no ops
    # before begin or after end — `return x` emits no op, so the tail is clean).
    begins = [i for i, op in enumerate(body) if op.op_name == _BEGIN]
    ends = [i for i, op in enumerate(body) if op.op_name == _END]
    if len(begins) != 1 or len(ends) != 1:
        return None
    bi, ei = begins[0], ends[0]
    if bi != 0 or ei != len(body) - 1 or ei <= bi:
        return None

    begin = body[bi]
    trip = begin.kwargs.get("trip_count")
    if not isinstance(trip, int) or begin.kwargs.get("kind") == "dynamic":
        return None  # dynamic / non-static trip → not v1

    body_ops = body[bi + 1:ei]
    if not body_ops:
        return None

    # Carry = the single function arg that is both read in the body and re-bound
    # by a body op (its SSA gets a `base__N` version). next_carry = that op's
    # result. Multiple re-bound args → multi-carry → reject.
    read_ssas = {_strip(o) for op in body_ops for o in op.operands}
    carry_bases: Dict[str, str] = {}  # base arg -> latest body result ssa
    for op in body_ops:
        if op.result is None:
            continue
        base = op.result.split("__", 1)[0]
        if base in arg_names and base in read_ssas:
            carry_bases[base] = op.result
    if len(carry_bases) != 1:
        return None
    carry_base, next_carry_ssa = next(iter(carry_bases.items()))

    return LoopShape(
        arg_names=arg_names,
        carry_arg_index=arg_names.index(carry_base),
        carry_base=carry_base,
        next_carry_ssa=next_carry_ssa,
        trip=int(trip),
        body_ops=tuple(body_ops),
    )


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
    entry = _OP_TABLE.get(op.op_name)
    if entry is None:
        raise error_cls(
            f"@jit(target='apple_gpu') loop body op {op.op_name!r} cannot be "
            f"lowered to the Apple GPU control_for path (supported: "
            f"{', '.join(sorted(_OP_TABLE))})"
        )
    method, kind = entry
    fn = getattr(g, method)
    try:
        ins = [env[_strip(o)] for o in op.operands]
    except KeyError as e:
        raise error_cls(
            f"@jit(target='apple_gpu') loop op {op.op_name!r} references a value "
            f"not in scope: {e} (v1: args, the carry, and earlier body ops only)"
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


def build_graphfn(shape: LoopShape, arg_shapes: List[Tuple[int, ...]], elem: str):
    """Translate a :class:`LoopShape` into an un-executed ``GraphFn`` carrying a
    ``tessera.control_for``. ``arg_shapes`` are the concrete call shapes (in arg
    order); ``elem`` is the GraphFn element type (``"f32"`` / ``"bf16"``)."""
    from .._jit_boundary import GraphFn, TesseraJitError

    g = GraphFn(name="tessera_jit_loop", elem=elem, target="apple_gpu")
    arg_vals = {name: g.arg(tuple(sh))
                for name, sh in zip(shape.arg_names, arg_shapes)}
    carry_init = arg_vals[shape.carry_base]

    def body(carry_val):
        env = dict(arg_vals)
        env[shape.carry_base] = carry_val
        for op in shape.body_ops:
            env[op.result] = _apply_op(g, op, env, TesseraJitError)
        return env[shape.next_carry_ssa]

    out = g.for_loop(shape.trip, init=carry_init, body=body)
    g.ret(out)
    return g


def run_bridged_loop(jitfn: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
    """Execute a ``@jit(target='apple_gpu')`` bounded-loop function through the
    GraphFn control_for path. Builds a shape-specialized GraphFn (cached on the
    JitFn per arg-shape/dtype key) and runs it via the Target-IR control_loop path
    (or the direct in-memory path when ``tessera-opt`` is unavailable)."""
    import numpy as np

    from .._jit_boundary import _find_tessera_opt

    shape: LoopShape = jitfn._loop_shape
    # Resolve call args into arg order (positional first, then keyword by name).
    ordered: List[Any] = list(args)
    if len(ordered) < len(shape.arg_names):
        for name in shape.arg_names[len(ordered):]:
            if name not in kwargs:
                raise TypeError(
                    f"{jitfn._fn.__name__}() missing argument {name!r}")
            ordered.append(kwargs[name])
    arrays = [np.asarray(a) for a in ordered[:len(shape.arg_names)]]

    dt = arrays[shape.carry_arg_index].dtype
    elem = "bf16" if str(dt) in ("bfloat16",) else "f32"
    key = tuple((a.shape, str(a.dtype)) for a in arrays)
    cache = jitfn._bridge_cache
    g = cache.get(key)
    if g is None:
        g = build_graphfn(shape, [a.shape for a in arrays], elem)
        cache[key] = g

    if _find_tessera_opt() is not None:
        return g.run_via_target_ir(*arrays)
    return g.run(*arrays)
