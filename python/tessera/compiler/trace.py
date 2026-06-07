"""Phase-F abstract-interpretation tracing lift (F1 — straight-line core).

Interpret a function *once by running it* with abstract ``Tracer`` values that
record graph_ir ops, instead of pattern-matching its AST. Every ``tessera.ops.*``
call already routes through the autodiff op wrapper
(``autodiff/tape.py::_make_wrapper``); that wrapper now consults
``_trace_hook.active_tracer()`` first, so a ``TraceBuilder`` set as the active
tracer records each op (and returns a fresh ``Tracer`` whose shape comes from a
rule) without running numpy.

F1 covers straight-line ``tessera.ops`` functions over the GraphFn-executable
subset and proves the round-trip: ``trace(fn, *specs)`` → graph_ir → ``to_graphfn``
→ execute, matching numpy. Control flow (F2), full-vocab shape rules and the @jit
wiring (F3–F5) build on this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import _trace_hook
from .graph_ir import IROp
from .op_catalog import graph_name_for

# ── abstract value ────────────────────────────────────────────────────────── #


@dataclass(frozen=True)
class Tracer:
    """An abstract value flowing through a trace: shape + dtype + graph SSA."""

    shape: Tuple[int, ...]
    dtype: str
    ssa: str


@dataclass
class TracedFunction:
    """The result of a trace: typed args, a straight-line graph_ir body, and the
    SSA names of the outputs."""

    args: List[Tuple[str, Tuple[int, ...], str]]  # (ssa, shape, dtype)
    body: List[IROp]
    outputs: List[str]                            # output SSA names


# ── shape rules (executable subset; widened in F6) ────────────────────────── #


def _matmul_shape(ins: List[Tuple[int, ...]], kw: dict) -> Tuple[int, ...]:
    a, b = ins[0], ins[1]
    if len(a) < 2 or len(b) < 2:
        raise TesseraTraceError(f"matmul needs rank>=2 operands, got {a} @ {b}")
    m, k = a[-2], a[-1]
    k2, n = b[-2], b[-1]
    if k != k2:
        raise TesseraTraceError(f"matmul inner-dim mismatch: {a} @ {b}")
    batch = a[:-2] if len(a) >= len(b) else b[:-2]
    return (*batch, m, n)


def _transpose_shape(ins: List[Tuple[int, ...]], kw: dict) -> Tuple[int, ...]:
    s = ins[0]
    if len(s) < 2:
        return s
    return (*s[:-2], s[-1], s[-2])


def _broadcast_shape(ins: List[Tuple[int, ...]], kw: dict) -> Tuple[int, ...]:
    return tuple(np.broadcast_shapes(*ins))


def _first_shape(ins: List[Tuple[int, ...]], kw: dict) -> Tuple[int, ...]:
    return ins[0]


# public-name -> shape rule. Anything not here falls back by arity (unary →
# shape-preserving, binary → broadcast); unknown/ambiguous → diagnostic.
_SHAPE_RULES: Dict[str, Callable[[List[Tuple[int, ...]], dict], Tuple[int, ...]]] = {
    "matmul": _matmul_shape,
    "gemm": _matmul_shape,
    "transpose": _transpose_shape,
    "add": _broadcast_shape, "sub": _broadcast_shape,
    "mul": _broadcast_shape, "div": _broadcast_shape,
    "silu": _first_shape, "relu": _first_shape, "sigmoid": _first_shape,
    "tanh": _first_shape, "gelu": _first_shape,
    "rmsnorm": _first_shape, "layer_norm": _first_shape, "softmax": _first_shape,
}


class TesseraTraceError(Exception):
    """Raised when a function cannot be abstractly traced (no shape rule, a
    non-Tracer positional operand, an op outside the catalog, ...)."""


def register_shape_rule(name: str, rule) -> None:
    """Register/override the abstract shape rule for op ``name`` (F6 widening)."""
    _SHAPE_RULES[name] = rule


def _infer_shape(name: str, in_shapes: List[Tuple[int, ...]], kw: dict
                 ) -> Tuple[int, ...]:
    rule = _SHAPE_RULES.get(name)
    if rule is not None:
        return rule(in_shapes, kw)
    if len(in_shapes) == 1:           # unary fallback: shape-preserving
        return in_shapes[0]
    if len(in_shapes) == 2:           # binary fallback: broadcast
        return tuple(np.broadcast_shapes(*in_shapes))
    raise TesseraTraceError(
        f"trace: no shape rule for op {name!r} ({len(in_shapes)} tensor inputs); "
        f"register one via tessera.compiler.trace.register_shape_rule")


# ── trace builder ─────────────────────────────────────────────────────────── #


def _ty(shape: Tuple[int, ...], dtype: str) -> str:
    dims = "x".join(str(d) for d in shape) if shape else ""
    return f"tensor<{dims}x{dtype}>" if dims else f"tensor<{dtype}>"


@dataclass
class TraceBuilder:
    """Accumulates the graph_ir body as ``tessera.ops`` calls record themselves
    via :meth:`record_op` while this builder is the active tracer."""

    args: List[Tuple[str, Tuple[int, ...], str]] = field(default_factory=list)
    body: List[IROp] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    _counter: int = 0

    def arg(self, ssa: str, shape, dtype: str) -> Tracer:
        sh = tuple(int(d) for d in shape)
        self.args.append((ssa, sh, dtype))
        return Tracer(sh, dtype, ssa)

    def _fresh(self) -> str:
        n = self._counter
        self._counter += 1
        return f"v{n}"

    def record_op(self, name: str, args: tuple, kwargs: dict) -> Tracer:
        graph_name = graph_name_for(name)
        if graph_name is None:
            raise TesseraTraceError(f"trace: op {name!r} is not in the op catalog")
        tracer_args = [a for a in args if isinstance(a, Tracer)]
        for a in args:
            if not isinstance(a, Tracer):
                raise TesseraTraceError(
                    f"trace: op {name!r} got a non-Tracer positional operand "
                    f"({type(a).__name__}); F1 requires tensor inputs to be "
                    "traced values (pass constants as keyword args)")
        in_shapes = [t.shape for t in tracer_args]
        out_shape = _infer_shape(name, in_shapes, kwargs)
        dtype = tracer_args[0].dtype if tracer_args else "fp32"
        ssa = self._fresh()
        self.body.append(IROp(
            result=ssa,
            op_name=graph_name,
            operands=[f"%{t.ssa}" for t in tracer_args],
            operand_types=[_ty(t.shape, t.dtype) for t in tracer_args],
            result_type=_ty(out_shape, dtype),
            kwargs=dict(kwargs),
        ))
        return Tracer(out_shape, dtype, ssa)

    def set_outputs(self, outs: List[str]) -> None:
        self.outputs = list(outs)

    def finish(self) -> TracedFunction:
        return TracedFunction(args=list(self.args), body=list(self.body),
                              outputs=list(self.outputs))


# ── trace entry points ────────────────────────────────────────────────────── #


def _spec_shape_dtype(spec: Any) -> Tuple[Tuple[int, ...], str]:
    if isinstance(spec, np.ndarray):
        return tuple(spec.shape), _np_dtype_to_elem(spec.dtype)
    if isinstance(spec, tuple) and len(spec) == 2 and not isinstance(spec[1], int):
        shape, dtype = spec
        return tuple(int(d) for d in shape), str(dtype)
    # a bare shape tuple
    return tuple(int(d) for d in spec), "fp32"


def _np_dtype_to_elem(dt) -> str:
    name = str(dt)
    if name == "bfloat16":
        return "bf16"
    if name in ("float16", "half"):
        return "f16"
    return "f32"


def trace(fn: Callable, *example_specs: Any) -> TracedFunction:
    """Abstractly interpret ``fn`` over ``Tracer`` args, returning the recorded
    :class:`TracedFunction`. ``example_specs`` are arrays, ``(shape, dtype)``
    pairs, or bare shape tuples — only shape/dtype are used."""
    tb = TraceBuilder()
    arg_tracers = []
    for i, spec in enumerate(example_specs):
        shape, dtype = _spec_shape_dtype(spec)
        arg_tracers.append(tb.arg(f"a{i}", shape, dtype))
    token = _trace_hook.set_active_tracer(tb)
    try:
        result = fn(*arg_tracers)
    finally:
        _trace_hook.reset_active_tracer(token)
    outs = result if isinstance(result, tuple) else (result,)
    for o in outs:
        if not isinstance(o, Tracer):
            raise TesseraTraceError(
                "trace: function must return Tracer value(s); got "
                f"{type(o).__name__}")
    tb.set_outputs([o.ssa for o in outs])
    return tb.finish()


# ── Layer 2 (straight-line) — traced graph_ir → executable GraphFn ────────── #


def to_graphfn(traced: TracedFunction, *, elem: str = "f32",
               target: str = "apple_gpu"):
    """Translate a straight-line :class:`TracedFunction` into an executable
    ``GraphFn`` by replaying each recorded op through the GraphFn builder (reusing
    ``graphfn_bridge._apply_op`` + ``_OP_TABLE``). An op outside the executable
    subset raises the same hard diagnostic the AST bridge uses (Decision #21)."""
    from .._jit_boundary import GraphFn, TesseraJitError
    from .graphfn_bridge import _apply_op

    if len(traced.outputs) != 1:
        raise TesseraJitError(
            "trace→GraphFn (F1) supports a single output; multi-output is F6")
    g = GraphFn(name="tessera_trace", elem=elem, target=target)
    env: Dict[str, Any] = {}
    for (ssa, shape, _dt) in traced.args:
        env[ssa] = g.arg(shape)
    for op in traced.body:
        assert op.result is not None  # traced ops always bind a result SSA
        env[op.result] = _apply_op(g, op, env, TesseraJitError)
    g.ret(env[traced.outputs[0]])
    return g


def run_traced(fn: Callable, *arrays: np.ndarray, target: str = "apple_gpu"):
    """Convenience: trace ``fn`` with ``arrays`` (as shape/dtype specs), lower to a
    ``GraphFn``, and execute — returning the result as ``np.ndarray``. The
    straight-line counterpart to ``jit_fori_loop`` for arbitrary ``tessera.ops``
    bodies. (Straight-line graphs have no control_for, so ``run()`` is the
    executor on both lanes.)"""
    arrs = [np.asarray(a) for a in arrays]
    traced = trace(fn, *arrs)
    elem = "bf16" if (arrs and _np_dtype_to_elem(arrs[0].dtype) == "bf16") else "f32"
    g = to_graphfn(traced, elem=elem, target=target)
    return g.run(*arrs)
