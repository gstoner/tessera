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

    def _trace_region(self, run) -> Any:
        """Run ``run()`` with a fresh sub-builder active (sharing this builder's
        SSA counter so names stay globally unique) and return ``(sub_body, value)``
        where ``sub_body`` is the recorded op-list and ``value`` is whatever
        ``run`` returned (a Tracer or tuple of Tracers)."""
        sub = TraceBuilder()
        sub._counter = self._counter
        token = _trace_hook.set_active_tracer(sub)
        try:
            value = run()
        finally:
            _trace_hook.reset_active_tracer(token)
        self._counter = sub._counter
        return sub.body, value

    def record_for_loop(self, lower: int, upper: int, body_fun, init_carry
                        ) -> "Tracer":
        """Trace a bounded ``fori_loop`` into a ``tessera.control_for`` IROp.
        ``body_fun(i, carry)`` is traced once with ``i=0`` (the control_for ABI is
        index-independent); the carry is captured by the body's returned Tracer."""
        if not isinstance(init_carry, Tracer):
            raise TesseraTraceError("traced fori_loop: init_val must be a Tracer")
        trip = int(upper) - int(lower)
        carry_ssa = self._fresh()
        carry = Tracer(init_carry.shape, init_carry.dtype, carry_ssa)
        sub_body, nxt = self._trace_region(lambda: body_fun(0, carry))
        if not isinstance(nxt, Tracer):
            raise TesseraTraceError("traced fori_loop: body must return a Tracer")
        if nxt.shape != init_carry.shape:
            raise TesseraTraceError(
                "traced fori_loop: body must preserve the carry shape")
        res = self._fresh()
        self.body.append(IROp(
            result=res, op_name="tessera.control_for",
            operands=[f"%{init_carry.ssa}"],
            operand_types=[_ty(init_carry.shape, init_carry.dtype)],
            result_type=_ty(init_carry.shape, init_carry.dtype),
            kwargs={"_region": "for", "_trip": trip, "_carry_ssa": carry_ssa,
                    "_next_ssa": nxt.ssa, "_body": sub_body},
        ))
        return Tracer(init_carry.shape, init_carry.dtype, res)

    def record_cond(self, pred, true_fun, false_fun, operands) -> "Tracer":
        """Trace a ``cond`` into a ``tessera.control_if`` IROp."""
        if not isinstance(pred, Tracer):
            raise TesseraTraceError("traced cond: pred must be a Tracer")
        then_body, tval = self._trace_region(lambda: true_fun(*operands))
        else_body, fval = self._trace_region(lambda: false_fun(*operands))
        if not (isinstance(tval, Tracer) and isinstance(fval, Tracer)):
            raise TesseraTraceError("traced cond: branches must return a Tracer")
        if tval.shape != fval.shape:
            raise TesseraTraceError("traced cond: branches must share a shape")
        res = self._fresh()
        self.body.append(IROp(
            result=res, op_name="tessera.control_if",
            operands=[f"%{pred.ssa}"],
            operand_types=[_ty(pred.shape, pred.dtype)],
            result_type=_ty(tval.shape, tval.dtype),
            kwargs={"_region": "if", "_flag_ssa": pred.ssa,
                    "_then_body": then_body, "_then_ssa": tval.ssa,
                    "_else_body": else_body, "_else_ssa": fval.ssa},
        ))
        return Tracer(tval.shape, tval.dtype, res)

    def record_while(self, cond_fun, body_fun, init, max_steps) -> "Tracer":
        """Trace a bounded ``while_loop`` into a ``tessera.control_while`` IROp."""
        if not isinstance(init, Tracer):
            raise TesseraTraceError("traced while_loop: init_val must be a Tracer")
        if max_steps is None:
            raise TesseraTraceError(
                "traced while_loop needs a bound: pass max_steps=N")
        carry_ssa = self._fresh()
        carry = Tracer(init.shape, init.dtype, carry_ssa)
        body_ops, nxt = self._trace_region(lambda: body_fun(carry))
        cond_ops, pred = self._trace_region(lambda: cond_fun(carry))
        if not (isinstance(nxt, Tracer) and isinstance(pred, Tracer)):
            raise TesseraTraceError(
                "traced while_loop: cond/body must return a Tracer")
        if nxt.shape != init.shape:
            raise TesseraTraceError(
                "traced while_loop: body must preserve the carry shape")
        res = self._fresh()
        self.body.append(IROp(
            result=res, op_name="tessera.control_while",
            operands=[f"%{init.ssa}"],
            operand_types=[_ty(init.shape, init.dtype)],
            result_type=_ty(init.shape, init.dtype),
            kwargs={"_region": "while", "_max_iters": int(max_steps),
                    "_carry_ssa": carry_ssa,
                    "_body": body_ops, "_next_ssa": nxt.ssa,
                    "_cond": cond_ops, "_pred_ssa": pred.ssa},
        ))
        return Tracer(init.shape, init.dtype, res)

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
    from .graphfn_bridge import _apply_op, _strip

    if len(traced.outputs) != 1:
        raise TesseraJitError(
            "trace→GraphFn supports a single output; multi-output is F6")
    g = GraphFn(name="tessera_trace", elem=elem, target=target)
    env: Dict[str, Any] = {}
    for (ssa, shape, _dt) in traced.args:
        env[ssa] = g.arg(shape)

    def _replay(ops, base_env):
        """Replay a straight-line sub op-list (a control-flow region body) over a
        copy of the enclosing env; return the env so the caller can read outputs."""
        e = dict(base_env)
        for bop in ops:
            assert bop.result is not None
            e[bop.result] = _apply_op(g, bop, e, TesseraJitError)
        return e

    for op in traced.body:
        assert op.result is not None
        if op.op_name == "tessera.control_for":
            kw = op.kwargs
            init = env[_strip(op.operands[0])]

            def _body(carry, kw=kw):
                e = dict(env)
                e[kw["_carry_ssa"]] = carry
                e = _replay(kw["_body"], e)
                return e[kw["_next_ssa"]]

            env[op.result] = g.for_loop(kw["_trip"], init=init, body=_body)
        elif op.op_name == "tessera.control_if":
            kw = op.kwargs
            flag = env[kw["_flag_ssa"]]
            env[op.result] = g.cond(
                flag,
                then_fn=lambda kw=kw: _replay(kw["_then_body"], env)[kw["_then_ssa"]],
                else_fn=lambda kw=kw: _replay(kw["_else_body"], env)[kw["_else_ssa"]])
        elif op.op_name == "tessera.control_while":
            kw = op.kwargs
            init = env[_strip(op.operands[0])]

            def _wbody(carry, kw=kw):
                e = dict(env)
                e[kw["_carry_ssa"]] = carry
                return _replay(kw["_body"], e)[kw["_next_ssa"]]

            def _wcond(carry, kw=kw):
                e = dict(env)
                e[kw["_carry_ssa"]] = carry
                return _replay(kw["_cond"], e)[kw["_pred_ssa"]]

            env[op.result] = g.while_loop(
                kw["_max_iters"], cond=_wcond, body=_wbody, init=init)
        else:
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
