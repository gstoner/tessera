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

import contextlib
import inspect
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from . import _trace_hook
from .graph_ir import IROp, apply_presence_flags
from .op_catalog import graph_name_for

if TYPE_CHECKING:
    from .graph_ir import GraphIRModule

# ── abstract value ────────────────────────────────────────────────────────── #


@dataclass(frozen=True)
class Tracer:
    """An abstract value flowing through a trace: shape + dtype + graph SSA."""

    shape: Tuple[int, ...]
    dtype: str
    ssa: str
    # F6 concrete tracing — the op's numpy result, threaded so shape/dtype come
    # from real execution (full-vocab inference, no per-op rule). ``None`` for
    # value-less ``(shape, dtype)`` specs (the abstract / shape-rule path).
    # Excluded from eq/hash (a numpy array is unhashable).
    value: Any = field(default=None, compare=False, repr=False)

    def __bool__(self):
        # A raw Python ``if tracer:`` / ``while tracer:`` would silently take one
        # path (a frozen dataclass is truthy) — the classic abstract-trace hazard.
        # Force the explicit data-dependent control-flow surface instead.
        raise TesseraTraceError(
            "cannot branch on a traced value with a Python `if`/`while`; use "
            "tessera.control.cond / tessera.control.while_loop for data-dependent "
            "control flow (a Python `for _ in range(N)` over a static N unrolls)")


@dataclass
class TracedFunction:
    """The result of a trace: typed args, a straight-line graph_ir body, and the
    SSA names of the outputs."""

    args: List[Tuple[str, Tuple[int, ...], str]]  # (ssa, shape, dtype)
    body: List[IROp]
    outputs: List[str]                            # output SSA names
    output_values: Tuple[Any, ...] = field(default=(), compare=False, repr=False)


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
    from .graph_ir import tensor_ir_type

    return str(tensor_ir_type(tuple(str(dim) for dim in shape), dtype))


@dataclass
class TraceBuilder:
    """Accumulates the graph_ir body as ``tessera.ops`` calls record themselves
    via :meth:`record_op` while this builder is the active tracer."""

    args: List[Tuple[str, Tuple[int, ...], str]] = field(default_factory=list)
    body: List[IROp] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    _counter: int = 0

    def arg(self, ssa: str, shape, dtype: str, value: Any = None) -> Tracer:
        sh = tuple(int(d) for d in shape)
        self.args.append((ssa, sh, dtype))
        return Tracer(sh, dtype, ssa, value)

    def _fresh(self) -> str:
        n = self._counter
        self._counter += 1
        return f"v{n}"

    def record_op(
        self, name: str, original, args: tuple, kwargs: dict
    ) -> Tracer | tuple[Tracer, ...]:
        graph_name = graph_name_for(name)
        if graph_name is None:
            raise TesseraTraceError(f"trace: op {name!r} is not in the op catalog")
        # `call_args` mirrors `args` with Tracer→value substituted lazily; it
        # preserves variadic-operand *grouping* (a list/tuple of Tracers, the
        # cat/stack pattern) so the concrete-execution path can reconstruct the
        # original call.  `tracer_args` is the flattened operand set used for the
        # IR (one SSA ref per tensor) and shape inference.
        tracer_args: List[Tracer] = []
        call_args: List[Any] = []
        call_kwargs = dict(kwargs)
        ir_kwargs = dict(kwargs)
        for a in args:
            if isinstance(a, Tracer):
                tracer_args.append(a)
                call_args.append(a)
            elif (isinstance(a, (list, tuple)) and a
                  and all(isinstance(x, Tracer) for x in a)):
                # Variadic tensor-list operand (e.g. ``cat([a, b], axis=…)``):
                # flatten its tracers into the operand set but keep the group so
                # ``original`` is still called as ``op([v0, v1], …)``.
                tracer_args.extend(a)
                call_args.append(list(a))
            else:
                raise TesseraTraceError(
                    f"trace: op {name!r} got a non-Tracer positional operand "
                    f"({type(a).__name__}); tensor inputs must be traced values "
                    "(pass constants as keyword args)")
        # Tensor-valued keyword operands are dataflow, not attributes. Preserve
        # the Python signature's parameter order so ``gamma=``/``beta=`` and
        # similar optional operands cannot change the ODS operand ABI merely by
        # changing call-site keyword order.
        try:
            parameter_order = tuple(inspect.signature(original).parameters)
        except (TypeError, ValueError):
            parameter_order = tuple(kwargs)
        ordered_keys = [key for key in parameter_order if key in kwargs]
        ordered_keys.extend(key for key in kwargs if key not in ordered_keys)
        positional_operand_count = len(tracer_args)
        keyword_operand_names: List[str] = []
        for key in ordered_keys:
            item = kwargs[key]
            if isinstance(item, Tracer):
                tracer_args.append(item)
                ir_kwargs.pop(key, None)
                keyword_operand_names.append(key)
            elif (isinstance(item, (list, tuple)) and item
                  and all(isinstance(value, Tracer) for value in item)):
                tracer_args.extend(item)
                ir_kwargs.pop(key, None)
                keyword_operand_names.append(key)
        # Presence of each optional operand, for ops whose operand list cannot
        # be decoded from position alone (`graph_ir._PRESENCE_FLAGGED_OPERANDS`).
        #
        # Emitted here as well as in the AST frontend because the two are held
        # to structural parity: a fact recorded by one and not the other is a
        # frontend divergence, and the differential certificate rejects it --
        # which is exactly how this omission was caught.
        apply_presence_flags(
            graph_name, ir_kwargs, positional_operand_count,
            keyword_operand_names)
        # F6 concrete tracing: when every input carries a concrete value, run the
        # real numpy op to get the result's shape/dtype (works for ANY op, no
        # per-op shape rule). Falls back to the shape-rule path for value-less
        # specs (the executable subset only).
        if tracer_args and all(t.value is not None for t in tracer_args):
            def _concrete(x: Any) -> Any:
                if isinstance(x, Tracer):
                    return x.value
                return [t.value for t in x]  # variadic group
            concrete_kwargs = {
                key: _concrete(item)
                if isinstance(item, Tracer) or (
                    isinstance(item, (list, tuple)) and item
                    and all(isinstance(value, Tracer) for value in item)
                )
                else item
                for key, item in call_kwargs.items()
            }
            # The catalog operation is the trace boundary.  Its eager
            # implementation may itself call other registered operations
            # (GQA expands K/V and calls flash attention, for example).  Run
            # that implementation with tracing suspended so those private
            # implementation details do not become duplicate Graph nodes or
            # receive already-materialized ndarray operands.
            token = _trace_hook.set_active_tracer(None)
            try:
                out = original(
                    *[_concrete(x) for x in call_args], **concrete_kwargs
                )
            finally:
                _trace_hook.reset_active_tracer(token)
            concrete_outputs = out if isinstance(out, tuple) else (out,)
            if not concrete_outputs:
                raise TesseraTraceError(
                    f"trace: op {name!r} returned an empty result tuple"
                )
            arrays = tuple(np.asarray(value) for value in concrete_outputs)
            out_shapes = tuple(tuple(value.shape) for value in arrays)
            dtypes = tuple(_np_dtype_to_elem(value.dtype) for value in arrays)
            values: tuple[Any, ...] = arrays
        else:
            out_shapes = (
                _infer_shape(name, [t.shape for t in tracer_args], ir_kwargs),
            )
            dtypes = (tracer_args[0].dtype if tracer_args else "fp32",)
            values = (None,)
        if graph_name == "tessera.reduce" and name in {"sum", "mean"}:
            ir_kwargs.setdefault("kind", name)
        from .graph_ir import _canonicalize_spectral_attrs, tensor_ir_type
        operand_ir_types = [
            tensor_ir_type(tuple(str(dim) for dim in item.shape), item.dtype)
            for item in tracer_args
        ]
        _canonicalize_spectral_attrs(graph_name, operand_ir_types, ir_kwargs)
        ssas = tuple(self._fresh() for _ in out_shapes)
        result_ir_types = tuple(
            tensor_ir_type(tuple(str(dim) for dim in shape), dtype)
            for shape, dtype in zip(out_shapes, dtypes, strict=True)
        )
        result_type = (
            str(result_ir_types[0])
            if len(result_ir_types) == 1
            else "(" + ", ".join(map(str, result_ir_types)) + ")"
        )
        self.body.append(IROp(
            result=",".join(ssas),
            op_name=graph_name,
            operands=[f"%{t.ssa}" for t in tracer_args],
            operand_types=[_ty(t.shape, t.dtype) for t in tracer_args],
            result_type=result_type,
            kwargs=ir_kwargs,
            inferred_type=result_ir_types[0],
            inferred_types=result_ir_types,
        ))
        traced_outputs = tuple(
            Tracer(shape, dtype, ssa, value)
            for shape, dtype, ssa, value in zip(
                out_shapes, dtypes, ssas, values, strict=True
            )
        )
        return traced_outputs[0] if len(traced_outputs) == 1 else traced_outputs

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
        carry = Tracer(init_carry.shape, init_carry.dtype, carry_ssa,
                       init_carry.value)
        sub_body, nxt = self._trace_region(lambda: body_fun(0, carry))
        if not isinstance(nxt, Tracer):
            raise TesseraTraceError("traced fori_loop: body must return a Tracer")
        if nxt.shape != init_carry.shape:
            raise TesseraTraceError(
                "traced fori_loop: body must preserve the carry shape")
        # CF1: match ControlForOp::verify — the loop result type must equal the
        # carried iter_arg type (dtype as well as shape, not just shape).
        if nxt.dtype != init_carry.dtype:
            raise TesseraTraceError(
                "traced fori_loop: body must preserve the carry dtype "
                f"(carry {init_carry.dtype}, body returned {nxt.dtype})")
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

    def record_cond(self, pred, true_fun, false_fun, operands) -> Any:
        """Trace a ``cond`` into a ``tessera.control_if`` IROp."""
        if not isinstance(pred, Tracer):
            raise TesseraTraceError("traced cond: pred must be a Tracer")
        then_body, tval = self._trace_region(lambda: true_fun(*operands))
        else_body, fval = self._trace_region(lambda: false_fun(*operands))
        then_values = tval if isinstance(tval, tuple) else (tval,)
        else_values = fval if isinstance(fval, tuple) else (fval,)
        if not then_values or not all(isinstance(value, Tracer) for value in then_values):
            raise TesseraTraceError("traced cond: branches must return Tracers")
        if len(then_values) != len(else_values) or not all(
            isinstance(value, Tracer) for value in else_values
        ):
            raise TesseraTraceError("traced cond: branches must return equal-arity state")
        for index, (then_value, else_value) in enumerate(zip(then_values, else_values)):
            if then_value.shape != else_value.shape:
                raise TesseraTraceError(
                    "traced cond: branch result "
                    f"{index} must share a shape "
                    f"(then {then_value.shape}/{then_value.dtype}, "
                    f"else {else_value.shape}/{else_value.dtype})"
                )
            if then_value.dtype != else_value.dtype:
                raise TesseraTraceError(
                    "traced cond: branch result "
                    f"{index} must share a dtype "
                    f"(then {then_value.shape}/{then_value.dtype}, "
                    f"else {else_value.shape}/{else_value.dtype})"
                )
        results = tuple(self._fresh() for _ in then_values)
        result_types = tuple(_ty(value.shape, value.dtype) for value in then_values)
        self.body.append(IROp(
            result=",".join(results), op_name="tessera.control_if",
            operands=[f"%{pred.ssa}"],
            operand_types=[_ty(pred.shape, pred.dtype)],
            result_type=(
                result_types[0]
                if len(result_types) == 1
                else "(" + ", ".join(result_types) + ")"
            ),
            kwargs={"_region": "if", "_flag_ssa": pred.ssa,
                    "_then_body": then_body,
                    "_then_ssas": tuple(value.ssa for value in then_values),
                    "_then_ssa": then_values[0].ssa,
                    "_else_body": else_body,
                    "_else_ssas": tuple(value.ssa for value in else_values),
                    "_else_ssa": else_values[0].ssa},
        ))
        traced_results = tuple(
            Tracer(value.shape, value.dtype, result)
            for value, result in zip(then_values, results)
        )
        return traced_results[0] if len(traced_results) == 1 else traced_results

    def record_while(self, cond_fun, body_fun, init, max_steps) -> "Tracer":
        """Trace a bounded ``while_loop`` into a ``tessera.control_while`` IROp."""
        if not isinstance(init, Tracer):
            raise TesseraTraceError("traced while_loop: init_val must be a Tracer")
        if max_steps is None:
            raise TesseraTraceError(
                "traced while_loop needs a bound: pass max_steps=N")
        carry_ssa = self._fresh()
        carry = Tracer(init.shape, init.dtype, carry_ssa, init.value)
        body_ops, nxt = self._trace_region(lambda: body_fun(carry))
        cond_ops, pred = self._trace_region(lambda: cond_fun(carry))
        if not (isinstance(nxt, Tracer) and isinstance(pred, Tracer)):
            raise TesseraTraceError(
                "traced while_loop: cond/body must return a Tracer")
        if nxt.shape != init.shape:
            raise TesseraTraceError(
                "traced while_loop: body must preserve the carry shape")
        # CF1: match ControlWhileOp::verify — the while result type must equal
        # the carried iter_arg type (dtype as well as shape).
        if nxt.dtype != init.dtype:
            raise TesseraTraceError(
                "traced while_loop: body must preserve the carry dtype "
                f"(carry {init.dtype}, body returned {nxt.dtype})")
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

    def record_scan(self, fn, init, xs, length) -> "Tuple[Tracer, Tracer]":
        """Trace a fused ``scan`` into a ``tessera.control_scan`` IROp (H3b).
        ``fn(carry, x_t) -> (carry, y)`` is traced once over the carry + a per-step
        ``x_t`` slice; the op carries the body op-list + the carry/x_t/ys SSAs so
        ``execute_traced`` can dispatch ``run_graph_scan_f32``. Returns the carry
        Tracer; the stacked-ys Tracer is stashed on the op (``_ys_ssa``) and
        returned by ``scan`` (see the wrapper below)."""
        if not (isinstance(init, Tracer) and isinstance(xs, Tracer)):
            raise TesseraTraceError("traced scan: init and xs must be Tracers")
        if len(xs.shape) < 1:
            raise TesseraTraceError("traced scan: xs must have a leading scan axis")
        trip = int(length) if length is not None else int(xs.shape[0])
        x_inner = tuple(xs.shape[1:])
        carry_ssa, xt_ssa = self._fresh(), self._fresh()
        carry = Tracer(init.shape, init.dtype, carry_ssa, init.value)
        xt_val = (np.asarray(xs.value)[0] if xs.value is not None else None)
        xt = Tracer(x_inner, xs.dtype, xt_ssa, xt_val)
        body_ops, out = self._trace_region(lambda: fn(carry, xt))
        if not (isinstance(out, tuple) and len(out) == 2
                and isinstance(out[0], Tracer) and isinstance(out[1], Tracer)):
            raise TesseraTraceError("traced scan: fn must return (carry, y)")
        nxt, y = out
        if nxt.shape != init.shape:
            raise TesseraTraceError("traced scan: body must preserve carry shape")
        # CF1: the scan carry dtype must be stable across the step (the recurrent
        # state type is fixed), matching the loop ops' carry-type contract.
        if nxt.dtype != init.dtype:
            raise TesseraTraceError(
                "traced scan: body must preserve carry dtype "
                f"(carry {init.dtype}, body returned {nxt.dtype})")
        res_c, res_y = self._fresh(), self._fresh()
        ys_shape = (trip, *y.shape)
        self.body.append(IROp(
            result=res_c, op_name="tessera.control_scan",
            operands=[f"%{init.ssa}", f"%{xs.ssa}"],
            operand_types=[_ty(init.shape, init.dtype), _ty(xs.shape, xs.dtype)],
            result_type=_ty(init.shape, init.dtype),
            kwargs={"_region": "scan", "_trip": trip,
                    "_carry_ssa": carry_ssa, "_xt_ssa": xt_ssa,
                    "_init_ssa": init.ssa, "_xs_ssa": xs.ssa,
                    "_body": body_ops, "_next_ssa": nxt.ssa, "_y_ssa": y.ssa,
                    "_ys_ssa": res_y, "_carry_shape": tuple(init.shape),
                    "_x_shape": x_inner, "_y_shape": tuple(y.shape)},
        ))
        # carry value = one-step (shape-correct); ys value is a shape-only
        # placeholder (GPU computes the real values in execute_traced).
        ys_val = (np.zeros(ys_shape, dtype=np.float32)
                  if init.value is not None else None)
        carry_t = Tracer(init.shape, init.dtype, res_c, nxt.value)
        ys_t = Tracer(ys_shape, y.dtype, res_y, ys_val)
        return carry_t, ys_t

    def set_outputs(self, outs: List[str]) -> None:
        self.outputs = list(outs)

    def finish(self, output_values: Tuple[Any, ...] = ()) -> TracedFunction:
        return TracedFunction(args=list(self.args), body=list(self.body),
                              outputs=list(self.outputs), output_values=output_values)


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
    if name == "complex64":
        return "complex64"
    if name == "complex128":
        return "complex128"
    if name == "float64":
        return "f64"
    return "f32"


def trace(
    fn: Callable,
    *example_specs: Any,
    arg_names: Sequence[str] | None = None,
) -> TracedFunction:
    """Interpret ``fn`` over ``Tracer`` args, returning the recorded
    :class:`TracedFunction`. ``example_specs`` are arrays (concrete tracing —
    shapes/dtypes come from real numpy execution, full vocab), ``(shape, dtype)``
    pairs, or bare shape tuples (abstract tracing — shape rules, executable
    subset only)."""
    if arg_names is not None and len(arg_names) != len(example_specs):
        raise TesseraTraceError("trace arg_names must match the example arity")
    tb = TraceBuilder()
    arg_tracers = []
    for i, spec in enumerate(example_specs):
        shape, dtype = _spec_shape_dtype(spec)
        value = np.ascontiguousarray(spec) if isinstance(spec, np.ndarray) else None
        name = str(arg_names[i]) if arg_names is not None else f"a{i}"
        arg_tracers.append(tb.arg(name, shape, dtype, value))
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
    return tb.finish(tuple(o.value for o in outs))


def to_graph_ir_module(
    traced: TracedFunction,
    *,
    name: str = "traced",
    source_hash: str | None = None,
) -> "GraphIRModule":
    """Promote a trace directly to the canonical Graph IR object model.

    This is the production frontend boundary.  It intentionally does not replay
    through ``GraphFn`` or rebuild operations from Python names, so operand SSA,
    registered effects, stochastic identity, and region bodies retain the
    identity recorded by the tracer.
    """
    from .graph_ir import GraphIRFunction, GraphIRModule, IRArg, IRType, tensor_ir_type
    from .structured_cfg import recover_structured_cfg

    result_types: list[IRType] = []
    by_result = {
        result: op
        for op in traced.body
        for result in op.result_names
    }
    for output in traced.outputs:
        operation = by_result.get(output)
        if operation is None or not operation.result_type:
            raise TesseraTraceError(
                f"trace output %{output} has no typed Graph IR definition"
            )
        result_types.append(IRType(operation.result_type))
    structured_cfg = recover_structured_cfg(traced.body)
    function = GraphIRFunction(
        name=name,
        args=[
            IRArg(ssa, tensor_ir_type(tuple(str(dim) for dim in shape), dtype))
            for ssa, shape, dtype in traced.args
        ],
        result_types=result_types,
        body=list(traced.body),
        fn_attrs={
            "tessera.frontend.authority": '"tracer"',
            "tessera.structured_cfg.schema": '"tessera.structured_cfg.v1"',
            "tessera.structured_cfg.digest": f'"{structured_cfg.digest}"',
            "tessera.structured_cfg.blocks": str(len(structured_cfg.blocks)),
        },
        return_values=[f"%{output}" for output in traced.outputs],
        source_hash=source_hash,
        structured_cfg=structured_cfg,
    )
    module = GraphIRModule(
        functions=[function],
        module_attrs={
            "tessera.ir.version": '"1.0"',
            "tessera.frontend.authority": '"tracer"',
        },
    )
    verification = module.verify()
    if not verification.ok:
        raise TesseraTraceError(verification.format())
    return module


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
            if len(op.result_names) != 1:
                raise TesseraJitError(
                    "trace→GraphFn requires a native variadic control_if consumer"
                )
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


# ── Layer 2 (general) — concrete interpreter over a traced function ──────── #
#
# F3 lifts the GraphFn-executor constraints (return == construct result, loop
# init == function arg) that limit the to_graphfn path: this walks the traced op
# list with a concrete env, running straight-line ops as per-op Apple GPU kernels
# (``agb.gpu_*``) and each control region as ONE fused ``run_graph_*`` dispatch
# whose "args" are the live concrete values the region's body references — so a
# control construct can sit anywhere, with straight-line code before and after.


def _has_control_flow(traced: TracedFunction) -> bool:
    return any(op.op_name.startswith("tessera.control_") for op in traced.body)


def _branch_dicts(body_ops, idof: Dict[str, int], base: int):
    """Serialize a region's straight-line body to the run_graph op-list ABI.
    ``idof`` maps already-bound SSAs (live args + carry) to ids; op ``j`` binds id
    ``base + j``. Returns ``(dicts, idof)``."""
    from .graphfn_bridge import _OP_TABLE, _strip

    idof = dict(idof)
    out = []
    for j, op in enumerate(body_ops):
        entry = _OP_TABLE.get(op.op_name)
        if entry is None:
            raise TesseraTraceError(
                f"trace exec: region op {op.op_name!r} is not executable on "
                "apple_gpu")
        name = entry[0]
        e: dict = {"op": name, "in0": idof[_strip(op.operands[0])]}
        if name == "matmul":
            e["in1"] = idof[_strip(op.operands[1])]
            e["transpose_a"] = bool(op.kwargs.get("transpose_a"))
            e["transpose_b"] = bool(op.kwargs.get("transpose_b"))
        elif name in ("add", "sub", "mul", "div"):
            e["in1"] = idof[_strip(op.operands[1])]
        elif name in ("rmsnorm", "layer_norm"):
            e["eps"] = float(op.kwargs.get("eps", 1e-5))
        out.append(e)
        assert op.result is not None
        idof[op.result] = base + j
    return out, idof


def _live_refs(bodies, exclude: set, extra_refs=()) -> List[str]:
    """Ordered external SSAs referenced across ``bodies`` (region op-lists) plus
    ``extra_refs`` (region OUTPUT ssas — a branch/cond can return a bare external
    arg with no ops), excluding region-internal results and ``exclude`` (carry)."""
    from .graphfn_bridge import _strip

    internal = {op.result for body in bodies for op in body}
    live: List[str] = []
    seen: set = set()

    def consider(s: str) -> None:
        if s not in internal and s not in exclude and s not in seen:
            seen.add(s)
            live.append(s)

    for body in bodies:
        for op in body:
            for o in op.operands:
                consider(_strip(o))
    for s in extra_refs:
        consider(_strip(s))
    return live


def _region_flat(ops) -> bool:
    """A region body/branch is *flat* (serializable to a single fused
    ``run_graph_*`` op-list) iff it contains no nested control op."""
    return not any(op.op_name.startswith("tessera.control_") for op in ops)


def execute_traced(traced: TracedFunction, arrays: List[np.ndarray]):
    """Concrete Apple GPU interpreter over a traced function (F3 + H1). Walks the
    op list with a concrete env; ``exec_op`` dispatches each op:

    * straight-line → a per-op kernel (``_gpu_straightline_op``).
    * a control region whose body/branches are **flat** → ONE fused
      ``run_graph_*`` dispatch over its live concrete inputs.
    * a control region whose body **contains a nested control op** (H1) →
      host-orchestration: run the region's trip/branches as a Python loop,
      threading the concrete carry, recursively calling ``exec_op`` (so the inner
      construct still fuses while the outer runs per-step on the host).
    """
    from tessera import _apple_gpu_backend as agb
    from tessera import apple_mlpkg as mp

    from .graphfn_bridge import _strip

    def shapes(names, e):
        return [tuple(e[s].shape) for s in names]

    def _replay(ops, base_env):
        e = dict(base_env)
        for bop in ops:
            assert bop.result is not None
            e[bop.result] = exec_op(bop, e)
        return e

    def _pred_true(arr) -> bool:
        return float(np.asarray(arr).reshape(-1)[0]) > 0.0

    def exec_op(op, env):
        nm = op.op_name
        if nm == "tessera.control_for":
            kw = op.kwargs
            carry_ssa, init_ssa = kw["_carry_ssa"], _strip(op.operands[0])
            if _region_flat(kw["_body"]):
                live = _live_refs([kw["_body"]], {carry_ssa}, [kw["_next_ssa"]])
                args = [env[s] for s in live] + [env[init_ssa]]
                n_args = len(args)
                idof = {s: i for i, s in enumerate(live)}
                idof[carry_ssa] = n_args
                dicts, local = _branch_dicts(kw["_body"], idof, n_args + 1)
                return mp.run_graph_loop_f32(
                    args, shapes(live, env) + [tuple(env[init_ssa].shape)],
                    len(live), kw["_trip"], dicts, local[kw["_next_ssa"]],
                    tuple(env[init_ssa].shape))
            # H1 — nested control in the body: host-orchestrate the trip.
            carry = env[init_ssa]
            for _ in range(int(kw["_trip"])):
                benv = _replay(kw["_body"], {**env, carry_ssa: carry})
                carry = benv[kw["_next_ssa"]]
            return carry
        if nm == "tessera.control_if":
            if len(op.result_names) != 1:
                raise TesseraTraceError(
                    "traced execution requires a native variadic control_if consumer"
                )
            kw = op.kwargs
            flag_ssa = kw["_flag_ssa"]
            if _region_flat(kw["_then_body"]) and _region_flat(kw["_else_body"]):
                live = _live_refs([kw["_then_body"], kw["_else_body"]], set(),
                                  [flag_ssa, kw["_then_ssa"], kw["_else_ssa"]])
                if flag_ssa not in live:
                    live = [flag_ssa] + live
                idof = {s: i for i, s in enumerate(live)}
                tdicts, tlocal = _branch_dicts(kw["_then_body"], idof, len(live))
                edicts, elocal = _branch_dicts(kw["_else_body"], idof, len(live))
                out_shape = tuple(env[kw["_then_ssa"]].shape) \
                    if kw["_then_ssa"] in env else _result_shape(op)
                return mp.run_graph_cond_f32(
                    [env[s] for s in live], shapes(live, env),
                    live.index(flag_ssa), tdicts, tlocal[kw["_then_ssa"]],
                    edicts, elocal[kw["_else_ssa"]], out_shape)
            # H1 — nested control in a branch: evaluate the flag on the host and
            # run only the taken branch (divergent semantics, like the fused op).
            if _pred_true(env[flag_ssa]):
                return _replay(kw["_then_body"], env)[kw["_then_ssa"]]
            return _replay(kw["_else_body"], env)[kw["_else_ssa"]]
        if nm == "tessera.control_while":
            kw = op.kwargs
            carry_ssa, init_ssa = kw["_carry_ssa"], _strip(op.operands[0])
            if _region_flat(kw["_body"]) and _region_flat(kw["_cond"]):
                live = _live_refs([kw["_body"], kw["_cond"]], {carry_ssa},
                                  [kw["_next_ssa"], kw["_pred_ssa"]])
                args = [env[s] for s in live] + [env[init_ssa]]
                n_args = len(args)
                idof = {s: i for i, s in enumerate(live)}
                idof[carry_ssa] = n_args
                bdicts, blocal = _branch_dicts(kw["_body"], idof, n_args + 1)
                cdicts, clocal = _branch_dicts(kw["_cond"], idof, n_args + 1)
                return mp.run_graph_while_f32(
                    args, shapes(live, env) + [tuple(env[init_ssa].shape)],
                    len(live), kw["_max_iters"], bdicts, blocal[kw["_next_ssa"]],
                    cdicts, clocal[kw["_pred_ssa"]], tuple(env[init_ssa].shape))
            # H1 — nested control: host-orchestrate (cond-then-body; freeze on
            # false, matching the fused forLoop+select-masking semantics).
            carry = env[init_ssa]
            for _ in range(int(kw["_max_iters"])):
                cenv = _replay(kw["_cond"], {**env, carry_ssa: carry})
                if not _pred_true(cenv[kw["_pred_ssa"]]):
                    break
                benv = _replay(kw["_body"], {**env, carry_ssa: carry})
                carry = benv[kw["_next_ssa"]]
            return carry
        if nm == "tessera.control_scan":
            # H3b — fused scan via run_graph_scan_f32 (or host-orchestrate when the
            # body nests control). exec_op returns the carry and stashes the
            # stacked ys in env under `_ys_ssa`.
            kw = op.kwargs
            carry_ssa, xt_ssa = kw["_carry_ssa"], kw["_xt_ssa"]
            init_ssa, xs_ssa = _strip(op.operands[0]), _strip(op.operands[1])
            trip = int(kw["_trip"])
            xs = env[xs_ssa]
            if _region_flat(kw["_body"]):
                live = _live_refs([kw["_body"]],
                                  {carry_ssa, xt_ssa, xs_ssa, init_ssa},
                                  [kw["_next_ssa"], kw["_y_ssa"]])
                # consts = [carry init] + the body's external refs; ids:
                # init=0, live[i]=i+1, carry=nc, x_t=nc+1, body op j=nc+2+j.
                const_arrays = [env[init_ssa]] + [env[s] for s in live]
                const_shapes = [tuple(env[init_ssa].shape)] + shapes(live, env)
                nc = len(const_arrays)
                idof = {init_ssa: 0}
                for i, s in enumerate(live):
                    idof[s] = i + 1
                idof[carry_ssa] = nc
                idof[xt_ssa] = nc + 1
                dicts, local = _branch_dicts(kw["_body"], idof, nc + 2)
                res = mp.run_graph_scan_f32(
                    const_arrays, const_shapes, 0, xs, trip,
                    tuple(kw["_x_shape"]), dicts, local[kw["_next_ssa"]],
                    local[kw["_y_ssa"]], tuple(kw["_carry_shape"]),
                    tuple(kw["_y_shape"]))
                if res is None:
                    raise TesseraTraceError("trace exec: scan dispatch failed")
                carry, ys = res
                env[kw["_ys_ssa"]] = ys
                return carry
            # nested control in the scan body → host-orchestrate.
            carry = env[init_ssa]
            ys_list = []
            for t in range(trip):
                benv = _replay(kw["_body"],
                               {**env, carry_ssa: carry, xt_ssa: xs[t]})
                carry = benv[kw["_next_ssa"]]
                ys_list.append(benv[kw["_y_ssa"]])
            env[kw["_ys_ssa"]] = np.stack(ys_list)
            return carry
        return _gpu_straightline_op(agb, op, env)

    env: Dict[str, np.ndarray] = {}
    for (ssa, _shape, _dt), arr in zip(traced.args, arrays):
        env[ssa] = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    for op in traced.body:
        assert op.result is not None
        env[op.result] = exec_op(op, env)
        if env[op.result] is None:
            raise TesseraTraceError(
                f"trace exec: op {op.op_name!r} dispatch failed / runtime "
                "unavailable")
    # Multi-output (e.g. scan → (carry, ys)) returns a tuple; single → the array.
    outs = [env[s].copy() for s in traced.outputs]
    return outs[0] if len(outs) == 1 else tuple(outs)


def _result_shape(op: IROp) -> Tuple[int, ...]:
    rt = op.result_type or ""
    inside = rt[rt.find("<") + 1:rt.rfind("x")] if "<" in rt else ""
    try:
        return tuple(int(d) for d in inside.split("x") if d)
    except ValueError:
        return ()


def _gpu_straightline_op(agb, op: IROp, env) -> np.ndarray:
    from .graphfn_bridge import _strip

    nm = op.op_name
    ins = [env[_strip(o)] for o in op.operands]
    kw = op.kwargs or {}
    if nm in ("tessera.matmul", "tessera.gemm"):
        a = ins[0].T if kw.get("transpose_a") else ins[0]
        b = ins[1].T if kw.get("transpose_b") else ins[1]
        return agb.gpu_matmul(np.ascontiguousarray(a), np.ascontiguousarray(b))
    if nm == "tessera.softmax":
        return agb.gpu_softmax(ins[0])
    if nm == "tessera.gelu":
        return agb.gpu_gelu(ins[0])
    if nm in ("tessera.relu", "tessera.sigmoid", "tessera.tanh", "tessera.silu"):
        return agb.gpu_unary(nm.split(".", 1)[1], ins[0])
    if nm == "tessera.rmsnorm":
        return agb.gpu_rmsnorm(ins[0], eps=float(kw.get("eps", 1e-5)))
    if nm == "tessera.layer_norm":
        return agb.gpu_layer_norm(ins[0], eps=float(kw.get("eps", 1e-5)))
    if nm in ("tessera.add", "tessera.sub", "tessera.mul", "tessera.div"):
        return agb.gpu_binary(nm.split(".", 1)[1], ins[0], ins[1])
    if nm == "tessera.transpose":
        return np.ascontiguousarray(ins[0].T)
    raise TesseraTraceError(
        f"trace exec: op {nm!r} is not executable on apple_gpu")


# ── F4 — wire @jit(target="apple_gpu") to trace-by-running ────────────────── #
#
# Behind a flag while F4 oracles parity against the AST bridge; F5 flips it to the
# default and retires the AST `_OpExtractor` + detect_loop_fn/detect_cond_fn path.

# F5: default ON. Only control-flow apple_gpu @jit functions are intercepted
# (see ``function_needs_tracer`` + the surgical gate in ``JitFn.__call__``);
# straight-line falls through to the existing package/auto_batch/canonical path
# untouched. Set ``TESSERA_JIT_TRACE=0`` to force the legacy behavior.
_JIT_TRACE = [os.environ.get("TESSERA_JIT_TRACE", "1") not in ("0", "false", "no")]


def jit_trace_enabled() -> bool:
    """Whether `@jit(target="apple_gpu")` routes **control-flow** functions through
    the tracer (F5 — default on). Toggle with ``set_jit_trace`` / the ``jit_trace``
    context manager / the ``TESSERA_JIT_TRACE`` env var."""
    return _JIT_TRACE[0]


def function_needs_tracer(graph_ir_module: Any, fn) -> bool:
    """Whether ``fn`` uses control flow and so must route through the tracer
    (F5 surgical gate): a raw Python ``for``/``if`` (its graph_ir carries
    ``tessera.scf.*`` markers) or an explicit ``tessera.control.*`` call. Pure
    straight-line functions return ``False`` and keep the existing apple_gpu path
    (package lane / auto_batch / canonical runtime) untouched."""
    import inspect
    import re

    try:
        for f in getattr(graph_ir_module, "functions", []) or []:
            for op in getattr(f, "body", []):
                if op.op_name.startswith("tessera.scf."):
                    return True
    except Exception:
        pass
    try:
        src = inspect.getsource(fn)
        if re.search(r"\bcontrol\.(fori_loop|cond|while_loop|scan)\b", src):
            return True
    except (OSError, TypeError):
        pass
    return False


def set_jit_trace(on: bool) -> None:
    _JIT_TRACE[0] = bool(on)


@contextlib.contextmanager
def jit_trace(on: bool = True):
    """Context manager toggling @jit trace-by-running (used by parity tests)."""
    prev = _JIT_TRACE[0]
    _JIT_TRACE[0] = bool(on)
    try:
        yield
    finally:
        _JIT_TRACE[0] = prev


class TesseraCallBindingError(TypeError):
    """A call the function's own signature rejects.

    Subclasses ``TypeError`` so ``except TypeError`` keeps working exactly as it
    does for a plain Python call -- the point is to preserve those semantics,
    not to invent a new contract. It is a distinct type only so the apple_gpu
    tracer's diagnostic wrapper can tell a caller error from a tracer failure
    and let it through unwrapped; reporting "the tracer could not execute this"
    for a misspelled keyword would name the wrong culprit.
    """


def _reject_invalid_call_binding(fn: Callable, args: tuple, kwargs: dict) -> None:
    """Raise for a call Python itself would reject, before the tracer sees it.

    ``run_jit_traced`` rebuilds the tracer's positional inputs by matching
    keyword names against the recovered ABI, and a keyword matching no parameter
    is simply never appended. On its own that turns a misspelled option into a
    silent trace against the parameter's DEFAULT — ``f(x=x, scael=2)`` on
    ``def f(x, scale=1)`` returned a result computed with ``scale=1`` where a
    plain Python call raises ``TypeError``. A wrong number is worse than an
    error, so bind against the real signature first: that restores unexpected
    keyword, duplicate, missing-required, and positional-/keyword-only errors in
    one step instead of re-deriving each rule here (Decision #30).

    Fail-soft on introspection only. A callable whose signature cannot be read
    (a builtin, a C extension, a ``*args``-only wrapper) is left exactly as
    permissive as it was rather than newly rejected.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return
    try:
        signature.bind(*args, **kwargs)
    except TypeError as exc:
        raise TesseraCallBindingError(str(exc)) from None


def run_jit_traced(jitfn: Any, args: tuple, kwargs: dict):
    """Execute a control-flow `@jit(target="apple_gpu")` function via the tracer
    (F5). Reached only for functions ``function_needs_tracer`` flagged — straight-
    line functions never get here; they keep the existing apple_gpu path. Trace
    ``jitfn._fn`` (concrete, full vocab); explicit ``tessera.control.*`` runs via
    ``execute_traced`` (fused ``run_graph_*``); a raw ``for _ in range(N)``
    unrolls to a straight-line trace and runs via the GraphFn lane."""
    _reject_invalid_call_binding(jitfn._fn, args, kwargs)
    names = list(getattr(jitfn, "arg_names", []) or [])
    ordered: List[Any] = list(args)
    for nm in names[len(ordered):]:
        if nm in kwargs:
            ordered.append(kwargs[nm])
    arrays = [np.asarray(a) for a in ordered]
    traced = trace(jitfn._fn, *arrays)
    if _has_control_flow(traced):
        return execute_traced(traced, arrays)
    elem = "bf16" if (arrays and _np_dtype_to_elem(arrays[0].dtype) == "bf16") \
        else "f32"
    return to_graphfn(traced, elem=elem, target="apple_gpu").run(*arrays)


def run_traced(fn: Callable, *arrays: np.ndarray, target: str = "apple_gpu"):
    """Convenience: trace ``fn`` with ``arrays`` (as shape/dtype specs) and execute,
    returning the result as ``np.ndarray``. The general successor to
    ``jit_fori_loop`` for arbitrary ``tessera.ops`` + ``tessera.control`` bodies.

    Straight-line traces lower to a ``GraphFn`` (fusion); traces containing
    control flow use the F3 concrete interpreter (``execute_traced``), which lifts
    the GraphFn-executor constraints so control flow can sit anywhere with
    surrounding straight-line code."""
    arrs = [np.asarray(a) for a in arrays]
    traced = trace(fn, *arrs)
    if target == "apple_gpu" and _has_control_flow(traced):
        return execute_traced(traced, arrs)
    elem = "bf16" if (arrs and _np_dtype_to_elem(arrs[0].dtype) == "bf16") else "f32"
    g = to_graphfn(traced, elem=elem, target=target)
    return g.run(*arrs)
