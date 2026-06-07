"""S5 control-flow and transform reference primitives.

These are Tessera-owned semantics, not wrappers around JAX. The v1
implementation is CPU/reference Python so model authors can express recurrent
and transform-heavy code while Graph IR, Schedule IR, and backend lowering
catch up.
"""

from __future__ import annotations

import contextvars
import functools
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .autodiff import checkpoint, grad, jvp, rematerialize, vmap
from .autodiff.mixed_precision import autocast
from .autodiff.tape import tape


@dataclass(frozen=True)
class AxisFrame:
    name: str
    index: int
    size: int


_AXIS_STACK: contextvars.ContextVar[tuple[AxisFrame, ...]] = contextvars.ContextVar(
    "_tessera_axis_stack",
    default=(),
)


def _push_axis(name: str, index: int, size: int):
    stack = _AXIS_STACK.get()
    return _AXIS_STACK.set(stack + (AxisFrame(str(name), int(index), int(size)),))


def _pop_axis(token) -> None:
    _AXIS_STACK.reset(token)


def _find_axis(name: str | None = None) -> AxisFrame:
    stack = _AXIS_STACK.get()
    if name is None:
        if not stack:
            raise RuntimeError("axis_index/axis_size called outside mapped axis context")
        return stack[-1]
    for frame in reversed(stack):
        if frame.name == name:
            return frame
    raise RuntimeError(f"axis {name!r} is not active")


def axis_index(axis_name: str | None = None) -> int:
    return _find_axis(axis_name).index


def axis_size(axis_name: str | None = None) -> int:
    return _find_axis(axis_name).size


def axis_name() -> str:
    return _find_axis(None).name


def _slice_tree(xs: Any, i: int, axis: int = 0) -> Any:
    if xs is None:
        return None
    if isinstance(xs, tuple):
        return tuple(_slice_tree(x, i, axis=axis) for x in xs)
    if isinstance(xs, list):
        return [_slice_tree(x, i, axis=axis) for x in xs]
    from . import ops
    return ops.take(xs, np.asarray(i), axis=axis)


def _tree_length(xs: Any, axis: int = 0) -> int:
    if xs is None:
        raise ValueError("length is required when scan/map xs is None")
    if isinstance(xs, (tuple, list)):
        if not xs:
            raise ValueError("xs must not be empty")
        return _tree_length(xs[0], axis=axis)
    return int(np.asarray(xs).shape[axis])


def _stack_outputs(outputs: list[Any], axis: int = 0) -> Any:
    if not outputs:
        return np.asarray([])
    first = outputs[0]
    if isinstance(first, tuple):
        return tuple(_stack_outputs([out[i] for out in outputs], axis=axis)
                     for i in range(len(first)))
    if isinstance(first, list):
        return [_stack_outputs([out[i] for out in outputs], axis=axis)
                for i in range(len(first))]
    return np.stack([np.asarray(out) for out in outputs], axis=axis)


def scan(
    fn: Callable[[Any, Any], tuple[Any, Any]],
    init: Any,
    xs: Any = None,
    *,
    length: int | None = None,
    reverse: bool = False,
    axis_name: str = "scan",
) -> tuple[Any, Any]:
    """Sequential scan: ``(carry, ys) = scan(fn, init, xs)``."""
    n = int(length) if length is not None else _tree_length(xs)
    order = range(n - 1, -1, -1) if reverse else range(n)
    carry = init
    ys: list[Any] = []
    for logical_i, i in enumerate(order):
        token = _push_axis(axis_name, logical_i, n)
        try:
            x_i = None if xs is None else _slice_tree(xs, i)
            carry, y = fn(carry, x_i)
        finally:
            _pop_axis(token)
        ys.append(y)
    if reverse:
        ys.reverse()
    return carry, _stack_outputs(ys)


def associative_scan(
    fn: Callable[[Any, Any], Any],
    xs: Any,
    *,
    reverse: bool = False,
    axis: int = 0,
) -> Any:
    """Prefix scan for associative binary ``fn``."""
    n = _tree_length(xs, axis=axis)
    order = range(n - 1, -1, -1) if reverse else range(n)
    acc = None
    out: list[Any] = []
    for i in order:
        x_i = _slice_tree(xs, i, axis=axis)
        acc = x_i if acc is None else fn(acc, x_i)
        out.append(acc)
    if reverse:
        out.reverse()
    return _stack_outputs(out, axis=axis)


def _active_trace_builder():
    """The active Phase-F trace builder, or None. Under trace, the control-flow
    primitives emit a `tessera.control_*` op instead of running a host loop."""
    from .compiler._trace_hook import active_tracer

    return active_tracer()


def while_loop(
    cond_fun: Callable[[Any], bool],
    body_fun: Callable[[Any], Any],
    init_val: Any,
    *,
    max_steps: int | None = None,
) -> Any:
    tr = _active_trace_builder()
    if tr is not None and hasattr(tr, "record_while"):
        return tr.record_while(cond_fun, body_fun, init_val, max_steps)
    value = init_val
    steps = 0
    while bool(cond_fun(value)):
        if max_steps is not None and steps >= max_steps:
            raise RuntimeError(f"while_loop exceeded max_steps={max_steps}")
        value = body_fun(value)
        steps += 1
    return value


def fori_loop(lower: int, upper: int, body_fun: Callable[[int, Any], Any], init_val: Any) -> Any:
    tr = _active_trace_builder()
    if tr is not None and hasattr(tr, "record_for_loop"):
        return tr.record_for_loop(lower, upper, body_fun, init_val)
    value = init_val
    for i in range(int(lower), int(upper)):
        value = body_fun(i, value)
    return value


def cond(
    pred: Any,  # bool on the host path; a Tracer under a Phase-F trace
    true_fun: Callable[..., Any],
    false_fun: Callable[..., Any],
    *operands: Any,
) -> Any:
    tr = _active_trace_builder()
    if tr is not None and hasattr(tr, "record_cond") and not isinstance(pred, bool):
        return tr.record_cond(pred, true_fun, false_fun, operands)
    return true_fun(*operands) if bool(pred) else false_fun(*operands)


def switch(index: int, branches: Sequence[Callable[..., Any]], *operands: Any) -> Any:
    if not branches:
        raise ValueError("switch requires at least one branch")
    i = int(index)
    if i < 0 or i >= len(branches):
        raise IndexError(f"switch index {i} out of range for {len(branches)} branches")
    return branches[i](*operands)


def map(fn: Callable[[Any], Any], xs: Any, *, axis_name: str = "map") -> Any:
    n = _tree_length(xs)
    outputs = []
    for i in range(n):
        token = _push_axis(axis_name, i, n)
        try:
            outputs.append(fn(_slice_tree(xs, i)))
        finally:
            _pop_axis(token)
    return _stack_outputs(outputs)


def pmap(
    fn: Callable,
    *,
    in_axes: int | Sequence[int | None] | None = 0,
    out_axes: int | None = 0,
    axis_name: str = "pmap",
) -> Callable:
    """Reference SPMD map with mesh-axis introspection."""

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        axes: tuple[int | None, ...]
        if in_axes is None:
            axes = tuple(None for _ in args)
        elif isinstance(in_axes, int):
            axes = tuple(in_axes for _ in args)
        else:
            axes = tuple(in_axes)
        if len(axes) != len(args):
            raise ValueError("pmap in_axes length must match positional args")
        sizes = [
            np.asarray(arg).shape[axis]
            for arg, axis in zip(args, axes)
            if axis is not None
        ]
        if not sizes:
            return fn(*args, **kwargs)
        if len(set(sizes)) != 1:
            raise ValueError(f"pmap inputs have inconsistent sizes: {sizes}")
        size = sizes[0]
        outputs = []
        for i in range(size):
            call_args = [
                arg if axis is None else np.take(np.asarray(arg), i, axis=axis)
                for arg, axis in zip(args, axes)
            ]
            token = _push_axis(axis_name, i, size)
            try:
                outputs.append(fn(*call_args, **kwargs))
            finally:
                _pop_axis(token)
        if out_axes is None:
            return outputs
        return _stack_outputs(outputs, axis=out_axes)

    return wrapped


def value_and_grad(fn: Callable, argnums: int | Sequence[int] = 0) -> Callable:
    grad_fn = grad(fn, argnums=argnums)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        return fn(*args, **kwargs), grad_fn(*args, **kwargs)

    return wrapped


def vjp(fn: Callable, *primals: Any):
    """Return ``(value, pullback)`` for a single-output numpy/tessera function."""
    with tape() as t:
        value = fn(*primals)

    def pullback(cotangent):
        t.backward(value, cotangent=cotangent, retain_graph=True, accumulate_param_grad=False)
        grads = []
        for primal in primals:
            arr = np.asarray(primal)
            grads.append(t.cotangent.get(id(arr), np.zeros_like(arr)))
        return tuple(grads)

    return value, pullback


remat = rematerialize


__all__ = [
    "AxisFrame",
    "associative_scan",
    "autocast",
    "axis_index",
    "axis_name",
    "axis_size",
    "checkpoint",
    "cond",
    "fori_loop",
    "grad",
    "jvp",
    "map",
    "pmap",
    "remat",
    "rematerialize",
    "scan",
    "switch",
    "value_and_grad",
    "vjp",
    "vmap",
    "while_loop",
]
