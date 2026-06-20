"""Tape, tape-active state, and `tessera.ops.*` op wrapping.

Tape design — flat list of `TapeEntry`s in forward-call order. Backward walks
the list in reverse, seeding the cotangent dict with `1.0` at the loss scalar
and propagating per-VJP into parents. `Parameter` provenance is recorded at
forward time so the backward pass can write `param.grad` directly.

See `docs/spec/AUTODIFF_SPEC.md` for the full design.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .vjp import get_vjp


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────


class TesseraAutodiffError(RuntimeError):
    """Raised on misuse of the autodiff machinery (no VJP, scalar shape, etc.)."""


# ─────────────────────────────────────────────────────────────────────────────
# Tape data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InputDesc:
    """Describes one forward-call input: its underlying numpy buffer + Parameter origin."""
    param: Any   # Parameter | None — typed loosely to avoid import cycle
    array_id: int
    array: np.ndarray
    is_literal: bool = False  # True for python-scalar operands (non-differentiable)


@dataclass
class TapeEntry:
    op: str
    inputs: tuple[InputDesc, ...]
    kwargs: dict
    output_id: int
    output: np.ndarray | np.generic
    vjp: Callable | None


@dataclass
class Tape:
    entries: list[TapeEntry] = field(default_factory=list)
    _consumed: bool = False
    # Final cotangent dict from `backward()` — exposed so that
    # `tessera.autodiff.rematerialize` can extract input cotangents from a
    # nested tape's backward pass without re-walking it.
    cotangent: dict[int, np.ndarray] = field(default_factory=dict)

    def record(
        self,
        op: str,
        inputs: tuple[InputDesc, ...],
        kwargs: dict,
        output: np.ndarray | np.generic,
        vjp: Callable | None,
    ) -> None:
        self.entries.append(
            TapeEntry(
                op=op,
                inputs=inputs,
                kwargs=kwargs,
                output_id=id(output),
                output=output,
                vjp=vjp,
            )
        )

    def backward(
        self,
        target: Any,
        *,
        cotangent: Any = None,
        retain_graph: bool = False,
        accumulate_param_grad: bool = True,
    ) -> None:
        """Seed the cotangent at `target` and propagate backward through the tape.

        Two patterns are supported:

        1. **Scalar loss** — `target` is a 0-d numpy array that came directly
           from a recorded op (the last tape entry's output). Default cotangent
           is `1.0`. Useful when the entire loss expression goes through `ops.*`.

        2. **Explicit cotangent** — pass `cotangent=dy` to seed at `id(target)`
           with a user-computed gradient. Use this when loss math sits outside
           the tape (raw numpy: `(y - t)**2`-style). `target` must still be a
           tape-recorded numpy array.

        Re-runnability (deferred-items plan, Item 4):
          * Default behavior raises if `backward()` is called twice on the
            same tape — preserves the original v1 contract.
          * Pass ``retain_graph=True`` to allow re-running backward with a
            different cotangent (e.g. for ``jacrev``, which seeds one
            basis vector per output dim). The tape state stays valid.
          * Pass ``accumulate_param_grad=False`` to skip writing into
            ``Parameter.grad`` slots — useful when ``grad()`` wants the
            cotangent map without mutating user-visible state.
        """
        if self._consumed and not retain_graph:
            raise TesseraAutodiffError(
                "tape.backward() called twice on the same tape; open a new "
                "tape() block, or pass retain_graph=True if you intentionally "
                "want to re-run with a different cotangent (jacrev / repeated "
                "backward)."
            )
        target_id = id(target)
        target_on_tape = any(e.output_id == target_id for e in self.entries)
        if not target_on_tape:
            raise TesseraAutodiffError(
                "backward target is not a tape-recorded output. Forward computation "
                "must produce `target` via tessera.ops.*; if your loss math runs in "
                "raw numpy, pass `cotangent=dy` and use the model output as `target`. "
                "See docs/spec/AUTODIFF_SPEC.md."
            )
        if cotangent is None:
            arr = np.asarray(target)
            if arr.size != 1:
                raise TesseraAutodiffError(
                    f"backward expects a scalar target when no cotangent is given; "
                    f"got shape {arr.shape}. Pass cotangent=... for non-scalar targets."
                )
            seed_dtype = arr.dtype if arr.dtype.kind == "f" else np.float32
            seed = np.ones((), dtype=seed_dtype)
        else:
            seed = np.asarray(cotangent)
            target_arr = np.asarray(target)
            if seed.shape != target_arr.shape:
                raise TesseraAutodiffError(
                    f"cotangent shape {seed.shape} does not match target shape {target_arr.shape}"
                )
        cotan: dict[int, np.ndarray] = {target_id: seed}

        for entry in reversed(self.entries):
            dout = cotan.get(entry.output_id)
            if dout is None:
                continue  # not on the path to the scalar

            if entry.vjp is None:
                raise TesseraAutodiffError(
                    f"op {entry.op!r} is not differentiable in v1 — encountered on "
                    f"the gradient path. Register a VJP via "
                    f"tessera.autodiff.custom_rule({entry.op!r}). "
                    f"See docs/spec/AUTODIFF_SPEC.md."
                )

            forward_args = tuple(d.array for d in entry.inputs)
            d_in = entry.vjp(dout, *forward_args, **entry.kwargs)
            if not isinstance(d_in, tuple):
                d_in = (d_in,)
            # Tolerate a VJP that omits cotangents for trailing python-scalar
            # *literal* operands (non-differentiable; e.g. ops.minimum(t, 1.2)).
            # Pad with None so the strict per-array count check below still
            # catches genuine VJP-author bugs for real array inputs.
            if len(d_in) < len(entry.inputs):
                trailing_literals = 0
                for desc in reversed(entry.inputs):
                    if desc.is_literal:
                        trailing_literals += 1
                    else:
                        break
                missing = len(entry.inputs) - len(d_in)
                if 0 < missing <= trailing_literals:
                    d_in = d_in + (None,) * missing
            if len(d_in) != len(entry.inputs):
                raise TesseraAutodiffError(
                    f"VJP for {entry.op!r} returned {len(d_in)} cotangents, "
                    f"expected {len(entry.inputs)} (one per input)"
                )

            for desc, g in zip(entry.inputs, d_in):
                if g is None:
                    continue
                g = np.asarray(g)
                # Accumulate into the cotangent dict for downstream backward steps
                if desc.array_id in cotan:
                    cotan[desc.array_id] = cotan[desc.array_id] + g
                else:
                    cotan[desc.array_id] = g
                # If this input came from a Parameter, accumulate into .grad
                # (skippable for `grad()` callsites that want the cotangent
                # map without mutating user-visible Parameter state).
                if desc.param is not None and accumulate_param_grad:
                    _accumulate_param_grad(desc.param, g)

        # Store final cotangent map for downstream consumers (rematerialize,
        # grad, jacrev). `_consumed` is set to True regardless — but
        # `retain_graph=True` lets a future backward bypass the consumed
        # check at call time.
        self.cotangent = cotan
        self._consumed = True


# ─────────────────────────────────────────────────────────────────────────────
# Tape-active state (contextvar — async/thread safe)
# ─────────────────────────────────────────────────────────────────────────────


_ACTIVE_TAPE: contextvars.ContextVar[Tape | None] = contextvars.ContextVar(
    "_tessera_autodiff_tape", default=None
)


@contextmanager
def tape():
    """Context manager that opens a new tape and binds it as active.

    Inside the `with` block, every tape-aware `tessera.ops.<name>` call is
    recorded. Outside, ops behave normally.
    """
    t = Tape()
    token = _ACTIVE_TAPE.set(t)
    try:
        yield t
    finally:
        _ACTIVE_TAPE.reset(token)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter integration — local import to avoid cycle
# ─────────────────────────────────────────────────────────────────────────────


def _parameter_class():
    from ..nn.module import Parameter
    return Parameter


def _accumulate_param_grad(param, grad: np.ndarray) -> None:
    """Add `grad` into `param.grad`, creating a new buffer if `.grad` is None.

    Gradients accumulate in at least fp32 regardless of the parameter's storage
    dtype: fp16/bf16 accumulation loses precision and can underflow across many
    tape entries (the optimizer master-weight path assumes fp32 grads).
    """
    g = np.asarray(grad)
    if g.dtype.kind == "f" and g.dtype.itemsize < 4:
        g = g.astype(np.float32)
    if param.grad is None:
        param.grad = g.copy()  # setter wraps in a fresh DistributedArray
    else:
        # `param.grad._data` IS the numpy buffer — write in place.
        param.grad._data[...] = param.grad._data + g


# Sentinel for positional args that are not array-like (e.g., dtype strings,
# bools, Python ints/floats). These are passed through to the original op as-is
# and excluded from tape inputs / VJP gradient lists.
_NON_ARRAY = object()


def _describe(arg: Any):
    """Convert a forward argument into an `InputDesc` (array-like) or `_NON_ARRAY`.

    Array-like resolution (in order):
      1. `Parameter` — direct.
      2. Underlying numpy buffer registered in `_PARAM_REGISTRY` (a Parameter's
         `.data._data` that flowed through `_as_array(param)`).
      3. `DistributedArray` (or duck-typed ._data: numpy holder).
      4. `numpy.ndarray`.

    Non-array (strings, plain Python scalars, None, ...) returns `_NON_ARRAY`
    so the wrapper can pass them through to the op without adding them to the
    tape input list.
    """
    Parameter = _parameter_class()
    from ..nn.module import parameter_for_buffer_id

    if isinstance(arg, Parameter):
        buf = arg._data._data
        return InputDesc(param=arg, array_id=id(buf), array=np.asarray(buf))

    if hasattr(arg, "_data") and not isinstance(arg, np.ndarray):
        inner = arg._data
        if isinstance(inner, np.ndarray):
            param = parameter_for_buffer_id(id(inner))
            return InputDesc(param=param, array_id=id(inner), array=inner)

    if isinstance(arg, np.ndarray):
        param = parameter_for_buffer_id(id(arg))
        return InputDesc(param=param, array_id=id(arg), array=arg)

    # numpy generic scalars (np.float32(...), np.int64(...), 0-d arrays from .sum())
    if isinstance(arg, np.generic):
        # Key on id(arg), NOT id(np.asarray(arg)): a producer op records its
        # output by id(output). When that output is a np.generic scalar (e.g.
        # reduce-to-scalar) and is then consumed by another op, keying on a
        # fresh array's id would sever the producer→consumer gradient link.
        # Using id(arg) keeps the chain intact for scalar-valued intermediates.
        return InputDesc(param=None, array_id=id(arg), array=np.asarray(arg))

    # Python numeric scalars passed *positionally* as differentiable operands
    # (e.g. ops.minimum(ratio, 1.2), ops.mul(x, -0.5)). Record them as
    # non-differentiable (param=None) literal inputs so the VJP receives the
    # operand value and the cotangent count matches. `bool` is excluded — it is
    # an int subclass used for flags, not a differentiable operand. Backward
    # tolerates a VJP that omits trailing literal cotangents (see Tape.backward).
    if isinstance(arg, (int, float)) and not isinstance(arg, bool):
        a = np.asarray(arg, dtype=np.float64)
        return InputDesc(param=None, array_id=id(arg), array=a, is_literal=True)

    return _NON_ARRAY


# ─────────────────────────────────────────────────────────────────────────────
# Op wrapping — install tape-aware wrappers on tessera.ops.<name>
# ─────────────────────────────────────────────────────────────────────────────


_WRAPPED: set[str] = set()


def _make_wrapper(name: str, original: Callable) -> Callable:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        # Late import — avoids a load-time cycle through autodiff.__init__.
        from .mixed_precision import autocast_dtype, autocast_keep_fp32

        # Abstract-interp trace (Phase F) — a sibling context to the tape. When a
        # tracer is active, record this op into the trace graph and return a
        # Tracer (shape from a rule); the numpy op is NOT run. Mutually exclusive
        # with the tape. Lightweight neutral hook → no import cycle.
        from ..compiler._trace_hook import active_tracer

        _tracer = active_tracer()
        if _tracer is not None:
            return _tracer.record_op(name, original, args, kwargs)

        active = _ACTIVE_TAPE.get()
        if active is None:
            # No-tape fast path — keep `fast_args` as its own variable so
            # the type-narrower doesn't propagate the tuple type into the
            # taped-path block below.
            fast_args: tuple[Any, ...] = tuple(_to_forward_arg(a) for a in args)
            cast_dtype = autocast_dtype()
            if cast_dtype is not None:
                if autocast_keep_fp32(name):
                    fast_args = tuple(_autocast_args(fast_args, "fp32"))
                else:
                    fast_args = tuple(_autocast_args(fast_args, cast_dtype))
            return original(*fast_args, **kwargs)

        # Tape active — describe each positional arg, pre-convert to numpy
        # for the forward call, and record only the array-like ones on the tape.
        descs_full = tuple(_describe(a) for a in args)
        forward_args: list[Any] = []
        array_descs: list[InputDesc] = []
        for a, d in zip(args, descs_full):
            if d is _NON_ARRAY:
                forward_args.append(a)
            else:
                forward_args.append(d.array)
                array_descs.append(d)

        cast_dtype = autocast_dtype()
        if cast_dtype is not None:
            if autocast_keep_fp32(name):
                forward_args = list(_autocast_args(forward_args, "fp32"))
            else:
                forward_args = list(_autocast_args(forward_args, cast_dtype))

        out = original(*forward_args, **kwargs)
        # Look up the VJP at call time so that `custom_rule` can register/override
        # after the wrapper was installed. `vjp_fn=None` is allowed; backward
        # raises only if this entry actually lies on the gradient path.
        vjp_fn = get_vjp(name)
        if isinstance(out, tuple):
            for i, component in enumerate(out):
                if isinstance(component, (np.ndarray, np.generic)):
                    component_kwargs = dict(kwargs)
                    component_kwargs["_output_index"] = i
                    active.record(
                        name,
                        tuple(array_descs),
                        component_kwargs,
                        component,
                        vjp_fn,
                    )
        else:
            active.record(name, tuple(array_descs), dict(kwargs), out, vjp_fn)
        return out

    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    # Preserve the op's `__name__` so callers that introspect (e.g., autotune)
    # still see e.g. "matmul" rather than "_traced_matmul".
    wrapped.__name__ = getattr(original, "__name__", name)
    return wrapped


_AUTOCAST_NUMPY_DTYPE = {
    "fp16": np.float16,
    "bf16": np.float32,  # bf16 stored as fp32 in the numpy reference
    "fp32": np.float32,
    "fp64": np.float64,
    # fp8 / fp6 / fp4 / nvfp4 are handled by routing through their
    # `ops.quantize_*` op rather than a plain numpy cast — see
    # `_autocast_args` below. The dict entries below are placeholders so
    # the lookup doesn't fall through to the plain-cast branch.
    "fp8_e4m3": np.float32,
    "fp8_e5m2": np.float32,
    "fp6_e2m3": np.float32,
    "fp6_e3m2": np.float32,
    "fp4_e2m1": np.float32,
    "nvfp4": np.float32,
}


def _autocast_args(args, dtype: str):
    """Cast each ndarray in ``args`` to ``dtype``; pass non-arrays through.

    For low-precision floats (fp8 / fp6 / fp4 / nvfp4), arrays are
    quantized to the requested format via the matching ``ops.quantize_*``
    op on the boundary. The result is fp32 storage that's numerically
    equal to its low-precision-rounded value — downstream ops can keep
    using float arithmetic without bridge code.
    """
    if dtype.startswith(("fp8_", "fp6_", "fp4_")) or dtype == "nvfp4":
        from .. import ops as _ops  # noqa: WPS433
        # Pick the matching quantize op + extract the bare format suffix.
        if dtype.startswith("fp8_"):
            quantize = getattr(_ops.quantize_fp8, "__wrapped__",
                               _ops.quantize_fp8)
            fmt = dtype[len("fp8_"):]
            kwargs = {"format": fmt}
        elif dtype.startswith("fp6_"):
            quantize = getattr(_ops.quantize_fp6, "__wrapped__",
                               _ops.quantize_fp6)
            fmt = dtype[len("fp6_"):]
            kwargs = {"format": fmt}
        elif dtype.startswith("fp4_"):
            quantize = getattr(_ops.quantize_fp4, "__wrapped__",
                               _ops.quantize_fp4)
            fmt = dtype[len("fp4_"):]
            kwargs = {"format": fmt}
        else:  # nvfp4
            quantize = getattr(_ops.quantize_nvfp4, "__wrapped__",
                               _ops.quantize_nvfp4)
            kwargs = {}  # default block_size

        out = []
        for a in args:
            if isinstance(a, np.ndarray) and a.dtype.kind in "fc":
                try:
                    q, _scale = quantize(a, **kwargs)
                except ValueError:
                    # NVFP4 requires last dim divisible by block_size; if
                    # the input doesn't satisfy that, pass through as fp32
                    # rather than crash the autocast region.
                    out.append(a.astype(np.float32, copy=False))
                    continue
                out.append(q)
            else:
                out.append(a)
        return out

    np_dtype = _AUTOCAST_NUMPY_DTYPE.get(dtype, np.float32)
    out = []
    for a in args:
        if isinstance(a, np.ndarray) and a.dtype.kind in "fc":
            out.append(a.astype(np_dtype, copy=False))
        else:
            out.append(a)
    return out


def _to_forward_arg(arg):
    """Pre-convert Parameter / DistributedArray to a numpy buffer.

    Used outside an active tape so the underlying ops (which look for
    `_data` one level deep) don't trip on a Parameter whose `._data` is a
    DistributedArray rather than a numpy array.
    """
    Parameter = _parameter_class()
    if isinstance(arg, Parameter):
        return arg._data._data
    if hasattr(arg, "_data") and not isinstance(arg, np.ndarray):
        inner = arg._data
        if isinstance(inner, np.ndarray):
            return inner
    return arg


def install_op_wrappers() -> None:
    """Install tape-aware wrappers on every registered `tessera.ops.<name>`.

    Wraps **all** ops, not just ops in `_VJPS`. Ops without a VJP are still
    recorded on the tape; `Tape.backward` only raises if the gradient path
    actually reaches such an entry. This lets users call unsupported ops
    inside a tape as long as they don't try to differentiate through them.

    Idempotent.
    """
    from .. import ops  # local import — autodiff is loaded after ops

    # Iterate over the registry so any op anyone registered (including
    # custom_rule additions) gets wrapped.
    op_names = list(ops.registry._entries.keys())
    for name in op_names:
        if name in _WRAPPED:
            continue
        original = getattr(ops, name, None)
        if original is None:
            continue
        wrapped = _make_wrapper(name, original)
        setattr(ops, name, wrapped)
        # Keep the registry's reference entry in sync so `registry.dispatch`
        # routes through the wrapper too.
        ops.registry._entries[name].reference = wrapped
        _WRAPPED.add(name)


__all__ = [
    "Tape",
    "TapeEntry",
    "InputDesc",
    "TesseraAutodiffError",
    "tape",
    "install_op_wrappers",
]
