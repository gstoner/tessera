"""Tape, tape-active state, and `tessera.ops.*` op wrapping.

Tape design — flat list of `TapeEntry`s in forward-call order. Backward walks
the list in reverse, seeding the cotangent dict with `1.0` at the loss scalar
and propagating per-VJP into parents. `Parameter` provenance is recorded at
forward time so the backward pass can write `param.grad` directly.

See `docs/spec/AUTODIFF_SPEC.md` for the full design.
"""

from __future__ import annotations

import contextvars
import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, NoReturn

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────
# Defined in the leaf module `errors.py` and re-exported here: `vjp.py` (which
# this module imports) needs the same class object for the degeneracy guards,
# and defining it here would make that a circular import. Re-exporting keeps
# every existing `from .tape import TesseraAutodiffError` naming one class.
from .errors import TesseraAutodiffError  # noqa: F401  (re-export)

from .vjp import get_vjp


# ─────────────────────────────────────────────────────────────────────────────
# Tape data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InputDesc:
    """Describes one forward-call input: its underlying numpy buffer + Parameter origin."""

    param: Any  # Parameter | None — typed loosely to avoid import cycle
    array_id: int
    array: Any  # np.ndarray, or list[np.ndarray] for a sequence operand
    is_literal: bool = False  # True for python-scalar operands (non-differentiable)
    # Sequence operand (`cat`/`stack` take a *list* of arrays as ONE slot):
    # per-element descriptors so backward can fan the rule's list-valued
    # gradient out to each element's own producer link. `None` for ordinary
    # single-array operands. AD-LAW-1j finding (2026-08-19): `_describe`
    # previously returned `_NON_ARRAY` for a list of arrays, so the tape
    # recorded NO operand and replay called `vjp_cat(dout)` — reverse mode
    # through `ops.cat`/`ops.stack` was broken outright. Pinned by
    # ``test_tape_differentiates_sequence_operands``.
    elements: "tuple[InputDesc, ...] | None" = None


@dataclass
class TapeEntry:
    op: str
    inputs: tuple[InputDesc, ...]
    kwargs: dict
    output_id: int
    output: np.ndarray | np.generic
    vjp: Callable | None
    #: How `inputs` regroup into the rule's *positional* arguments. `None` (the
    #: common case) means one input per positional argument. Otherwise one entry
    #: per positional argument: `None` for a plain operand, or an int `k` for a
    #: **sequence** operand built from the next `k` inputs — the shape
    #: `ops.cat`/`ops.stack` need, where one argument carries many tensors.
    #: Without this the tape recorded a sequence operand as neither an operand
    #: nor a kwarg, and `vjp_cat` was called with no `xs` at all.
    input_groups: tuple[int | None, ...] | None = None


@dataclass
class Tape:
    entries: list[TapeEntry] = field(default_factory=list)
    _consumed: bool = False
    # Final cotangent dict from `backward()` — exposed so that
    # `tessera.autodiff.rematerialize` can extract input cotangents from a
    # nested tape's backward pass without re-walking it.
    cotangent: dict[int, np.ndarray] = field(default_factory=dict)
    #: Set when an `ops.*` call was diverted to an enclosing forward-mode trace
    #: while this tape was open. The trace is torn down before `backward` runs
    #: (reverse-over-forward), so liveness alone cannot detect the nesting —
    #: without this flag that case reported the generic "raw numpy" message.
    diverted_to_forward_mode: bool = False
    #: Set when an inner `tape()` opened while THIS tape's forward pass was
    #: still running. `_ACTIVE_TAPE` is a single contextvar, so an inner tape
    #: shadows the outer one completely: every `ops.*` call goes to the inner
    #: tape and the outer records nothing for that stretch. This is the
    #: reverse-mode twin of `diverted_to_forward_mode` above, and it exists for
    #: the same reason — without it, an input whose path was swallowed by a
    #: nested tape is indistinguishable from an input the function genuinely
    #: does not use, and the zero-gradient answer is reported as fact.
    #:
    #: **Per buffer id, not a bare flag** (review on #678). Setting a single
    #: bool on context-manager ENTRY made any nested tape poison the whole
    #: outer pass: a function that opens an inner tape for an unrelated
    #: diagnostic and genuinely ignores one of its arguments had that
    #: argument's legitimate ZERO gradient turned into a refusal. The question
    #: the zero branch actually needs to ask is narrower -- "was *this*
    #: value's path swallowed?" -- so record the ids the inner tape consumed
    #: and let each parameter be judged on its own evidence.
    shadowed_buffer_ids: set[int] = field(default_factory=set)
    #: Set the first time `backward()` is entered. After that the forward pass
    #: is definitively over, so a nested tape opened from a VJP rule (which is
    #: what `rematerialize` legitimately does) must NOT set the flag above.
    _forward_closed: bool = False
    #: Buffer ids that received a cotangent during `backward` but had no
    #: producing entry on THIS tape and are known to have been consumed by a
    #: nested tape -- i.e. the reverse sweep ran into the shadowed region and
    #: stopped there.
    #:
    #: **Why the parameter's own id is not enough** (review on #679). A nested
    #: tape can swallow an INTERMEDIATE rather than the differentiated input:
    #: `y = ops.mul(x, x)` on the outer tape, `z = ops.sin(y)` inside an inner
    #: one, `ops.sum(z)` back on the outer. Then `shadowed_buffer_ids` holds
    #: y and z but NOT x, the sweep dies at z, and x's missing cotangent was
    #: reported as a zero gradient. The question is where the PATH broke, not
    #: whether the parameter itself was touched.
    truncated_at_shadow: set[int] = field(default_factory=set)

    def consumed_buffer_ids(self) -> set[int]:
        """Buffer ids this tape read as a differentiable input.

        Literal operands are excluded: a python scalar carries no gradient, so
        an inner tape that only touched literals swallowed nothing. Outputs are
        included too -- a value the inner tape PRODUCED is equally unreachable
        from the outer tape, and an outer op consuming it would have recorded
        there instead.
        """
        ids: set[int] = set()
        for entry in self.entries:
            for desc in entry.inputs:
                if not desc.is_literal:
                    ids.add(desc.array_id)
            ids.add(entry.output_id)
        return ids

    def record(
        self,
        op: str,
        inputs: tuple[InputDesc, ...],
        kwargs: dict,
        output: np.ndarray | np.generic,
        vjp: Callable | None,
        input_groups: tuple[int | None, ...] | None = None,
    ) -> None:
        self.entries.append(
            TapeEntry(
                op=op,
                inputs=inputs,
                kwargs=kwargs,
                output_id=id(output),
                output=output,
                vjp=vjp,
                input_groups=input_groups,
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
        # The forward pass is over for good once backward is entered — true
        # even if this call raises, and even under retain_graph. Nested tapes
        # opened from here on are VJP-internal (`rematerialize`) and legitimate.
        self._forward_closed = True
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
            from .jvp import active_jvp_trace

            if active_jvp_trace() is not None or self.diverted_to_forward_mode:
                # Mode nesting, not a raw-numpy loss. Every `ops.*` call inside
                # this tape was intercepted by the enclosing forward-mode trace,
                # so the tape is empty and there is nothing to walk back. Saying
                # "your loss math runs in raw numpy" here sends the reader to
                # look for a bug that does not exist (MC6).
                raise TesseraAutodiffError(
                    "cannot run reverse mode inside an active forward-mode "
                    "trace: every tessera.ops.* call was captured by the "
                    "enclosing jvp() trace, so this tape recorded nothing.\n"
                    "\n"
                    "This is what blocks forward-over-reverse (`jvp(grad(f), "
                    "x, v)`) and reverse-over-forward (`grad(lambda x: "
                    "jvp(f, x, v)[1])`) alike: each mode's rules are numpy "
                    "functions, not `ops.*` calls, so neither mode can trace "
                    "through the other's rules. Composing them needs one "
                    "derivative datum evaluated in a higher-order algebra "
                    "(AUTODIFF_NEXTGEN_PLAN.md AD-WEIL-1), not a deeper tape.\n"
                    "\n"
                    "Available today: `tessera.autodiff.hvp(f, x, v)` for "
                    "Hessian-vector products (central difference of the "
                    "gradient), or `JitFn.compiled_hvp_ir` for the exact "
                    "compiled forward-over-reverse path."
                )
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
            # Consume the output's adjoint. Every consumer of this output sits
            # later on the tape and has already contributed, so `dout` is final
            # here. Popping matters when an op returns one of its inputs
            # verbatim (`ops.clamp(x)` with no bounds, `ops.reshape` to the same
            # shape): the entry's output_id and that input's array_id are then
            # the *same* key, and accumulating below would add the passthrough
            # cotangent on top of the `dout` we just consumed, doubling every
            # upstream gradient silently. Re-seeding the key from the rule's
            # own result is the only correct reading of an aliased edge.
            del cotan[entry.output_id]

            if entry.vjp is None:
                raise TesseraAutodiffError(
                    f"op {entry.op!r} is not differentiable in v1 — encountered on "
                    f"the gradient path. Register a VJP via "
                    f"tessera.autodiff.custom_rule({entry.op!r}). "
                    f"See docs/spec/AUTODIFF_SPEC.md."
                )

            forward_args = _regroup_inputs(entry)
            d_in = entry.vjp(dout, *forward_args, **entry.kwargs)
            if not isinstance(d_in, tuple):
                d_in = (d_in,)
            if entry.input_groups is not None:
                d_in = _scatter_grouped_cotangents(entry, d_in)
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
                if desc.elements is not None:
                    # Sequence operand: the rule returns one list/tuple of
                    # per-element gradients for this slot; fan each out to
                    # its own element's producer link.
                    if not isinstance(g, (list, tuple)) \
                            or len(g) != len(desc.elements):
                        got = (f"a {len(g)}-element sequence"
                               if isinstance(g, (list, tuple))
                               else type(g).__name__)
                        raise TesseraAutodiffError(
                            f"VJP for {entry.op!r} must return one gradient "
                            f"per element for its {len(desc.elements)}-element "
                            f"sequence operand; got {got}"
                        )
                    for edesc, eg in zip(desc.elements, g):
                        if eg is None:
                            continue
                        eg = np.asarray(eg)
                        if edesc.array_id in cotan:
                            cotan[edesc.array_id] = cotan[edesc.array_id] + eg
                        else:
                            cotan[edesc.array_id] = eg
                        if edesc.param is not None and accumulate_param_grad:
                            _accumulate_param_grad(edesc.param, eg)
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
        # Where did the sweep stop? A leftover cotangent whose id no entry on
        # this tape produced is a leaf. A leaf that a nested tape consumed is
        # not a leaf at all -- it is a severed edge, and every input upstream
        # of it is now unprovable rather than zero.
        produced = {e.output_id for e in self.entries}
        self.truncated_at_shadow = {
            i for i in cotan
            if i not in produced and i in self.shadowed_buffer_ids
        }
        self.cotangent = cotan
        self._consumed = True


# ─────────────────────────────────────────────────────────────────────────────
# Tape-active state (contextvar — async/thread safe)
# ─────────────────────────────────────────────────────────────────────────────


_ACTIVE_TAPE: contextvars.ContextVar[Tape | None] = contextvars.ContextVar("_tessera_autodiff_tape", default=None)

# Opt-in primitive-execution counter (R1 / Baur–Strassen oracle). When a box is
# bound, every actual `tessera.ops.<name>` *execution* (not a tracer record)
# bumps it. Used to measure the cost ratio between a plain forward and a full
# gradient/Jacobian — a ratio far above the Baur–Strassen constant flags an
# implementation that re-runs the forward pass (the jacrev/jacfwd defect B1/B2).
_EXEC_COUNT: contextvars.ContextVar[list | None] = contextvars.ContextVar("_tessera_exec_count", default=None)


@contextmanager
def count_primitive_executions():
    """Count primitive op *executions* inside the block.

    Yields a single-element ``[count]`` box; read ``box[0]`` after the block.
    Only counts ops that actually run (eager or taped); tracer records do not
    execute a primitive and are not counted.
    """
    box = [0]
    token = _EXEC_COUNT.set(box)
    try:
        yield box
    finally:
        _EXEC_COUNT.reset(token)


@contextmanager
def tape():
    """Context manager that opens a new tape and binds it as active.

    Inside the `with` block, every tape-aware `tessera.ops.<name>` call is
    recorded. Outside, ops behave normally.
    """
    t = Tape()
    outer = _ACTIVE_TAPE.get()
    # `_forward_closed` distinguishes a nested tape opened from a VJP rule
    # (which `rematerialize` legitimately does, after the forward pass is over)
    # from one that shadows a running forward pass.
    shadows_outer = outer is not None and not outer._forward_closed
    token = _ACTIVE_TAPE.set(t)
    try:
        yield t
    finally:
        _ACTIVE_TAPE.reset(token)
        if shadows_outer and outer is not None:
            # Decided on EXIT, from what the inner tape actually consumed --
            # not on entry from the mere fact that a tape opened. An inner tape
            # that recorded nothing, or touched only values the outer pass was
            # not tracking, swallowed no gradient path and must not turn a
            # legitimate zero into a refusal.
            outer.shadowed_buffer_ids |= t.consumed_buffer_ids()


def shadow_blocks_zero_claim(t: "Tape", buf_id: int) -> bool:
    """Whether a missing cotangent for `buf_id` is unprovable rather than zero.

    Two ways a nested tape invalidates the zero:

    1. It consumed this buffer directly -- the value's own path was swallowed.
    2. It severed the chain somewhere upstream (`truncated_at_shadow`) and this
       buffer took part in the surviving forward pass. The sweep never reached
       it, so "no cotangent" says nothing about its gradient.

    A buffer that appears in neither -- never recorded here, never swallowed --
    is genuinely unused, and its zero is provable. That distinction is the whole
    point: refusing everything is as wrong as returning zeros for everything.
    """
    if buf_id in t.shadowed_buffer_ids:
        return True
    return bool(t.truncated_at_shadow) and buf_id in t.consumed_buffer_ids()


def raise_nested_tape_shadowed(surface: str) -> NoReturn:
    """Refuse to report a zero gradient that a nested tape is responsible for.

    Called from the zero-gradient branch of `grad` / `jacrev` when the tape was
    shadowed during its forward pass (`Tape.shadowed_by_nested_tape`). At that
    point the honest answer is not zero and not a guess — it is that this lane
    cannot compose the two reverse passes, which is exactly what the sibling
    forward-mode message in `Tape.backward` already says.
    """
    raise TesseraAutodiffError(
        f"{surface} cannot resolve this gradient: an inner tape() opened while "
        "the outer forward pass was still running, so every tessera.ops.* call "
        "in that stretch was recorded on the inner tape and the outer tape "
        "never saw this input.\n"
        "\n"
        "This is what blocks reverse-over-reverse (`grad(grad(f))`, "
        "`jacrev(grad(f))`). It is the mirror of the forward-mode nesting "
        "error raised by tape.backward(): each mode's rules are numpy "
        "functions rather than `ops.*` calls, so neither mode can trace "
        "through the other's. Composing them needs one derivative datum "
        "evaluated in a higher-order algebra (AUTODIFF_NEXTGEN_PLAN.md "
        "AD-WEIL-1), not a deeper tape.\n"
        "\n"
        "Returning zeros here would report that the function is constant in "
        "this input, which is a different and false statement.\n"
        "\n"
        "Available today: `tessera.autodiff.hvp(f, x, v)` for Hessian-vector "
        "products (central difference of the gradient), or "
        "`JitFn.compiled_hvp_ir` for the exact compiled forward-over-reverse "
        "path."
    )


def record_custom_vjp_call(
    name: str,
    forward: Callable[..., Any],
    vjp_fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute and, when a tape is active, record a Python custom-VJP call.

    This is the narrow reference-lane bridge for operations whose forward is a
    Python callable rather than an entry in ``tessera.ops``.  It deliberately
    shares the tape's input-description and recording rules with ordinary ops,
    so a custom VJP composes with ``grad`` instead of merely exposing a manual
    callback on the decorated function.

    ``vjp_fn`` follows the normal convention ``(dout, *array_inputs, **kwargs)
    -> tuple[cotangent, ...]``.  Non-array positional arguments are passed to
    ``forward`` but are not recorded as differentiable inputs.
    """
    active = _ACTIVE_TAPE.get()
    if active is None:
        return forward(*(_to_forward_arg(a) for a in args), **kwargs)

    descs_full = tuple(_describe(a) for a in args)
    forward_args: list[Any] = []
    array_descs: list[InputDesc] = []
    for arg, desc in zip(args, descs_full):
        # `_describe` turns a python int/float into a *literal* descriptor whose
        # `.array` is a float64 0-d array — a convenience for ops like
        # `ops.minimum(x, 1.2)`, where the literal really is a numeric operand.
        # It is wrong here: this function's contract is that non-array
        # positionals reach `forward` unchanged and are not recorded, and a
        # python int is exactly that. Passing the coerced form turned an
        # `axis=1` argument into np.float64(1.0), so both `forward` and the
        # rule raised on any callable that indexes with it.
        if desc is _NON_ARRAY or desc.is_literal:
            forward_args.append(arg)
        else:
            forward_args.append(desc.array)
            array_descs.append(desc)

    out = forward(*forward_args, **kwargs)
    if not isinstance(out, (np.ndarray, np.generic)):
        raise TesseraAutodiffError(
            f"custom VJP call {name!r} must return an ndarray or numpy scalar; got {type(out).__name__}"
        )
    active.record(name, tuple(array_descs), dict(kwargs), out, vjp_fn)
    return out


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


def _describe_sequence(arg: Any) -> list[InputDesc] | None:
    """Describe a list/tuple **of tensors** as a sequence operand, else `None`.

    Every element must be a real array-like — not a literal. That is what keeps
    a configuration list (`ops.pad(x, [1, 2])`, whose elements describe as
    python-scalar literals or not at all) from being mistaken for a sequence of
    operands.
    """
    if not isinstance(arg, (list, tuple)) or not arg:
        return None
    descs: list[InputDesc] = []
    for item in arg:
        d = _describe(item)
        if d is _NON_ARRAY or d.is_literal:
            return None
        descs.append(d)
    return descs


def _regroup_inputs(entry: "TapeEntry") -> tuple:
    """Rebuild the rule's positional arguments from the flat `inputs` list.

    Without `input_groups` this is the identity: one input per argument. With
    it, a sequence operand is handed back to the rule as one list of arrays —
    which is what `vjp_cat(dout, xs, ...)` reads.
    """
    if entry.input_groups is None:
        return tuple(d.array for d in entry.inputs)
    args: list[Any] = []
    cursor = 0
    for group in entry.input_groups:
        if group is None:
            args.append(entry.inputs[cursor].array)
            cursor += 1
        else:
            args.append([entry.inputs[cursor + k].array for k in range(group)])
            cursor += group
    return tuple(args)


def _scatter_grouped_cotangents(entry: "TapeEntry", d_in: tuple) -> tuple:
    """Flatten a rule's per-argument cotangents back to one per recorded input.

    A sequence operand gets one cotangent *per element* from the rule (that is
    what `vjp_cat`/`vjp_stack` already return), so the group is expanded here
    and the ordinary per-input accumulation downstream needs no special case.
    """
    groups = entry.input_groups
    assert groups is not None
    if len(d_in) != len(groups):
        raise TesseraAutodiffError(
            f"VJP for {entry.op!r} returned {len(d_in)} cotangents, expected "
            f"{len(groups)} (one per positional operand, counting a sequence "
            f"operand as one)"
        )
    flat: list[Any] = []
    for g, group in zip(d_in, groups):
        if group is None:
            flat.append(g)
            continue
        if g is None:
            flat.extend([None] * group)
            continue
        parts = list(g)
        if len(parts) != group:
            raise TesseraAutodiffError(
                f"VJP for {entry.op!r} returned {len(parts)} cotangents for a "
                f"sequence operand of {group} tensors; it must return exactly "
                f"one per element so each input receives its own gradient"
            )
        flat.extend(parts)
    return tuple(flat)


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
        # NOTE (W0.3): do NOT resolve a whole-array view to its base here.
        # It looks like a clean fix for accessors that hand out a fresh
        # `arr.view()` per access (`Multivector.coefficients` does), but it is
        # asymmetric: `Tape.record` keys an op's OUTPUT on `id(output)`, so
        # redirecting only the INPUT side severs producer→consumer links and
        # silently drops gradients (measured: 12 Clifford/MoE autodiff
        # failures). Callers that need to recover a cotangent for a buffer
        # they own should match by buffer identity after `backward` —
        # see `tessera.ebm.geo_sampling._cotangent_for_buffer`.
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

    # Sequence operand: a list/tuple whose elements are ALL genuine
    # array-likes (not int/float literals — a `(3, 4)` shape tuple or a
    # `(0.1, 0.2)` mean tuple stays configuration). `cat`/`stack` take one
    # such slot; the rule returns one list-valued gradient for it, and
    # backward fans that out per element (see `Tape.backward`).
    if isinstance(arg, (list, tuple)) and arg:
        elem_descs = []
        for e in arg:
            d = _describe(e)
            if d is _NON_ARRAY or d.is_literal or d.elements is not None:
                break
            elem_descs.append(d)
        else:
            return InputDesc(
                param=None,
                array_id=id(arg),
                array=[d.array for d in elem_descs],
                elements=tuple(elem_descs),
            )

    return _NON_ARRAY


# ─────────────────────────────────────────────────────────────────────────────
# Positional-argument routing — operands stay positional, config becomes kwargs
# ─────────────────────────────────────────────────────────────────────────────
#
# The replay contract is `rule(dout, *recorded_inputs, **recorded_kwargs)`, and
# `_describe` only records array-likes and python int/float. So a canonical
# forward call that passes a *configuration* argument positionally used to be
# mishandled two ways, both of which this routing closes:
#
#   * `ops.reduce(x, "mean")` — a str/bool/None positional is `_NON_ARRAY`, so
#     it reached the forward but was recorded nowhere. The rule then ran on its
#     own default (`op="sum"`) and returned a **silently wrong** gradient.
#   * `ops.sum(x, 0)` — an int positional became a float64 literal `InputDesc`,
#     which was then handed to the *forward* (`np.sum(x, axis=array(0.0))`) and
#     replayed as an extra positional the keyword-only rule could not bind.
#
# Which positional slots are genuine differentiable operands is not guessable
# from the forward signature alone: `mul(x, y=None)` and `clamp(x, min=None,
# max=None)` are spelled identically and mean opposite things. It IS derivable
# from the registered rule, which is the consumer of the contract (Decision
# #30): the rule's own positional slots after `dout` enumerate the operands, in
# order and by name. Anything a slot does not claim is configuration.

# Keyed by `id(fn)`, and the value keeps a strong reference to `fn` itself:
# without it a collected temporary (a `custom_rule` lambda) could free its id
# for reuse and hand the next function a stale signature.
_FORWARD_SIG_CACHE: dict[int, tuple[Callable, list[str] | None, dict[str, Any]]] = {}


def _innermost(fn: Callable) -> Callable:
    """Unwrap to the implementation the call actually binds against.

    `tessera.ops.<name>` stacks the tape wrapper over the ops-namespace dtype
    wrapper over the reference implementation, and both use `functools.wraps`.
    Only the innermost function's signature is the real binding contract —
    `inspect.signature` alone can report an outer, wider one.
    """
    seen: set[int] = set()
    while True:
        inner = getattr(fn, "__wrapped__", None)
        if inner is None or id(inner) in seen:
            return fn
        seen.add(id(fn))
        fn = inner


def _signature_facts(fn: Callable) -> tuple[list[str] | None, dict[str, Any]]:
    """Positional parameter names of `fn` and its per-parameter defaults.

    One parse per callable, for every question the recording paths ask of a
    signature: `inspect.signature` costs microseconds that a taped call pays
    again on every iteration of a training loop, with an identical answer.
    The entry holds a strong reference to `fn` so the `id()` key cannot be
    reused by a later object.
    """
    key = id(fn)
    cached = _FORWARD_SIG_CACHE.get(key)
    if cached is not None:
        return cached[1], cached[2]
    defaults: dict[str, Any] = {}
    try:
        sig = inspect.signature(_innermost(fn), follow_wrapped=False)
    except (TypeError, ValueError):
        names = None
    else:
        params = list(sig.parameters.values())
        defaults = {p.name: p.default for p in params}
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
            names = None  # `*args` — no stable name for a positional slot
        else:
            names = [
                p.name
                for p in params
                if p.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
    _FORWARD_SIG_CACHE[key] = (fn, names, defaults)
    return names, defaults


def _positional_params(fn: Callable) -> list[str] | None:
    """Positional parameter names of `fn`, or None if it takes `*args`/is opaque."""
    return _signature_facts(fn)[0]


def _route_positional(name: str, original: Callable, args: tuple, vjp_fn: Callable | None):
    """Split `args` into (forward_args, recorded input descs, routed kwargs).

    Arrays are always operands. A non-array positional (int/float literal, str,
    bool, None, ...) stays an operand only when the rule declares a positional
    slot of that name at that operand index; otherwise it is configuration and
    is returned as a kwarg keyed by its canonical forward parameter name.
    """
    fwd_names = _positional_params(original)
    rule_slots = _positional_params(vjp_fn) if vjp_fn is not None else None
    # `rule_slots[0]` is `dout`; operand k sits at `rule_slots[k + 1]`.
    operand_slots = rule_slots[1:] if rule_slots else None

    forward_args: list[Any] = []
    array_descs: list[InputDesc] = []
    groups: list[int | None] = []
    routed: dict[str, Any] = {}
    for i, a in enumerate(args):
        seq = _describe_sequence(a)
        if seq is not None:
            # A sequence operand (`ops.cat([a, b])`): every element becomes a
            # recorded input, and `input_groups` remembers they belong to one
            # positional argument. Before this the whole list was neither an
            # operand nor a kwarg, so the rule was called without it.
            forward_args.append([d.array for d in seq])
            array_descs.extend(seq)
            groups.append(len(seq))
            continue
        d = _describe(a)
        if d is not _NON_ARRAY and not d.is_literal:
            forward_args.append(d.array)  # a real array is always an operand
            array_descs.append(d)
            groups.append(None)
            continue
        pname = fwd_names[i] if fwd_names is not None and i < len(fwd_names) else None
        k = len(array_descs)  # the operand index this arg would occupy
        claimed = (
            pname is not None
            and operand_slots is not None
            and k < len(operand_slots)
            # The slot is this arg's iff the rule names it the same way, or —
            # for the many rules that rename operands (`vjp_mod(dout, a, b)`
            # over `mod(x, y)`) — iff no earlier positional arg was skipped, so
            # the two index spaces still line up. Requiring `i == k` after a
            # skip is what stops `conv2d(x, w, None, 1, 0)` from binding the
            # stride to the rule's `bias` slot.
            and (operand_slots[k] == pname or i == k)
        )
        if claimed and a is None:
            # Preserve an omitted optional operand when a later keyword tensor
            # is promoted into the positional operand sequence.  The literal
            # placeholder is replayed to the rule, receives a ``None``
            # cotangent, and keeps every later tensor aligned with its declared
            # slot (for example GEMM residual without bias).
            forward_args.append(None)
            array_descs.append(
                InputDesc(param=None, array_id=id(a), array=None, is_literal=True)
            )
            groups.append(None)
            continue
        if claimed or pname is None:
            # A differentiable scalar operand (`ops.mul(x, -0.5)`), or an op
            # whose binding contract we cannot read — keep the legacy handling.
            if d is _NON_ARRAY:
                forward_args.append(a)
            else:
                forward_args.append(d.array)
                array_descs.append(d)
                groups.append(None)
            continue
        # Configuration: the forward gets the ORIGINAL value in its original
        # position (an int stays an int), and the rule gets it by name.
        forward_args.append(a)
        routed[pname] = a
    # `None` (the common case) keeps the fast path: one input per argument.
    input_groups = tuple(groups) if any(g is not None for g in groups) else None
    return forward_args, array_descs, routed, input_groups


def promote_operand_kwargs(
    original: Callable,
    args: tuple,
    kwargs: dict,
    rule_fn: Callable | None,
):
    """Move keyword-spelled operands into the positional operand list.

    `ops.rmsnorm(x, gamma=g)` and `ops.mul(x, y=y)` spell a differentiable
    operand by keyword. The recording paths only inspect positional args,
    so a keyword operand was invisible to the record (never a primal, never
    an `InputDesc`) while the rule still bound it by name from the recorded
    kwargs and returned a cotangent for it — the measured failure is
    "VJP returned 2 cotangents, expected 1", and forward mode ran the
    rule's operand-less branch (PR #604 review, P1).

    Same authority as `_route_positional` (Decision #30): the VJP rule's
    positional slots after `dout` enumerate the operands. A kwarg is
    promoted iff it fills the NEXT unfilled slot, its name matches BOTH
    the rule slot and the forward's positional parameter at that index
    (so passing it positionally to the forward is meaning-preserving —
    rules that rename operands simply keep the old behavior), and its
    value is operand-shaped (an array or a non-bool scalar; None/str/bool
    stay configuration). Keyword-only rule parameters (`*, eps=...`) are
    never slots, so configuration can never be promoted.
    """
    if not kwargs or rule_fn is None:
        return args, kwargs
    slots = _positional_params(rule_fn)
    fwd_names, fwd_defaults = _signature_facts(original)
    if not slots or fwd_names is None:
        return args, kwargs
    operand_slots = slots[1:]  # slots[0] is `dout`
    out = list(args)
    remaining = dict(kwargs)

    def has_none_default(name: str) -> bool:
        return fwd_defaults.get(name, inspect.Parameter.empty) is None

    def later_operand_exists(start: int) -> bool:
        for later in range(start, len(operand_slots)):
            name = operand_slots[later]
            if name not in remaining:
                continue
            value = remaining[name]
            if value is None or isinstance(value, (bool, str)):
                continue
            if _describe(value) is not _NON_ARRAY:
                return True
        return False

    for k in range(len(args), len(operand_slots)):
        pname = operand_slots[k]
        if k >= len(fwd_names) or fwd_names[k] != pname:
            break
        if pname not in remaining:
            if has_none_default(pname) and later_operand_exists(k + 1):
                out.append(None)
                continue
            break
        v = remaining[pname]
        if v is None or isinstance(v, (bool, str)):
            break
        if _describe(v) is _NON_ARRAY:
            break
        out.append(remaining.pop(pname))
    if len(out) == len(args):
        return args, kwargs
    return tuple(out), remaining


def split_positional_config(name: str, original: Callable, args: tuple):
    """Forward-mode view of the same split: `(operands, config_kwargs)`.

    Reverse mode keeps configuration in its positional place for the forward
    call and records it by name; forward mode has no separate record, so the
    configuration is *moved* out of the primal tuple and into the kwargs the
    rule (and the forward itself) already accept by name.
    """
    from .vjp import get_vjp

    _, _, routed, _groups = _route_positional(name, original, args, get_vjp(name))
    if not routed:
        return args, {}
    fwd_names = _positional_params(original) or []
    operands = tuple(
        a for i, a in enumerate(args)
        if not (i < len(fwd_names) and fwd_names[i] in routed)
    )
    return operands, routed


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

        # Compiler/public forward mode uses the same canonical op wrapper as
        # reverse mode. Keeping this hook here avoids global monkey-patching and
        # makes nested thread/async contexts safe through ContextVar ownership.
        from .jvp import active_jvp_trace

        # R1 cost oracle: count this primitive execution if a counter is bound.
        # This must happen BEFORE the forward-mode dispatch below: the JVP trace
        # `return`s, so counting after it made every forward-mode execution
        # invisible and the Baur-Strassen ratio read ~1x for a `jacfwd` that was
        # really running one forward pass per input dimension (MC7).
        _exec_box = _EXEC_COUNT.get()
        if _exec_box is not None:
            _exec_box[0] += 1

        _jvp_trace = active_jvp_trace()
        if _jvp_trace is not None:
            _open_tape = _ACTIVE_TAPE.get()
            if _open_tape is not None:
                _open_tape.diverted_to_forward_mode = True
            return _jvp_trace.record_op(name, original, args, kwargs)

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
        # Positionally-passed *configuration* arguments are routed into the
        # recorded kwargs under their canonical parameter name so the rule
        # sees them exactly as it would from a keyword call.
        vjp_fn = get_vjp(name)
        # A keyword-spelled operand (`ops.rmsnorm(x, gamma=g)`) joins the
        # positional operands first, or it is never recorded while the rule
        # still answers for it — see `promote_operand_kwargs`.
        args, kwargs = promote_operand_kwargs(original, args, kwargs, vjp_fn)
        forward_args, array_descs, routed_kwargs, input_groups = _route_positional(
            name, original, args, vjp_fn
        )

        cast_dtype = autocast_dtype()
        if cast_dtype is not None:
            if autocast_keep_fp32(name):
                forward_args = list(_autocast_args(forward_args, "fp32"))
            else:
                forward_args = list(_autocast_args(forward_args, cast_dtype))

        out = original(*forward_args, **kwargs)
        # `vjp_fn` was looked up above, at call time, so that `custom_rule` can
        # register/override after the wrapper was installed. `vjp_fn=None` is
        # allowed; backward raises only if this entry actually lies on the
        # gradient path.
        if routed_kwargs:
            kwargs = {**routed_kwargs, **kwargs}
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
                        input_groups,
                    )
        else:
            active.record(name, tuple(array_descs), dict(kwargs), out, vjp_fn,
                          input_groups)
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
            quantize = getattr(_ops.quantize_fp8, "__wrapped__", _ops.quantize_fp8)
            fmt = dtype[len("fp8_") :]
            kwargs = {"format": fmt}
        elif dtype.startswith("fp6_"):
            quantize = getattr(_ops.quantize_fp6, "__wrapped__", _ops.quantize_fp6)
            fmt = dtype[len("fp6_") :]
            kwargs = {"format": fmt}
        elif dtype.startswith("fp4_"):
            quantize = getattr(_ops.quantize_fp4, "__wrapped__", _ops.quantize_fp4)
            fmt = dtype[len("fp4_") :]
            kwargs = {"format": fmt}
        else:  # nvfp4
            quantize = getattr(_ops.quantize_nvfp4, "__wrapped__", _ops.quantize_nvfp4)
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
