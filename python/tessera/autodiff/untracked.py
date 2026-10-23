"""Refuse a gradient whose path ran through values the tape never saw.

The reverse-mode tape intercepts ``tessera.ops.*`` calls and nothing else. A
tracked value that passes through raw numpy or Python -- ``ops.sin(x) * x``
with numpy's ``*`` -- comes back as a plain array no recorded op produced, and
when a later ``ops.*`` call consumes it the tape records it as an *operand*:
indistinguishable, by identity, from a constant. The backward sweep stops
there, the differentiated argument receives no cotangent along that path, and
the gradient is wrong without looking wrong -- for ``ops.sum(ops.sin(x) * x)``
every component reads zero.

The tape cannot tell a laundered intermediate from a genuine constant
(``ops.mul(x, W)``) by identity alone, so this module asks the question it
actually needs answered: **does the operand's value depend on the
differentiated arguments?** It replays ``fn`` under fresh tapes with the
arguments moved by a scale-aware step ``h`` and by ``h/2``, and compares the
operand at the same position of the structurally identical replay; see
:func:`refuse_untracked_dependence` for the full test.

Only positive evidence refuses. An operand that is stable under the step is a
constant to first order (a captured weight, ``np.sign(x)``), and one whose
change does not scale with the step is a threshold jump (a mask crossing its
boundary), whose derivative is zero. A replay whose op sequence differs
(data-dependent Python control flow), or an operand that changes between two
*unperturbed* runs (unseeded randomness), is inconclusive and left alone:
refusing there would reject correct programs, and the residual is documented
in ``docs/spec/AUTODIFF_SPEC.md``.

**Replays re-execute caller code.** The probe runs only when the sweep reached
an untracked operand that is not an array the caller handed in (an argument,
keyword or closure/global value), so programs written entirely in ``ops.*``
over captured constants never replay; a correct program with a locally built
constant replays once. Each replay starts from the global numpy and Python
``random`` states the original forward pass saw, and the post-forward states
are restored afterwards, so the caller's random streams are unchanged. Other
side effects (counters, buffer mutation, I/O) repeat: ``fn`` must be pure, the
same contract ``jacfwd``, ``vmap`` and ``hvp`` already rely on by
re-executing it.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .tape import Tape, TesseraAutodiffError, tape

#: One operand position on a tape: (entry index, flat operand index).
_Position = tuple[int, int]


def _flat_operands(entry: Any) -> list[Any]:
    """An entry's differentiable operand descriptors, sequences flattened."""
    out: list[Any] = []
    for desc in entry.inputs:
        if desc.elements is not None:
            out.extend(desc.elements)
        else:
            out.append(desc)
    return out


def _caller_array_ids(fn: Callable, args: Sequence[Any], kwargs: dict,
                      argnums: Iterable[int]) -> set[int]:
    """Ids of arrays the caller supplied rather than ``fn`` computed.

    These are constants with respect to the differentiated arguments by
    construction, so they never need a replay. Containers are searched two
    levels deep; anything missed here is merely probed, not misjudged.
    """
    ids: set[int] = set()

    def visit(value: Any, depth: int) -> None:
        if isinstance(value, np.ndarray):
            ids.add(id(value))
            return
        if isinstance(value, (type, type(inspect))):
            return
        try:
            data = getattr(value, "_data", None)
        except Exception:
            return
        if isinstance(data, np.ndarray):
            ids.add(id(data))
            return
        if depth <= 0:
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item, depth - 1)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, depth - 1)

    skip = set(argnums)
    for index, value in enumerate(args):
        if index not in skip:
            visit(value, 2)
    for value in kwargs.values():
        visit(value, 2)
    try:
        closure = inspect.getclosurevars(inspect.unwrap(fn))
    except (TypeError, ValueError):
        closure = None
    if closure is not None:
        for value in list(closure.nonlocals.values()) + list(closure.globals.values()):
            visit(value, 2)
    return ids


def untracked_operand_positions(t: Tape, nonzero_ids: set[int],
                                caller_ids: set[int]) -> list[_Position]:
    """Operands no recorded op produced, that the sweep reached with a nonzero
    cotangent, and that are not Parameters or caller-supplied arrays."""
    produced = {entry.output_id for entry in t.entries}
    positions: list[_Position] = []
    for k, entry in enumerate(t.entries):
        for j, desc in enumerate(_flat_operands(entry)):
            if desc.is_literal or desc.param is not None:
                continue
            if desc.array_id in produced or desc.array_id in caller_ids:
                continue
            if desc.array_id in nonzero_ids:
                positions.append((k, j))
    return positions


def nonzero_cotangent_ids(t: Tape) -> set[int]:
    """Buffer ids whose cotangent from the last sweep is not identically zero."""
    return {
        buf_id for buf_id, value in t.cotangent.items()
        if value is not None and np.any(np.asarray(value) != 0)
    }


def rng_snapshot() -> tuple[Any, Any]:
    """The global numpy and Python ``random`` states, for :func:`refuse_untracked_dependence`."""
    import random

    return np.random.get_state(), random.getstate()


def _rng_restore(snapshot: tuple[Any, Any]) -> None:
    import random

    np.random.set_state(snapshot[0])
    random.setstate(snapshot[1])


def _perturbed(value: Any, fraction: float) -> np.ndarray | None:
    arr = np.array(value, copy=True)
    if not np.issubdtype(arr.dtype, np.floating):
        return None
    # A scale-aware step (sqrt(eps) relative, the finite-difference crossover)
    # keeps the probe local: a threshold a coarse step would cross stays put.
    # Deterministic, nonzero at every element; a local generator so no RNG
    # stream is consumed.
    scale = fraction * float(np.sqrt(np.finfo(arr.dtype).eps))
    sign = np.where(np.random.default_rng(0x7E55).random(arr.shape) < 0.5, -1.0, 1.0)
    return (arr + scale * (1.0 + np.abs(arr)) * sign).astype(arr.dtype)


def _replay(fn: Callable, args: Sequence[Any], kwargs: dict,
            argnums: Sequence[int], fraction: float, signature: list[str],
            positions: list[_Position],
            rng_before: tuple[Any, Any] | None) -> list[np.ndarray] | None:
    """Re-run ``fn`` with the differentiated arguments moved by ``fraction``
    of the probe step (0 = unchanged) and return the operands at
    ``positions``, or ``None`` when the replay is not comparable."""
    from .grad import _wrap_as_parameter

    new_args = list(args)
    for i in argnums:
        value = np.array(args[i], copy=True)
        if fraction:
            moved = _perturbed(value, fraction)
            if moved is None:
                return None
            value = moved
        new_args[i] = _wrap_as_parameter(value)
    if rng_before is not None:
        # The replay draws the same global random stream the original
        # forward pass drew, so seeded-by-state noise reads as the constant
        # it is to the derivative.
        _rng_restore(rng_before)
    try:
        with tape() as replay:
            fn(*new_args, **kwargs)
    except Exception:
        return None
    if [entry.op for entry in replay.entries] != signature:
        return None
    return [np.array(_flat_operands(replay.entries[k])[j].array, copy=True)
            for k, j in positions]


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def _change(base: np.ndarray, moved: np.ndarray) -> float:
    return float(np.linalg.norm(
        np.asarray(moved, dtype=np.float64) - np.asarray(base, dtype=np.float64)))


def refuse_untracked_dependence(
    surface: str,
    fn: Callable,
    args: Sequence[Any],
    kwargs: dict,
    argnums: Sequence[int],
    t: Tape,
    nonzero_ids: set[int],
    rng_before: tuple[Any, Any] | None = None,
) -> None:
    """Raise when a gradient path ran through a value the tape never saw.

    ``nonzero_ids`` is the union, over every backward sweep the caller ran, of
    :func:`nonzero_cotangent_ids` -- ``jacrev`` sweeps once per output element.
    ``rng_before`` is :func:`rng_snapshot` taken before the original forward
    pass; replays start from it and the post-forward state is restored after.

    Evidence is a *derivative*, not a difference. An operand counts as
    depending on the arguments only when it moves under a step ``h``, moves
    again under ``h/2``, the two changes scale linearly (ratio ~2, which a
    threshold jump cannot produce), and an unchanged replay reproduces the
    original operand (which unseeded randomness cannot). Replays run lazily
    and stop at the first test that clears every operand, so a correct
    program with a locally built constant costs one replay, not three.
    """
    caller_ids = _caller_array_ids(fn, args, kwargs, argnums)
    positions = untracked_operand_positions(t, nonzero_ids, caller_ids)
    if not positions:
        return
    signature = [entry.op for entry in t.entries]
    base = {pos: np.array(_flat_operands(t.entries[pos[0]])[pos[1]].array, copy=True)
            for pos in positions}
    after = rng_snapshot()
    try:
        full = _replay(fn, args, kwargs, argnums, 1.0, signature, positions, rng_before)
        if full is None:
            return
        moved = {pos: _change(base[pos], value)
                 for pos, value in zip(positions, full) if not _same(base[pos], value)}
        if not moved:
            return
        candidates = list(moved)
        half = _replay(fn, args, kwargs, argnums, 0.5, signature, candidates, rng_before)
        if half is None:
            return
        linear = []
        for pos, value in zip(candidates, half):
            step_half = _change(base[pos], value)
            if step_half > 0.0 and 1.5 <= moved[pos] / step_half <= 2.5:
                linear.append(pos)
        if not linear:
            return
        repeat = _replay(fn, args, kwargs, argnums, 0.0, signature, linear, rng_before)
        if repeat is None:
            return
        dependent = [pos for pos, value in zip(linear, repeat) if _same(base[pos], value)]
    finally:
        _rng_restore(after)
    if not dependent:
        return
    k, j = dependent[0]
    raise TesseraAutodiffError(
        f"{surface}: operand {j} of the recorded tessera.ops.{t.entries[k].op} "
        f"call (tape entry {k}) depends on the differentiated argument(s), but "
        f"no recorded op produced it -- it was computed in raw numpy or Python "
        f"(e.g. `ops.sin(x) * x` with numpy's `*`). The tape cannot see that "
        f"path, so the gradient along it would be silently missing "
        f"({len(dependent)} such operand(s) found). Express the computation "
        f"with tessera.ops.* (ops.mul, ops.add, ...), or, for loss math that "
        f"runs entirely after the last recorded op, seed backward with an "
        f"explicit cotangent. See docs/spec/AUTODIFF_SPEC.md."
    )


__all__ = [
    "nonzero_cotangent_ids",
    "rng_snapshot",
    "refuse_untracked_dependence",
    "untracked_operand_positions",
]
