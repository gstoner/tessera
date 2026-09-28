"""Per-call route receipts for the public GA / EBM primitives.

EVIDENCE-PACKET-1 (docs/audit/compiler/INTEGRATED_COMPILER_PLAN.md), sync
``EVIDENCE-PACKET-1-2026-09-27``.

A public ``tessera.ga`` / ``tessera.ebm`` primitive decides at call time
whether it runs natively -- the Apple GPU runtime's MSL kernels, the x86
AVX-512 kernels, a compiler-generated ROCm kernel, the sm_120 row program --
or on its NumPy reference. Each native lane is one ``_try_<target>_*`` helper
that returns the native result or ``None``; the primitive falls back on
``None``. Nothing recorded which happened. The composition benchmarks
(``benchmarks/{clifford,energy}_core``) could therefore only label their rows
``unattributed``, and the jit_bridge trace (``take_dispatch_trace``) saw only
the Apple manifest lane, so on an x86 or ROCm host it would have reported "no
native dispatch" for a call that ran on AVX-512.

This module closes that:

* :func:`native_attempt` decorates every ``_try_<target>_*`` helper. When the
  helper returns a result (not ``None``) it notes a native dispatch for its
  target, derived from the helper's name -- never from the host.
* :func:`public_route` decorates every public primitive that reaches such a
  helper. Each call opens a frame; when it returns, the frame becomes a
  :class:`RouteReceipt` naming the route its own native dispatches and its
  nested public calls produced.
* :func:`capture_route_receipts` collects receipts for a span, per thread.
  Outside a capture both decorators cost one thread-local read.

Fail-closed rules (Decision #21a): a native dispatch noted while no public
frame is open is an **orphan** -- some native work ran that no receipt
names -- and the capture's attribution is ``incomplete``; a capture with no
receipts is ``incomplete`` too. An incomplete capture's route is
``unattributed``, never a guess. ``tests/unit/test_route_receipts.py`` fails
if a ``_try_*`` helper or a public caller of one in ``tessera.ga`` /
``tessera.ebm`` is left undecorated, or if native runtime work is reached
outside a ``_try_*`` helper.

What a receipt does *not* claim: Python glue inside a public primitive (the
code around its native calls) is not separately attributed, and a receipt is
not a timing or promotion certificate. The ``rocm`` target names the ROCm
lane, not a chip: the chip is a fact about the recording host.
"""

from __future__ import annotations

import functools
import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, TypeVar

ROUTE_RECEIPT_SCHEMA = "tessera.route_receipts.v1"

ROUTE_PYTHON_REFERENCE = "python_reference"
ROUTE_MIXED = "mixed"
ROUTE_UNATTRIBUTED = "unattributed"

#: ``_try_<prefix>`` -> the native target it dispatches to. Ordered so the
#: longest prefix matches first.
NATIVE_TARGETS: tuple[tuple[str, str], ...] = (
    ("_try_apple_gpu_", "apple_gpu_runtime"),
    ("_try_cuda_gpu_", "cuda"),
    ("_try_x86_", "x86_avx512"),
    ("_try_rocm_", "rocm"),
)

F = TypeVar("F", bound=Callable[..., Any])


def native_target_for(helper_name: str) -> str:
    """The native target a ``_try_*`` helper's name declares."""
    for prefix, target in NATIVE_TARGETS:
        if helper_name.startswith(prefix):
            return target
    raise ValueError(
        f"{helper_name!r} names no declared native target; a native lane helper "
        f"must be called _try_<target>_... with one of {[p for p, _ in NATIVE_TARGETS]}")


@dataclass(frozen=True)
class RouteReceipt:
    """One public call's route.

    ``native`` lists ``(target, helper)`` for the dispatches the call made
    itself; ``nested_routes`` the routes of public calls it made. ``route`` is
    ``python_reference`` when nothing under the call ran natively, the single
    target when every native piece ran there and no nested public call took
    a reference or mixed route, and ``mixed`` otherwise.
    """

    op: str
    route: str
    depth: int
    native: tuple[tuple[str, str], ...]
    nested_routes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "route": self.route, "depth": self.depth,
                "native": [list(n) for n in self.native],
                "nested_routes": list(self.nested_routes)}


@dataclass
class _Frame:
    op: str
    depth: int
    native: list[tuple[str, str]] = field(default_factory=list)
    nested: list[str] = field(default_factory=list)


@dataclass
class RouteReceiptLog:
    """Everything one :func:`capture_route_receipts` span saw."""

    receipts: list[RouteReceipt] = field(default_factory=list)
    orphan_dispatches: list[tuple[str, str]] = field(default_factory=list)

    def top_level(self) -> list[RouteReceipt]:
        return [r for r in self.receipts if r.depth == 0]

    def refusal(self) -> str | None:
        """Why this capture cannot attribute its span, or ``None``."""
        if self.orphan_dispatches:
            helpers = sorted({h for _, h in self.orphan_dispatches})
            return (f"ROUTE_RECEIPT_ORPHAN_DISPATCH: {len(self.orphan_dispatches)} native "
                    f"dispatch(es) outside any public primitive ({helpers})")
        if not self.top_level():
            return "ROUTE_RECEIPT_EMPTY: no public GA/EBM primitive was called in the span"
        return None

    def route(self) -> str:
        if self.refusal() is not None:
            return ROUTE_UNATTRIBUTED
        routes = {r.route for r in self.top_level()}
        return routes.pop() if len(routes) == 1 else ROUTE_MIXED

    def summary(self) -> dict[str, Any]:
        top = self.top_level()
        ops: dict[str, Counter[str]] = {}
        for r in top:
            ops.setdefault(r.op, Counter())[r.route] += 1
        native: Counter[str] = Counter()
        for r in self.receipts:
            for target, helper in r.native:
                native[f"{target}:{helper}"] += 1
        refusal = self.refusal()
        return {
            "schema": ROUTE_RECEIPT_SCHEMA,
            "attribution": "incomplete" if refusal else "complete",
            "refusal": refusal,
            "route": self.route(),
            "calls": len(top),
            "routes": dict(sorted(Counter(r.route for r in top).items())),
            "ops": {op: dict(sorted(c.items())) for op, c in sorted(ops.items())},
            "native_dispatches": dict(sorted(native.items())),
            "orphan_dispatches": len(self.orphan_dispatches),
        }


class _State(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.captures: list[RouteReceiptLog] = []
        self.frames: list[_Frame] = []


_STATE = _State()


@contextmanager
def capture_route_receipts() -> Iterator[RouteReceiptLog]:
    """Collect every receipt made on this thread inside the ``with`` block."""
    log = RouteReceiptLog()
    _STATE.captures.append(log)
    try:
        yield log
    finally:
        # Stack discipline by identity: RouteReceiptLog is a dataclass, so
        # ``list.remove`` would match an *equal* log (two empty nested
        # captures) and pop the outer one (Codex review, PR #869).
        if not _STATE.captures or _STATE.captures[-1] is not log:
            raise RuntimeError("route receipt captures closed out of order")
        _STATE.captures.pop()


def note_native_dispatch(target: str, helper: str) -> None:
    """Record that native work for ``target`` ran (via ``helper``)."""
    state = _STATE
    if not state.captures:
        return
    if state.frames:
        state.frames[-1].native.append((target, helper))
    else:
        for log in state.captures:
            log.orphan_dispatches.append((target, helper))


def _route_of(frame: _Frame) -> str:
    targets = {t for t, _ in frame.native}
    nested = set(frame.nested)
    nested_native = nested - {ROUTE_PYTHON_REFERENCE}
    if not targets and not nested_native:
        return ROUTE_PYTHON_REFERENCE
    if ROUTE_PYTHON_REFERENCE in nested or ROUTE_MIXED in nested:
        return ROUTE_MIXED
    everything = targets | nested_native
    return everything.pop() if len(everything) == 1 else ROUTE_MIXED


def public_route(op: str) -> Callable[[F], F]:
    """Decorate a public primitive so each call leaves a :class:`RouteReceipt`.

    ``op`` is ``<package>:<name>`` (``"ebm:inner_step"``), never a dotted
    ``tessera.ebm.inner_step``: that spelling is also an ODS op name, and the
    ODS consumer audit would read the label as a compiler consumer of the op.
    """
    if "." in op or ":" not in op:
        raise ValueError(f"public_route label {op!r} must be '<package>:<name>' with no dots")

    def wrap(fn: F) -> F:
        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            state = _STATE
            if not state.captures:
                return fn(*args, **kwargs)
            frame = _Frame(op=op, depth=len(state.frames))
            state.frames.append(frame)
            try:
                result = fn(*args, **kwargs)
            finally:
                state.frames.pop()
            # Only a call that returned leaves a receipt; an exception reaches
            # the caller, which is attribution enough.
            receipt = RouteReceipt(op=op, route=_route_of(frame), depth=frame.depth,
                                   native=tuple(frame.native), nested_routes=tuple(frame.nested))
            if state.frames:
                state.frames[-1].nested.append(receipt.route)
            for log in state.captures:
                log.receipts.append(receipt)
            return result

        inner.__tessera_public_route__ = op  # type: ignore[attr-defined]
        return inner  # type: ignore[return-value]

    return wrap


def native_attempt(fn: F) -> F:
    """Decorate a ``_try_<target>_*`` helper: a non-``None`` return is a
    native dispatch for the target its name declares."""
    target = native_target_for(fn.__name__)
    helper = fn.__name__

    @functools.wraps(fn)
    def inner(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        if result is not None and _STATE.captures:
            note_native_dispatch(target, helper)
        return result

    inner.__tessera_native_target__ = target  # type: ignore[attr-defined]
    return inner  # type: ignore[return-value]


def validate_receipt_summary(summary: Any) -> str:
    """Re-derive a stored :meth:`RouteReceiptLog.summary` and return its route.

    A consumer of a recorded row calls this instead of trusting its ``route``:
    the route, attribution and call count must follow from the per-route and
    per-op counts it states, every route must be declared, and a complete
    attribution may carry no orphan dispatch. Raises ``ValueError``.
    """
    if not isinstance(summary, dict) or summary.get("schema") != ROUTE_RECEIPT_SCHEMA:
        raise ValueError("route receipts: not a tessera.route_receipts.v1 summary")
    declared = {ROUTE_PYTHON_REFERENCE, ROUTE_MIXED} | {t for _, t in NATIVE_TARGETS}
    routes, ops = summary.get("routes"), summary.get("ops")
    if not isinstance(routes, dict) or not isinstance(ops, dict):
        raise ValueError("route receipts: routes and ops must be objects")
    if set(routes) - declared:
        raise ValueError(f"route receipts: undeclared route(s) {sorted(set(routes) - declared)}")
    counts = [v for v in routes.values()]
    if not all(type(v) is int and v > 0 for v in counts):
        raise ValueError("route receipts: route counts must be positive integers")
    per_op: Counter[str] = Counter()
    for op, by_route in ops.items():
        if not isinstance(by_route, dict):
            raise ValueError(f"route receipts: op {op!r} has no route counts")
        per_op.update(by_route)
    if dict(per_op) != routes:
        raise ValueError("route receipts: per-op counts do not sum to the route counts")
    orphans = summary.get("orphan_dispatches")
    if type(orphans) is not int or orphans < 0:
        raise ValueError("route receipts: orphan_dispatches must be a count")
    if summary.get("calls") != sum(counts):
        raise ValueError("route receipts: calls differ from the route counts")
    complete = bool(counts) and orphans == 0
    expected = (ROUTE_UNATTRIBUTED if not complete
                else next(iter(routes)) if len(routes) == 1 else ROUTE_MIXED)
    if summary.get("attribution") != ("complete" if complete else "incomplete"):
        raise ValueError("route receipts: attribution differs from what the counts derive")
    if (summary.get("refusal") is None) != complete:
        raise ValueError("route receipts: refusal present exactly when attribution is incomplete")
    if summary.get("route") != expected:
        raise ValueError(f"route receipts: route {summary.get('route')!r} is not the derived {expected!r}")
    return expected


def device_label(route: str) -> str:
    """The benchmark ``device`` a composition's route supports.

    Composition glue always runs on the host, so any native route is reported
    beside ``cpu``; an unattributed span stays ``unattributed``.
    """
    if route == ROUTE_UNATTRIBUTED:
        return ROUTE_UNATTRIBUTED
    if route == ROUTE_PYTHON_REFERENCE:
        return "cpu"
    if route == ROUTE_MIXED:
        return "mixed+cpu"
    return f"{route}+cpu"


__all__ = [
    "NATIVE_TARGETS",
    "ROUTE_MIXED",
    "ROUTE_PYTHON_REFERENCE",
    "ROUTE_RECEIPT_SCHEMA",
    "ROUTE_UNATTRIBUTED",
    "RouteReceipt",
    "RouteReceiptLog",
    "capture_route_receipts",
    "device_label",
    "native_attempt",
    "native_target_for",
    "note_native_dispatch",
    "public_route",
    "validate_receipt_summary",
]
