"""Memory-model verifier — M4 deliverable.

Tessera's memory model claims live in
``docs/spec/MEMORY_MODEL_SPEC.md`` and across the Schedule/Tile IR
verifiers.  The existing :class:`tessera.compiler.tile_ir.TileIRVerifier`
checks structural validity (``async_copy.stage >= 0``,
``queue.create.depth >= 1``, duplicate queues, etc.) but **not**
ordering claims like "every ``wait_async`` for stage S happens
after at least one ``async_copy`` at stage S" or "every
``queue.pop`` happens after a matching ``queue.push``".

M4 adds those ordering checks as a second pass that walks a
:class:`TileIRModule` and emits diagnostics for:

  - ``wait_async`` with no preceding ``async_copy`` at the same stage.
  - ``queue.pop`` / ``queue.barrier`` with no preceding ``queue.push``
    on the same queue.
  - ``queue.push`` with no ``queue.create`` already in scope.
  - ``async_copy`` whose ``source`` / ``dest`` memory-space attrs
    name an illegal transition (e.g., ``shared`` → ``shared``,
    which would normally be a register copy).

The verifier emits :class:`MemoryModelDiagnostic` records that
reuse the same severity / code surface as
:class:`tile_ir.TileIRDiagnostic` so callers can route the output
through one diagnostic pipeline.

Source spans (op attrs ``loc_line`` / ``loc_col``) are preserved
in the diagnostic ``where`` payload when present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .tile_ir import TileFunction, TileIRModule, TileOp


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic surface
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MemoryModelDiagnostic:
    """One memory-model verifier finding.

    ``code`` is a stable string the test surface can match on
    (e.g., ``"MEM_WAIT_WITHOUT_COPY"``).  ``where`` carries the
    source-span attrs when the offending op exposed them.
    """
    severity: str
    message: str
    code: str
    op_name: str
    where: Optional[dict] = None

    def format(self) -> str:
        scope = ""
        if self.where:
            scope = f"  loc={self.where}"
        return f"{self.severity.upper()} [{self.code}]: {self.message}{scope}"


@dataclass(frozen=True)
class MemoryModelVerificationResult:
    diagnostics: tuple[MemoryModelDiagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)

    @property
    def errors(self) -> tuple[MemoryModelDiagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity == "error")

    def format(self) -> str:
        return "\n".join(d.format() for d in self.diagnostics)


class MemoryModelVerificationError(ValueError):
    """Raised when :func:`verify_memory_model` is asked to assert
    success and the program failed verification."""


# ─────────────────────────────────────────────────────────────────────────────
# Memory-space transitions
# ─────────────────────────────────────────────────────────────────────────────

# Canonical Tessera memory-space names.  ``async_copy`` source/dest
# attrs use these strings (when set).  See
# ``docs/spec/MEMORY_MODEL_SPEC.md`` §Memory spaces.
VALID_MEMORY_SPACES: frozenset[str] = frozenset({
    "global", "shared", "register", "tmem", "constant", "host",
})

# Pairs (source -> dest) that are legal for ``async_copy``.  All
# other source/dest combos surface as
# ``MEM_INVALID_ASYNC_COPY_TRANSITION``.  This list captures what
# the current Apple GPU + NVIDIA SM_90+ paths actually use.
LEGAL_ASYNC_COPY_TRANSITIONS: frozenset[tuple[str, str]] = frozenset({
    ("global",   "shared"),    # the canonical SM_80+ cp.async
    ("global",   "tmem"),      # Blackwell TMEM / TMA descriptor land
    ("shared",   "register"),  # tile-local prologue
    ("global",   "register"),  # bypass-shared TMA on Hopper
    ("host",     "global"),    # ingest path (rarely used at Tile IR level)
})


# ─────────────────────────────────────────────────────────────────────────────
# Verifier
# ─────────────────────────────────────────────────────────────────────────────

class _MemoryStateTracker:
    """Per-function happens-before state."""

    __slots__ = (
        "issued_copy_stages", "open_queues", "queue_outstanding",
        "diagnostics",
    )

    def __init__(self) -> None:
        # Stages with at least one outstanding async_copy.
        self.issued_copy_stages: set[int] = set()
        # Queues that have been ``queue.create``d in this scope.
        self.open_queues: set[int] = set()
        # Per-queue count of unmatched ``push`` events.  Negative
        # means more pops than pushes (an error).
        self.queue_outstanding: dict[int, int] = {}
        self.diagnostics: list[MemoryModelDiagnostic] = []


def _location(op: TileOp) -> Optional[dict]:
    attrs = op.attrs
    if "loc_line" in attrs or "loc_col" in attrs or "loc" in attrs:
        return {k: attrs[k] for k in ("loc", "loc_line", "loc_col") if k in attrs}
    return None


def _verify_async_copy(op: TileOp, state: _MemoryStateTracker) -> None:
    stage = int(op.attrs.get("stage", 0))
    state.issued_copy_stages.add(stage)
    src = op.attrs.get("source_space")
    dst = op.attrs.get("dest_space")
    if src is None or dst is None:
        return  # caller didn't annotate spaces; nothing to verify
    src = str(src)
    dst = str(dst)
    if src not in VALID_MEMORY_SPACES or dst not in VALID_MEMORY_SPACES:
        state.diagnostics.append(MemoryModelDiagnostic(
            severity="error",
            message=(
                f"async_copy source_space={src!r} or dest_space={dst!r} "
                f"is not in the canonical set {sorted(VALID_MEMORY_SPACES)}"
            ),
            code="MEM_UNKNOWN_MEMORY_SPACE",
            op_name=op.op_name,
            where=_location(op),
        ))
        return
    if (src, dst) not in LEGAL_ASYNC_COPY_TRANSITIONS:
        state.diagnostics.append(MemoryModelDiagnostic(
            severity="error",
            message=(
                f"async_copy {src!r} -> {dst!r} is not a legal transition; "
                f"see LEGAL_ASYNC_COPY_TRANSITIONS"
            ),
            code="MEM_INVALID_ASYNC_COPY_TRANSITION",
            op_name=op.op_name,
            where=_location(op),
        ))


def _verify_wait_async(op: TileOp, state: _MemoryStateTracker) -> None:
    stage = int(op.attrs.get("stage", 0))
    if stage not in state.issued_copy_stages:
        state.diagnostics.append(MemoryModelDiagnostic(
            severity="error",
            message=(
                f"wait_async(stage={stage}) has no preceding async_copy "
                f"at the same stage; observed stages={sorted(state.issued_copy_stages)}"
            ),
            code="MEM_WAIT_WITHOUT_COPY",
            op_name=op.op_name,
            where=_location(op),
        ))
        return
    # wait consumes one outstanding copy for that stage.  We use a
    # set rather than a count so this stays cheap; subsequent waits
    # for the same stage are still allowed (multiple consumers).


def _verify_queue_create(op: TileOp, state: _MemoryStateTracker) -> None:
    queue_id = int(op.attrs.get("queue_id", -1))
    if queue_id < 0:
        return  # structural error handled by TileIRVerifier
    state.open_queues.add(queue_id)
    state.queue_outstanding.setdefault(queue_id, 0)


def _verify_queue_push(op: TileOp, state: _MemoryStateTracker) -> None:
    queue_id = int(op.attrs.get("queue_id", -1))
    if queue_id < 0:
        return
    if queue_id not in state.open_queues:
        state.diagnostics.append(MemoryModelDiagnostic(
            severity="error",
            message=f"queue.push references queue {queue_id} that was not created in scope",
            code="MEM_QUEUE_PUSH_WITHOUT_CREATE",
            op_name=op.op_name,
            where=_location(op),
        ))
        return
    state.queue_outstanding[queue_id] = state.queue_outstanding.get(queue_id, 0) + 1


def _verify_queue_pop_or_barrier(op: TileOp, state: _MemoryStateTracker) -> None:
    queue_id = int(op.attrs.get("queue_id", -1))
    if queue_id < 0:
        return
    if queue_id not in state.open_queues:
        state.diagnostics.append(MemoryModelDiagnostic(
            severity="error",
            message=f"{op.op_name} references queue {queue_id} that was not created in scope",
            code="MEM_QUEUE_OP_WITHOUT_CREATE",
            op_name=op.op_name,
            where=_location(op),
        ))
        return
    if state.queue_outstanding.get(queue_id, 0) <= 0:
        state.diagnostics.append(MemoryModelDiagnostic(
            severity="error",
            message=(
                f"{op.op_name}(queue_id={queue_id}) has no preceding "
                "queue.push in scope (would block indefinitely)"
            ),
            code="MEM_QUEUE_POP_WITHOUT_PUSH",
            op_name=op.op_name,
            where=_location(op),
        ))
        return
    if op.op_name == "tessera.queue.pop":
        state.queue_outstanding[queue_id] -= 1
    # `barrier` waits but does not consume — leave the count alone.


def _walk(ops: Iterable[TileOp], state: _MemoryStateTracker) -> None:
    for op in ops:
        if op.op_name == "tile.async_copy":
            _verify_async_copy(op, state)
        elif op.op_name == "tile.wait_async":
            _verify_wait_async(op, state)
        elif op.op_name == "tessera.queue.create":
            _verify_queue_create(op, state)
        elif op.op_name == "tessera.queue.push":
            _verify_queue_push(op, state)
        elif op.op_name in {"tessera.queue.pop", "tessera.queue.barrier"}:
            _verify_queue_pop_or_barrier(op, state)
        _walk(op.body, state)


def verify_memory_model(
    module_or_function: TileIRModule | TileFunction,
) -> MemoryModelVerificationResult:
    """Run the memory-model verifier over a :class:`TileIRModule` or
    a single :class:`TileFunction`.

    Returns a result whose ``ok`` field is ``False`` when any
    diagnostic has severity ``"error"``.  Use
    :func:`assert_memory_model_ok` for the convenience that raises
    instead of returning.
    """
    diagnostics: list[MemoryModelDiagnostic] = []
    functions: tuple[TileFunction, ...]
    if isinstance(module_or_function, TileFunction):
        functions = (module_or_function,)
    else:
        functions = tuple(module_or_function.functions)
    for fn in functions:
        state = _MemoryStateTracker()
        _walk(fn.body, state)
        diagnostics.extend(state.diagnostics)
    return MemoryModelVerificationResult(tuple(diagnostics))


def assert_memory_model_ok(
    module_or_function: TileIRModule | TileFunction,
) -> None:
    """Verify and raise :class:`MemoryModelVerificationError` on
    any error-severity diagnostic.  Use this at the boundary
    between Tile IR construction and Target IR codegen — invalid
    memory programs should never reach the target lane."""
    result = verify_memory_model(module_or_function)
    if not result.ok:
        raise MemoryModelVerificationError(result.format())


__all__ = [
    "MemoryModelDiagnostic",
    "MemoryModelVerificationResult",
    "MemoryModelVerificationError",
    "VALID_MEMORY_SPACES",
    "LEGAL_ASYNC_COPY_TRANSITIONS",
    "verify_memory_model",
    "assert_memory_model_ok",
]
