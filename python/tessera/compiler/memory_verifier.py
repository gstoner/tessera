"""Memory-model verifier — M4 deliverable.

Tessera's memory model claims live in
``docs/spec/MEMORY_MODEL_SPEC.md`` and across the Schedule/Tile IR
verifiers.  The existing :class:`tessera.compiler.tile_ir.TileIRVerifier`
checks structural validity (``async_copy.stage >= 0``,
``queue.create.depth >= 1``, duplicate queues, etc.) but **not**
ordering claims like "every ``wait_async`` consumes a token produced by a
preceding ``async_copy``".

M4 adds those ordering checks as a second pass that walks a
:class:`TileIRModule` and emits diagnostics for:

  - ``wait_async`` with no preceding token-producing ``async_copy``.
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

# Sprint M5 (2026-05-22) — atomic op attribute validation.
#
# Memory model §5 lists 5 valid memory orders and 6 valid scopes.
# The verifier rejects atomics that name an order/scope outside these
# sets and emits ``MEM_ATOMIC_INVALID_ORDER`` / ``MEM_ATOMIC_INVALID_SCOPE``.
# This is a *structural* attribute-validity check — it does not
# attempt to verify happens-before for atomic chains.
VALID_ATOMIC_ORDERS: frozenset[str] = frozenset({
    "relaxed", "acquire", "release", "acq_rel", "seq_cst",
})

VALID_SYNC_SCOPES: frozenset[str] = frozenset({
    "thread", "warp", "block", "cluster", "device", "mesh",
})

VALID_ATOMIC_OPS: frozenset[str] = frozenset({
    "add", "sub", "min", "max", "and", "or", "xor", "exchange", "cas",
})

# Sprint M5 (2026-05-22) — deterministic-profile reduction rule.
#
# Memory model §7 says deterministic / strict profiles must not use
# nondeterministic float atomic reductions for aggregation.  The
# verifier flags any atomic op with a float dtype + reduction-style
# op (add / sub / min / max) when the enclosing function carries
# ``deterministic=True``.  Integer atomics are deterministic by
# definition (atomicity is sufficient).
NONDETERMINISTIC_FLOAT_DTYPES: frozenset[str] = frozenset({
    "fp64", "fp32", "fp16", "bf16",
    "fp8_e4m3", "fp8_e5m2", "fp6_e2m3", "fp6_e3m2", "fp4_e2m1", "nvfp4",
})

REDUCTION_ATOMIC_OPS: frozenset[str] = frozenset({
    "add", "sub", "min", "max",
})


# ─────────────────────────────────────────────────────────────────────────────
# Verifier
# ─────────────────────────────────────────────────────────────────────────────

class _MemoryStateTracker:
    """Per-function happens-before state."""

    __slots__ = (
        "issued_tokens", "waited_tokens", "diagnostics",
    )

    def __init__(self) -> None:
        self.issued_tokens: set[str] = set()
        self.waited_tokens: set[str] = set()
        self.diagnostics: list[MemoryModelDiagnostic] = []


def _location(op: TileOp) -> Optional[dict]:
    attrs = op.attrs
    if "loc_line" in attrs or "loc_col" in attrs or "loc" in attrs:
        return {k: attrs[k] for k in ("loc", "loc_line", "loc_col") if k in attrs}
    return None


def _verify_async_copy(op: TileOp, state: _MemoryStateTracker) -> None:
    if op.result:
        state.issued_tokens.add(op.result)
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
    if len(op.operands) != 1:
        state.diagnostics.append(MemoryModelDiagnostic(
            severity="error",
            message="wait_async requires exactly one token operand",
            code="MEM_WAIT_WITHOUT_COPY",
            op_name=op.op_name,
            where=_location(op),
        ))
        return
    token = op.operands[0].removeprefix("%")
    if token not in state.issued_tokens:
        state.diagnostics.append(MemoryModelDiagnostic(
            severity="error",
            message=f"wait_async token %{token} has no preceding async_copy",
            code="MEM_WAIT_WITHOUT_COPY",
            op_name=op.op_name,
            where=_location(op),
        ))
        return
    if token in state.waited_tokens:
        state.diagnostics.append(MemoryModelDiagnostic(
            severity="error",
            message=f"async token %{token} is awaited more than once",
            code="MEM_DUPLICATE_WAIT",
            op_name=op.op_name,
            where=_location(op),
        ))
        return
    state.waited_tokens.add(token)


def _verify_atomic(
    op: TileOp, state: _MemoryStateTracker, *, deterministic: bool,
) -> None:
    """Sprint M5 (2026-05-22) — validate atomic op order/scope/op
    attributes and flag nondeterministic float-atomic reductions under
    deterministic profiles.  Memory model §5 + §7."""
    op_kind = op.attrs.get("atomic_op")
    if op_kind is not None and str(op_kind) not in VALID_ATOMIC_OPS:
        state.diagnostics.append(MemoryModelDiagnostic(
            severity="error",
            message=(
                f"atomic op {op_kind!r} is not in the canonical set "
                f"{sorted(VALID_ATOMIC_OPS)}"
            ),
            code="MEM_ATOMIC_INVALID_OP",
            op_name=op.op_name,
            where=_location(op),
        ))
    order = op.attrs.get("order")
    if order is not None and str(order) not in VALID_ATOMIC_ORDERS:
        state.diagnostics.append(MemoryModelDiagnostic(
            severity="error",
            message=(
                f"atomic order {order!r} is not in the canonical set "
                f"{sorted(VALID_ATOMIC_ORDERS)}; see MEMORY_MODEL_SPEC §5"
            ),
            code="MEM_ATOMIC_INVALID_ORDER",
            op_name=op.op_name,
            where=_location(op),
        ))
    scope = op.attrs.get("scope")
    if scope is not None and str(scope) not in VALID_SYNC_SCOPES:
        state.diagnostics.append(MemoryModelDiagnostic(
            severity="error",
            message=(
                f"atomic scope {scope!r} is not in the canonical set "
                f"{sorted(VALID_SYNC_SCOPES)}; see MEMORY_MODEL_SPEC §3"
            ),
            code="MEM_ATOMIC_INVALID_SCOPE",
            op_name=op.op_name,
            where=_location(op),
        ))
    if deterministic and op_kind is not None:
        dtype = op.attrs.get("dtype")
        if (
            dtype is not None
            and str(dtype) in NONDETERMINISTIC_FLOAT_DTYPES
            and str(op_kind) in REDUCTION_ATOMIC_OPS
        ):
            state.diagnostics.append(MemoryModelDiagnostic(
                severity="error",
                message=(
                    f"float atomic {op_kind!r} on dtype {dtype!r} is "
                    "nondeterministic and rejected under "
                    "deterministic / strict numeric profiles; use a "
                    "fixed reduction tree (see MEMORY_MODEL_SPEC §7)"
                ),
                code="MEM_DETERMINISTIC_NONDETERMINISTIC_REDUCTION",
                op_name=op.op_name,
                where=_location(op),
            ))


def _verify_fence(op: TileOp, state: _MemoryStateTracker) -> None:
    """Sprint M5 (2026-05-22) — validate fence scope attribute.
    Memory model §3 + §4."""
    scope = op.attrs.get("scope")
    if scope is None:
        return  # caller didn't annotate; nothing to verify
    if str(scope) not in VALID_SYNC_SCOPES:
        state.diagnostics.append(MemoryModelDiagnostic(
            severity="error",
            message=(
                f"fence scope {scope!r} is not in the canonical set "
                f"{sorted(VALID_SYNC_SCOPES)}; see MEMORY_MODEL_SPEC §3"
            ),
            code="MEM_FENCE_INVALID_SCOPE",
            op_name=op.op_name,
            where=_location(op),
        ))


def _function_is_deterministic(fn: TileFunction) -> bool:
    """A TileFunction carries `deterministic=True` either as a top
    attribute or via its target's numeric profile.  We accept both
    surfaces so user code can opt in at either layer.
    """
    attrs = getattr(fn, "attrs", None) or {}
    if attrs.get("deterministic") is True:
        return True
    if attrs.get("numeric_profile") in ("deterministic", "strict"):
        return True
    return False


def _walk(
    ops: Iterable[TileOp],
    state: _MemoryStateTracker,
    *,
    deterministic: bool = False,
) -> None:
    for op in ops:
        if op.op_name == "tile.async_copy":
            _verify_async_copy(op, state)
        elif op.op_name == "tile.wait_async":
            _verify_wait_async(op, state)
        elif op.op_name in {"tile.atomic", "tessera.atomic", "atomic"}:
            _verify_atomic(op, state, deterministic=deterministic)
        elif op.op_name in {"tile.fence", "tessera.fence", "fence.device"}:
            _verify_fence(op, state)
        _walk(op.body, state, deterministic=deterministic)


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
        _walk(fn.body, state, deterministic=_function_is_deterministic(fn))
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
    "VALID_ATOMIC_ORDERS",
    "VALID_SYNC_SCOPES",
    "VALID_ATOMIC_OPS",
    "NONDETERMINISTIC_FLOAT_DTYPES",
    "REDUCTION_ATOMIC_OPS",
    "verify_memory_model",
    "assert_memory_model_ok",
]
