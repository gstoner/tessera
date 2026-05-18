"""Tessera CompileReport — M1 deliverable.

One JSON envelope that records, for any frontend-to-codegen path:

  - Source origin (function qualname, file path, or `"inline"`).
  - Frontend lane (`tessera.jit`, `textual`, or `clifford_jit`).
  - **Value kind** — `tensor` / `multivector` / `mixed` (per Decision
    #15a, Multivector is a sibling value kind, not a 7th tensor
    attribute).
  - Target name and the capability decision that picked it.
  - Stable per-layer IR digests (Graph IR, Schedule IR, Tile IR,
    Target IR — populated when the lane emits them).
  - Diagnostics (warnings + errors collected during compile).
  - Fallback reason when native execution didn't fire.
  - **Proof routes** — extended :class:`JitBridgeRoute` rows (M1
    explicitly reuses this rather than introducing a parallel
    `NativeExecutionProof` struct).
  - Timing envelope (separated from the hash so reports are
    bit-deterministic except for the timing block).
  - Correctness envelope (e.g., max-abs-err vs the reference path).

The envelope is JSON-serializable and renders identically across
runs given identical inputs.  The drift-friendly fields (`timing_ms`,
`tessera_version`) are excluded from `report_hash()` so two runs of
the same program emit the same hash.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Optional

from . import jit_bridge as _bridge
from .fallback import FallbackReason


# Canonical strings — guarded by tests so misspellings get caught.
FRONTEND_TESSERA_JIT = "tessera.jit"
FRONTEND_TEXTUAL = "textual"
FRONTEND_CLIFFORD_JIT = "clifford_jit"
VALID_FRONTENDS: frozenset[str] = frozenset({
    FRONTEND_TESSERA_JIT, FRONTEND_TEXTUAL, FRONTEND_CLIFFORD_JIT,
})

VALUE_KIND_TENSOR = "tensor"
VALUE_KIND_MULTIVECTOR = "multivector"
VALUE_KIND_MIXED = "mixed"
VALID_VALUE_KINDS: frozenset[str] = frozenset({
    VALUE_KIND_TENSOR, VALUE_KIND_MULTIVECTOR, VALUE_KIND_MIXED,
})

# The four IR layers a report may carry a digest for.  Reports
# emitted by lanes that don't produce a given layer simply omit it
# (the `ir_hashes` map can be empty for "no IR digest available").
IR_LAYERS: tuple[str, ...] = ("graph_ir", "schedule_ir", "tile_ir", "target_ir")


@dataclass(frozen=True)
class CompileReport:
    """One frontend→codegen path inspectable in one format.

    The dataclass is JSON-serializable via :meth:`as_dict` /
    :meth:`as_json` and stable-hashable via :meth:`report_hash`.
    """

    program_id: str
    source: str
    frontend: str
    value_kind: str
    target: str
    tessera_version: str = ""
    ir_hashes: Mapping[str, str] = field(default_factory=dict)
    target_decision: Mapping[str, str] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()
    fallback_reason: Optional[str | FallbackReason] = None
    proof_routes: tuple[_bridge.JitBridgeRoute, ...] = ()
    timing_ms: Optional[Mapping[str, float]] = None
    correctness: Optional[Mapping[str, float]] = None

    def __post_init__(self) -> None:
        if self.frontend not in VALID_FRONTENDS:
            raise ValueError(
                f"CompileReport: frontend {self.frontend!r} not in "
                f"{sorted(VALID_FRONTENDS)}"
            )
        if self.value_kind not in VALID_VALUE_KINDS:
            raise ValueError(
                f"CompileReport: value_kind {self.value_kind!r} not in "
                f"{sorted(VALID_VALUE_KINDS)}"
            )
        unknown_layers = set(self.ir_hashes) - set(IR_LAYERS)
        if unknown_layers:
            raise ValueError(
                f"CompileReport: ir_hashes has unknown layers "
                f"{sorted(unknown_layers)} (valid: {IR_LAYERS})"
            )

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly nested dict.  Routes flatten to plain dicts so
        the envelope round-trips through :mod:`json` unchanged."""
        return {
            "program_id": self.program_id,
            "source": self.source,
            "frontend": self.frontend,
            "value_kind": self.value_kind,
            "target": self.target,
            "tessera_version": self.tessera_version,
            "ir_hashes": dict(sorted(self.ir_hashes.items())),
            "target_decision": dict(sorted(self.target_decision.items())),
            "diagnostics": list(self.diagnostics),
            "fallback_reason": (
                self.fallback_reason.value
                if isinstance(self.fallback_reason, FallbackReason)
                else self.fallback_reason
            ),
            "proof_routes": [
                {
                    "op_name": r.op_name,
                    "target": r.target,
                    "status": r.status,
                    "symbol": r.symbol,
                    "context": r.context,
                    "latency_ms": r.latency_ms,
                    "args_summary": list(r.args_summary),
                }
                for r in self.proof_routes
            ],
            "timing_ms": dict(self.timing_ms) if self.timing_ms else None,
            "correctness": dict(self.correctness) if self.correctness else None,
        }

    def as_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=False)

    def report_hash(self) -> str:
        """Stable 16-char sha256 prefix over the report's
        non-timing-dependent content.

        Excludes ``timing_ms``, the per-route ``latency_ms``, and
        ``tessera_version`` so two runs of the same program produce
        the same hash.
        """
        canonical: dict[str, Any] = {
            "program_id": self.program_id,
            "source": self.source,
            "frontend": self.frontend,
            "value_kind": self.value_kind,
            "target": self.target,
            "ir_hashes": dict(sorted(self.ir_hashes.items())),
            "target_decision": dict(sorted(self.target_decision.items())),
            "diagnostics": list(self.diagnostics),
            "fallback_reason": (
                self.fallback_reason.value
                if isinstance(self.fallback_reason, FallbackReason)
                else self.fallback_reason
            ),
            "proof_routes": [
                {
                    "op_name": r.op_name,
                    "target": r.target,
                    "status": r.status,
                    "symbol": r.symbol,
                    "context": r.context,
                    "args_summary": list(r.args_summary),
                }
                for r in self.proof_routes
            ],
            "correctness": dict(self.correctness) if self.correctness else None,
        }
        blob = json.dumps(canonical, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience builders
# ─────────────────────────────────────────────────────────────────────────────

def hash_ir_text(text: str) -> str:
    """Stable 16-char digest of any IR text — used for the
    ``ir_hashes`` map.  Whitespace-stripped + sorted to remove
    cosmetic noise."""
    canonical = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def routes_from_thread_trace() -> tuple[_bridge.JitBridgeRoute, ...]:
    """Snapshot the bridge's thread-local trace without clearing it.

    Callers typically toggle tracing on, run the program, then call
    this to attach the captured routes to a CompileReport.
    """
    return _bridge.current_dispatch_trace()


# ─────────────────────────────────────────────────────────────────────────────
# Auto-emission sink — step 4 of the 2026-05-18 post-reassessment plan.
#
# Each frontend's compiled-callable can call :func:`emit_compile_report`
# at the end of its ``__call__``.  The sink is a contextvar so emission
# is thread-/task-safe and is a no-op unless a caller has opened a
# :func:`capture_compile_reports` scope.  This means every frontend
# can opt into uniform emission without any runtime overhead in the
# default case.
# ─────────────────────────────────────────────────────────────────────────────

_SINK_VAR: contextvars.ContextVar[Optional[list["CompileReport"]]] = \
    contextvars.ContextVar("tessera.compile_report_sink", default=None)


@contextmanager
def capture_compile_reports() -> Iterator[list["CompileReport"]]:
    """Open a scope in which every frontend's ``__call__`` appends
    its :class:`CompileReport` to the returned list.

    Usage::

        with capture_compile_reports() as reports:
            f(x, y)              # @tessera.jit fn
            clifford_demo(...)   # @clifford_jit fn
        # reports now contains one CompileReport per call
    """
    sink: list[CompileReport] = []
    token = _SINK_VAR.set(sink)
    try:
        yield sink
    finally:
        _SINK_VAR.reset(token)


def emit_compile_report(report: "CompileReport") -> None:
    """Push a :class:`CompileReport` to the active sink, if any.

    Called by every frontend's ``__call__`` after the wrapped
    function returns; outside of :func:`capture_compile_reports`
    this is a no-op so the hot path stays cheap.
    """
    sink = _SINK_VAR.get()
    if sink is not None:
        sink.append(report)


def active_sink_is_capturing() -> bool:
    """Cheap probe — frontends can use this to skip the cost of
    building a CompileReport when nobody is listening."""
    return _SINK_VAR.get() is not None


__all__ = [
    "FRONTEND_TESSERA_JIT",
    "FRONTEND_TEXTUAL",
    "FRONTEND_CLIFFORD_JIT",
    "VALID_FRONTENDS",
    "VALUE_KIND_TENSOR",
    "VALUE_KIND_MULTIVECTOR",
    "VALUE_KIND_MIXED",
    "VALID_VALUE_KINDS",
    "IR_LAYERS",
    "CompileReport",
    "hash_ir_text",
    "routes_from_thread_trace",
    "capture_compile_reports",
    "emit_compile_report",
    "active_sink_is_capturing",
]
