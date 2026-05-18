"""M2 — shared compile-session object.

The session is the stateful counterpart to :class:`CompileReport`:

- :class:`CompileReport` is a single-shot, immutable snapshot of
  one frontend → codegen path.
- :class:`CompileSession` is the **scope** that wraps zero or more
  reports and adds session-level state — target capability cache,
  artifact-hash index, merged diagnostics, and a derived
  ``value_kind`` reduction that says whether the session ran a
  pure tensor program, a pure multivector program, or a mixed
  program.

M2 acceptance criteria the session enforces:

  1. **Schema parity** — every frontend that emits a CompileReport
     drops it into the session; the session never re-derives
     ``frontend`` / ``value_kind`` / ``target`` fields by reading
     dtypes or other proxies.  ``value_kind`` is a normative
     field, not an attribute.  (Decision #15a, locked.)
  2. **Capability cache** — a single ``(op, target)`` lookup
     suffices for the whole session; later lowering stages don't
     re-query the manifest.
  3. **Artifact-hash index** — reports producing the same IR hash
     are de-duplicated for the audit + memoization stories M5
     and M6 Step 3 both depend on.

The session is intentionally *small*: no compiler logic, no
codegen, no scheduling — it's the join between the three
frontends.  All compiler logic stays in the frontends + IR
modules; the session just makes the join legible.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

from . import backend_manifest as _bm
from .compile_report import CompileReport, capture_compile_reports


# ─────────────────────────────────────────────────────────────────────────────
# Session diagnostic — separate from CompileReport.diagnostics so
# session-level findings (mixed-op boundary failures, cross-frontend
# inconsistencies) are distinguishable from per-call ones.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SessionDiagnostic:
    """One session-level finding.

    ``code`` is a stable string the test surface can grep for
    (``M2_VALUE_KIND_MISMATCH``, ``M2_UNSUPPORTED_MIXED_OP``, …).
    ``source_span`` carries ``(line, col)`` when the offending
    call exposed it, ``None`` otherwise.
    """
    severity: str
    message: str
    code: str
    source_span: Optional[tuple[int, int]] = None

    def format(self) -> str:
        loc = ""
        if self.source_span is not None:
            loc = f"  at line {self.source_span[0]}, col {self.source_span[1]}"
        return f"{self.severity.upper()} [{self.code}]: {self.message}{loc}"


# ─────────────────────────────────────────────────────────────────────────────
# CompileSession
# ─────────────────────────────────────────────────────────────────────────────

# Canonical session-level value-kind values.
SESSION_VALUE_KIND_TENSOR = "tensor"
SESSION_VALUE_KIND_MULTIVECTOR = "multivector"
SESSION_VALUE_KIND_MIXED = "mixed"
SESSION_VALUE_KIND_EMPTY = "empty"


@dataclass
class CompileSession:
    """Stateful aggregate of zero or more :class:`CompileReport`s
    captured within a :func:`compile_session` scope.

    Fields are mutable so frontends + diagnostic emitters can
    append into a live session; downstream consumers should not
    rely on field identity across sessions.
    """
    reports: list[CompileReport] = field(default_factory=list)
    diagnostics: list[SessionDiagnostic] = field(default_factory=list)
    # Capability cache — (op, target) → manifest status string.
    target_decisions: dict[tuple[str, str], str] = field(default_factory=dict)
    # IR-hash → list of program_ids that emitted that hash.
    artifact_index: dict[str, list[str]] = field(default_factory=dict)

    # ── derived views ────────────────────────────────────────────

    def value_kind(self) -> str:
        """Reduce per-report value_kinds to a session-level kind.

        Empty session ⇒ ``"empty"``.  Single distinct kind across
        every report ⇒ that kind.  More than one kind ⇒
        ``"mixed"``.  Mixed is also the result if any individual
        report declared ``value_kind="mixed"`` (e.g., the
        ``rotor_sandwich_ebt_tiny`` composite canonical).
        """
        if not self.reports:
            return SESSION_VALUE_KIND_EMPTY
        kinds = {r.value_kind for r in self.reports}
        if SESSION_VALUE_KIND_MIXED in kinds:
            return SESSION_VALUE_KIND_MIXED
        if len(kinds) == 1:
            return next(iter(kinds))
        return SESSION_VALUE_KIND_MIXED

    def frontends(self) -> set[str]:
        """Set of distinct frontends that wrote into this session."""
        return {r.frontend for r in self.reports}

    def targets(self) -> set[str]:
        """Set of distinct targets seen in this session."""
        return {r.target for r in self.reports}

    def has_mixed_boundary(self) -> bool:
        """``True`` iff this session needs explicit boundary ops to
        cross between value kinds (i.e., it's mixed)."""
        return self.value_kind() == SESSION_VALUE_KIND_MIXED

    def refresh(self) -> None:
        """Recompute the artifact-hash index and the target-decision
        cache from ``self.reports``.  Called automatically when the
        session scope closes; callers may call it eagerly mid-scope."""
        # Artifact index: ir_hash → [program_ids]
        index: dict[str, list[str]] = {}
        for r in self.reports:
            for layer, h in r.ir_hashes.items():
                key = f"{layer}:{h}"
                index.setdefault(key, []).append(r.program_id)
        self.artifact_index = index
        # Target-decision cache.  We don't re-query the manifest —
        # we read from each report's `target_decision` field and
        # union them.  This mirrors what M0's `audit.py` would
        # surface but indexed by (op, target).
        cache: dict[tuple[str, str], str] = self.target_decisions
        for r in self.reports:
            for target, decision in r.target_decision.items():
                # Use program_id as a proxy for `op` — the report's
                # target_decision is per-(target), not per-(op, target),
                # so we key on the program identifier.
                cache.setdefault((r.program_id, target), decision)

    # ── diagnostics surface ──────────────────────────────────────

    def emit_diagnostic(
        self,
        *,
        severity: str = "error",
        message: str,
        code: str,
        source_span: Optional[tuple[int, int]] = None,
    ) -> None:
        """Append a session-level diagnostic.  Use for findings
        the per-call CompileReport can't naturally carry (mixed
        boundary errors, cross-frontend inconsistencies, etc.)."""
        self.diagnostics.append(SessionDiagnostic(
            severity=severity,
            message=message,
            code=code,
            source_span=source_span,
        ))

    @property
    def has_errors(self) -> bool:
        return any(d.severity == "error" for d in self.diagnostics)


# ─────────────────────────────────────────────────────────────────────────────
# Context manager
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def compile_session() -> Iterator[CompileSession]:
    """Open a :class:`CompileSession` scope.

    Inside the scope, every CompileReport emitted by any frontend
    (``@tessera.jit``, textual, ``@clifford_jit``) appends to the
    session's report list.  When the scope closes, the session's
    derived views (artifact index, target-decision cache) are
    refreshed.

    Usage::

        with compile_session() as session:
            run_tensor_program(...)
            run_clifford_program(...)
            run_textual_program(...)

        session.value_kind()      # 'mixed' (or 'tensor' / 'multivector')
        session.frontends()       # {'tessera.jit', 'clifford_jit', ...}
        session.targets()         # {'apple_gpu', 'cpu', ...}
        session.has_mixed_boundary()  # True iff value_kind == 'mixed'
    """
    session = CompileSession()
    with capture_compile_reports() as sink:
        # Share the underlying list so the session sees reports as
        # they're emitted, not just at close time.  The sink IS the
        # session's reports list — a single allocation.
        session.reports = sink
        try:
            yield session
        finally:
            session.refresh()


__all__ = [
    "CompileSession",
    "SessionDiagnostic",
    "compile_session",
    "SESSION_VALUE_KIND_TENSOR",
    "SESSION_VALUE_KIND_MULTIVECTOR",
    "SESSION_VALUE_KIND_MIXED",
    "SESSION_VALUE_KIND_EMPTY",
]
