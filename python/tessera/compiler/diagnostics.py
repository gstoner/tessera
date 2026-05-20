"""Public diagnostic-code taxonomy for the Tessera compiler.

Tessera emits diagnostics from two structurally different places:

1. **Frontend / lowering** (:class:`JitDiagnosticCode`).  Stable
   codes for AST-side issues — eager fallback paths, source
   availability, lowering rejections.  Emitted by the JIT pipeline
   in :mod:`tessera.compiler.matmul_pipeline` /
   :mod:`tessera.compiler.jit`.

2. **Execution-side fallback** (:class:`FallbackReason`, re-exported
   from :mod:`tessera.compiler.fallback`).  Stable codes for "why
   did this run on numpy instead of MSL?".  Emitted by the runtime
   dispatcher.

This module is the canonical home for both vocabularies.  Each code
is a string-typed enum so callers can:

  * Compare with ``is`` (``code is JitDiagnosticCode.SOURCE_UNAVAILABLE``)
  * Serialize to JSON without conversion
    (``json.dumps(code.value)``)
  * Match in tests against stable values (``code.value ==
    "JIT_SOURCE_UNAVAILABLE"``)

The companion :class:`Diagnostic` dataclass normalizes the two
vocabularies into a single shape consumable by ``JitFn.explain()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Union

from .fallback import (
    FallbackDecision,
    FallbackReason,
    TesseraNativeRequiredError,
    classify_host,
    message_for as fallback_message_for,
)


class JitDiagnosticCode(str, Enum):
    """Frontend / lowering diagnostic codes.

    Stability contract: the ``.value`` strings are public API.  CI
    can match against these in tests and they appear verbatim in
    ``CompileReport`` JSON exports.  Adding a new code is a minor
    change; renaming or removing one is a breaking change.
    """

    #: Source could not be read for AST inspection (REPL / heredoc /
    #: dynamically-generated lambdas).  Native @jit dispatch is
    #: refused; caller can pass ``source=...`` or
    #: ``source_path=...`` to override.
    SOURCE_UNAVAILABLE = "JIT_SOURCE_UNAVAILABLE"

    #: Caller passed an explicit ``source=`` string for inspection.
    #: Emitted as info-level so the source-of-truth is auditable.
    SOURCE_PROVIDED = "JIT_SOURCE_PROVIDED"

    #: The lowered Graph IR module is empty (no functions emitted).
    #: Eager Python fallback runs.
    EAGER_FALLBACK_EMPTY = "JIT_EAGER_FALLBACK_EMPTY"

    #: Source contains an op that isn't in the canonical catalog;
    #: eager fallback runs.
    EAGER_FALLBACK_UNSUPPORTED_OP = "JIT_EAGER_FALLBACK_UNSUPPORTED_OP"

    #: Op arity at call site doesn't match the catalog signature.
    EAGER_FALLBACK_ARITY = "JIT_EAGER_FALLBACK_ARITY"

    #: Function body contains a construct (e.g., nested def,
    #: comprehension) that the AST → Graph IR lowering doesn't
    #: handle yet.
    EAGER_FALLBACK_UNSUPPORTED_BODY = "JIT_EAGER_FALLBACK_UNSUPPORTED_BODY"

    #: Info-level: the JIT successfully compiled through Graph IR →
    #: Schedule IR → Tile IR → CPU Target IR and dispatched on the
    #: CPU path.
    COMPILED_CPU = "JIT_COMPILED_CPU"

    #: Info-level: the JIT emitted a Target IR artifact for a non-CPU
    #: target but no native execution is wired today.  Caller sees
    #: ``execution_kind == "artifact_only"``.
    TARGET_IR_ARTIFACT_ONLY = "JIT_TARGET_IR_ARTIFACT_ONLY"


# Union type for "any stable diagnostic code".  ``str`` is allowed
# for backwards compatibility — the existing ``JitDiagnostic.code``
# field is typed ``str`` and we don't want to break that contract.
DiagnosticCode = Union[JitDiagnosticCode, FallbackReason, str]


@dataclass(frozen=True)
class Diagnostic:
    """Normalized diagnostic for ``JitFn.explain().diagnostics``.

    Bridges the two vocabularies (JIT-side + fallback-side) so the
    explain front door has a single shape to render.
    """

    severity: str
    """``info`` / ``warning`` / ``error``."""

    code: DiagnosticCode
    """Stable code — :class:`JitDiagnosticCode`,
    :class:`FallbackReason`, or a free-form string for callers that
    haven't migrated yet."""

    message: str
    """Human-readable message."""

    detail: Mapping[str, Any] = field(default_factory=dict)
    """Optional structured detail (target name, op name, shape, etc.)."""

    @property
    def code_value(self) -> str:
        """Code as a string regardless of whether it's an enum
        member or a raw string."""

        code = self.code
        if isinstance(code, Enum):
            return code.value
        return str(code)

    @classmethod
    def from_fallback(
        cls,
        reason: FallbackReason,
        *,
        severity: str = "warning",
        detail: Mapping[str, Any] | None = None,
    ) -> "Diagnostic":
        """Lift a :class:`FallbackReason` into a :class:`Diagnostic`."""

        return cls(
            severity=severity,
            code=reason,
            message=reason.message(),
            detail=dict(detail or {}),
        )

    @classmethod
    def from_jit(
        cls,
        code: JitDiagnosticCode | str,
        message: str,
        *,
        severity: str = "warning",
        detail: Mapping[str, Any] | None = None,
    ) -> "Diagnostic":
        """Lift a JIT-side code into a :class:`Diagnostic`."""

        return cls(
            severity=severity,
            code=code,
            message=message,
            detail=dict(detail or {}),
        )


__all__ = [
    "Diagnostic",
    "DiagnosticCode",
    "FallbackDecision",
    "FallbackReason",
    "JitDiagnosticCode",
    "TesseraNativeRequiredError",
    "classify_host",
    "fallback_message_for",
]
