"""Public diagnostic-code taxonomy for the Tessera compiler.

Tessera has three deliberate frontend lanes (Python ``@tessera.jit``,
the textual DSL, and the constrained math lanes — ``@clifford_jit`` /
``@complex_jit`` / ``@energy_jit``) plus an execution-side fallback
classifier.  This module is the canonical home for **every** stable
diagnostic vocabulary they emit:

1. :class:`JitDiagnosticCode` — Python ``@tessera.jit`` lane.
   Source-availability, eager-fallback, lowering rejections.
2. :class:`FrontendDiagnosticCode` — textual DSL lane.
   Parse errors, semantic errors (unknown op, arity mismatch).
3. :class:`ConstrainedDiagnosticCode` — Clifford / Complex / Energy
   lanes.  Whitelist rejections, holomorphicity failures,
   forbidden-op errors.
4. :class:`FallbackReason` — execution-side, re-exported from
   :mod:`tessera.compiler.fallback`.  Why did this run on numpy
   instead of MSL?

Each code is a string-typed enum so callers can:

  * Compare with ``is`` (``code is JitDiagnosticCode.SOURCE_UNAVAILABLE``)
  * Serialize to JSON without conversion
    (``json.dumps(code.value)``)
  * Match in tests against stable values (``code.value ==
    "JIT_SOURCE_UNAVAILABLE"``)

The companion :class:`Diagnostic` dataclass normalizes every
vocabulary into a single shape consumable by ``JitFn.explain()`` —
including the optional :class:`SourceLocation` (re-exported from
:mod:`tessera.compiler.graph_ir`) so positions plumb cleanly
through every lane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Union

from .fallback import (
    FallbackDecision,
    FallbackReason,
    TesseraNativeRequiredError,
    classify_host,
    message_for as fallback_message_for,
)
from .graph_ir import SourceSpan as SourceLocation


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

    #: Info-level: the function body contains structured control flow
    #: markers (``tessera.scf.if.*`` / ``tessera.scf.for.*`` /
    #: ``tessera.scf.while.*``) that D.1/D.2/D.3 lowered correctly.
    #: No backend currently emits executable code for scf at the
    #: TileIR level, so execution falls back to eager Python — which is
    #: numerically correct, just not optimized. Distinguished from
    #: ``EAGER_FALLBACK_UNSUPPORTED_OP`` so dashboards can show this as
    #: an *expected* eager path rather than a generic "we don't know
    #: this op" miss. Audit follow-up A.2 (2026-05-31).
    EAGER_FALLBACK_CONTROL_FLOW = "JIT_EAGER_FALLBACK_CONTROL_FLOW"

    #: Info-level: the JIT successfully device_verified_jit through Graph IR →
    #: Schedule IR → Tile IR → CPU Target IR and dispatched on the
    #: CPU path.
    COMPILED_CPU = "JIT_COMPILED_CPU"

    #: Info-level: the JIT emitted Target IR for a target whose complete op
    #: program has a runtime dispatch lane.  Individual launch contracts (for
    #: example Apple optimizer f32 contiguous p/g/m/v) can still select an
    #: explicit reference override.
    COMPILED_TARGET_RUNTIME = "JIT_COMPILED_TARGET_RUNTIME"

    #: Info-level: the JIT emitted a Target IR artifact for a non-CPU
    #: target but no native execution is wired today.  Caller sees
    #: ``execution_kind == "artifact_only"``.
    TARGET_IR_ARTIFACT_ONLY = "JIT_TARGET_IR_ARTIFACT_ONLY"

    #: Warning-level: AST Graph IR emission failed for an ``apple_gpu``
    #: function, but decoration did NOT hard-fail — the Phase-F tracer
    #: executes the function by running it (it never reads the AST
    #: graph_ir), so a body the AST can't emit still decorates and runs
    #: via the tracer at call time. Non-apple_gpu targets still raise.
    APPLE_GPU_TRACE_DEFERRED = "JIT_APPLE_GPU_TRACE_DEFERRED"

    #: P3 (2026-06-09) — the apple_gpu one-command-buffer route (auto_batch)
    #: is active (explicit or auto-detected as a recognized decode chain), so
    #: the unused AST Graph IR emission was skipped (the tracer runs the body).
    APPLE_GPU_AUTO_BATCH = "JIT_APPLE_GPU_AUTO_BATCH"


class FrontendDiagnosticCode(str, Enum):
    """Textual-DSL frontend (``tessera.compiler.frontend.parser``).

    Stable codes for the regex-lexer + recursive-descent parser pair
    that consumes the textual IR-like surface used by lit fixtures
    and serialized IR round-trips.
    """

    #: Lexer encountered an unrecognized character.
    LEX_UNEXPECTED_CHAR = "TEXTUAL_LEX_UNEXPECTED_CHAR"

    #: Parser saw an unexpected token (not what the grammar expected).
    PARSE_UNEXPECTED_TOKEN = "TEXTUAL_PARSE_UNEXPECTED_TOKEN"

    #: Parser hit EOF in the middle of a statement / module / function.
    PARSE_UNEXPECTED_EOF = "TEXTUAL_PARSE_UNEXPECTED_EOF"

    #: Parser expected an identifier and got something else.
    PARSE_EXPECTED_IDENTIFIER = "TEXTUAL_PARSE_EXPECTED_IDENTIFIER"

    #: Parser found a module-declaration keyword it doesn't recognize.
    PARSE_UNSUPPORTED_MODULE_DECL = "TEXTUAL_PARSE_UNSUPPORTED_MODULE_DECL"

    #: Semantic pass: op name is not in the canonical catalog.
    SEMANTIC_UNKNOWN_OP = "TEXTUAL_SEMANTIC_UNKNOWN_OP"

    #: Semantic pass: op operand count doesn't match the catalog signature.
    SEMANTIC_ARITY_MISMATCH = "TEXTUAL_SEMANTIC_ARITY_MISMATCH"

    #: Semantic pass: result type couldn't be inferred for an op call.
    SEMANTIC_RESULT_TYPE_UNRESOLVED = "TEXTUAL_SEMANTIC_RESULT_TYPE_UNRESOLVED"


class ConstrainedDiagnosticCode(str, Enum):
    """Constrained math lanes — ``@clifford_jit``, ``@complex_jit``,
    ``@energy_jit``.

    Each lane runs a whitelist-based AST → IR lowering with stricter
    contracts than ``@tessera.jit`` (CR-verification, GA op
    membership, forbidden-op rejection).  These codes appear when
    the lane refuses to compile.
    """

    # ── Clifford / GA lane ────────────────────────────────────────────
    #: An op outside the GA whitelist appears in the function body.
    CLIFFORD_OP_NOT_WHITELISTED = "CLIFFORD_OP_NOT_WHITELISTED"

    #: ``@clifford_jit`` traced an empty op plan — the function body
    #: never called any ``tessera.ga.*`` op.
    CLIFFORD_EMPTY_OP_PLAN = "CLIFFORD_EMPTY_OP_PLAN"

    #: An op routed to a target other than the one ``@clifford_jit``
    #: was decorated with.
    CLIFFORD_TARGET_MISMATCH = "CLIFFORD_TARGET_MISMATCH"

    #: ``@clifford_jit(target=...)`` got an unsupported target name.
    CLIFFORD_UNSUPPORTED_TARGET = "CLIFFORD_UNSUPPORTED_TARGET"

    #: A ``@clifford_jit`` callable was invoked with a Multivector whose
    #: algebra signature has no GPU kernel.  v1 ships only ``Cl(3,0)``
    #: (``cl30``) kernels; other signatures (e.g. spacetime ``Cl(1,3)``)
    #: are front-end-expressible but have no backend, so the lane refuses
    #: rather than silently running the numpy reference.  Use the plain
    #: ``tessera.ga.*`` lane for non-Cl(3,0) algebras.
    CLIFFORD_UNSUPPORTED_SIGNATURE = "CLIFFORD_UNSUPPORTED_SIGNATURE"

    # ── Complex / Visual Complex lane ─────────────────────────────────
    #: ``@complex_jit`` / ``@analytic_symbolic`` rejected the function:
    #: it contains a non-holomorphic op (``complex_conjugate``,
    #: ``complex_abs``, ``complex_arg``, etc.).
    COMPLEX_NON_HOLOMORPHIC = "COMPLEX_NON_HOLOMORPHIC"

    #: Numerical ``@analytic`` decorator probed at random points and
    #: found a Cauchy-Riemann residual exceeding ``atol``.
    COMPLEX_CR_RESIDUAL_TOO_LARGE = "COMPLEX_CR_RESIDUAL_TOO_LARGE"

    # ── Energy lane ───────────────────────────────────────────────────
    #: ``@energy_jit`` rejected a forbidden op (the v1 whitelist is
    #: small — most numpy/aten ops are excluded by design).
    ENERGY_FORBIDDEN_OP = "ENERGY_FORBIDDEN_OP"

    #: ``@energy_jit`` got a dtype it doesn't support (v1 only ships
    #: ``f32``).
    ENERGY_UNSUPPORTED_DTYPE = "ENERGY_UNSUPPORTED_DTYPE"


# Union type for "any stable diagnostic code".  ``str`` is allowed
# for backwards compatibility — the existing ``JitDiagnostic.code``
# field is typed ``str`` and we don't want to break that contract.
DiagnosticCode = Union[
    JitDiagnosticCode,
    FrontendDiagnosticCode,
    ConstrainedDiagnosticCode,
    FallbackReason,
    str,
]


@dataclass(frozen=True)
class Diagnostic:
    """Normalized diagnostic for ``JitFn.explain().diagnostics``.

    Bridges every diagnostic vocabulary (JIT-side, textual-DSL-side,
    constrained-math-side, fallback-side) into one shape the explain
    front door can render.
    """

    severity: str
    """``info`` / ``warning`` / ``error``."""

    code: DiagnosticCode
    """Stable code — :class:`JitDiagnosticCode`,
    :class:`FrontendDiagnosticCode`, :class:`ConstrainedDiagnosticCode`,
    :class:`FallbackReason`, or a free-form string for callers that
    haven't migrated yet."""

    message: str
    """Human-readable message."""

    detail: Mapping[str, Any] = field(default_factory=dict)
    """Optional structured detail (target name, op name, shape, etc.)."""

    source_position: Optional[SourceLocation] = None
    """Optional source location (file/line/col).  Populated by every
    emission site that has access to AST or lexer position info —
    CPython AST nodes carry ``lineno``/``col_offset``; the textual
    lexer tracks position per token; the constrained math lanes use
    the same CPython AST shape as ``@tessera.jit``."""

    lane: Optional[str] = None
    """Optional frontend-lane provenance — one of ``tessera_jit`` /
    ``textual_dsl`` / ``clifford_jit`` / ``complex_jit`` /
    ``energy_jit``.  Lets ``.explain()`` distinguish "the Python
    lane rejected this" from "the GA lane rejected this" without
    consumers having to inspect the code prefix."""

    @property
    def code_value(self) -> str:
        """Code as a string regardless of whether it's an enum
        member or a raw string."""

        code = self.code
        if isinstance(code, Enum):
            return code.value
        return str(code)

    def format_position(self) -> str:
        """Human-readable position string (``"line N col M"``), or
        empty when no position is attached."""

        if self.source_position is None:
            return ""
        return self.source_position.format()

    @classmethod
    def from_fallback(
        cls,
        reason: FallbackReason,
        *,
        severity: str = "warning",
        detail: Mapping[str, Any] | None = None,
        source_position: Optional[SourceLocation] = None,
        lane: Optional[str] = None,
    ) -> "Diagnostic":
        """Lift a :class:`FallbackReason` into a :class:`Diagnostic`."""

        return cls(
            severity=severity,
            code=reason,
            message=reason.message(),
            detail=dict(detail or {}),
            source_position=source_position,
            lane=lane,
        )

    @classmethod
    def from_jit(
        cls,
        code: JitDiagnosticCode | str,
        message: str,
        *,
        severity: str = "warning",
        detail: Mapping[str, Any] | None = None,
        source_position: Optional[SourceLocation] = None,
    ) -> "Diagnostic":
        """Lift a Python ``@tessera.jit`` lane code into a Diagnostic."""

        return cls(
            severity=severity,
            code=code,
            message=message,
            detail=dict(detail or {}),
            source_position=source_position,
            lane="tessera_jit",
        )

    @classmethod
    def from_frontend(
        cls,
        code: FrontendDiagnosticCode | str,
        message: str,
        *,
        severity: str = "error",
        detail: Mapping[str, Any] | None = None,
        source_position: Optional[SourceLocation] = None,
    ) -> "Diagnostic":
        """Lift a textual-DSL frontend code into a Diagnostic.

        Default severity is ``error`` because the textual DSL refuses
        to emit IR on a parse/semantic failure — there's no
        eager-fallback equivalent.
        """

        return cls(
            severity=severity,
            code=code,
            message=message,
            detail=dict(detail or {}),
            source_position=source_position,
            lane="textual_dsl",
        )

    @classmethod
    def from_constrained(
        cls,
        code: ConstrainedDiagnosticCode | str,
        message: str,
        *,
        lane: str,
        severity: str = "error",
        detail: Mapping[str, Any] | None = None,
        source_position: Optional[SourceLocation] = None,
    ) -> "Diagnostic":
        """Lift a constrained-math-lane code into a Diagnostic.

        ``lane`` is required and must be one of ``clifford_jit`` /
        ``complex_jit`` / ``energy_jit`` so the explain front door
        knows which whitelist caught the rejection.
        """

        return cls(
            severity=severity,
            code=code,
            message=message,
            detail=dict(detail or {}),
            source_position=source_position,
            lane=lane,
        )


__all__ = [
    "ConstrainedDiagnosticCode",
    "Diagnostic",
    "DiagnosticCode",
    "FallbackDecision",
    "FallbackReason",
    "FrontendDiagnosticCode",
    "JitDiagnosticCode",
    "SourceLocation",
    "TesseraNativeRequiredError",
    "classify_host",
    "fallback_message_for",
]
