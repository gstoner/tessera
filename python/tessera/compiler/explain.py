"""Opinionated single-call diagnostic — ``JitFn.explain()``.

P0-3 of the 2026-05-19 compiler-surface consolidation: developers
shouldn't have to learn six inspection methods (`ir_text`,
`compile_report`, `lowering_artifacts`, `runtime_artifact`,
`explain_lowering`, `graph_ir.to_mlir`) to answer four basic
questions about a JIT'd function:

  1. What ran?  (execution_kind: ``native_cpu`` / ``reference_cpu`` /
     ``native_gpu`` / ``artifact_only`` / ``fallback_eager``)
  2. Was it native / reference / artifact / fallback?
  3. Why?  (fallback reason, lowering diagnostics, target decision)
  4. What should I do next?  (hints with stable IDs)

The :func:`explain` function returns an :class:`Explain` object whose
``__str__`` answers the four questions in a 5-line summary suitable
for ``print(fn.explain())``.  Structured fields hang off the same
object: ``.ir``, ``.kernels``, ``.diagnostics``, ``.next_actions``.

The legacy methods stay as data sources; they're not deprecated —
they're now subordinate to the front door.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional

from .diagnostics import (
    ConstrainedDiagnosticCode,
    Diagnostic,
    FallbackReason,
    FrontendDiagnosticCode,
    JitDiagnosticCode,
)

if TYPE_CHECKING:  # pragma: no cover — imported only for typing
    from .jit import JitFn


# Stable next-action IDs.  Surfaced via ``Explain.next_actions`` so
# CI and docs can match against the code, not the prose.
NEXT_INSTALL_APPLE_GPU = "INSTALL_APPLE_GPU_RUNTIME"
NEXT_PROVIDE_SOURCE = "PROVIDE_SOURCE_FOR_NOTEBOOK"
NEXT_NATIVE_REQUIRED = "USE_NATIVE_REQUIRED_TO_DIAGNOSE"
NEXT_INSPECT_IR = "INSPECT_IR_LAYERS"
NEXT_CHECK_SUPPORT = "CHECK_TS_COMPILER_SUPPORT"
NEXT_NO_ACTION = "NO_ACTION_REQUIRED"

# Lane-specific actionable hints — U3 (2026-05-19).  Each maps a
# diagnostic code to the canonical "what should I do?" answer.
NEXT_REWRITE_TO_GA_OPS = "REWRITE_USING_GA_OPS"
NEXT_REWRITE_HOLOMORPHIC = "REWRITE_USING_HOLOMORPHIC_OPS"
NEXT_USE_ENERGY_WHITELIST = "USE_ENERGY_WHITELIST_OPS"
NEXT_SWITCH_LANE_TO_TESSERA_JIT = "SWITCH_LANE_TO_TESSERA_JIT"
NEXT_USE_FP32_FOR_ENERGY = "USE_FP32_FOR_ENERGY_JIT"
NEXT_USE_APPLE_GPU_TARGET = "USE_APPLE_GPU_TARGET_FOR_CLIFFORD"
NEXT_FIX_TEXTUAL_SYNTAX = "FIX_TEXTUAL_SYNTAX"

# Diagnostic-code → NextAction message lookup.  Stable codes flow
# in; humanized prescriptions flow out.  Keep this table flat — it
# should be readable as "if you see X, do Y" without traversal.
#
# The keys are the ``.value`` of the lane-specific enum members so
# this table doesn't have to import every enum (avoids circular
# imports during type-checking).
_DIAGNOSTIC_CODE_NEXT_ACTIONS: dict[str, tuple[str, str]] = {
    # ── Constrained math lanes ───────────────────────────────────────
    ConstrainedDiagnosticCode.CLIFFORD_OP_NOT_WHITELISTED.value: (
        NEXT_REWRITE_TO_GA_OPS,
        "Rewrite using tessera.ga ops "
        "(geometric_product / wedge / inner / rotor_sandwich), "
        "or switch to @tessera.jit for general tensor work.",
    ),
    ConstrainedDiagnosticCode.CLIFFORD_EMPTY_OP_PLAN.value: (
        NEXT_REWRITE_TO_GA_OPS,
        "Function body did not call any tessera.ga.* op. "
        "Add at least one fused-on-target call, or switch to "
        "@tessera.jit if GA isn't actually needed.",
    ),
    ConstrainedDiagnosticCode.CLIFFORD_TARGET_MISMATCH.value: (
        NEXT_USE_APPLE_GPU_TARGET,
        "@clifford_jit currently only fuses on apple_gpu. "
        "Either use target='apple_gpu' (default) or switch to "
        "@tessera.jit for cross-target support.",
    ),
    ConstrainedDiagnosticCode.CLIFFORD_UNSUPPORTED_TARGET.value: (
        NEXT_USE_APPLE_GPU_TARGET,
        "@clifford_jit only supports target='apple_gpu' in v1. "
        "Use @tessera.jit for cross-target compilation.",
    ),
    ConstrainedDiagnosticCode.COMPLEX_NON_HOLOMORPHIC.value: (
        NEXT_REWRITE_HOLOMORPHIC,
        "Function contains a non-holomorphic op "
        "(complex_conjugate / complex_abs / complex_arg / dbar). "
        "Use the holomorphic subset (complex_mul / complex_exp / "
        "complex_log / mobius / stereographic), or switch to "
        "@tessera.jit if anti-holomorphic ops are intentional.",
    ),
    ConstrainedDiagnosticCode.COMPLEX_CR_RESIDUAL_TOO_LARGE.value: (
        NEXT_REWRITE_HOLOMORPHIC,
        "Cauchy-Riemann residual exceeded tolerance. "
        "Check that every branch of the function is holomorphic, "
        "or relax the @analytic atol parameter if the residual is "
        "numerically explainable.",
    ),
    ConstrainedDiagnosticCode.ENERGY_FORBIDDEN_OP.value: (
        NEXT_USE_ENERGY_WHITELIST,
        "Function uses an op outside the @energy_jit v1 whitelist. "
        "Use only quadratic / softmax_energy / partition / "
        "log_partition ops, or switch to @tessera.jit for the "
        "general surface.",
    ),
    ConstrainedDiagnosticCode.ENERGY_UNSUPPORTED_DTYPE.value: (
        NEXT_USE_FP32_FOR_ENERGY,
        "@energy_jit v1 only ships fp32. Use dtype='f32' on the "
        "decorator, or wait for v2 which will add fp16/bf16.",
    ),
    # ── Textual DSL ───────────────────────────────────────────────────
    FrontendDiagnosticCode.PARSE_UNEXPECTED_TOKEN.value: (
        NEXT_FIX_TEXTUAL_SYNTAX,
        "Textual DSL parser saw an unexpected token. "
        "Check the BNF in docs/spec/LANGUAGE_AND_IR_SPEC.md for "
        "the expected grammar at this position.",
    ),
    FrontendDiagnosticCode.PARSE_UNEXPECTED_EOF.value: (
        NEXT_FIX_TEXTUAL_SYNTAX,
        "Textual DSL parser hit EOF mid-statement. "
        "Likely a missing closing brace, paren, or semicolon.",
    ),
    FrontendDiagnosticCode.SEMANTIC_UNKNOWN_OP.value: (
        NEXT_CHECK_SUPPORT,
        "Op name not found in the canonical catalog. "
        "Check tessera.compiler.support(op) for the per-op "
        "readiness picture, and confirm the op name matches the "
        "support_table.md vocabulary.",
    ),
    # ── JIT lane ──────────────────────────────────────────────────────
    JitDiagnosticCode.SOURCE_UNAVAILABLE.value: (
        NEXT_PROVIDE_SOURCE,
        "AST inspection failed — likely a REPL / notebook / heredoc. "
        "Use tessera.from_text(source) for notebook-safe "
        "construction, or pass source=... to @tessera.jit.",
    ),
}


def next_action_for_diagnostic(
    diagnostic: Diagnostic,
) -> Optional["NextAction"]:
    """Look up the canonical NextAction for ``diagnostic.code_value``.

    Returns ``None`` when the code has no registered next-action
    (e.g., info-level codes like ``JIT_COMPILED_CPU`` that don't
    need a fix).  Lane-specific entries (Clifford / Complex /
    Energy / textual DSL) are populated in
    :data:`_DIAGNOSTIC_CODE_NEXT_ACTIONS`.
    """

    pair = _DIAGNOSTIC_CODE_NEXT_ACTIONS.get(diagnostic.code_value)
    if pair is None:
        return None
    return NextAction(code=pair[0], message=pair[1])


@dataclass(frozen=True)
class NextAction:
    """One actionable hint with a stable code."""

    code: str
    message: str


@dataclass(frozen=True)
class IRLayers:
    """The four IR layers as a single namespace.

    Each field is the layer's textual rendering, or ``None`` when
    that layer wasn't emitted (e.g., fallback_eager skips
    schedule/tile/target).
    """

    graph: Optional[str] = None
    schedule: Optional[str] = None
    tile: Optional[str] = None
    target: Optional[str] = None

    def all_present(self) -> bool:
        return all(
            x is not None for x in (self.graph, self.schedule, self.tile, self.target)
        )

    def layers(self) -> tuple[tuple[str, Optional[str]], ...]:
        return (
            ("graph", self.graph),
            ("schedule", self.schedule),
            ("tile", self.tile),
            ("target", self.target),
        )


@dataclass(frozen=True)
class Kernel:
    """Per-op resolution: what the JIT decided to dispatch."""

    op_name: str
    runtime_status: str
    """E.g., ``ready`` / ``reference`` / ``unknown``."""
    source: str
    """Where this resolution came from (op_catalog / manifest / etc.)."""


@dataclass(frozen=True)
class Explain:
    """The single inspection front door for a JIT'd function.

    Construct via :func:`explain` (which is wired to
    :meth:`tessera.compiler.jit.JitFn.explain`).  Pretty-printed
    summary is opinionated and short; structured fields are below.
    """

    function_name: str
    target: str
    execution_kind: str
    ir: IRLayers
    kernels: tuple[Kernel, ...]
    diagnostics: tuple[Diagnostic, ...]
    next_actions: tuple[NextAction, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    lane: str = "tessera_jit"
    """The frontend lane that lowered this function — one of
    ``tessera_jit`` / ``textual_dsl`` / ``clifford_jit`` /
    ``complex_jit`` / ``energy_jit``.  Surfaced in the first line of
    the summary so developers see which verification rules applied."""

    @property
    def fallback_reason(self) -> Optional[FallbackReason]:
        for d in self.diagnostics:
            if isinstance(d.code, FallbackReason):
                return d.code
        return None

    @property
    def is_native(self) -> bool:
        return self.execution_kind in ("native_cpu", "native_gpu")

    @property
    def is_reference(self) -> bool:
        return self.execution_kind == "reference_cpu"

    @property
    def is_artifact_only(self) -> bool:
        return self.execution_kind == "artifact_only"

    @property
    def is_fallback(self) -> bool:
        return self.execution_kind == "fallback_eager"

    def summary(self) -> str:
        """Return the 5-line opinionated summary."""

        lines: list[str] = []
        # Line 1: what ran.
        kind_label = {
            "native_cpu": "✓ native CPU dispatch",
            "native_gpu": "✓ native GPU dispatch",
            "reference_cpu": "○ reference CPU (numpy backend)",
            "artifact_only": "△ artifact-only (IR emitted; not executed)",
            "fallback_eager": "× eager Python fallback (no compilation)",
        }.get(self.execution_kind, f"? {self.execution_kind}")
        lane_label = f"[{self.lane}]" if self.lane and self.lane != "tessera_jit" else ""
        lines.append(
            f"tessera.jit{lane_label}[{self.function_name!r} → {self.target}]: {kind_label}"
        )
        # Line 2: IR coverage.
        present = [name for name, value in self.ir.layers() if value is not None]
        if present:
            lines.append(f"  IR layers: {', '.join(present)}")
        else:
            lines.append("  IR layers: (none — eager fallback)")
        # Line 3: kernel count.
        lines.append(
            f"  Kernels resolved: {len(self.kernels)} "
            f"({sum(1 for k in self.kernels if k.runtime_status == 'ready')} ready, "
            f"{sum(1 for k in self.kernels if k.runtime_status == 'reference')} reference)"
        )
        # Line 4: diagnostics.
        if not self.diagnostics:
            lines.append("  Diagnostics: clean")
        else:
            counts: dict[str, int] = {}
            for d in self.diagnostics:
                counts[d.severity] = counts.get(d.severity, 0) + 1
            parts = [f"{n} {sev}" for sev, n in sorted(counts.items())]
            top = self.diagnostics[0]
            # Prefix the diagnostic detail with source position when
            # available so `print(fn.explain())` shows users exactly
            # where the issue sits.
            pos = top.format_position()
            pos_prefix = f"@{pos} " if pos else ""
            lines.append(
                f"  Diagnostics: {', '.join(parts)} "
                f"(first: {pos_prefix}{top.code_value} — {top.message[:60]}"
                + ("..." if len(top.message) > 60 else "")
                + ")"
            )
        # Line 5: next action.
        if self.next_actions:
            top_action = self.next_actions[0]
            lines.append(f"  Next: [{top_action.code}] {top_action.message}")
        else:
            lines.append(f"  Next: [{NEXT_NO_ACTION}] —")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON ingestion."""

        return {
            "function_name": self.function_name,
            "target": self.target,
            "execution_kind": self.execution_kind,
            "lane": self.lane,
            "is_native": self.is_native,
            "is_reference": self.is_reference,
            "is_artifact_only": self.is_artifact_only,
            "is_fallback": self.is_fallback,
            "ir": {
                "graph_present": self.ir.graph is not None,
                "schedule_present": self.ir.schedule is not None,
                "tile_present": self.ir.tile is not None,
                "target_present": self.ir.target is not None,
            },
            "kernels": [
                {
                    "op_name": k.op_name,
                    "runtime_status": k.runtime_status,
                    "source": k.source,
                }
                for k in self.kernels
            ],
            "diagnostics": [
                {
                    "severity": d.severity,
                    "code": d.code_value,
                    "message": d.message,
                    "detail": dict(d.detail),
                    "source_position": (
                        d.source_position.format()
                        if d.source_position is not None else None
                    ),
                    "lane": d.lane,
                }
                for d in self.diagnostics
            ],
            "next_actions": [
                {"code": n.code, "message": n.message}
                for n in self.next_actions
            ],
            "metadata": dict(self.metadata),
        }


def _build_kernels(fn: "JitFn") -> tuple[Kernel, ...]:
    """Build per-op resolution from the JIT's Graph IR body."""

    out: list[Kernel] = []
    if not fn.graph_ir.functions:
        return ()
    body = fn.graph_ir.functions[0].body
    for op in body:
        # Strip ``tessera.`` prefix when present so it matches the
        # support-query API's expected op name.
        name = op.op_name
        if name.startswith("tessera."):
            name = name[len("tessera."):]
        # The runtime_status field is best-effort: we read what the
        # support-query API knows about this op (if it's in OP_SPECS).
        runtime = "unknown"
        source = "graph_ir"
        # Use the module-qualified path explicitly — ``from .
        # import support`` would shadow the function name when
        # the compiler package re-exports ``support``.
        from .support import support as _support_query
        try:
            info = _support_query(name)
            runtime = info.runtime
            source = f"support_table[{info.family}]"
        except KeyError:
            pass
        out.append(Kernel(op_name=name, runtime_status=runtime, source=source))
    return tuple(out)


def _build_diagnostics(fn: "JitFn") -> tuple[Diagnostic, ...]:
    """Lift the JIT's lowering diagnostics + fallback reason into the
    normalized :class:`Diagnostic` shape.

    Every emission from ``@tessera.jit`` is stamped with
    ``lane="tessera_jit"`` so consumers can distinguish it from
    textual-DSL / constrained-math-lane diagnostics without parsing
    the code prefix.  The execution-side fallback diagnostic is
    not lane-stamped because it fires *after* the frontend choice
    is made.
    """

    out: list[Diagnostic] = []
    for ld in fn.lowering_diagnostics:
        out.append(Diagnostic(
            severity=ld.severity,
            code=ld.code,
            message=ld.message,
            lane="tessera_jit",
        ))
    if fn.last_fallback_reason is not None:
        out.append(Diagnostic.from_fallback(fn.last_fallback_reason))
    return tuple(out)


def _build_next_actions(
    fn: "JitFn",
    diagnostics: Iterable[Diagnostic],
) -> tuple[NextAction, ...]:
    """Derive actionable hints from execution kind + diagnostics.

    U3 (2026-05-19) — every diagnostic with a registered next-action
    in :data:`_DIAGNOSTIC_CODE_NEXT_ACTIONS` contributes a targeted
    hint.  Lane-specific codes (Clifford whitelist rejection,
    Complex non-holomorphic, Energy forbidden op, textual DSL parse
    errors) produce *actionable* prescriptions instead of the
    generic ``INSPECT_IR_LAYERS`` fallback.

    Hints are deduplicated by code so the same advice doesn't fire
    multiple times when a function emits several diagnostics with
    the same root cause.
    """

    actions: list[NextAction] = []
    seen_codes: set[str] = set()
    diags = list(diagnostics)

    # Pass 1 — pull targeted hints out of the lane-aware lookup
    # table.  Every diagnostic that has a registered next-action
    # contributes one; duplicates by code are skipped.
    for d in diags:
        hint = next_action_for_diagnostic(d)
        if hint is None:
            continue
        if hint.code in seen_codes:
            continue
        seen_codes.add(hint.code)
        actions.append(hint)

    # Fallback reasons → install / opt-out hints.  These are
    # NOT in the lookup table because they depend on the
    # ``last_fallback_reason`` attribute of the JitFn, not on a
    # standalone diagnostic.
    if fn.last_fallback_reason is FallbackReason.NON_DARWIN_HOST:
        if NEXT_INSTALL_APPLE_GPU not in seen_codes:
            actions.append(NextAction(
                NEXT_INSTALL_APPLE_GPU,
                "Apple GPU runtime requires macOS.  Set target='cpu' or run "
                "on a Darwin host.",
            ))
            seen_codes.add(NEXT_INSTALL_APPLE_GPU)
    elif fn.last_fallback_reason is FallbackReason.APPLE_GPU_RUNTIME_UNAVAILABLE:
        if NEXT_INSTALL_APPLE_GPU not in seen_codes:
            actions.append(NextAction(
                NEXT_INSTALL_APPLE_GPU,
                "Apple GPU runtime failed to initialize.  Check `metal-cli` "
                "and re-run; otherwise stay on the reference path.",
            ))
            seen_codes.add(NEXT_INSTALL_APPLE_GPU)

    # If artifact_only, surface IR inspection as the next move.
    if fn.execution_kind == "artifact_only" and NEXT_INSPECT_IR not in seen_codes:
        actions.append(NextAction(
            NEXT_INSPECT_IR,
            "Target IR was emitted but no native dispatch is wired. "
            "Inspect `.ir.target` or check `ts.compiler.support(op).targets`.",
        ))
        seen_codes.add(NEXT_INSPECT_IR)

    # If reference + user might have wanted native, suggest native_required
    if (
        fn.execution_kind == "reference_cpu"
        and fn.target not in (None, "cpu", "auto")
        and NEXT_NATIVE_REQUIRED not in seen_codes
    ):
        actions.append(NextAction(
            NEXT_NATIVE_REQUIRED,
            "Compiled to reference CPU even though target was set. "
            "Pass `native_required=True` to surface the exact fallback reason.",
        ))
        seen_codes.add(NEXT_NATIVE_REQUIRED)

    # If clean native, hint at the support API for next-step exploration.
    if not actions and fn.execution_kind in ("native_cpu", "native_gpu"):
        actions.append(NextAction(
            NEXT_CHECK_SUPPORT,
            "Native dispatch is clean. Use `ts.compiler.support(op)` for "
            "the per-target readiness picture.",
        ))

    if not actions:
        actions.append(NextAction(NEXT_NO_ACTION, ""))
    return tuple(actions)


def build_explain(fn: "JitFn") -> Explain:
    """Front-door builder consumed by ``JitFn.explain``.

    Pure data assembly — no compilation side effects.  Reads the
    JIT's already-materialized state (``graph_ir``,
    ``compile_bundle``, ``cpu_plan``, ``lowering_diagnostics``,
    ``last_fallback_reason``) and lifts it into the
    :class:`Explain` shape.
    """

    ir = IRLayers(
        graph=fn.ir_text() if fn.graph_ir.functions else None,
        schedule=fn.schedule_ir,
        tile=fn.tile_ir,
        target=fn.target_ir,
    )
    diagnostics = _build_diagnostics(fn)
    kernels = _build_kernels(fn)
    next_actions = _build_next_actions(fn, diagnostics)
    target = fn.target if fn.target else "cpu"
    if hasattr(target, "isa"):
        target = str(target)
    # Derive lane from the GraphIR function attribute when present;
    # fall back to "tessera_jit" for any JitFn that doesn't carry
    # explicit lane provenance (the common case).
    lane = "tessera_jit"
    if fn.graph_ir.functions:
        lane = getattr(fn.graph_ir.functions[0], "lane", "tessera_jit")
    return Explain(
        function_name=getattr(fn._fn, "__qualname__", "<jit_fn>"),
        target=str(target),
        execution_kind=fn.execution_kind,
        ir=ir,
        kernels=kernels,
        diagnostics=diagnostics,
        next_actions=next_actions,
        lane=lane,
        metadata={
            "is_executable": fn.is_executable,
            "has_target_artifacts": fn.has_target_artifacts,
        },
    )


__all__ = [
    "Explain",
    "IRLayers",
    "Kernel",
    "NextAction",
    "NEXT_CHECK_SUPPORT",
    "NEXT_FIX_TEXTUAL_SYNTAX",
    "NEXT_INSPECT_IR",
    "NEXT_INSTALL_APPLE_GPU",
    "NEXT_NATIVE_REQUIRED",
    "NEXT_NO_ACTION",
    "NEXT_PROVIDE_SOURCE",
    "NEXT_REWRITE_HOLOMORPHIC",
    "NEXT_REWRITE_TO_GA_OPS",
    "NEXT_SWITCH_LANE_TO_TESSERA_JIT",
    "NEXT_USE_APPLE_GPU_TARGET",
    "NEXT_USE_ENERGY_WHITELIST",
    "NEXT_USE_FP32_FOR_ENERGY",
    "build_explain",
    "next_action_for_diagnostic",
]
