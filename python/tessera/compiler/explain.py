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
    Diagnostic,
    FallbackReason,
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
        lines.append(
            f"tessera.jit[{self.function_name!r} → {self.target}]: {kind_label}"
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
            lines.append(
                f"  Diagnostics: {', '.join(parts)} "
                f"(first: {top.code_value} — {top.message[:60]}"
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
    normalized :class:`Diagnostic` shape."""

    out: list[Diagnostic] = []
    for ld in fn.lowering_diagnostics:
        out.append(Diagnostic(
            severity=ld.severity,
            code=ld.code,
            message=ld.message,
        ))
    if fn.last_fallback_reason is not None:
        out.append(Diagnostic.from_fallback(fn.last_fallback_reason))
    return tuple(out)


def _build_next_actions(
    fn: "JitFn",
    diagnostics: Iterable[Diagnostic],
) -> tuple[NextAction, ...]:
    """Derive actionable hints from execution kind + diagnostics."""

    actions: list[NextAction] = []
    diags = list(diagnostics)
    # Source unavailable → say so explicitly.
    for d in diags:
        if d.code is JitDiagnosticCode.SOURCE_UNAVAILABLE or (
            isinstance(d.code, str) and d.code == "JIT_SOURCE_UNAVAILABLE"
        ):
            actions.append(NextAction(
                NEXT_PROVIDE_SOURCE,
                "Pass `source=...` to @jit, or use `tessera.from_text(...)` "
                "for notebook-safe construction.",
            ))
            break

    # Fallback reasons → install / opt-out hints.
    if fn.last_fallback_reason is FallbackReason.NON_DARWIN_HOST:
        actions.append(NextAction(
            NEXT_INSTALL_APPLE_GPU,
            "Apple GPU runtime requires macOS.  Set target='cpu' or run "
            "on a Darwin host.",
        ))
    elif fn.last_fallback_reason is FallbackReason.APPLE_GPU_RUNTIME_UNAVAILABLE:
        actions.append(NextAction(
            NEXT_INSTALL_APPLE_GPU,
            "Apple GPU runtime failed to initialize.  Check `metal-cli` "
            "and re-run; otherwise stay on the reference path.",
        ))

    # If artifact_only, surface IR inspection as the next move.
    if fn.execution_kind == "artifact_only":
        actions.append(NextAction(
            NEXT_INSPECT_IR,
            "Target IR was emitted but no native dispatch is wired. "
            "Inspect `.ir.target` or check `ts.compiler.support(op).targets`.",
        ))

    # If reference + user might have wanted native, suggest native_required
    if fn.execution_kind == "reference_cpu" and fn.target not in (None, "cpu", "auto"):
        actions.append(NextAction(
            NEXT_NATIVE_REQUIRED,
            "Compiled to reference CPU even though target was set. "
            "Pass `native_required=True` to surface the exact fallback reason.",
        ))

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
    return Explain(
        function_name=getattr(fn._fn, "__qualname__", "<jit_fn>"),
        target=str(target),
        execution_kind=fn.execution_kind,
        ir=ir,
        kernels=kernels,
        diagnostics=diagnostics,
        next_actions=next_actions,
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
    "NEXT_INSPECT_IR",
    "NEXT_INSTALL_APPLE_GPU",
    "NEXT_NATIVE_REQUIRED",
    "NEXT_NO_ACTION",
    "NEXT_PROVIDE_SOURCE",
    "build_explain",
]
