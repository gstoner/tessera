"""Canonical compiler driver — audit recommendation **C** (C.1: the wrapper).

The audit's framing:

    Make one canonical compiler driver own the whole ladder: Graph module
    in, typed artifacts plus execution capability out. Right now driver.py,
    matmul_pipeline.py, backend manifests, target maps, and runtime dispatch
    each hold part of the truth.

What's true today:

* ``driver.compile_graph_module(...)`` is structurally already "the real
  ladder runner" — it accepts a ``GraphIRModule`` + target and returns a
  ``CompileArtifactBundle`` with all four IR levels populated, plus
  ``executable``/``runtime_status``/``execution_kind``.
* But the executable answer ``CompileArtifactBundle`` produces is decided
  by ``cpu_plan`` presence inside the driver — it does **not** consult the
  audit-named gate ladder (B) or surface the per-cell proof matrix (A).
* ``runtime.launch()`` separately re-derives an executable answer from
  artifact metadata; ``execution_matrix.executor_for_metadata`` does the
  dispatch lookup; ``backend_manifest.manifest_for`` is the per-target
  kernel truth. **Five owners. One question.**

This module is the canonical wrapper that reconciles them:

    canonical_compile(module, *, target="cpu", ...) → CompileResult

with the audit's contract:

    typed_artifacts + capability_set + (executable | reason)

It is **additive** — every existing caller of ``compile_graph_module`` keeps
working unchanged. New callers (the runtime, tests, examples) should reach
for ``canonical_compile`` because:

* The ``executable`` answer is the AND of the bundle's executable and "no
  gate fails," which is the audit's whole point.
* The ``reason`` string leads with the audit-named first failing gate
  (``"first failing gate `toolchain` — nvcc not on PATH"``) so the caller
  doesn't have to know about the gate evaluator separately.
* The ``CompileResult`` is one typed surface that carries everything the
  five today-owners produce, with a single shared truth.

Drift-guarded by ``tests/unit/test_canonical_compile.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from tessera.compiler import pipeline_gates as _pg
from tessera.compiler.driver import (
    CompileArtifactBundle,
    compile_graph_module as _compile_graph_module,
)
from tessera.compiler.graph_ir import GraphIRModule


# --- CompileResult --------------------------------------------------------

@dataclass(frozen=True)
class CompileResult:
    """One typed surface for the canonical compile.

    Carries:

    * ``bundle`` — the full ``CompileArtifactBundle`` from the existing
      driver (Graph / Schedule / Tile / Target IR, diagnostics, trace).
    * ``gate_results`` — the seven named pipeline gates (B layer).
    * ``first_failing_gate`` — the audit-named gate (or None if every gate
      passes).
    * ``executable`` — synthesized AND of bundle + gates. The two must
      agree for the answer to be ``True``.
    * ``reason`` — empty string when executable; the audit-named reason
      otherwise. Leads with ``first failing gate `<name>` — <detail>``.

    Convenience accessors (``graph_ir``, ``schedule_ir``, ``tile_ir``,
    ``target_ir``) return the IR text from the bundle so callers don't have
    to thread through ``.bundle.graph.text`` every time.
    """

    bundle: CompileArtifactBundle
    gate_results: tuple[_pg.GateResult, ...]
    first_failing_gate: Optional[_pg.GateResult]
    executable: bool
    reason: str
    primary_op: Optional[str]
    target: str

    @property
    def graph_ir(self) -> str:
        return self.bundle.graph.text if self.bundle.graph else ""

    @property
    def schedule_ir(self) -> str:
        return self.bundle.schedule.text if self.bundle.schedule else ""

    @property
    def tile_ir(self) -> str:
        return self.bundle.tile.text if self.bundle.tile else ""

    @property
    def target_ir(self) -> str:
        return self.bundle.target_ir.text if self.bundle.target_ir else ""

    @property
    def runtime_status(self) -> str:
        return self.bundle.runtime_status

    @property
    def execution_kind(self) -> str:
        return self.bundle.execution_kind

    @property
    def execution_mode(self) -> str:
        return self.bundle.execution_mode

    @property
    def compiler_path(self) -> str:
        return self.bundle.request.pipeline_name

    def gate_status(self, gate_name: str) -> str:
        """Return the status string for one named gate, or ``"unknown"``."""
        for r in self.gate_results:
            if r.gate == gate_name:
                return r.status
        return "unknown"

    def to_runtime_artifact(self):
        """Project the canonical answer into a :class:`RuntimeArtifact`.

        The runtime ABI consumes ``RuntimeArtifact``s, not ``CompileResult``s
        (legacy ergonomic). This method stamps the canonical answer
        (``executable`` / ``reason`` / ``first_failing_gate`` / gates) into
        ``RuntimeArtifact.metadata`` so :func:`tessera.runtime.launch` can
        **trust** the upstream answer instead of re-running the gate
        evaluator at launch time.

        The artifact is fully self-describing — every consumer who needs
        the seven-gate truth can read it from the artifact without
        re-importing the compiler. Audit recommendation C.2 in
        ``docs/audit/compiler_layer_gap_remediation.md``.
        """
        # Import here to keep the compiler→runtime dependency one-way at
        # import time (canonical_compile is pulled by jit.py very early).
        from tessera.runtime import RuntimeArtifact

        meta: dict[str, Any] = dict(self.bundle.to_metadata())
        meta.update({
            # Audit-named answer (C.1 → C.2). Two channels: a top-level
            # boolean + reason string, and a structured first-failing-gate
            # mirror that the runtime can short-circuit on.
            "canonical_executable": self.executable,
            "canonical_reason": self.reason,
            "canonical_first_failing_gate": (
                self.first_failing_gate.gate
                if self.first_failing_gate else None
            ),
            "canonical_first_failing_gate_detail": (
                self.first_failing_gate.detail
                if self.first_failing_gate else ""
            ),
            "canonical_gates": [
                {"gate": r.gate, "status": r.status, "detail": r.detail}
                for r in self.gate_results
            ],
            "canonical_primary_op": self.primary_op,
            # The bundle's executable claim — preserved so callers that
            # need to compare bundle-side vs canonical-side decisions can.
            "bundle_executable": self.bundle.executable,
            # Canonical answer is authoritative for `executable`.
            "executable": self.executable,
        })
        return RuntimeArtifact(
            graph_ir=self.graph_ir,
            schedule_ir=self.schedule_ir,
            tile_ir=self.tile_ir,
            target_ir=self.target_ir,
            metadata=meta,
            abi_signature=f"tessera.canonical.v1.{self.target}",
        )

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable summary — useful for telemetry, dashboards,
        compile-report integration, and tests that want a stable surface.

        For Apple targets, also includes ``apple_gpu_capabilities``
        (per-feature flag dict) and ``apple_gpu_archive`` (MTL4Archive
        cache state) — the Apple-sample Actions 1 + 6 surface. The
        snapshot is a best-effort read; on hosts where the runtime
        isn't available, the keys are present but mostly empty
        (``runtime_available=False``), so consumers don't need to
        special-case the missing case.
        """
        out: dict[str, Any] = {
            "target": self.target,
            "primary_op": self.primary_op,
            "compiler_path": self.compiler_path,
            "executable": self.executable,
            "reason": self.reason,
            "runtime_status": self.runtime_status,
            "execution_kind": self.execution_kind,
            "execution_mode": self.execution_mode,
            "first_failing_gate": (self.first_failing_gate.gate
                                   if self.first_failing_gate else None),
            "first_failing_gate_detail": (self.first_failing_gate.detail
                                          if self.first_failing_gate else ""),
            "gates": [
                {"gate": r.gate, "status": r.status, "detail": r.detail}
                for r in self.gate_results
            ],
            "artifact_hashes": self.bundle.artifact_hashes,
        }
        # Apple-sample Actions 1 + 6 — capability + archive telemetry on
        # Apple targets. The compile result is the natural carrier: a
        # dashboard or test that wants "what's lit up on this host"
        # reads it from one place. Lazy-imported so the canonical
        # compile module stays a pure aggregator over its declared
        # truth sources.
        if self.target in ("apple_gpu", "apple_cpu"):
            try:
                from tessera._apple_gpu_dispatch import (
                    apple_gpu_capabilities_snapshot,
                )
                snap = apple_gpu_capabilities_snapshot()
                out["apple_gpu_capabilities"] = snap.get("capabilities", {})
                out["apple_gpu_capabilities_raw"] = snap.get(
                    "capabilities_raw", 0)
                out["apple_gpu_mtl4_full"] = snap.get("mtl4_full", False)
                out["apple_gpu_archive"] = snap.get("archive", {})
                out["apple_gpu_runtime_available"] = snap.get(
                    "runtime_available", False)
            except Exception:
                # Defensive — capability telemetry must never break
                # ``to_dict``. Leave the keys absent on import failure.
                pass
        return out


# --- Primary-op extraction -----------------------------------------------

def _extract_primary_op(module: GraphIRModule) -> Optional[str]:
    """Pick the op name that the gate evaluator should treat as primary.

    Today: the *first* op in the module's first function. Gate evaluation
    is op-specific for ``codegen`` / ``link`` / ``numerical`` — those want
    a concrete op like ``matmul`` to consult the backend manifest. For
    multi-op programs this is best-effort; the conformance matrix's
    component-ops model is the long-term fit. Returned **without** the
    ``tessera.`` prefix so the gate evaluator's internal registry matches
    (the gate module strips the prefix when it sees it).
    """
    if not module.functions:
        return None
    fn = module.functions[0]
    if not fn.body:
        return None
    op_name = fn.body[0].op_name or ""
    if op_name.startswith("tessera."):
        op_name = op_name[len("tessera."):]
    return op_name or None


# --- Synthesizers --------------------------------------------------------

def _result_from_bundle(
    bundle: CompileArtifactBundle,
    primary_op: Optional[str],
) -> CompileResult:
    """Shared "bundle + gates → CompileResult" reconciliation. Both
    :func:`canonical_compile` (which runs the ladder first) and
    :func:`compile_result_from_bundle` (for callers like ``@jit`` who
    already have a bundle in hand) flow through here, so the
    executable/reason synthesis lives in exactly one place."""
    target = bundle.request.target
    gate_results = _pg.evaluate(target, primary_op)
    first_fail = _pg.first_failing_gate(target, primary_op)

    if bundle.executable and first_fail is None:
        executable = True
        reason = ""
    elif first_fail is not None:
        executable = False
        reason = (
            f"first failing gate `{first_fail.gate}` — {first_fail.detail}."
            f" (see docs/audit/op_target_conformance.md)"
        )
    else:
        executable = False
        reason = (
            f"bundle reports non-executable artifact: "
            f"runtime_status={bundle.runtime_status} "
            f"execution_kind={bundle.execution_kind}"
        )

    return CompileResult(
        bundle=bundle,
        gate_results=gate_results,
        first_failing_gate=first_fail,
        executable=executable,
        reason=reason,
        primary_op=primary_op,
        target=target,
    )


def compile_result_from_bundle(
    bundle: CompileArtifactBundle,
    *,
    module: Optional[GraphIRModule] = None,
    primary_op: Optional[str] = None,
) -> CompileResult:
    """Build a :class:`CompileResult` from an already-computed
    :class:`CompileArtifactBundle`.

    Use case (C.3): ``@tessera.jit`` calls ``compile_graph_module`` directly
    today and rewraps the bundle to inject pre-compile diagnostics. Calling
    :func:`canonical_compile` from ``@jit`` would re-run the whole ladder.
    Instead, ``@jit`` finishes its bundle construction and then synthesizes
    the canonical answer from it via this helper — same C.1 truth, no
    duplicated compile work.

    ``primary_op`` may be passed explicitly; otherwise it's extracted from
    ``module``. If neither is provided, the gate evaluation runs
    target-level only (which is still useful — toolchain / hardware_smoke
    don't need an op name).
    """
    if primary_op is None and module is not None:
        primary_op = _extract_primary_op(module)
    return _result_from_bundle(bundle, primary_op)


# --- Canonical compile() -------------------------------------------------

def canonical_compile(
    module: GraphIRModule,
    *,
    target: str = "cpu",
    source_origin: str = "<canonical>",
    options: Mapping[str, Any] | None = None,
    cpu_tile: tuple[int, int, int] = (128, 128, 64),
    enable_tool_validation: bool = True,
) -> CompileResult:
    """Run the full Tessera compile ladder and return a typed result.

    Composes:

    * :func:`tessera.compiler.driver.compile_graph_module` — the existing
      ladder runner (Graph→Schedule→Tile→Target lowering + diagnostics).
    * :func:`tessera.compiler.pipeline_gates.evaluate` — the audit's seven
      named capability gates (B layer).

    Returns a :class:`CompileResult` whose ``executable`` is the
    AND of both layers' answers — the bundle's structural "we made an
    executable artifact" AND "no audit gate fails." When ``executable`` is
    False, ``reason`` leads with the audit-named first failing gate so the
    diagnostic is actionable.

    No new compiler logic lives here — this is a *pure composition* of the
    existing surfaces. Audit recommendation C's whole point: one place to
    look for the answer.
    """
    bundle = _compile_graph_module(
        module,
        source_origin=source_origin,
        target=target,
        cpu_tile=cpu_tile,
        options=options or {},
        enable_tool_validation=enable_tool_validation,
    )
    return _result_from_bundle(bundle, _extract_primary_op(module))


__all__ = [
    "CompileResult",
    "canonical_compile",
    "compile_result_from_bundle",
]
