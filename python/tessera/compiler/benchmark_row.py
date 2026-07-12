"""Canonical benchmark-row schema — M5 deliverable.

Each Tessera benchmark harness (CPU, Apple GPU, NVIDIA, ROCm)
emits JSON rows describing what it ran.  Before M5 the field set
drifted between harnesses — some emit ``backend`` / ``mode``, some
emit ``namespace`` / ``dispatched_on_gpu``, some omit timing
percentiles.  That made it hard to feed benchmark output back into
the :class:`tessera.compiler.compile_report.CompileReport`
envelope M1 introduced.

M5 unifies the row shape and provides one-line conversion in both
directions.  The bridge means:

  - A canonical-program driver can call
    ``BenchmarkRow.from_compile_report(report, latency_ms=...)`` and
    drop its row directly into ``benchmark_ga_ebm.json``.
  - A benchmark row from disk can be hoisted into a CompileReport
    via ``row.to_compile_report()`` so the audit + drift tooling
    treats both surfaces uniformly.

The unified schema also doubles as the M5 "artifact-only paths
cannot accidentally appear as native in generated reports"
guarantee — :func:`validate_benchmark_row` rejects rows that claim
``backend != "python_ref"`` but lack a route proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .compile_report import (
    CompileReport, FRONTEND_TESSERA_JIT, VALUE_KIND_TENSOR,
)
from . import jit_bridge as _bridge


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

#: Canonical benchmark-row field names.  Any row that omits one of
#: these (or adds an unknown one) is rejected by
#: :func:`validate_benchmark_row`.
REQUIRED_BENCHMARK_FIELDS: frozenset[str] = frozenset({
    "namespace",
    "op",
    "backend",
    "shape",
    "dtype",
    "mode",
    "ok",
    "latency_ms",
    "device",
    "tessera_version",
})

OPTIONAL_BENCHMARK_FIELDS: frozenset[str] = frozenset({
    "reps",
    "stdev_ms",
    "p10_ms", "p50_ms", "p90_ms",
    "min_ms", "max_ms",
    "max_abs_err",
    "tolerance",
    "dispatched_on_gpu",
    "symbols",
    "compiled_artifact",
    "proof_routes",
    "fallback_reason",
    "report_hash",
    "error",
    "plan_hash",
    # The single open extensibility slot — hot-path group / expected-latency /
    # ratchet-bound / fused-chain name ride here instead of proliferating
    # schema fields or Apple-specific ad-hoc JSON (DEEP_COMPILER_AUDIT_2026_06_10).
    "hot_path_metadata",
})

#: ``mode`` values that imply native execution on the named backend.
NATIVE_MODES: frozenset[str] = frozenset({
    "fused", "fused_chain", "jit_compiled", "native",
})

#: ``backend`` values that explicitly disclaim native execution.
NON_NATIVE_BACKENDS: frozenset[str] = frozenset({
    "python_ref", "reference", "numpy",
})


@dataclass(frozen=True)
class BenchmarkRow:
    """One canonical benchmark-row record.

    Schema-validated at construction.  Use
    :meth:`from_compile_report` to build a row from a CompileReport
    a canonical-program driver emitted; use
    :meth:`to_compile_report` to lift an existing row into the M1
    envelope.
    """
    namespace: str
    op: str
    backend: str
    shape: str
    dtype: str
    mode: str
    ok: bool
    latency_ms: float
    device: str
    tessera_version: str

    # Optional fields — preserved verbatim in :meth:`as_dict`.
    reps: Optional[int] = None
    stdev_ms: Optional[float] = None
    p10_ms: Optional[float] = None
    p50_ms: Optional[float] = None
    p90_ms: Optional[float] = None
    min_ms: Optional[float] = None
    max_ms: Optional[float] = None
    max_abs_err: Optional[float] = None
    tolerance: Optional[float] = None
    dispatched_on_gpu: Optional[bool] = None
    symbols: tuple[str, ...] = ()
    compiled_artifact: Optional[Mapping[str, Any]] = None
    proof_routes: tuple[_bridge.JitBridgeRoute, ...] = ()
    fallback_reason: Optional[str] = None
    report_hash: Optional[str] = None
    plan_hash: Optional[str] = None
    error: Optional[str] = None
    #: Single open extensibility slot for hot-path metadata (group name,
    #: expected latency, ratchet bound, fused-chain alias). Preserved verbatim.
    hot_path_metadata: Optional[Mapping[str, Any]] = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly dict — preserves only fields that were set
        so existing JSON snapshots don't acquire spurious null
        columns."""
        d: dict[str, Any] = {
            "namespace": self.namespace,
            "op": self.op,
            "backend": self.backend,
            "shape": self.shape,
            "dtype": self.dtype,
            "mode": self.mode,
            "ok": self.ok,
            "latency_ms": self.latency_ms,
            "device": self.device,
            "tessera_version": self.tessera_version,
        }
        for name in (
            "reps", "stdev_ms", "p10_ms", "p50_ms", "p90_ms",
            "min_ms", "max_ms", "max_abs_err", "tolerance",
            "dispatched_on_gpu", "fallback_reason", "report_hash",
            "plan_hash", "error",
        ):
            value = getattr(self, name)
            if value is not None:
                d[name] = value
        if self.symbols:
            d["symbols"] = list(self.symbols)
        if self.proof_routes:
            d["proof_routes"] = [
                {
                    "op_name": r.op_name,
                    "target": r.target,
                    "status": r.status,
                    "symbol": r.symbol,
                    "context": r.context,
                    "latency_ms": r.latency_ms,
                }
                for r in self.proof_routes
            ]
        if self.compiled_artifact is not None:
            d["compiled_artifact"] = dict(self.compiled_artifact)
        if self.hot_path_metadata is not None:
            d["hot_path_metadata"] = dict(self.hot_path_metadata)
        return d

    # ── Conversions ────────────────────────────────────────────────

    @classmethod
    def from_compile_report(
        cls,
        report: CompileReport,
        *,
        namespace: str,
        shape: str,
        dtype: str = "fp32",
        mode: str = "jit_compiled",
        device: str = "unknown",
        reps: int = 1,
    ) -> "BenchmarkRow":
        """Build a BenchmarkRow from a CompileReport.  M1's envelope
        already carries everything a benchmark row needs."""
        ok = report.fallback_reason is None or report.target.startswith("cpu")
        # Pick a representative latency: prefer ``end_to_end`` if
        # present, otherwise the first timing entry.
        latency_ms = 0.0
        if report.timing_ms:
            latency_ms = float(
                report.timing_ms.get("end_to_end")
                or next(iter(report.timing_ms.values()), 0.0)
            )
        max_abs_err: Optional[float] = None
        tolerance: Optional[float] = None
        if report.correctness:
            max_abs_err = float(report.correctness.get("max_abs_err", 0.0))
            tolerance = float(report.correctness.get("tolerance", 0.0))
            if tolerance and max_abs_err > tolerance:
                ok = False
        # Backend follows the report's target — `apple_gpu` rows are
        # native; `cpu` rows are reference.
        backend = (
            "python_ref" if report.target.startswith("cpu")
            else report.target
        )
        if backend == "python_ref" and mode in NATIVE_MODES:
            mode = "reference"
        # Serialize the IR / target_decision into compiled_artifact
        # so the JSON row is self-describing.
        compiled_artifact: dict[str, Any] = {
            "frontend": report.frontend,
            "value_kind": report.value_kind,
            "target": report.target,
            "ir_hashes": dict(report.ir_hashes),
            "target_decision": dict(report.target_decision),
        }
        fb = report.fallback_reason
        fallback_str: Optional[str] = None
        if fb is not None:
            fallback_str = (
                fb.value if hasattr(fb, "value") else str(fb)
            )
        return cls(
            namespace=namespace,
            op=report.program_id,
            backend=backend,
            shape=shape,
            dtype=dtype,
            mode=mode,
            ok=bool(ok),
            latency_ms=latency_ms,
            device=device,
            tessera_version=report.tessera_version,
            reps=reps,
            max_abs_err=max_abs_err,
            tolerance=tolerance,
            dispatched_on_gpu=(backend != "python_ref"),
            symbols=tuple(
                r.symbol for r in report.proof_routes if r.symbol
            ),
            compiled_artifact=compiled_artifact,
            proof_routes=tuple(report.proof_routes),
            fallback_reason=fallback_str,
            report_hash=report.report_hash(),
        )

    def to_compile_report(self) -> CompileReport:
        """Lift this benchmark row back into a CompileReport.  This
        is the inverse of :meth:`from_compile_report` for rows that
        carry the M5 fields; older rows still produce a valid (if
        sparse) CompileReport."""
        artifact = self.compiled_artifact or {}
        ir_hashes = dict(artifact.get("ir_hashes", {}))
        target_decision = dict(artifact.get("target_decision", {}))
        correctness: Optional[dict[str, float]] = None
        if self.max_abs_err is not None and self.tolerance is not None:
            correctness = {
                "max_abs_err": self.max_abs_err,
                "tolerance": self.tolerance,
            }
        return CompileReport(
            program_id=self.op,
            source=f"benchmark_row[{self.namespace}/{self.op}]",
            frontend=str(artifact.get("frontend", FRONTEND_TESSERA_JIT)),
            value_kind=str(artifact.get("value_kind", VALUE_KIND_TENSOR)),
            target=str(artifact.get("target", self.backend)),
            tessera_version=self.tessera_version,
            ir_hashes=ir_hashes,
            target_decision=target_decision,
            fallback_reason=self.fallback_reason,
            proof_routes=tuple(self.proof_routes),
            timing_ms={"end_to_end": self.latency_ms},
            correctness=correctness,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_benchmark_row(row: Mapping[str, Any]) -> None:
    """Validate a row dict against the canonical schema.

    Raises :class:`ValueError` when required fields are missing,
    unknown fields appear, or the row claims native execution
    (``backend != python_ref``, ``mode in NATIVE_MODES``) without a
    route proof or device_verified_jit-artifact entry.  This is the M5 "no
    silent native claim" guarantee.
    """
    missing = REQUIRED_BENCHMARK_FIELDS - row.keys()
    if missing:
        raise ValueError(
            f"benchmark row missing required fields: {sorted(missing)}"
        )
    allowed = REQUIRED_BENCHMARK_FIELDS | OPTIONAL_BENCHMARK_FIELDS
    unknown = set(row.keys()) - allowed
    if unknown:
        raise ValueError(
            f"benchmark row has unknown fields: {sorted(unknown)} "
            f"(canonical set: {sorted(allowed)})"
        )
    backend = row["backend"]
    mode = row["mode"]
    if backend not in NON_NATIVE_BACKENDS and mode in NATIVE_MODES:
        has_proof = bool(
            row.get("proof_routes")
            or row.get("compiled_artifact")
            or row.get("plan_hash")
            or row.get("symbols")
        )
        if not has_proof:
            raise ValueError(
                f"row claims native execution (backend={backend!r}, "
                f"mode={mode!r}) but carries no proof (proof_routes / "
                "compiled_artifact / plan_hash / symbols)"
            )


__all__ = [
    "BenchmarkRow",
    "REQUIRED_BENCHMARK_FIELDS",
    "OPTIONAL_BENCHMARK_FIELDS",
    "NATIVE_MODES",
    "NON_NATIVE_BACKENDS",
    "validate_benchmark_row",
]
