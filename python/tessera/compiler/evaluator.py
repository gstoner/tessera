"""Execution-derived Evaluator — per-backend rung verdicts (Phase E1).

See ``docs/audit/compiler/EVALUATOR_PLAN.md`` for the full architecture.

**Why this exists.** ``accelerator_proof.classify()`` declares an op ``proven``
by *envelope membership* — a static map lookup. That is the "registry models
reality" gap in miniature (DEEP_COMPILER_AUDIT_2026_06_10 Method note): a green
classification proves the op is *routed*, not that a real program using it
*executed and matched a reference*. This module derives the claim dynamically:
run a (already-jitted) program, read the actual ``execution_mode`` provenance
signal, compare against an oracle, and record the **highest honest rung** the
backend reached.

The load-bearing invariant: **if the backend genuinely executed
(``provenance_ok``), the result must match the oracle — otherwise it is a
miscompile, not a pass; and a silent fallback to the numpy reference can never
earn an execution rung.** Numerical agreement and "ran on the demanded backend"
are separate gates (anti-silent-fallback, per TritonRL / Decision #21).

This first slice covers the verdict logic (portable) and the Apple GPU
execution path (Darwin-only). The horizontal-equivalence oracle, the
NVIDIA/ROCm ``ptxas``/``hipcc`` assembly rung, and the conformance-matrix
derivation are the next slices in the plan.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class Rung(IntEnum):
    """Per-backend status ladder (EVALUATOR_PLAN.md §2). Higher = stronger,
    and the integer order is meaningful (``verdict.rung >= Rung.EXECUTES``).
    The names are the contract; rely on ordering, not specific integers."""

    ARTIFACT_ONLY = 1       # IR emitted
    LOWERS_CLEAN = 2        # Target IR passes the MLIR verifier, no unsupported-diag
    EMITS_ASM_TEXT = 3      # backend emits real PTX/AMDGCN assembler text (rung 2.5)
    ASSEMBLES = 4           # emitted text actually assembles (ptxas/hipcc) — CI
    CODEGEN_STABLE = 5      # same IR @ two opt levels → structurally-equivalent code
    NUMERICAL_SYMBOLIC = 6  # microkernel ≡ reference via finite-field / SMT
    EXECUTES = 7            # ran on the demanded backend (real silicon)
    HARDWARE_VERIFIED = 8   # ran AND oracle-matched on real silicon


# ``execution_kind`` values (from ``runtime.launch``) that mean the program
# produced a result but NOT on the demanded native backend — a silent fallback.
# Read the *runtime* verdict, never the compile-time intent: an op can compile
# with ``execution_mode="metal_runtime"`` yet fall back at runtime through the
# strict-dispatch funnel, and only the runtime result reveals that.
_FALLBACK_EXECUTION_KINDS: frozenset[str] = frozenset(
    {"fallback_eager", "eager_fallback", "reference_cpu", "reference"}
)


@dataclass(frozen=True)
class BackendVerdict:
    """The highest honest rung a single backend reached for one program."""

    target: str
    rung: Rung
    execution_kind: str          # runtime.launch's execution_kind (native_gpu, fallback_eager, ...)
    runtime_status: str          # success | unimplemented | invalid_artifact | missing_backend
    provenance_ok: bool          # the demanded backend ran (not a silent fallback)
    correctness: str             # "pass" | "fail" | "unproven"
    detail: str = ""

    @property
    def is_silent_fallback(self) -> bool:
        """True when the program produced a result but NOT on the demanded
        backend — the exact degeneracy a green cell must never be earned by."""
        return not self.provenance_ok


def verdict_for(
    target: str,
    execution_kind: str,
    runtime_status: str,
    oracle_match: bool | None,
) -> BackendVerdict:
    """Pure decision function over the **runtime** signal → verdict.

    Provenance is earned only when the runtime genuinely executed
    (``runtime_status == "success"``) on the demanded backend (the
    ``execution_kind`` is not a fallback). ``oracle_match`` is ``True``/``False``
    from a reference compare, or ``None`` when no reference was available. This
    function has no I/O and is exhaustively unit-testable: it is the
    anti-silent-fallback contract in one place.
    """
    provenance_ok = (
        runtime_status == "success"
        and execution_kind not in _FALLBACK_EXECUTION_KINDS
    )

    if not provenance_ok:
        # The demanded backend did not execute. Whatever number came back is a
        # fallback, so we make NO execution/correctness claim — and we do NOT
        # claim rung 2 (`lowers_clean`) either, which requires positive evidence
        # the Target IR passed the MLIR verifier. The honest floor is rung 1
        # (`artifact_only`): IR was emitted; nothing stronger is proven. This is
        # exactly where NVIDIA/ROCm sit today (Target IR artifact, no execution).
        return BackendVerdict(
            target=target, rung=Rung.ARTIFACT_ONLY,
            execution_kind=execution_kind, runtime_status=runtime_status,
            provenance_ok=False, correctness="unproven",
            detail=f"did not execute natively on {target} "
                   f"(kind={execution_kind!r}, status={runtime_status!r}); "
                   "artifact only — silent fallback cannot earn an execution rung",
        )

    if oracle_match is None:
        return BackendVerdict(
            target=target, rung=Rung.EXECUTES,
            execution_kind=execution_kind, runtime_status=runtime_status,
            provenance_ok=True, correctness="unproven",
            detail="executed natively; no reference available to compare",
        )

    if oracle_match:
        return BackendVerdict(
            target=target, rung=Rung.HARDWARE_VERIFIED,
            execution_kind=execution_kind, runtime_status=runtime_status,
            provenance_ok=True, correctness="pass",
            detail="executed natively and matched the reference oracle",
        )

    # Ran on the real backend but disagreed with the oracle → a genuine
    # miscompile signal, NOT a pass. This is the bug the Evaluator exists to find.
    return BackendVerdict(
        target=target, rung=Rung.EXECUTES,
        execution_kind=execution_kind, runtime_status=runtime_status,
        provenance_ok=True, correctness="fail",
        detail="executed natively but DISAGREED with the reference oracle "
               "(miscompile candidate)",
    )


def evaluate(
    target: str,
    jitted_fn: Any,
    args: tuple[Any, ...],
    oracle: Any,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    exact: bool = False,
) -> BackendVerdict:
    """Run one ``@jit(target=...)`` program through the runtime and derive its
    backend verdict.

    Drives execution through ``runtime.launch`` (the authoritative runtime
    path), which returns the real ``execution_kind`` / ``runtime_status`` *and*
    the native ``output``. ``oracle`` is the independent reference array;
    ``exact`` requests integer/bit equality. The verdict combines the runtime
    provenance with the oracle compare via :func:`verdict_for`.
    """
    import numpy as np

    from tessera import runtime as _rt

    res = _rt.launch(jitted_fn.runtime_artifact(), args)
    execution_kind = str(res.get("execution_kind", "unknown"))
    runtime_status = str(res.get("runtime_status", "unknown"))

    match: bool | None = None
    if runtime_status == "success" and "output" in res:
        got = np.asarray(res["output"])
        ref = np.asarray(oracle)
        if not np.all(np.isfinite(got)) or got.shape != ref.shape:
            match = None
        elif exact:
            match = bool(np.array_equal(got, ref))
        else:
            match = bool(np.allclose(got, ref, rtol=rtol, atol=atol))

    return verdict_for(target, execution_kind, runtime_status, match)


# ── Horizontal-equivalence oracle (PolyJuice, OOPSLA'24) ─────────────────────
#
# The vertical oracle compares the executed result against an external numpy
# reference. The *horizontal* oracle compares two equivalent program
# representations on the SAME backend — a fused chain run as one kernel vs. the
# same math composed from separately-executed native ops. It needs no trusted
# external reference (self-consistency) and isolates the fusion rewrite as the
# suspect: if it diverges, the fusion is not semantics-preserving. This is the
# check that hardens Tessera's fused chains (matmul→softmax[→matmul], etc.).


def run_native(target: str, fn: Any, args: tuple[Any, ...]) -> tuple[Any | None, bool]:
    """Execute one jitted program and return ``(output, native_ok)``.

    ``native_ok`` is True only when the demanded backend genuinely ran (not a
    silent fallback). ``output`` is ``None`` when it did not run natively, so a
    fallback result can never be silently used as a comparison operand.
    """
    from tessera import runtime as _rt

    res = _rt.launch(fn.runtime_artifact(), args)
    native = (
        res.get("runtime_status") == "success"
        and str(res.get("execution_kind", "")) not in _FALLBACK_EXECUTION_KINDS
    )
    return (res.get("output") if native else None), bool(native)


@dataclass(frozen=True)
class HorizontalVerdict:
    """Result of comparing a fused chain against its unfused equivalent."""

    target: str
    relation: str            # "equivalent" | "divergent" | "inconclusive"
    max_abs_err: float | None
    detail: str = ""

    @property
    def is_divergent(self) -> bool:
        return self.relation == "divergent"


def _horizontal_relation(
    native_fused: bool,
    native_unfused: bool,
    max_abs_err: float | None,
    *,
    tol: float,
) -> tuple[str, str]:
    """Pure classifier for the horizontal verdict (portably unit-testable)."""
    if not native_fused or not native_unfused:
        which = []
        if not native_fused:
            which.append("fused")
        if not native_unfused:
            which.append("unfused")
        return "inconclusive", (
            f"{'/'.join(which)} side did not execute natively — cannot compare "
            "two representations on the same backend"
        )
    if max_abs_err is None:
        return "inconclusive", "outputs not comparable (shape mismatch / non-finite)"
    if max_abs_err <= tol:
        return "equivalent", f"fused ≡ unfused (max_abs_err={max_abs_err:.3e} ≤ {tol:.1e})"
    return "divergent", (
        f"fused ≠ unfused (max_abs_err={max_abs_err:.3e} > {tol:.1e}) — the "
        "fusion rewrite is not semantics-preserving on this backend"
    )


def horizontal_equivalence(
    target: str,
    fused_fn: Any,
    args: tuple[Any, ...],
    unfused: Callable[[tuple[Any, ...]], tuple[Any, bool]],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> HorizontalVerdict:
    """Assert a fused chain equals its unfused equivalent on the same backend.

    ``fused_fn`` is the single-jit chain (e.g. ``gelu(matmul(a, b))``).
    ``unfused`` is a callable that recomputes the same value by composing
    separately-executed native ops and returns ``(value, all_native)`` — build
    it with :func:`run_native` so each component's provenance is checked. The
    verdict is ``inconclusive`` unless *both* representations ran natively.
    """
    import numpy as np

    fused_out, native_fused = run_native(target, fused_fn, args)
    unfused_out, native_unfused = unfused(args)

    max_abs_err: float | None = None
    if native_fused and native_unfused and fused_out is not None and unfused_out is not None:
        a = np.asarray(fused_out, dtype=np.float64)
        b = np.asarray(unfused_out, dtype=np.float64)
        if a.shape == b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            max_abs_err = float(np.max(np.abs(a - b)))

    tol = atol + rtol * (
        float(np.max(np.abs(np.asarray(unfused_out, dtype=np.float64))))
        if unfused_out is not None and native_unfused else 1.0
    )
    relation, detail = _horizontal_relation(
        native_fused, native_unfused, max_abs_err, tol=tol
    )
    return HorizontalVerdict(target, relation, max_abs_err, detail)


# ── NVIDIA/ROCm emission rung (rung 2.5) ─────────────────────────────────────
#
# These backends do not execute here; their honest forward progress is measured
# by whether the lowering emits real assembler text. The lowering attaches the
# emitted PTX + its structural-validation status to the artifact metadata
# (jit.py target_ir_artifact branch); this reads it and reports the rung —
# metadata only, no toolchain, no GPU. Assembly (rung 4) is the CI gate.


def nvidia_emission_verdict(jitted_fn: Any) -> BackendVerdict:
    """Derive an NVIDIA program's emission rung from its artifact metadata.

    ``EMITS_ASM_TEXT`` when the lowering attached structurally-valid WGMMA PTX;
    ``ARTIFACT_ONLY`` otherwise. Never claims execution or assembly — an
    artifact-only backend cannot earn a higher rung here.
    """
    try:
        meta = jitted_fn.runtime_artifact().metadata
    except Exception:
        meta = {}
    target = str(meta.get("target", "nvidia"))
    if "nvidia_ptx" in meta and bool(meta.get("nvidia_ptx_valid", False)):
        return BackendVerdict(
            target=target, rung=Rung.EMITS_ASM_TEXT,
            execution_kind="ptx_emitted", runtime_status="artifact_only",
            provenance_ok=False, correctness="unproven",
            detail="emits structurally-valid WGMMA PTX assembler text (skeleton); "
                   "ptxas assembly is the rung-4 CI gate",
        )
    return BackendVerdict(
        target=target, rung=Rung.ARTIFACT_ONLY,
        execution_kind="none", runtime_status="artifact_only",
        provenance_ok=False, correctness="unproven",
        detail="no emitted assembler text — Target IR artifact only",
    )
