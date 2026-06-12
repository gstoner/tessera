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

from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class Rung(IntEnum):
    """Per-backend status ladder (EVALUATOR_PLAN.md §2). Higher = stronger,
    and the integer order is meaningful (``verdict.rung >= Rung.EXECUTES``)."""

    ARTIFACT_ONLY = 1       # IR emitted
    LOWERS_CLEAN = 2        # backend pipeline lowered the program, no diagnostics
    ASSEMBLES = 3           # emitted PTX/AMDGCN actually assembles (ptxas/hipcc)
    CODEGEN_STABLE = 4      # same IR @ two opt levels → structurally-equivalent code
    NUMERICAL_SYMBOLIC = 5  # microkernel ≡ reference via finite-field / SMT
    EXECUTES = 6            # ran on the demanded backend (real silicon)
    HARDWARE_VERIFIED = 7   # ran AND oracle-matched on real silicon


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
        # fallback, so we make NO execution/correctness claim about this backend
        # — it caps at "lowered" regardless of numerical agreement.
        return BackendVerdict(
            target=target, rung=Rung.LOWERS_CLEAN,
            execution_kind=execution_kind, runtime_status=runtime_status,
            provenance_ok=False, correctness="unproven",
            detail=f"did not execute natively on {target} "
                   f"(kind={execution_kind!r}, status={runtime_status!r}); "
                   "silent fallback cannot earn an execution rung",
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
