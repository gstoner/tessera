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


# ── DESIL — cross-path differential oracle ───────────────────────────────────
#
# Run the SAME program through two+ independent executable lowering paths
# (apple_gpu/Metal, apple_cpu/Accelerate, cpu/JIT) and require they agree. No
# external reference (the paths cross-check each other) and it exercises distinct
# compilers, so a miscompile in any single lowering path is caught — DESIL's
# "differential across lowering paths" realized via Tessera's multiple backends.


@dataclass(frozen=True)
class CrossPathVerdict:
    """Agreement of one program across independent lowering paths."""

    relation: str                # "equivalent" | "divergent" | "inconclusive"
    paths: tuple[str, ...]       # the paths that ran natively and were compared
    max_abs_err: float | None
    detail: str = ""

    @property
    def is_divergent(self) -> bool:
        return self.relation == "divergent"


def _cross_path_relation(
    n_native: int, max_abs_err: float | None, *, tol: float
) -> tuple[str, str]:
    """Pure classifier for the cross-path verdict (portably unit-testable)."""
    if n_native < 2:
        return "inconclusive", (
            f"only {n_native} path(s) ran natively — need ≥2 to cross-check"
        )
    if max_abs_err is None:
        return "inconclusive", "paths produced incomparable outputs (shape/non-finite)"
    if max_abs_err <= tol:
        return "equivalent", f"paths agree (max_abs_err={max_abs_err:.3e} ≤ {tol:.1e})"
    return "divergent", (
        f"lowering paths DISAGREE (max_abs_err={max_abs_err:.3e} > {tol:.1e}) — "
        "a miscompile in one path"
    )


def cross_path_equivalence(
    paths: list[tuple[str, Any]],
    args: tuple[Any, ...],
    *,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> CrossPathVerdict:
    """Assert a program agrees across independent lowering paths.

    ``paths`` is ``[(target, jitted_fn), ...]`` for the *same* computation (each
    jitted for its target). Only natively-executed paths are compared; the
    verdict is ``inconclusive`` unless ≥2 ran natively.
    """
    import numpy as np

    outs: list[tuple[str, Any]] = []
    for target, fn in paths:
        out, native = run_native(target, fn, args)
        if native and out is not None:
            arr = np.asarray(out, dtype=np.float64)
            if np.all(np.isfinite(arr)):
                outs.append((target, arr))

    max_abs_err: float | None = None
    ref_scale = 1.0
    if len(outs) >= 2:
        ref = outs[0][1]
        errs = [
            float(np.max(np.abs(o - ref)))
            for _, o in outs[1:]
            if o.shape == ref.shape
        ]
        if len(errs) == len(outs) - 1:           # every path was comparable
            max_abs_err = max(errs)
            ref_scale = float(np.max(np.abs(ref))) or 1.0

    tol = atol + rtol * ref_scale
    relation, detail = _cross_path_relation(len(outs), max_abs_err, tol=tol)
    return CrossPathVerdict(relation, tuple(t for t, _ in outs), max_abs_err, detail)


# ── E2: opt-level checksum oracle (DESIL checksum-across-opt-levels) ──────────
#
# A correct compiler produces the same result for the same program regardless of
# optimization level. DESIL (OOPSLA'25) exploits this: compile one program at two
# opt levels, run both, compare a stable checksum — a divergence is a miscompile
# at one level. Here the "opt levels" are two independent lowerings of the same
# math (e.g. fusion-on vs fusion-off, autotune variant A vs B). The checksum is
# tolerance-rounded so benign float-reordering across fusions doesn't false-alarm,
# while a real divergence (a dropped term, a wrong tile) still trips it.


def opt_level_checksum(value: Any, *, decimals: int = 4) -> int:
    """A stable integer checksum of an array, robust to benign float reordering.

    Rounds to ``decimals`` then sums the scaled integers — identical for two
    numerically-equivalent lowerings, divergent for a real miscompile. The sum is
    accumulated in Python ``int`` (arbitrary precision) so a large / high-magnitude
    tensor cannot silently wrap an int64 accumulator into a spurious mismatch."""
    import numpy as np

    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return -1  # non-finite output is never "stable"
    scaled = np.rint(arr * (10 ** decimals))
    # int(np.float64) is exact for |x| < 2^53; fold via Python ints to avoid the
    # int64 overflow np.sum() would hit on large tensors.
    return int(sum(int(x) for x in scaled.ravel().tolist()))


@dataclass(frozen=True)
class OptLevelVerdict:
    """Agreement of one program's checksum across optimization levels."""

    relation: str                 # "stable" | "divergent" | "inconclusive"
    levels: tuple[str, ...]        # the opt levels that ran natively
    checksums: tuple[int, ...]
    detail: str = ""

    @property
    def is_stable(self) -> bool:
        return self.relation == "stable"


def opt_level_equivalence(
    variants: list[tuple[str, Any]],
    args: tuple[Any, ...],
    *,
    decimals: int = 4,
) -> OptLevelVerdict:
    """Assert a program checksums identically across optimization levels.

    ``variants`` is ``[(level_name, jitted_fn), ...]`` — the *same* computation
    lowered at different opt levels (fusion on/off, autotune A/B). Only natively-
    executed variants are compared, and only against others that ran on the **same
    backend**: comparing an apple_gpu lowering to a cpu one would conflate a
    backend float difference with an opt-level miscompile. ``inconclusive`` unless
    ≥2 variants ran natively on a single backend. A differing checksum within one
    backend is a miscompile at one opt level (rung 5 CODEGEN_STABLE).
    """
    ran: list[tuple[str, str, int]] = []  # (level, backend, checksum)
    for level, fn in variants:
        out, backend = None, None
        for cand in ("apple_gpu", "cpu"):
            try:
                o, native = run_native(cand, fn, args)
            except Exception:
                continue  # this backend can't build/run the variant — try the next
            if native and o is not None:
                out, backend = o, cand
                break
        if out is not None and backend is not None:
            ran.append((level, backend, opt_level_checksum(out, decimals=decimals)))

    # Cross-check only within the largest same-backend group.
    by_backend: dict[str, list[tuple[str, int]]] = {}
    for level, backend, chk in ran:
        by_backend.setdefault(backend, []).append((level, chk))
    group_backend, group = max(
        by_backend.items(), key=lambda kv: len(kv[1]), default=(None, []))

    if len(group) < 2:
        return OptLevelVerdict(
            "inconclusive", tuple(l for l, _ in group), tuple(c for _, c in group),
            f"only {len(group)} variant(s) ran natively on a single backend "
            f"({group_backend}) — need ≥2 on the same backend to cross-check")
    checks = [c for _, c in group]
    relation = "stable" if len(set(checks)) == 1 else "divergent"
    detail = (f"checksums agree across {len(group)} opt levels on "
              f"{group_backend} ({checks[0]})" if relation == "stable" else
              f"opt levels DISAGREE on {group_backend}: {dict(group)} — "
              "a miscompile at one level")
    return OptLevelVerdict(relation, tuple(l for l, _ in group), tuple(checks),
                           detail)


# ── DESIL at the KV-ABI level — PagedKVState differential oracle ─────────────
#
# Workstream A. The same logical KV sequence held by independent substrates
# (contiguous KVCacheHandle, tiered cap=1, tiered cap=all) must produce identical
# attention. Two invariants in one verdict: (1) a paged layout ≡ a contiguous one,
# and (2) the prefetch/residency *schedule* must not change numerics — a
# miscompiled PagedAttentionLoweringPass that, say, gathered a stale page would
# diverge here. Reuses the cross-path classifier so it speaks the evaluator's
# vocabulary. See docs/audit/roadmap/CONTRACT_PASS_PLAN.md (Workstream A).


def paged_kv_equivalence(
    states: list[tuple[str, Any]],
    Q: Any,
    *,
    causal: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> CrossPathVerdict:
    """Assert ``paged_attention(Q, state)`` agrees across independent KV substrates.

    ``states`` is ``[(name, kv_state), ...]`` — each a different physical layout
    (contiguous / tiered with varying residency capacity / latent) holding the
    *same* logical sequence. The verdict is ``inconclusive`` with <2 states.
    """
    import numpy as np

    from ..cache.paged_kv import paged_attention

    outs: list[tuple[str, Any]] = []
    for name, state in states:
        out = np.asarray(paged_attention(Q, state, causal=causal), dtype=np.float64)
        if np.all(np.isfinite(out)):
            outs.append((name, out))

    max_abs_err: float | None = None
    ref_scale = 1.0
    if len(outs) >= 2:
        ref = outs[0][1]
        errs = [float(np.max(np.abs(o - ref))) for _, o in outs[1:]
                if o.shape == ref.shape]
        if len(errs) == len(outs) - 1:
            max_abs_err = max(errs)
            ref_scale = float(np.max(np.abs(ref))) or 1.0

    tol = atol + rtol * ref_scale
    relation, detail = _cross_path_relation(len(outs), max_abs_err, tol=tol)
    return CrossPathVerdict(relation, tuple(n for n, _ in outs), max_abs_err, detail)


def paged_kv_native_equivalence(
    kv_state: Any,
    Q: Any,
    *,
    causal: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> CrossPathVerdict:
    """Native rung for the PagedKVState ABI (#8): the Apple-GPU paged-attention
    path must run on ``metal_runtime`` **and** agree with the numpy reference.

    Two genuinely-independent lowering paths over the same staged KV — the Metal
    fused matmul→softmax→matmul kernel vs numpy — cross-check each other (DESIL).
    The verdict is ``inconclusive`` unless the GPU path actually fired (provenance
    gate: a silent fallback cannot earn the native rung), and ``divergent`` if the
    two paths disagree (a Metal miscompile).
    """
    import numpy as np

    from ..cache.paged_kv import paged_attention

    ref = np.asarray(paged_attention(Q, kv_state, causal=causal,
                                     backend="reference"), dtype=np.float64)
    gpu_out, exe = paged_attention(Q, kv_state, causal=causal,
                                   backend="apple_gpu", return_execution=True)
    native_ran = (exe == "metal_runtime")
    gpu = np.asarray(gpu_out, dtype=np.float64)

    if not native_ran:
        return CrossPathVerdict(
            "inconclusive", ("reference",), None,
            f"apple_gpu paged attention fell back ({exe!r}) — native rung unearned")
    max_abs_err = float(np.max(np.abs(gpu - ref))) if gpu.shape == ref.shape else None
    ref_scale = float(np.max(np.abs(ref))) or 1.0
    tol = atol + rtol * ref_scale
    relation, detail = _cross_path_relation(2, max_abs_err, tol=tol)
    return CrossPathVerdict(relation, ("apple_gpu:metal_runtime", "reference"),
                            max_abs_err, detail)


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


def rocm_emission_verdict(arch: str = "gfx1151", dtype: str = "f16") -> BackendVerdict:
    """Derive an AMD program's emission rung from `rocdl_emit`.

    Unlike NVIDIA (where ``ptxas`` is Linux-CI-only, capping the host at rung 3
    ``EMITS_ASM_TEXT``), the LLVM 22 **AMDGPU backend ships in the host toolchain**,
    so ``llc -mcpu=<gfx>`` lowers ``llvm.amdgcn.wmma.*`` to a real ``v_wmma_*``
    instruction *here* — a genuine rung-4 ``ASSEMBLES`` with no GPU. Rungs 6–7
    (execute / hardware-verified) still gate on real silicon.

    Ladder:
      * ``ASSEMBLES`` — ``llc`` lowered the WMMA GEMM IR to AMDGCN containing the
        ``v_wmma_*`` instruction on this host.
      * ``EMITS_ASM_TEXT`` — IR emitted + structurally valid, but ``llc`` absent.
      * ``ARTIFACT_ONLY`` — neither.
    """
    target = f"rocm:{arch}"
    try:
        from . import rocdl_emit as _r
        ir = _r.emit_wmma_gemm_llvmir(dtype, arch=arch)
        validation = _r.validate_wmma_gemm_structure(ir, dtype=dtype, arch=arch)
    except Exception as exc:
        return BackendVerdict(
            target=target, rung=Rung.ARTIFACT_ONLY, execution_kind="none",
            runtime_status="invalid_artifact", provenance_ok=False,
            correctness="unproven", detail=f"emission failed: {exc!r}")

    if not validation.ok:
        return BackendVerdict(
            target=target, rung=Rung.ARTIFACT_ONLY, execution_kind="none",
            runtime_status="invalid_artifact", provenance_ok=False,
            correctness="unproven",
            detail=f"WMMA IR structurally invalid: {validation.reasons}")

    res = _r.llc_assemble(ir, arch=arch)
    if getattr(res, "status", "") == "ok" and getattr(res, "wmma_instruction", ""):
        return BackendVerdict(
            target=target, rung=Rung.ASSEMBLES,
            execution_kind="amdgcn_assembled", runtime_status="assembled",
            provenance_ok=False, correctness="unproven",
            detail=(f"llc lowered WMMA GEMM to {arch} AMDGCN on this host: "
                    f"{res.wmma_instruction!r}; execution (rungs 6-7) gates on "
                    "real silicon"))
    return BackendVerdict(
        target=target, rung=Rung.EMITS_ASM_TEXT,
        execution_kind="amdgcn_emitted", runtime_status="artifact_only",
        provenance_ok=False, correctness="unproven",
        detail="emits structurally-valid llvm.amdgcn.wmma LLVM-IR; llc assembly "
               "unavailable on this host")


# ── E2: legal-by-construction inputs (DESIL UB-elim / NNSmith safe inputs) ────
#
# Generated-program inputs must be legal by construction so a tolerance compare
# isn't dominated by NaN/Inf garbage and optimizers' non-UB assumptions hold.


def safe_input(kind: str, shape: tuple[int, ...], rng: Any) -> Any:
    """A finite, in-domain f32 input for one op family. ``kind``:
    ``real`` (bounded, no overflow), ``positive`` (>0 — log/sqrt/rsqrt domain),
    ``nonzero`` (away from 0 — division denominators), ``unit`` (bounded ~[-1,1])."""
    import numpy as np

    if kind == "real":
        return (rng.standard_normal(shape) / 4).astype(np.float32)
    if kind == "positive":
        return (np.abs(rng.standard_normal(shape)) + 0.5).astype(np.float32)
    if kind == "nonzero":
        x = (rng.standard_normal(shape) / 4).astype(np.float32)
        x[np.abs(x) < 0.1] = 0.1
        return x
    if kind == "unit":
        return (rng.standard_normal(shape).clip(-3, 3) / 3).astype(np.float32)
    raise ValueError(f"unknown safe-input kind {kind!r}")


# ── E2: metamorphic-equivalence oracle (algebraic invariants) ────────────────
#
# A reference-free relation oracle: ``output_map(fn(args_a)) ≡ fn(args_b)`` on
# the same backend, for an algebraic invariant the compiler MUST preserve (e.g.
# softmax shift-invariance). Catches numerical/algebraic miscompiles the vertical
# numpy oracle can mask (both sides run on the real backend, no external ref).


def metamorphic_equivalence(
    target: str,
    fn: Any,
    args_a: tuple[Any, ...],
    args_b: tuple[Any, ...],
    *,
    output_map: Callable[[Any], Any] | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> HorizontalVerdict:
    """Assert ``output_map(fn(args_a)) ≡ fn(args_b)`` natively on ``target``.

    Example (softmax shift-invariance): ``args_a=(x,)``, ``args_b=(x+c,)``,
    ``output_map=None`` (identity) — softmax(x) must equal softmax(x+c).
    ``inconclusive`` unless both runs are native.
    """
    import numpy as np

    out_a, na = run_native(target, fn, args_a)
    out_b, nb = run_native(target, fn, args_b)

    max_abs_err: float | None = None
    ref_scale = 1.0
    if na and nb and out_a is not None and out_b is not None:
        a = np.asarray(out_a, dtype=np.float64)
        if output_map is not None:
            a = np.asarray(output_map(a), dtype=np.float64)
        b = np.asarray(out_b, dtype=np.float64)
        if a.shape == b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            max_abs_err = float(np.max(np.abs(a - b)))
            ref_scale = float(np.max(np.abs(b))) or 1.0

    tol = atol + rtol * ref_scale
    relation, detail = _horizontal_relation(na, nb, max_abs_err, tol=tol)
    return HorizontalVerdict(target, relation, max_abs_err, detail)
