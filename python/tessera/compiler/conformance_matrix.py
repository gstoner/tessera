"""Op×target conformance matrix — pure aggregator over existing truth sources.

Audit response (recommendation **A** in
``docs/audit/compiler/COMPILER_AUDIT.md``): expose a per-(op, target)
view of where each op is on the **seven-step proof ladder**:

    graph_emitted → schedule_legal → tile_legal → target_legal
    → backend_compile → runtime_execute → numerical_check

The point is to make the gap between *architecture-implied capability* and
*executable capability* explicit and drift-gated, so claims like "Tessera
supports matmul on NVIDIA" can be replaced with the seven concrete proof
columns that distinguish "Graph IR knows about it" from "the runtime can
actually launch a numerically-validated kernel today."

Strict no-duplicate-truth rule: this module **only reads** from existing
sources, never restates them. Sources:

* :mod:`tessera.compiler.primitive_coverage` — per-op 12-axis contract status
  (math/shape/dtype/vjp/jvp/batching/transpose/sharding/masking/lowering/
  backend_kernel/tests); ``metadata.graph_ir_lowering`` for the
  Graph IR registration status.
* :mod:`tessera.compiler.backend_manifest` — per-target kernel status
  (``fused`` / ``reference`` / ``compileable`` / ``artifact_only`` / ``planned``).
* :mod:`tessera.compiler.execution_matrix` — per-target runtime executors.
* :mod:`tessera.compiler.driver` ``_APPLE_GPU_{MPS,MSL,MPSGRAPH}_OPS`` —
  per-op Apple-GPU runtime envelope (what actually has an MPS/MSL/MPSGraph
  dispatcher).
* Filesystem — ``tests/unit/test_*.py`` filenames as a coarse signal for
  numerical-check presence.

Rendered to ``docs/audit/op_target_conformance.md``; drift-gated by
``tests/unit/test_op_target_conformance.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from tessera.compiler import backend_manifest as _bm
from tessera.compiler import execution_matrix as _em
from tessera.compiler import pipeline_gates as _pg
from tessera.compiler import primitive_coverage as _pc


# --- Proof status enum ----------------------------------------------------

#: Concrete numerical/lit-validated success on this (op, target).
PROOF_COMPLETE = "complete"
#: Works but with a known caveat — e.g. reference (correct but unoptimized),
#: composes from primitives instead of running as a fused single kernel, or a
#: contract axis still ``partial``.
PROOF_PARTIAL = "partial"
#: IR emits a target artifact (lit-clean) but no native backend compile /
#: launch path lights it up at runtime today.
PROOF_ARTIFACT_ONLY = "artifact_only"
#: Declared in the registry / manifest but not yet implemented.
PROOF_PLANNED = "planned"
#: Not declared on this target.
PROOF_MISSING = "missing"
#: Concept does not apply to this target (e.g. cooperative warp ops on CPU).
PROOF_NA = "not_applicable"

#: Order used for rendering and for the "weakest column wins" overall status.
_STATUS_ORDER = (
    PROOF_COMPLETE,
    PROOF_PARTIAL,
    PROOF_ARTIFACT_ONLY,
    PROOF_PLANNED,
    PROOF_MISSING,
    PROOF_NA,
)

_STATUS_SYMBOL = {
    PROOF_COMPLETE: "✅",
    PROOF_PARTIAL: "⚙️",
    PROOF_ARTIFACT_ONLY: "⚠️",
    PROOF_PLANNED: "📋",
    PROOF_MISSING: "❌",
    PROOF_NA: "➖",
}


def _weakest(*statuses: str) -> str:
    """Return the strictest (=lowest-rank) status. NA is treated as transparent."""
    materialized = [s for s in statuses if s != PROOF_NA]
    if not materialized:
        return PROOF_NA
    return max(materialized, key=lambda s: _STATUS_ORDER.index(s))


# --- Conformance ops + targets --------------------------------------------

@dataclass(frozen=True)
class ConformanceOp:
    """A row in the dashboard.

    ``component_ops`` lists the primitives the row is composed from. For a
    single-primitive row it has length 1; for a fused chain it lists every
    participant. The matrix uses this to decide whether a row needs a *fusion
    pass* (more than one component) or only the per-op kernel.
    """
    name: str
    component_ops: tuple[str, ...]
    fusion_targets: frozenset[str] = field(default_factory=frozenset)
    notes: str = ""


# The seven ops the audit called out, with realistic fusion expectations:
#   * matmul_softmax has a real fused MSL kernel on apple_gpu (see
#     docs/apple_backend.md). On other targets it is compose-only.
#   * matmul_relu is compose-only on every target today — there is no
#     matmul→relu fusion pass in any backend. Surfacing this distinction is
#     the point of the matrix.
CONFORMANCE_OPS: tuple[ConformanceOp, ...] = (
    ConformanceOp(
        name="matmul",
        component_ops=("matmul",),
    ),
    ConformanceOp(
        name="matmul_relu",
        component_ops=("matmul", "relu"),
        fusion_targets=frozenset(),  # no native fusion pass for matmul+relu today
        notes="composes from primitives; no fused single-kernel today",
    ),
    ConformanceOp(
        name="softmax",
        component_ops=("softmax",),
    ),
    ConformanceOp(
        name="matmul_softmax",
        component_ops=("matmul", "softmax"),
        fusion_targets=frozenset({"apple_gpu"}),
        notes="fused MSL kernel on apple_gpu (single-kernel scores); compose elsewhere",
    ),
    ConformanceOp(
        name="conv2d",
        component_ops=("conv2d",),
    ),
    ConformanceOp(
        name="flash_attn",
        component_ops=("flash_attn",),
    ),
    ConformanceOp(
        name="kv_cache_read",
        component_ops=("kv_cache_read",),
    ),
)

CONFORMANCE_TARGETS: tuple[str, ...] = (
    "cpu",
    "apple_cpu",
    "apple_gpu",
    "nvidia",
    "rocm",
)


# --- Proof cell -----------------------------------------------------------

@dataclass(frozen=True)
class ProofCell:
    op: str
    target: str
    graph_emitted: str
    schedule_legal: str
    tile_legal: str
    target_legal: str
    backend_compile: str
    runtime_execute: str
    numerical_check: str
    #: First failing gate from ``pipeline_gates.first_failing_gate``, or
    #: ``None`` if every gate passes / is non-blocking. The presence of a
    #: ``first_failing_gate`` is the *audit-named reason* the cell is not
    #: complete; the seven proof columns are the post-hoc breakdown.
    first_failing_gate: Optional[str] = None
    first_failing_gate_detail: str = ""
    notes: tuple[str, ...] = ()

    @property
    def overall(self) -> str:
        return _weakest(
            self.graph_emitted,
            self.schedule_legal,
            self.tile_legal,
            self.target_legal,
            self.backend_compile,
            self.runtime_execute,
            self.numerical_check,
        )


# --- Source-of-truth lookups (read-only) ----------------------------------

def _coverage_for(op: str):
    return _pc.all_primitive_coverages().get(op)


def _manifest_for_target(op: str, target: str) -> list[_bm.BackendKernelEntry]:
    """Return manifest entries for ``op`` that resolve to the dashboard
    target. NVIDIA aggregates sm80/sm90/sm100/sm120 into ``nvidia``."""
    out: list[_bm.BackendKernelEntry] = []
    for e in _bm.manifest_for(op):
        t = e.target
        if target == "nvidia" and t.startswith("nvidia_"):
            out.append(e)
        elif t == target:
            out.append(e)
    return out


def _best_status(entries: Iterable[_bm.BackendKernelEntry]) -> str | None:
    """Return the *best* status across multiple entries (e.g. sm80..sm120).
    "best" = lowest rank in _STATUS_ORDER."""
    statuses = {e.status for e in entries}
    if not statuses:
        return None
    for s in _STATUS_ORDER:
        if s in statuses:
            return s
    return next(iter(statuses))


def _execution_matrix_targets() -> set[str]:
    return {row.target for row in _em.all_rows()}


def _apple_gpu_envelope_ops() -> set[str]:
    """Apple-GPU ops whose **@jit→launch** path executes (without the ``tessera.``
    prefix).

    NOTE (2026-06-19): this deliberately reads only the lanes the standard
    ``runtime.launch`` path dispatches end-to-end — a launch-wired *proxy*, not
    the broader ``_APPLE_GPU_RUNTIME_OPS`` union. An op with only a *direct*
    dispatcher (no ``@jit→launch`` integration) stays out, because widening the
    proxy prematurely would flip its cell to ``complete`` while the conformance
    Evaluator (correctly) refuses to corroborate it at rung 7.

    UPDATE (2026-06-20): ``conv2d`` now executes through ``@jit→launch`` — the
    driver's executable gate accepts a single ``tessera.conv2d_nhwc`` op, the
    runtime per-op path dispatches it to the Metal conv lane, and provenance is
    honest (``native_gpu`` only when the Metal symbol ran; ``reference`` on host
    fallback). So ``_APPLE_GPU_CONV_OPS`` joins the proxy and the
    ``conv2d``/``apple_gpu`` cell's ``runtime_execute`` flips to complete,
    corroborated by ``conformance_evaluator`` (the generic Evaluator reaches
    HARDWARE_VERIFIED for conv2d on this host). conv3d stays out until its launch
    path is verified the same way.
    """
    from tessera.compiler import driver as _drv
    from tessera.compiler.apple_gpu_envelope import _APPLE_GPU_CONV_OPS

    out: set[str] = set()
    for attr in ("_APPLE_GPU_MPS_OPS", "_APPLE_GPU_MSL_OPS",
                 "_APPLE_GPU_MPSGRAPH_OPS"):
        for name in getattr(_drv, attr, ()):
            out.add(name[len("tessera."):] if name.startswith("tessera.") else name)
    # conv2d is launch-wired (2026-06-20); conv3d is not yet, so add conv2d only.
    for name in ("tessera.conv2d", "tessera.conv2d_nhwc"):
        if name in _APPLE_GPU_CONV_OPS:
            out.add(name[len("tessera."):])
    return out


_TARGET_KEYWORDS = {
    "cpu": ("cpu", "x86", "amx", "avx"),
    "apple_cpu": ("apple_cpu", "accelerate"),
    "apple_gpu": ("apple_gpu", "metal", "mps", "mpsgraph", "msl"),
    "nvidia": ("nvidia", "cuda", "sm80", "sm90", "sm100", "sm120",
               "wgmma", "tma", "tcgen05"),
    "rocm": ("rocm", "mfma", "hip"),
}


_TEST_TEXT_CACHE: dict[str, str] | None = None


def _test_text_index() -> dict[str, str]:
    """Cache of ``tests/unit/test_*.py`` filename → lowercased content
    (filename + body). Built once per process; the dashboard regenerates
    rarely enough that re-reading per call is wasteful."""
    global _TEST_TEXT_CACHE
    if _TEST_TEXT_CACHE is not None:
        return _TEST_TEXT_CACHE
    repo = Path(__file__).resolve().parents[3]
    test_dir = repo / "tests" / "unit"
    out: dict[str, str] = {}
    if test_dir.is_dir():
        for f in test_dir.glob("test_*.py"):
            try:
                body = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                body = ""
            out[f.name] = (f.name + "\n" + body).lower()
    _TEST_TEXT_CACHE = out
    return out


def _numerical_proof_source(op: str, target: str) -> Optional[str]:
    """How the ``numerical_check`` for ``(op, target)`` is satisfied:

    - ``"fixture"`` — a manifest-declared ``execute_compare_fixture`` that
      exists on disk. This is a **real** execute-and-compare proof and the
      only source that may justify a ``complete`` claim (the P0
      "claimed-complete must be proven" gate).
    - ``"heuristic"`` — the legacy filename/keyword scan: some
      ``tests/unit/test_*.py`` mentions both the op and the target. This is
      *circumstantial* — it keeps a cell from regressing to ``missing`` but
      can never, on its own, justify ``complete``.
    - ``None`` — no proof of any kind.

    **Audit follow-up A.3 (2026-05-31) — manifest-first.** The first-class
    answer is ``BackendKernelEntry.execute_compare_fixture``, surfaced via
    ``backend_manifest.manifest_for(op)`` and verified on disk. The keyword
    scan is the legacy fallback the fixture map is steadily subsuming.
    """
    # Step 1a — directly declared fixture in the numerical-fixtures map.
    # This is the canonical "a fixture is declared for this (op, target)"
    # test and is reachable even for *fused composite* rows (e.g.
    # ``matmul_softmax``) that have no standalone manifest ``BackendKernelEntry``
    # for ``_attach_numerical_fixtures`` to decorate.
    repo = Path(__file__).resolve().parents[3]
    direct = _bm._NUMERICAL_FIXTURES.get((op, target))
    if direct and (repo / direct).is_file():
        return "fixture"
    # Step 1b — manifest entry carrying a fixture (covers per-arch
    # nvidia/rocm rows aggregated by ``_manifest_for_target``, and
    # any entry that set the fixture via its constructor).
    for entry in _manifest_for_target(op, target):
        fixture = entry.execute_compare_fixture
        if fixture and (repo / fixture).is_file():
            return "fixture"
    # Step 2 — legacy heuristic for cells not yet covered by the manifest.
    target_keys = _TARGET_KEYWORDS.get(target, (target,))
    op_keys: tuple[str, ...] = (op,)
    if "_" in op:
        # matmul_softmax → also match "matmul" or "softmax" alone
        op_keys = op_keys + tuple(op.split("_"))
    for text in _test_text_index().values():
        if any(k in text for k in target_keys) and any(
            k in text for k in op_keys
        ):
            return "heuristic"
    return None


def _numerical_check_present(op: str, target: str) -> bool:
    """Whether ``(op, target)`` has *any* numerical-check signal (fixture or
    heuristic). Presence grants at most ``partial``; only a real fixture
    (``_numerical_proof_source == "fixture"``) may upgrade to ``complete``."""
    return _numerical_proof_source(op, target) is not None


# --- Per-cell deriver -----------------------------------------------------

def _proof_cell(op: ConformanceOp, target: str) -> ProofCell:
    components = op.component_ops
    notes: list[str] = []

    # --- graph_emitted: every component must have a coverage row. ---
    component_covers = [(c, _coverage_for(c)) for c in components]
    missing_components = [c for c, cov in component_covers if cov is None]
    if missing_components:
        graph_emitted = PROOF_MISSING
        notes.append(f"missing component(s): {','.join(missing_components)}")
    else:
        graph_emitted = PROOF_COMPLETE

    # --- schedule_legal: lowering_rule complete on every component. ---
    if graph_emitted == PROOF_MISSING:
        schedule_legal = PROOF_MISSING
    else:
        statuses = [cov.contract_status.get("lowering_rule", "planned")
                    for _, cov in component_covers]
        schedule_legal = _proof_status_from_axis(statuses)

    # --- tile_legal: graph_ir_lowering metadata == registered (or N/A). ---
    if graph_emitted == PROOF_MISSING:
        tile_legal = PROOF_MISSING
    else:
        gil = [cov.metadata.get("graph_ir_lowering", "missing")
               for _, cov in component_covers]
        tile_legal = _proof_status_from_graph_ir_lowering(gil)

    # --- target_legal: every component has a manifest entry on this target
    #     with a status that is not just 'planned'.
    if graph_emitted == PROOF_MISSING:
        target_legal = PROOF_MISSING
        backend_compile = PROOF_MISSING
    else:
        comp_target_statuses: list[str] = []
        for c in components:
            entries = _manifest_for_target(c, target)
            best = _best_status(entries)
            if best is None:
                comp_target_statuses.append("missing")
            else:
                comp_target_statuses.append(best)
        target_legal = _proof_status_from_target_legal(comp_target_statuses)
        backend_compile = _proof_status_from_backend_compile(comp_target_statuses)

    # --- Composition / fusion adjustment for multi-component rows ---
    if len(components) > 1:
        if target in op.fusion_targets:
            notes.append("fused single-kernel on this target")
        else:
            notes.append(
                "composes from per-op kernels (no fusion pass on this target)"
            )
        # Composition is a real end-to-end proof when every component has a
        # declared execute/compare fixture. Fusion is a performance property,
        # not a correctness prerequisite, so do not demote a fully compiled
        # chain merely because it launches more than one kernel.

    # --- runtime_execute: target must have an execution_matrix row, AND for
    #     CPU/Apple targets every component must be in the runtime envelope.
    em_targets = _execution_matrix_targets()
    if target not in em_targets:
        runtime_execute = PROOF_MISSING
    elif graph_emitted == PROOF_MISSING:
        runtime_execute = PROOF_MISSING
    else:
        runtime_execute = _proof_status_from_runtime(target, components)

    # --- numerical_check: presence grants partial; only a real declared
    #     execute_compare_fixture may upgrade to complete (P0 gate —
    #     "claimed-complete must be proven"). The legacy keyword heuristic is
    #     circumstantial and caps the cell at partial. ---
    proof_source = _numerical_proof_source(op.name, target)
    if proof_source is not None:
        numerical_check = PROOF_PARTIAL  # presence ≠ rigor; partial by default
        # Upgrade to complete only when (a) a real fixture proves it AND
        # (b) every component's registered tests row is itself 'complete'.
        if proof_source == "fixture" and graph_emitted == PROOF_COMPLETE:
            statuses = [cov.contract_status.get("tests", "planned")
                        for _, cov in component_covers]
            if all(s == "complete" for s in statuses):
                numerical_check = PROOF_COMPLETE
    else:
        numerical_check = PROOF_MISSING

    # Cross-reference the named pipeline gate (audit recommendation B). The
    # gate is evaluated per primary component op — for fused/compose chains
    # we report the gate of the first component, which is what the runtime
    # would actually hit first.
    gate_result = _pg.first_failing_gate(target, components[0])
    if gate_result is not None:
        first_failing_gate = gate_result.gate
        first_failing_gate_detail = gate_result.detail
    else:
        first_failing_gate = None
        first_failing_gate_detail = ""

    return ProofCell(
        op=op.name,
        target=target,
        graph_emitted=graph_emitted,
        schedule_legal=schedule_legal,
        tile_legal=tile_legal,
        target_legal=target_legal,
        backend_compile=backend_compile,
        runtime_execute=runtime_execute,
        numerical_check=numerical_check,
        first_failing_gate=first_failing_gate,
        first_failing_gate_detail=first_failing_gate_detail,
        notes=tuple(notes),
    )


def _proof_status_from_axis(statuses: list[str]) -> str:
    """Map registry axis statuses (complete/partial/planned/not_applicable)
    to proof statuses."""
    if any(s == "missing" for s in statuses):
        return PROOF_MISSING
    if all(s == "complete" for s in statuses):
        return PROOF_COMPLETE
    if all(s in ("complete", "not_applicable") for s in statuses):
        return PROOF_COMPLETE
    if any(s == "planned" for s in statuses):
        return PROOF_PLANNED
    return PROOF_PARTIAL


def _proof_status_from_graph_ir_lowering(statuses: list[str]) -> str:
    """Graph IR registration is a binary-ish proof (registered / missing /
    host_materialized / runtime_only / legacy not_applicable / stub_required)."""
    if any(s == "missing" for s in statuses):
        return PROOF_MISSING
    if all(s in ("registered", "host_materialized", "runtime_only",
                 "not_applicable") for s in statuses):
        return PROOF_COMPLETE
    if any(s == "stub_required" for s in statuses):
        return PROOF_PARTIAL
    return PROOF_PARTIAL


def _proof_status_from_target_legal(statuses: list[str]) -> str:
    """For target_legal we accept any non-planned, non-missing manifest entry
    (= the backend can at least emit IR/artifact for this target)."""
    if any(s == "missing" for s in statuses):
        return PROOF_MISSING
    if any(s == "planned" for s in statuses):
        return PROOF_PLANNED
    return PROOF_COMPLETE


def _proof_status_from_backend_compile(statuses: list[str]) -> str:
    """fused/reference/compileable/hardware_verified/packaged count as real
    compile paths; artifact_only is the audit's 'IR emits but no link/launch'
    state.

    Project 3 (2026-06-01): ``hardware_verified`` is the top rung
    (fused + numerical-proof fixture), and ``packaged`` is the PK5
    ``.mtlpackage`` lane — both ship executable kernels and so count
    as ``complete`` in this column."""
    if any(s == "missing" for s in statuses):
        return PROOF_MISSING
    if any(s == "planned" for s in statuses):
        return PROOF_PLANNED
    # All entries report a non-planned status. The weakest one wins.
    # ``compiled`` (2026-06-25) is a real compile+execute path — a
    # compiler-generated hsaco that runs via runtime.launch() (one rung below
    # ``hardware_verified``: no shipped C symbol) — so it counts here.
    if all(s in ("fused", "reference", "compileable",
                  "hardware_verified", "compiled", "packaged")
           for s in statuses):
        return PROOF_COMPLETE
    if any(s == "artifact_only" for s in statuses):
        return PROOF_ARTIFACT_ONLY
    return PROOF_PARTIAL


def _proof_status_from_runtime(target: str, components: tuple[str, ...]) -> str:
    """A runtime executor exists for this target. For Apple-GPU specifically
    we additionally require every component to be in the runtime envelope
    (since execution_matrix is target-level, not op-level)."""
    if target == "apple_gpu":
        envelope = _apple_gpu_envelope_ops()
        missing = [c for c in components if c not in envelope]
        if missing:
            return PROOF_MISSING
        return PROOF_COMPLETE
    if target == "apple_cpu":
        # apple_cpu_accelerate dispatches matmul/gemm natively; everything else
        # composes through the JIT-CPU fallback (correct but unoptimized).
        accel = {"matmul", "gemm", "batched_gemm"}
        if all(c in accel for c in components):
            return PROOF_COMPLETE
        return PROOF_PARTIAL  # composes via JIT-CPU fallback
    if target == "cpu":
        # The host x86/CPU lane has both native_cpu and jit_cpu_numpy execution
        # rows. A correct reference executor is still a real end-to-end runtime
        # path; optimization level is not a conformance rung.
        return PROOF_COMPLETE
    if target == "rocm":
        # An op executes on ROCm (gfx1151) iff every component ships a real
        # executing kernel: ``hardware_verified`` (a shipped C-ABI runtime_symbol,
        # e.g. matmul / flash_attn) or ``compiled`` (a compiler-generated hsaco
        # launched via runtime.launch()). Both carry an execute_compare fixture.
        # Otherwise there is no rocm runtime path (artifact_only MFMA emit only).
        # Capability-based, matching apple_gpu (the committed dashboard does not
        # depend on the regen host having a GPU).
        executes = {"hardware_verified", "compiled"}
        for c in components:
            statuses = {e.status for e in _manifest_for_target(c, "rocm")}
            if not (statuses & executes):
                return PROOF_MISSING
        return PROOF_COMPLETE
    return PROOF_MISSING


# --- Public API -----------------------------------------------------------

def build_matrix() -> list[ProofCell]:
    """Generate the full matrix as a flat list of proof cells, ordered by
    (op, target) per ``CONFORMANCE_OPS`` × ``CONFORMANCE_TARGETS``.

    Built under ``deterministic_host_for_dashboard`` so the committed dashboard
    is byte-identical whether regenerated on a Mac or the Linux CI runner — the
    apple ``hardware_smoke`` gate otherwise flips between ``—`` and
    ``Apple silicon required`` depending on the host.
    """
    out: list[ProofCell] = []
    with _pg.deterministic_host_for_dashboard():
        for op in CONFORMANCE_OPS:
            for tgt in CONFORMANCE_TARGETS:
                out.append(_proof_cell(op, tgt))
    return out


def _surfaced_upstream_gaps(cells: list[ProofCell]) -> list[dict[str, str]]:
    """Return cells whose `missing` overall status is caused by a known
    upstream gap rather than a real "this target can't do it" answer.

    Today we detect two patterns:

      1. Apple-GPU runtime envelope mentions the op (it has an MPS / MSL /
         MPSGraph dispatcher), but `backend_manifest` has no entry for it on
         `apple_gpu`. The runtime works; the manifest is stale.
      2. Apple-GPU has *any* component with a runtime envelope entry but no
         manifest entry — same pattern for fused chains.
    """
    out: list[dict[str, str]] = []
    envelope = _apple_gpu_envelope_ops()
    seen: set[tuple[str, str, str]] = set()
    for cell in cells:
        if cell.overall != PROOF_MISSING:
            continue
        if cell.target != "apple_gpu":
            continue
        op_obj = next(o for o in CONFORMANCE_OPS if o.name == cell.op)
        for comp in op_obj.component_ops:
            if comp not in envelope:
                continue
            entries = _manifest_for_target(comp, "apple_gpu")
            if entries:
                continue
            key = (cell.op, cell.target, comp)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "op": cell.op,
                "target": cell.target,
                "source": f"backend_manifest entry for {comp!r}",
                "fix": (
                    f"add an `apple_gpu` `BackendKernelEntry` for {comp!r}"
                    f" in `backend_manifest.py` (runtime envelope already"
                    f" dispatches it)"
                ),
            })
    return out


def status_summary() -> dict[str, int]:
    counts = {s: 0 for s in _STATUS_ORDER}
    for cell in build_matrix():
        counts[cell.overall] += 1
    return counts


#: Stable CSV column order for the conformance matrix — append-only.
CONFORMANCE_CSV_COLUMNS: tuple[str, ...] = (
    "op", "target", "overall",
    "graph_emitted", "schedule_legal", "tile_legal", "target_legal",
    "backend_compile", "runtime_execute", "numerical_check",
    "first_failing_gate",
)


def render_csv() -> str:
    """Render the canonical machine-readable conformance matrix.

    One row per (op, target), sorted by ``(op, target)``.  This is the
    drift-gated artifact; the Markdown is the human companion.
    """
    import csv as _csv
    import io as _io

    cells = sorted(build_matrix(), key=lambda c: (c.op, c.target))
    buf = _io.StringIO()
    writer = _csv.writer(buf, lineterminator="\n")
    writer.writerow(CONFORMANCE_CSV_COLUMNS)
    for c in cells:
        writer.writerow([
            c.op, c.target, c.overall,
            c.graph_emitted, c.schedule_legal, c.tile_legal, c.target_legal,
            c.backend_compile, c.runtime_execute, c.numerical_check,
            c.first_failing_gate or "",
        ])
    return buf.getvalue()


def render_markdown() -> str:
    cells = build_matrix()
    by_op: dict[str, list[ProofCell]] = {}
    for cell in cells:
        by_op.setdefault(cell.op, []).append(cell)
    summary = status_summary()

    lines: list[str] = []
    lines.append("<!-- AUTO-GENERATED — DO NOT EDIT BY HAND. -->")
    lines.append(
        "<!-- Regenerate via: python -m tessera.cli.conformance_matrix"
        " --render -->"
    )
    lines.append("")
    lines.append("# Op×Target Conformance Matrix")
    lines.append("")
    lines.append(
        "This dashboard reports, per (op, target), where the op is on the"
        " seven-step proof ladder:"
    )
    lines.append("")
    lines.append(
        "  `graph_emitted` → `schedule_legal` → `tile_legal` →"
        " `target_legal` → `backend_compile` → `runtime_execute` →"
        " `numerical_check`"
    )
    lines.append("")
    lines.append(
        "The matrix is a **pure aggregator** over"
        " `primitive_coverage` (12-axis contracts), `backend_manifest`"
        " (per-target kernel status), `execution_matrix` (runtime"
        " executors), and the Apple-GPU runtime envelope sets. No proof"
        " column has its own private truth source — change the upstream"
        " status and the matrix regenerates."
    )
    lines.append("")
    lines.append(
        "Audit response to"
        " [docs/audit/compiler/COMPILER_AUDIT.md](compiler/COMPILER_AUDIT.md)"
        " recommendation **A**: the gap between *architecture-implied"
        " capability* and *executable capability* is now"
        " drift-gated rather than implicit."
    )
    lines.append("")

    # Legend
    lines.append("## Status legend")
    lines.append("")
    lines.append("| Symbol | Status | Meaning |")
    lines.append("|--------|--------|---------|")
    legend = [
        (PROOF_COMPLETE,
         "Real path lit up end-to-end on this target."),
        (PROOF_PARTIAL,
         "Works but with a known caveat (reference / composes / contract"
         " axis partial)."),
        (PROOF_ARTIFACT_ONLY,
         "IR emits a target artifact; no native compile / link / launch"
         " path yet (hardware-gated)."),
        (PROOF_PLANNED,
         "Declared in the registry / manifest, not yet implemented."),
        (PROOF_MISSING,
         "Not declared on this target."),
        (PROOF_NA,
         "Concept does not apply to this target."),
    ]
    for status, meaning in legend:
        lines.append(
            f"| {_STATUS_SYMBOL[status]} | `{status}` | {meaning} |"
        )
    lines.append("")

    # Top-level summary by overall status
    lines.append("## Overall counts")
    lines.append("")
    lines.append("| Overall (weakest column wins) | Count |")
    lines.append("|---|---:|")
    total = 0
    for s in _STATUS_ORDER:
        c = summary.get(s, 0)
        if c == 0 and s == PROOF_NA:
            continue
        lines.append(f"| {_STATUS_SYMBOL[s]} `{s}` | {c} |")
        total += c
    lines.append(f"| **total cells** | **{total}** |")
    lines.append("")

    # Surfaced upstream gaps — cells whose overall is `missing` because an
    # *upstream* truth source is incomplete (e.g. the runtime envelope lists
    # the op on this target but the backend_manifest does not). These are
    # actionable follow-ups: fix the upstream, regenerate, ratchet improves.
    surfaced = _surfaced_upstream_gaps(cells)
    if surfaced:
        lines.append("## Surfaced upstream gaps")
        lines.append("")
        lines.append(
            "These cells are `missing` because the upstream truth source is"
            " incomplete, not because the path doesn't exist. Each row is an"
            " actionable follow-up: fix the upstream entry and the matrix"
            " regenerates cleanly."
        )
        lines.append("")
        lines.append("| Op | Target | Upstream source | Fix |")
        lines.append("|----|--------|-----------------|-----|")
        for gap in surfaced:
            lines.append(
                f"| `{gap['op']}` | `{gap['target']}` | `{gap['source']}`"
                f" | {gap['fix']} |"
            )
        lines.append("")

    # Per-op section
    for op in CONFORMANCE_OPS:
        lines.append(f"## `{op.name}`")
        if op.notes:
            lines.append("")
            lines.append(f"_{op.notes}_")
        lines.append("")
        if len(op.component_ops) > 1:
            comps = ", ".join(f"`{c}`" for c in op.component_ops)
            fusion_targets = sorted(op.fusion_targets) or ["—"]
            lines.append(
                f"**Composition:** {comps}.  Fused-single-kernel targets:"
                f" {', '.join(fusion_targets)}."
            )
            lines.append("")
        header = ("| target | overall | graph | schedule | tile |"
                  " target_legal | backend_compile | runtime | numerical |"
                  " first failing gate (B) | notes |")
        sep = ("|--------|---------|-------|----------|------|"
               "--------------|-----------------|---------|-----------|"
               "------------------------|-------|")
        lines.append(header)
        lines.append(sep)
        for cell in by_op[op.name]:
            sym = lambda s: _STATUS_SYMBOL[s]  # noqa: E731
            note = "; ".join(cell.notes) if cell.notes else ""
            if cell.first_failing_gate is None:
                gate_cell = "—"
            else:
                detail = cell.first_failing_gate_detail
                if len(detail) > 60:
                    detail = detail[:57] + "…"
                gate_cell = f"`{cell.first_failing_gate}` — {detail}"
            lines.append(
                f"| `{cell.target}` |"
                f" {sym(cell.overall)} |"
                f" {sym(cell.graph_emitted)} |"
                f" {sym(cell.schedule_legal)} |"
                f" {sym(cell.tile_legal)} |"
                f" {sym(cell.target_legal)} |"
                f" {sym(cell.backend_compile)} |"
                f" {sym(cell.runtime_execute)} |"
                f" {sym(cell.numerical_check)} |"
                f" {gate_cell} |"
                f" {note} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


__all__ = [
    "PROOF_COMPLETE",
    "PROOF_PARTIAL",
    "PROOF_ARTIFACT_ONLY",
    "PROOF_PLANNED",
    "PROOF_MISSING",
    "PROOF_NA",
    "ConformanceOp",
    "ProofCell",
    "CONFORMANCE_OPS",
    "CONFORMANCE_TARGETS",
    "build_matrix",
    "status_summary",
    "render_markdown",
]
