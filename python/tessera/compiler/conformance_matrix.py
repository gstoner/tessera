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

Proof sources:

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
* The typed Graph/Schedule/Tile/Target lowering stack — actual emitted modules
  and verifier results for every curated program and exact target.
* Exact-target execute-and-compare fixtures — numerical proof; keyword scans
  are deliberately excluded.

Rendered to ``docs/audit/op_target_conformance.md``; drift-gated by
``tests/unit/test_op_target_conformance.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from functools import cache
from typing import Iterable, Optional

from tessera.compiler import backend_manifest as _bm
from tessera.compiler import conformance_evaluator as _ce
from tessera.compiler import execution_matrix as _em
from tessera.compiler import primitive_coverage as _pc
from tessera.compiler.schedule_ir import lower_graph_to_schedule_ir
from tessera.compiler.tile_ir import lower_schedule_to_tile_ir
from tessera.compiler.target_ir import lower_tile_to_target_ir


# --- Proof status enum ----------------------------------------------------

#: Concrete numerical/lit-validated success on this (op, target).
PROOF_COMPLETE = "complete"
#: Correct reference execution exists, but no target-native compile is claimed.
PROOF_REFERENCE = "reference"
#: The pinned compiler accepts the artifact; runtime execution is not proven.
PROOF_COMPILEABLE = "compileable"
#: Evidence exists but does not satisfy the rung's full contract.
PROOF_PARTIAL = "partial"
#: IR emits a target artifact but concrete backend compilation is absent.
PROOF_ARTIFACT_ONLY = "artifact_only"
#: Declared in the registry / manifest but not yet implemented.
PROOF_PLANNED = "planned"
#: Evidence required by this rung is absent.
PROOF_MISSING = "missing"
#: Concept does not apply to this target (e.g. cooperative warp ops on CPU).
PROOF_NA = "not_applicable"

#: Order used for rendering and for the "weakest column wins" overall status.
_STATUS_ORDER = (
    PROOF_COMPLETE,
    PROOF_REFERENCE,
    PROOF_COMPILEABLE,
    PROOF_PARTIAL,
    PROOF_ARTIFACT_ONLY,
    PROOF_PLANNED,
    PROOF_MISSING,
    PROOF_NA,
)

_STATUS_SYMBOL = {
    PROOF_COMPLETE: "✅",
    PROOF_REFERENCE: "🧪",
    PROOF_COMPILEABLE: "🔧",
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
#     docs/backends/apple/). On other targets it is compose-only.
#   * matmul_relu is fused on nvidia_sm120 through the canonical Tile
#     accumulator epilogue; other targets retain their independently proven
#     composition/fusion status.
CONFORMANCE_OPS: tuple[ConformanceOp, ...] = (
    ConformanceOp(
        name="matmul",
        component_ops=("matmul",),
    ),
    ConformanceOp(
        name="matmul_relu",
        component_ops=("matmul", "relu"),
        fusion_targets=frozenset({"nvidia_sm120"}),
        notes="fused Tile accumulator epilogue on nvidia_sm120; composes elsewhere",
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
    "x86",
    "apple_cpu",
    "apple_gpu",
    "rocm",
    "nvidia_sm80",
    "nvidia_sm90",
    "nvidia_sm100",
    "nvidia_sm120",
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
    """Return exact-target manifest entries for ``op``.

    Architecture families are never merged here. Family summaries are derived
    only after architecture-grain cells have been built.
    """
    return [e for e in _bm.manifest_for(op) if e.target == target]


def _best_status(entries: Iterable[_bm.BackendKernelEntry]) -> str | None:
    """Return the strongest concrete backend status for one exact target."""
    statuses = {e.status for e in entries}
    if not statuses:
        return None
    backend_order = (
        "device_verified_abi", "device_verified_jit", "packaged", "fused", "reference",
        "compileable", "artifact_only", "planned",
    )
    for s in backend_order:
        if s in statuses:
            return s
    raise ValueError(f"unknown backend status(es): {sorted(statuses)}")


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


def _numerical_proof_source(op: str, target: str) -> Optional[str]:
    """How the ``numerical_check`` for ``(op, target)`` is satisfied:

    - ``"fixture"`` — a manifest-declared ``execute_compare_fixture`` that
      exists on disk. This is a **real** execute-and-compare proof and the
      only source that may justify a ``complete`` claim (the P0
      "claimed-complete must be proven" gate).
    - ``None`` — no proof of any kind.

    **Audit follow-up A.3 (2026-07-12) — exact-target fixtures.** The only
    accepted evidence is an ``execute_compare_fixture`` declared for the
    exact ``(op, target)`` grain and verified on disk. There is deliberately
    no filename or keyword heuristic fallback.
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
    # Step 1b — exact-target manifest entry carrying a fixture.
    for entry in _manifest_for_target(op, target):
        fixture = entry.execute_compare_fixture
        if fixture and (repo / fixture).is_file():
            return "fixture"
    return None


def _numerical_check_present(op: str, target: str) -> bool:
    """Whether exact-target execute-and-compare evidence exists."""
    return _numerical_proof_source(op, target) is not None


@dataclass(frozen=True)
class IRProof:
    """Verifier-derived proof for the four emitted IR rungs."""

    graph_emitted: str
    schedule_legal: str
    tile_legal: str
    target_legal: str
    detail: str = ""


@cache
def _ir_proof(op: str, target: str) -> IRProof:
    """Compile the curated program and run the typed verifier at every rung.

    A registry row is not evidence. The rung completes only when the compiler
    emits that artifact and its in-process IR verifier accepts the typed module.
    """
    try:
        fn = _ce._jitted(op, target)
        bundle = fn.compile_bundle
        graph = fn.graph_ir
    except Exception as exc:  # noqa: BLE001 - audit converts failures to evidence
        return IRProof(PROOF_MISSING, PROOF_MISSING, PROOF_MISSING,
                       PROOF_MISSING, f"compile failed: {exc}")

    graph_result = graph.verify()
    if bundle.graph is None or not graph_result.ok:
        return IRProof(PROOF_MISSING, PROOF_MISSING, PROOF_MISSING,
                       PROOF_MISSING, graph_result.format())

    if bundle.schedule is None:
        return IRProof(PROOF_COMPLETE, PROOF_MISSING, PROOF_MISSING,
                       PROOF_MISSING, "compiler emitted no Schedule IR")
    schedule = lower_graph_to_schedule_ir(graph, target_kind=target)
    schedule_result = schedule.verify()
    if not schedule_result.ok:
        return IRProof(PROOF_COMPLETE, PROOF_PARTIAL, PROOF_MISSING,
                       PROOF_MISSING, schedule_result.format())

    if bundle.tile is None:
        return IRProof(PROOF_COMPLETE, PROOF_COMPLETE, PROOF_MISSING,
                       PROOF_MISSING, "compiler emitted no Tile IR")
    tile = lower_schedule_to_tile_ir(schedule, target_kind=target)
    tile_result = tile.verify()
    if not tile_result.ok:
        return IRProof(PROOF_COMPLETE, PROOF_COMPLETE, PROOF_PARTIAL,
                       PROOF_MISSING, tile_result.format())

    if bundle.target_ir is None:
        return IRProof(PROOF_COMPLETE, PROOF_COMPLETE, PROOF_COMPLETE,
                       PROOF_MISSING, "compiler emitted no Target IR")
    target_ir = lower_tile_to_target_ir(tile, target_kind=target)
    target_result = target_ir.verify()
    if not target_result.ok:
        return IRProof(PROOF_COMPLETE, PROOF_COMPLETE, PROOF_COMPLETE,
                       PROOF_PARTIAL, target_result.format())
    return IRProof(PROOF_COMPLETE, PROOF_COMPLETE, PROOF_COMPLETE,
                   PROOF_COMPLETE)


# --- Per-cell deriver -----------------------------------------------------

def _proof_cell(op: ConformanceOp, target: str) -> ProofCell:
    components = op.component_ops
    notes: list[str] = []

    # Registry coverage is used only to validate component identity and the
    # generic tests contract. IR rung status comes from actual compilation and
    # typed verifier results below.
    component_covers = [(c, _coverage_for(c)) for c in components]
    missing_components = [c for c, cov in component_covers if cov is None]
    if missing_components:
        notes.append(f"missing component(s): {','.join(missing_components)}")
    ir = (_ir_proof(op.name, target) if not missing_components else
          IRProof(PROOF_MISSING, PROOF_MISSING, PROOF_MISSING, PROOF_MISSING,
                  "component missing from primitive registry"))
    graph_emitted = ir.graph_emitted
    schedule_legal = ir.schedule_legal
    tile_legal = ir.tile_legal
    target_legal = ir.target_legal
    if ir.detail:
        notes.append(ir.detail)

    comp_target_statuses: list[str] = []
    for component in components:
        best = _best_status(_manifest_for_target(component, target))
        comp_target_statuses.append(best or "missing")
    backend_compile = _proof_status_from_backend_compile(
        comp_target_statuses, op.name, target
    )

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

    runtime_execute = _proof_status_from_runtime(
        op.name, target, components, comp_target_statuses
    )

    # --- numerical_check: only an exact-target execute_compare_fixture counts.
    proof_source = _numerical_proof_source(op.name, target)
    numerical_check = (PROOF_COMPLETE if proof_source == "fixture"
                       else PROOF_MISSING)

    proof_axes = (
        ("graph_emitted", graph_emitted),
        ("schedule_legal", schedule_legal),
        ("tile_legal", tile_legal),
        ("target_legal", target_legal),
        ("backend_compile", backend_compile),
        ("runtime_execute", runtime_execute),
        ("numerical_check", numerical_check),
    )
    first = next(((name, status) for name, status in proof_axes
                  if status != PROOF_COMPLETE), None)
    if first is None:
        first_failing_gate = None
        first_failing_gate_detail = ""
    else:
        first_failing_gate, status = first
        first_failing_gate_detail = (
            f"{first_failing_gate}={status}; components={','.join(components)}"
        )

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


def _proof_status_from_backend_compile(
    statuses: list[str], op: str, target: str
) -> str:
    """Map exact-target manifest evidence to backend compile proof.

    Reference and compileable are explicit non-complete states. A fused source
    row completes this rung only when an execute/compare fixture proves that the
    source was compiled and ran for the same exact target.
    """
    if any(s == "missing" for s in statuses):
        return PROOF_MISSING
    if any(s == "planned" for s in statuses):
        return PROOF_PLANNED
    if any(s == "artifact_only" for s in statuses):
        return PROOF_ARTIFACT_ONLY
    if any(s == "compileable" for s in statuses):
        return PROOF_COMPILEABLE
    if any(s == "reference" for s in statuses):
        return PROOF_REFERENCE
    native_statuses = {"device_verified_abi", "device_verified_jit", "packaged", "fused"}
    if all(s in native_statuses for s in statuses) and (
        all(s != "fused" for s in statuses)
        or _numerical_proof_source(op, target) == "fixture"
    ):
        return PROOF_COMPLETE
    if all(s in native_statuses for s in statuses):
        return PROOF_PARTIAL
    return PROOF_PARTIAL


def _proof_status_from_runtime(
    op: str,
    target: str,
    components: tuple[str, ...],
    statuses: list[str],
) -> str:
    """Require an executable target row plus exact-target op proof.

    ``ExecutionRow`` is target/path-grain, so the manifest status and the
    target-aligned execute/compare fixture provide the op-specific join.
    """
    executable_targets = {
        row.target for row in _em.all_rows() if row.executable
    }
    if target not in executable_targets:
        return PROOF_MISSING
    if _numerical_proof_source(op, target) != "fixture":
        return PROOF_MISSING
    if any(s in {"missing", "planned", "artifact_only", "compileable"}
           for s in statuses):
        return PROOF_MISSING
    if any(s == "reference" for s in statuses):
        return PROOF_REFERENCE
    if all(s in {"device_verified_abi", "device_verified_jit", "packaged", "fused"}
           for s in statuses):
        return PROOF_COMPLETE
    return PROOF_PARTIAL


# --- Public API -----------------------------------------------------------

def build_matrix() -> list[ProofCell]:
    """Generate the full matrix as a flat list of proof cells, ordered by
    (op, target) per ``CONFORMANCE_OPS`` × ``CONFORMANCE_TARGETS``.

    IR proof uses the in-process typed lowering/verifier stack. Runtime and
    numerical proof use checked-in exact-target evidence, so regeneration does
    not depend on the current host's optional accelerator toolchain.
    """
    out: list[ProofCell] = []
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
    "first_failing_gate", "first_failing_gate_detail",
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
            c.first_failing_gate_detail,
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
        "A cell is **complete** only when every proof column is `complete`."
        " Its `first_failing_gate` is then empty (`—`); otherwise that field"
        " names the first incomplete proof rung. Rows use exact manifest target"
        " grain. `cpu` is the portable host reference lane; `x86` is the native"
        " x86 lane; NVIDIA architectures are separate rows."
    )
    lines.append("")
    lines.append(
        "The four IR columns are derived by compiling each curated program and"
        " running the typed Graph/Schedule/Tile/Target verifiers. Backend and"
        " runtime columns join exact-target `backend_manifest` evidence to an"
        " executable `execution_matrix` target row. Numerical completion"
        " requires an exact-target execute-and-compare fixture."
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
        (PROOF_REFERENCE,
         "Correct reference execution; no target-native compile claim."),
        (PROOF_COMPILEABLE,
         "Pinned backend compiler accepts the artifact; execution unproven."),
        (PROOF_PARTIAL,
         "Evidence exists but does not satisfy the rung's full contract."),
        (PROOF_ARTIFACT_ONLY,
         "Target artifact emits; concrete backend compilation is absent."),
        (PROOF_PLANNED,
         "Declared in the registry / manifest, not yet implemented."),
        (PROOF_MISSING,
         "The evidence required by this rung is absent."),
        (PROOF_NA,
         "Concept does not apply to this target."),
    ]
    for status, meaning in legend:
        lines.append(
            f"| {_STATUS_SYMBOL[status]} | `{status}` | {meaning} |"
        )
    lines.append("")

    # Family summaries are derived from exact-target cells and never fed back
    # into row status.
    family_for = {
        "cpu": "host_reference",
        "x86": "x86",
        "apple_cpu": "apple",
        "apple_gpu": "apple",
        "rocm": "rocm",
        "nvidia_sm80": "nvidia",
        "nvidia_sm90": "nvidia",
        "nvidia_sm100": "nvidia",
        "nvidia_sm120": "nvidia",
    }
    family_counts: dict[str, dict[str, int]] = {}
    for cell in cells:
        family = family_for[cell.target]
        counts = family_counts.setdefault(family, {})
        counts[cell.overall] = counts.get(cell.overall, 0) + 1
    lines.append("## Derived family rollup")
    lines.append("")
    lines.append("| Family | Exact-target cells | Status counts |")
    lines.append("|---|---:|---|")
    for family in ("host_reference", "x86", "apple", "rocm", "nvidia"):
        counts = family_counts.get(family, {})
        total_family = sum(counts.values())
        detail = ", ".join(
            f"{status}={counts[status]}" for status in _STATUS_ORDER
            if counts.get(status)
        )
        lines.append(f"| `{family}` | {total_family} | {detail} |")
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
