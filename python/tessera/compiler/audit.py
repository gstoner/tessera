"""Tessera compiler support audit — M0 / M0.5 deliverable.

This module is the single renderer for the **eight-axis compiler
support taxonomy** decided in
``docs/audit/compiler_improvement_milestone_plan_2026_05_18.md``.

It does **not** invent a new schema.  It walks the four pre-existing
sources and surfaces them through one consistent table:

  - :mod:`tessera.compiler.op_catalog`         — *what the parser accepts*
  - :mod:`tessera.compiler.primitive_coverage` — *what the standalone compiler
                                                 contract says* (12 axes,
                                                 lowering metadata,
                                                 category classifiers)
  - :mod:`tessera.compiler.backend_manifest`   — *what each backend ships*
                                                 (per-target × per-dtype rows
                                                 with status + toolchain pins)
  - :mod:`tessera.compiler.capabilities`       — *what each target advertises*
                                                 (per-op runtime status)

The taxonomy is eight axes — one column per row of the rendered table:

  ``api``        : present in :data:`op_catalog.OP_SPECS` ?
  ``frontend``   : parseable by ``@tessera.jit`` / textual frontend ?
  ``graph_ir``   : primitive coverage ``metadata.graph_ir_lowering`` value
  ``schedule_ir``: ``contract_status['lowering_rule']`` (proxy until a
                   dedicated axis exists)
  ``tile_ir``    : ``contract_status['backend_kernel']`` (proxy)
  ``target_ir``  : best backend manifest status across all targets
                   (``fused`` / ``reference`` / ``compileable`` /
                   ``artifact_only`` / ``planned``)
  ``runtime``    : best capability runtime status across all targets
                   (``ready`` / ``reference`` / ``artifact_only`` /
                   ``planned`` / ``unsupported``)
  ``bench``      : present in any shipped benchmark inventory
                   (today: Apple GPU GA/EBM benchmark only)

Every row carries provenance (which source decided each axis) so the
table is auditable without re-running the walk.

CLI::

    python -m tessera.compiler.audit support_table
    python -m tessera.compiler.audit support_table --out docs/audit/generated/support_table.md
    python -m tessera.compiler.audit support_table --check

The ``--check`` form is the drift gate: regenerate the table and fail
when it differs from the checked-in copy.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

from . import backend_manifest as bm
from . import capabilities as cap
from . import primitive_coverage as pc
from .op_catalog import OP_SPECS


def _coverage_for(name: str):
    """Safe lookup wrapping `pc.coverage_for`'s raise-on-missing."""
    try:
        return pc.coverage_for(name)
    except KeyError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy
# ─────────────────────────────────────────────────────────────────────────────

#: The eight axes, in display order.
LAYER_AXES: tuple[str, ...] = (
    "api",
    "frontend",
    "graph_ir",
    "schedule_ir",
    "tile_ir",
    "target_ir",
    "runtime",
    "bench",
)

#: Canonical per-axis value sets — used by the drift gate to reject
#: unknown values.  Each value also carries a single-letter glyph for
#: the compact summary view.
AXIS_VALUE_GLYPHS: Mapping[str, str] = {
    # ─ api / frontend ─
    "public":           "P",
    "missing":          "·",
    # ─ graph_ir ─
    "registered":       "G",
    "stub_required":    "s",
    "not_applicable":   "n",
    # ─ schedule_ir / tile_ir / target_ir (contract / manifest statuses) ─
    "complete":         "C",
    "partial":          "p",
    "planned":          "·",
    "fused":            "F",
    "reference":        "R",
    "compileable":      "c",
    "artifact_only":    "A",
    # ─ runtime ─
    "ready":            "N",   # native
    "unsupported":      "X",
    "unknown":          "?",   # target has no opinion on this op (no
                                # explicit OpCapability entry and no
                                # generic reference-execution feature)
    # ─ bench ─
    "benchmarked":      "B",
    "none":             "·",
}


@dataclass(frozen=True)
class AxisCell:
    """One cell of the support table — status + source provenance."""
    status: str
    source: str

    def glyph(self) -> str:
        return AXIS_VALUE_GLYPHS.get(self.status, "?")


@dataclass(frozen=True)
class OpSupportRow:
    """One row of the support table — eight axis cells per op."""
    op_name: str
    family: str
    cells: Mapping[str, AxisCell] = field(default_factory=dict)

    def glyph_line(self) -> str:
        return "".join(self.cells[a].glyph() for a in LAYER_AXES)


# ─────────────────────────────────────────────────────────────────────────────
# Axis walkers
# ─────────────────────────────────────────────────────────────────────────────

# Ops that have benchmark inventory coverage today.  This is intentionally
# small and explicit so the bench axis doesn't silently inflate.  Source
# of truth: `benchmarks/apple_gpu/benchmark_ga_ebm.py` row catalog
# (17 GA + 9 native EBM, post 2026-05-17 close-out).
_BENCH_INVENTORY: frozenset[str] = frozenset({
    # GA primitives
    *[f"clifford_{n}" for n in (
        "geometric_product", "wedge", "left_contraction", "inner",
        "grade_projection", "reverse", "grade_involution", "conjugate",
        "norm", "exp", "log", "rotor_sandwich", "hodge_star",
        "ext_deriv", "vec_deriv", "codiff", "integral",
    )],
    # Native EBM primitives (9/9)
    "ebm_inner_step", "ebm_refinement", "ebm_langevin_step",
    "ebm_decode_init", "ebm_bivector_langevin", "ebm_sphere_langevin",
    "ebm_self_verify", "ebm_energy", "ebm_partition_exact",
})


# M7 Visual Complex Analysis primitives — public via tessera.complex.*.
# @complex_jit / @analytic are frontend lanes, not primitive rows. Same opt-in pattern as
# _BENCH_INVENTORY so M7 appears in the per-op support matrix without
# pulling in the entire planned long-tail of primitive_coverage.
# Source of truth: ``python/tessera/complex.py`` + the M7 rows in
# ``primitive_coverage.py`` (category ``visual_complex``).
#
# 2026-05-19: dropped ``complex_add`` from the inventory.  There is no
# ``def complex_add`` in ``python/tessera/complex.py`` — complex
# addition is just ``+`` on numpy arrays.  Keeping the row would have
# overclaimed ``api=public`` for a function that doesn't exist.
_M7_INVENTORY: frozenset[str] = frozenset({
    "complex_mul", "complex_div", "complex_exp",
    "complex_log", "complex_sqrt", "complex_pow", "complex_conjugate",
    "complex_abs", "complex_arg",
    "mobius", "mobius_from_three_points",
    "cross_ratio", "is_concyclic", "stereographic",
    "check_cauchy_riemann",
    "conformal_jacobian", "conformal_energy_on_sphere",
    "dz", "dbar", "laplacian_2d",
})


# Backend-manifest naming alias.  The public API in
# ``python/tessera/complex.py`` uses bare names (``mobius``,
# ``stereographic``) but the fused Apple GPU MSL kernels are
# registered under prefixed names (``complex_mobius``,
# ``complex_stereographic``) in ``backend_manifest._COMPLEX_APPLE_GPU_FUSED``
# so the C ABI symbol naming stays uniform with the other complex
# kernels (``complex_mul``, ``complex_exp``).  The audit walker
# consults this alias before looking up the backend manifest so the
# support table reflects the real fused-kernel coverage instead of
# silently underclaiming ``target_ir=planned``.
_M7_BACKEND_ALIASES: dict[str, str] = {
    "mobius": "complex_mobius",
    "stereographic": "complex_stereographic",
    # complex_mul / complex_exp use the same name on both sides.
}


def _backend_lookup_name(op_name: str) -> str:
    """Return the name to use when looking ``op_name`` up in the
    backend manifest / per-target fused-kernel sets."""

    return _M7_BACKEND_ALIASES.get(op_name, op_name)


def m7_fused_public_ops() -> frozenset[str]:
    """Return the public-facing M7 op names that ship a fused backend kernel.

    Derived from ``_M7_INVENTORY`` + ``_M7_BACKEND_ALIASES`` +
    ``backend_manifest._COMPLEX_APPLE_GPU_FUSED`` — never hardcoded.
    Tests use this as the canonical "M7 fused" list so a new fused
    complex kernel automatically widens every M7 regression guard.

    Today (2026-05-19) this returns ``{complex_mul, complex_exp,
    mobius, stereographic}``.  When a new entry lands in
    ``_COMPLEX_APPLE_GPU_FUSED`` (e.g., ``complex_log``,
    ``complex_sqrt``, a fused conformal Jacobian) it shows up here
    automatically — no test edits required.
    """

    out: set[str] = set()
    for public_name in _M7_INVENTORY:
        backend_name = _backend_lookup_name(public_name)
        if backend_name in bm._COMPLEX_APPLE_GPU_FUSED:
            out.add(public_name)
    return frozenset(out)


def _axis_api(op_name: str) -> AxisCell:
    if op_name in OP_SPECS:
        return AxisCell("public", "op_catalog")
    if _coverage_for(op_name) is not None:
        return AxisCell("public", "primitive_coverage")
    if op_name.startswith("clifford_") and op_name in bm._CLIFFORD_APPLE_GPU_FUSED:
        return AxisCell("public", "tessera.ga.*")
    if op_name.startswith("ebm_") and op_name in bm._EBM_APPLE_GPU_FUSED:
        return AxisCell("public", "tessera.ebm.*")
    if op_name in _M7_INVENTORY:
        return AxisCell("public", "tessera.complex.*")
    return AxisCell("missing", "op_catalog")


def _axis_frontend(op_name: str) -> AxisCell:
    # @tessera.jit walks the AST against OP_SPECS; the textual frontend
    # consults the same catalog.  GA / EBM ops have their own public
    # namespace + (for GA) the constrained @clifford_jit AST frontend.
    spec = OP_SPECS.get(op_name)
    if spec is not None:
        return AxisCell("public", "@tessera.jit / textual frontend")
    cov = _coverage_for(op_name)
    if cov is not None and cov.existing_op:
        return AxisCell("public", "primitive_coverage.existing_op")
    if op_name.startswith("clifford_") and op_name in bm._CLIFFORD_APPLE_GPU_FUSED:
        return AxisCell("public", "@clifford_jit / tessera.ga.*")
    if op_name.startswith("ebm_") and op_name in bm._EBM_APPLE_GPU_FUSED:
        return AxisCell("public", "tessera.ebm.*")
    if op_name in _M7_INVENTORY:
        return AxisCell("public", "tessera.complex.*")
    return AxisCell("missing", "op_catalog")


def _axis_graph_ir(op_name: str) -> AxisCell:
    # GA / EBM ops are dispatched through `jit_bridge → manifest →
    # shared loader`, not through a Graph IR ODS op.  That's a design
    # choice (manifest-backed dispatch), so the right value is
    # `not_applicable`, not `missing` — and we want to report that
    # even when the registry also has a placeholder row for the op.
    if op_name.startswith("clifford_") and op_name in bm._CLIFFORD_APPLE_GPU_FUSED:
        return AxisCell("not_applicable", "jit_bridge.manifest dispatch")
    if op_name.startswith("ebm_") and op_name in bm._EBM_APPLE_GPU_FUSED:
        return AxisCell("not_applicable", "jit_bridge.manifest dispatch")
    # E2 (partial-ops uplift, 2026-05-20).  M7 ``complex_*`` ops with a
    # fused MSL kernel are dispatched the same way as GA/EBM — through
    # the manifest, not through a Graph IR ODS op.  Translate via
    # ``_backend_lookup_name`` so the public names ``mobius`` /
    # ``stereographic`` resolve to ``complex_mobius`` /
    # ``complex_stereographic`` in the backend manifest.
    if _backend_lookup_name(op_name) in bm._COMPLEX_APPLE_GPU_FUSED:
        return AxisCell("not_applicable", "jit_bridge.manifest dispatch")
    cov = _coverage_for(op_name)
    if cov is not None:
        value = cov.metadata.get("graph_ir_lowering")
        if value is not None and str(value) in AXIS_VALUE_GLYPHS:
            return AxisCell(str(value), "primitive_coverage.metadata.graph_ir_lowering")
    return AxisCell("missing", "primitive_coverage")


def _axis_schedule_ir(op_name: str) -> AxisCell:
    cov = _coverage_for(op_name)
    if cov is not None:
        status = cov.contract_status.get("lowering_rule", "planned")
        return AxisCell(status, "primitive_coverage.contract_status.lowering_rule")
    # Manifest-dispatched ops (GA/EBM) don't carry a separate schedule
    # rule today — dispatch happens at the runtime layer.
    if op_name.startswith(("clifford_", "ebm_")):
        return AxisCell("not_applicable", "jit_bridge.manifest dispatch")
    return AxisCell("planned", "primitive_coverage")


def _axis_tile_ir(op_name: str) -> AxisCell:
    # Prefer the most concrete evidence: a fused manifest entry beats
    # a registry "partial" because the registry's backend_kernel axis
    # is intentionally the long-pole gate (Decision #25) and stays
    # `partial` until Phase G/H/I lights up distributed runtime.
    backend_name = _backend_lookup_name(op_name)
    if (backend_name in bm._CLIFFORD_APPLE_GPU_FUSED or
            backend_name in bm._EBM_APPLE_GPU_FUSED or
            backend_name in bm._COMPLEX_APPLE_GPU_FUSED):
        return AxisCell("fused", "backend_manifest fused entry")
    cov = _coverage_for(op_name)
    if cov is not None:
        status = cov.contract_status.get("backend_kernel", "planned")
        return AxisCell(status, "primitive_coverage.contract_status.backend_kernel")
    return AxisCell("planned", "primitive_coverage")


def _axis_target_ir(op_name: str) -> AxisCell:
    # M7 ops (mobius, stereographic) live under prefixed names in
    # backend_manifest; translate before lookup so the audit reflects
    # the fused-kernel coverage that actually ships.
    entries = bm.manifest_for(_backend_lookup_name(op_name))
    if not entries:
        return AxisCell("planned", "backend_manifest")
    # Best status across targets — rank ordering chooses the most concrete.
    rank = {"fused": 0, "reference": 1, "compileable": 2, "artifact_only": 3, "planned": 4}
    best_target = min(entries, key=lambda e: rank.get(e.status, 99))
    return AxisCell(
        best_target.status,
        f"backend_manifest[{best_target.target}]",
    )


def _axis_runtime(op_name: str) -> AxisCell:
    """Audit the runtime axis for ``op_name``.

    The cell value is the **most concrete runtime claim** the
    capability registry can make across all targets for this op.
    Critically: a missing per-op entry **does not promote to
    ``target_cap.default_runtime_status``** — that would
    overclaim ``runtime=ready`` for any primitive not explicitly
    registered with a target (e.g., pure-Python S-series
    primitives like ``tree_flatten`` / ``dataset_map``, where the
    only runtime is the Python data-structure code).

    Resolution per target:

      * Explicit ``OpCapability`` registered  → use its
        ``runtime_status`` verbatim.
      * No explicit entry, target advertises a generic reference
        fallback (the ``reference_execution`` feature) and the op
        is in the canonical op catalog  → ``reference``
        (the numpy/reference path runs it, but that's not a
        *native* claim).
      * Otherwise                              → ``unknown`` (the
        target has no opinion on this op; do not claim a status).
    """
    rank = {
        "ready": 0, "reference": 1, "fused": 0, "compileable": 2,
        "artifact_only": 3, "planned": 4, "unsupported": 5,
        "unknown": 6,
    }
    best_status = "unknown"
    best_target = "unknown"
    spec = OP_SPECS.get(op_name)
    graph_name = spec.graph_name if spec is not None else f"tessera.{op_name}"

    for target_name, target_cap in cap.TARGET_CAPABILITIES.items():
        op_caps = (
            target_cap.supported_ops.get(graph_name)
            or target_cap.supported_ops.get(op_name)
        )
        if op_caps is not None:
            status = op_caps.runtime_status
        elif (
            "reference_execution" in target_cap.features
            and op_name in OP_SPECS
        ):
            # CPU-style generic numpy fallback: covers any op in the
            # canonical catalog, but only via the reference path —
            # never claim "ready" for that.
            status = "reference"
        else:
            status = "unknown"
        if rank.get(status, 99) < rank.get(best_status, 99):
            best_status = status
            best_target = target_name
    return AxisCell(best_status, f"capabilities[{best_target}]")


def _axis_bench(op_name: str) -> AxisCell:
    if op_name in _BENCH_INVENTORY:
        return AxisCell("benchmarked", "benchmarks/apple_gpu/benchmark_ga_ebm.py")
    return AxisCell("none", "no benchmark row")


_AXIS_WALKERS = {
    "api":         _axis_api,
    "frontend":    _axis_frontend,
    "graph_ir":    _axis_graph_ir,
    "schedule_ir": _axis_schedule_ir,
    "tile_ir":     _axis_tile_ir,
    "target_ir":   _axis_target_ir,
    "runtime":     _axis_runtime,
    "bench":       _axis_bench,
}


# ─────────────────────────────────────────────────────────────────────────────
# Row construction
# ─────────────────────────────────────────────────────────────────────────────

def _family_for(op_name: str) -> str:
    """Bucket ops into a small set of families for table grouping."""
    if op_name.startswith("clifford_"):
        return "geometric_algebra"
    if op_name.startswith("ebm_"):
        return "energy_based_models"
    # E3 (2026-05-20): M7 Visual Complex ops are registered in OP_SPECS
    # with ``lowering="elementwise" / "stencil" / "stable_reduction"``
    # — the lowering tag identifies the *pass* that handles the op,
    # not its display family.  Pin them to the ``visual_complex``
    # family so support_table / e2e_coverage group them together.
    if op_name in _M7_INVENTORY:
        return "visual_complex"
    spec = OP_SPECS.get(op_name)
    if spec is not None:
        return spec.lowering or "elementwise"
    cov = _coverage_for(op_name)
    if cov is not None:
        return cov.category or "uncategorized"
    return "uncategorized"


def support_row_for(op_name: str) -> OpSupportRow:
    """Build the 8-axis row for a single op name."""
    cells = {axis: walker(op_name) for axis, walker in _AXIS_WALKERS.items()}
    return OpSupportRow(
        op_name=op_name,
        family=_family_for(op_name),
        cells=cells,
    )


def _candidate_op_names() -> list[str]:
    """The op-name population the table covers.

    Today: every op in the public catalog, plus every benchmarked GA/EBM
    primitive, plus the M7 Visual Complex Analysis surface.  This
    deliberately excludes the long-tail planned primitives in
    :mod:`primitive_coverage` so the table stays focused on ops the
    user can actually call.
    """
    names: set[str] = (
        set(OP_SPECS.keys())
        | set(_BENCH_INVENTORY)
        | set(_M7_INVENTORY)
    )
    return sorted(names)


def all_support_rows() -> list[OpSupportRow]:
    """Build the full table — one row per public + benchmarked op."""
    return [support_row_for(name) for name in _candidate_op_names()]


# ─────────────────────────────────────────────────────────────────────────────
# Markdown rendering
# ─────────────────────────────────────────────────────────────────────────────

def _canonical_program_section() -> str:
    """Render the M1.5 canonical-program registry as a section the
    support-table consumer can read alongside the per-op view."""
    # Import inside the function so the audit module stays light
    # for callers that only want the per-op table.
    from .canonical import CANONICAL_PROGRAMS

    lines = ["## Canonical end-to-end programs (M1 / M1.5)", ""]
    lines.append(
        "| Program | Family | Status | Owner file | Description |"
    )
    lines.append("|---|---|---|---|---|")
    for p in CANONICAL_PROGRAMS:
        lines.append(
            f"| `{p.program_id}` | {p.family} | **{p.status}** | "
            f"`{p.owner_file}` | {p.description} |"
        )
    lines.append("")
    return "\n".join(lines)


_LEGEND = """
**Axis glyph legend.**  `P` public · `G` registered Graph IR · `s` stub
required · `n` not applicable · `C` complete · `p` partial · `F` fused
kernel · `R` reference path · `c` compileable artifact · `A` artifact
only · `N` native runtime · `B` benchmarked · `·` planned / none / missing.
"""


def render_markdown(rows: Iterable[OpSupportRow] | None = None) -> str:
    """Render the support table as a deterministic Markdown document.

    The output is the canonical artifact under
    ``docs/audit/generated/support_table.md`` and is compared by
    ``--check`` for drift.
    """
    rows = list(rows) if rows is not None else all_support_rows()
    rows.sort(key=lambda r: (r.family, r.op_name))

    out: list[str] = []
    out.append("# Compiler support table — generated\n")
    out.append(
        "> **Generated by `python -m tessera.compiler.audit support_table`.**\n"
        "> Do not edit by hand — the M0 drift gate compares this file to a\n"
        "> regenerated copy in CI.  Source of truth: `op_catalog.OP_SPECS`,\n"
        "> `primitive_coverage`, `backend_manifest`, `capabilities`.\n"
    )
    out.append("")
    out.append(_LEGEND.strip())
    out.append("")
    out.append("## Per-op support matrix")
    out.append("")
    out.append("| Op | Family | " + " | ".join(LAYER_AXES) + " |")
    out.append("|----|--------|" + "|".join("---" for _ in LAYER_AXES) + "|")
    for row in rows:
        cells = " | ".join(row.cells[a].status for a in LAYER_AXES)
        out.append(f"| `{row.op_name}` | {row.family} | {cells} |")
    out.append("")
    out.append("## Summary by family")
    out.append("")
    out.append("| Family | Count | Glyphs (one column per op, axes packed L→R) |")
    out.append("|--------|------:|------|")
    families: dict[str, list[OpSupportRow]] = {}
    for row in rows:
        families.setdefault(row.family, []).append(row)
    for family in sorted(families):
        glyphs = " ".join(r.glyph_line() for r in families[family])
        out.append(f"| {family} | {len(families[family])} | {glyphs} |")
    out.append("")
    out.append(_canonical_program_section())
    out.append("## Axes")
    out.append("")
    for axis in LAYER_AXES:
        provenance = ""
        sample = next((row.cells[axis] for row in rows if row.cells[axis].status != "missing"), None)
        if sample is not None:
            provenance = f" (source: `{sample.source}`)"
        out.append(f"- **`{axis}`**{provenance}")
    out.append("")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_OUT = Path("docs/audit/generated/support_table.md")


def _public_doc_paths() -> list[Path]:
    """Public docs whose native-execution claims the lint scans.

    The list is intentionally short — these are the entry points
    that publicize the compiler's surface to readers who won't read
    the generated table.  Adding to this list is fine; the lint
    requires every claim to be grounded in the manifest.
    """
    return [
        Path("README.md"),
        Path("docs/README.md"),
        Path("docs/status/ga_ebm_milestone.md"),
        Path("docs/spec/GA_EBM_EXECUTION_STATUS.md"),
        Path("benchmarks/apple_gpu/README.md"),
    ]


# Pattern: `tessera_<target>_<op>_<dtype>` symbols.  When a doc names
# one of these, the manifest must have a `fused` entry for the (op,
# target) pair.  Anything else is silent over-claiming.
_TESSERA_SYMBOL_RE = re.compile(
    r"tessera_(?P<target>apple_gpu|apple_cpu|x86|nvidia|rocm|metalium)_"
    r"(?P<op>[A-Za-z0-9_]+?)_(?P<dtype>f32|f16|bf16|fp32|fp16)"
)

# Pattern: backtick-wrapped op names in the public-API form
# (`clifford_norm`, `ebm_partition_exact`, ...) accompanied by a
# native-claim word ("native", "fused", "hardware-runtime").
_NATIVE_CLAIM_RE = re.compile(
    r"`(?P<op>(?:clifford|ebm)_[a-z0-9_]+)`[^.\n]{0,160}?"
    r"\b(?P<claim>native|fused|hardware-runtime|hardware runtime)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClaimViolation:
    """One claim_lint finding — points at a public doc making a
    native claim the manifest can't substantiate."""
    doc_path: str
    line_no: int
    claim: str
    code: str

    def format(self) -> str:
        return f"{self.doc_path}:{self.line_no}: [{self.code}] {self.claim}"


def _scan_doc_for_claims(path: Path) -> list[ClaimViolation]:
    """Walk one doc and emit a ClaimViolation per unsupported claim."""
    if not path.exists():
        return []
    text = path.read_text()
    violations: list[ClaimViolation] = []

    # Symbol-form claims.
    for m in _TESSERA_SYMBOL_RE.finditer(text):
        target = m.group("target")
        op = m.group("op")
        # Compute the line number for the diagnostic.
        line_no = text.count("\n", 0, m.start()) + 1
        # Resolve the op name back to a manifest entry.  The symbol
        # encodes the op suffix (e.g., `clifford_rotor_sandwich_cl30`
        # → `clifford_rotor_sandwich`).  Strip the trailing `_cl{p}{q}`
        # suffix Clifford symbols carry.
        op_clean = re.sub(r"_cl\d+$", "", op)
        entries = bm.manifest_for(op_clean)
        target_entries = [e for e in entries if e.target == target]
        if not target_entries:
            violations.append(ClaimViolation(
                doc_path=str(path),
                line_no=line_no,
                claim=(
                    f"references `tessera_{target}_{op}_{m.group('dtype')}` "
                    f"but `backend_manifest` has no entry for "
                    f"op={op_clean!r} target={target!r}"
                ),
                code="CLAIM_LINT_SYMBOL_UNGROUNDED",
            ))
            continue
        # Symbol must map to a fused / reference entry — not `planned`.
        statuses = {e.status for e in target_entries}
        if statuses & {"planned", "artifact_only"} == statuses:
            violations.append(ClaimViolation(
                doc_path=str(path),
                line_no=line_no,
                claim=(
                    f"names symbol `tessera_{target}_{op}_{m.group('dtype')}` "
                    f"but manifest says status={sorted(statuses)} for "
                    f"({op_clean}, {target}) — not a native execution path"
                ),
                code="CLAIM_LINT_SYMBOL_NOT_FUSED",
            ))

    # Native-claim-form findings around op-name backticks.
    # Only check identifiers that map to a real manifest op — Python
    # function names and benchmark-report keys that happen to share
    # the `ebm_` / `clifford_` prefix are filtered out.
    known_ops = set(bm._CLIFFORD_APPLE_GPU_FUSED) | set(bm._EBM_APPLE_GPU_FUSED)
    for m in _NATIVE_CLAIM_RE.finditer(text):
        op = m.group("op")
        if op not in known_ops:
            continue
        line_no = text.count("\n", 0, m.start()) + 1
        entries = bm.manifest_for(op)
        if not any(e.status == "fused" for e in entries):
            violations.append(ClaimViolation(
                doc_path=str(path),
                line_no=line_no,
                claim=(
                    f"claims `{op}` is {m.group('claim')!r} but no "
                    f"manifest target ships a `fused` kernel "
                    f"(statuses: {sorted({e.status for e in entries})})"
                ),
                code="CLAIM_LINT_NO_FUSED_KERNEL",
            ))
    return violations


def run_claim_lint() -> list[ClaimViolation]:
    """Run claim_lint across every doc in :func:`_public_doc_paths`."""
    out: list[ClaimViolation] = []
    for path in _public_doc_paths():
        out.extend(_scan_doc_for_claims(path))
    return out


def _cmd_claim_lint(args: argparse.Namespace) -> int:
    violations = run_claim_lint()
    if not violations:
        print("claim_lint: no violations across "
              f"{len(_public_doc_paths())} doc(s)")
        return 0
    print(f"claim_lint: {len(violations)} violation(s)", file=sys.stderr)
    for v in violations:
        print(f"  {v.format()}", file=sys.stderr)
    return 1


def _cmd_support_table(args: argparse.Namespace) -> int:
    text = render_markdown()
    if args.check:
        if not args.out.exists():
            print(
                f"audit: --check requested but {args.out} does not exist",
                file=sys.stderr,
            )
            return 2
        on_disk = args.out.read_text()
        if on_disk != text:
            print(
                f"audit: drift detected in {args.out}\n"
                f"       regenerate with: python -m tessera.compiler.audit "
                f"support_table --out {args.out}",
                file=sys.stderr,
            )
            return 1
        print(f"audit: {args.out} matches generated output")
        return 0
    if args.out == Path("-"):
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"audit: wrote {args.out}", file=sys.stderr)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tessera.compiler.audit",
        description=__doc__.split("\n\n", 1)[0],
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_st = sub.add_parser(
        "support_table",
        help="generate the per-op support matrix",
    )
    p_st.add_argument(
        "--out", type=Path, default=_DEFAULT_OUT,
        help=f"output path (default: {_DEFAULT_OUT}; use `-` for stdout)",
    )
    p_st.add_argument(
        "--check", action="store_true",
        help="exit non-zero if the on-disk file differs from the regenerated output",
    )
    p_st.set_defaults(func=_cmd_support_table)

    p_cl = sub.add_parser(
        "claim_lint",
        help="scan public docs for unsupported native-execution claims",
    )
    p_cl.set_defaults(func=_cmd_claim_lint)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "LAYER_AXES",
    "AXIS_VALUE_GLYPHS",
    "AxisCell",
    "OpSupportRow",
    "ClaimViolation",
    "all_support_rows",
    "render_markdown",
    "run_claim_lint",
    "support_row_for",
    "main",
]
