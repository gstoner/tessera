"""TSOL-1 (2026-05-22) — Tessera Standard Operator Library coverage filter.

The Tessera Standard Operator Library (TSOL) is the curated portable
operator surface users call via ``tessera.ops.<name>``.  Its spec at
``docs/operations/Tessera_Standard_Operations.md`` enumerates the
canonical names per category (linear algebra / neural network
primitives / spectral / sparse / RNG / collectives / layout).

The primitive coverage registry at ``primitive_coverage.py`` tracks
432 entries × 12 contract axes for *every* Tessera primitive — TSOL
canonical names plus the long tail of internal / family-variant
primitives that aren't in the spec.

This module is the **TSOL filter**: it takes the full coverage
registry, narrows to the canonical TSOL names, and emits a focused
dashboard at ``docs/audit/generated/tsol_coverage.md`` so a reader
can answer "how complete is the TSOL surface today?" without
wading through the full 432-entry registry.

Drift gates at ``tests/unit/test_tsol_coverage.py``:

  * Every TSOL canonical name has a matching `PrimitiveCoverage` row
    (catches spec ops that lost their registry entry).
  * The generated dashboard is consistent with the live registry
    render (regenerate when the registry changes).
  * Per-axis floor counts at the TSOL slice (`complete` count never
    drops below the baseline).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DASHBOARD_PATH = (
    _REPO_ROOT / "docs" / "audit" / "generated" / "tsol_coverage.md"
)


# ─────────────────────────────────────────────────────────────────────────
# Canonical TSOL op list — sourced from
# docs/operations/Tessera_Standard_Operations.md catalog sections.
#
# Grouped by spec category so the dashboard reads in the same order
# as the spec.  When a new op is added to the spec, append it here.
# When an op is REMOVED from the spec, delete it here.  The drift
# gate will catch the inconsistency in either direction.
# ─────────────────────────────────────────────────────────────────────────


_TSOL_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Linear Algebra",
        (
            "gemm", "matmul", "batched_gemm", "einsum",
            "factorized_matmul", "tri_solve",
            "cholesky", "qr", "svd",
        ),
    ),
    (
        "Neural Network Primitives",
        (
            "conv2d", "conv3d",
            "layer_norm", "rmsnorm", "softmax",
            "gelu", "relu", "silu",
            "dropout", "qkv_projection",
            "flash_attn", "rope",
            "moe", "moe_dispatch", "moe_combine",
        ),
    ),
    (
        "Spectral Operators",
        (
            "fft", "ifft", "rfft", "irfft",
            "stft", "istft", "spectral_filter",
        ),
    ),
    (
        "Sparse, Segment, and Graph Operators",
        (
            "spmm_coo", "spmm_csr", "sddmm", "bsmm",
            "segment_reduce",
        ),
    ),
    (
        "RNG and Initialization",
        ("rng_uniform", "rng_normal", "dropout"),
    ),
    (
        "Collectives",
        (
            "all_reduce", "reduce_scatter",
            "all_gather", "all_to_all",
        ),
    ),
    (
        "Layout and Packing",
        (
            "transpose", "rearrange",
            "pack", "unpack",
            "tile_view",
        ),
    ),
)


def all_tsol_op_names() -> tuple[str, ...]:
    """Return every canonical TSOL op name from the catalog, deduped
    and sorted."""
    seen: set[str] = set()
    out: list[str] = []
    for _, names in _TSOL_CATEGORIES:
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
    return tuple(sorted(out))


# ─────────────────────────────────────────────────────────────────────────
# Coverage row construction
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TSOLRow:
    """One row of the TSOL coverage dashboard.

    Mirrors the per-axis status from PrimitiveCoverage, restricted to
    the axes that matter most for a spec-level "is this ready?" view.
    """

    name: str
    category: str
    has_registry_entry: bool
    math_semantics: str
    shape_rule: str
    dtype_layout_rule: str
    vjp: str
    jvp: str
    sharding_rule: str
    backend_kernel: str
    lowering_rule: str
    notes: str


def _category_of(name: str) -> str:
    """Return the spec category for a canonical TSOL op name.  When
    an op appears in multiple categories (e.g., ``dropout`` is in
    NN + RNG), return the first occurrence."""
    for cat, names in _TSOL_CATEGORIES:
        if name in names:
            return cat
    return "(uncategorised)"


def _row_for(name: str, coverages: dict) -> TSOLRow:
    """Build a TSOLRow for ``name`` from the full coverage map."""
    cov = coverages.get(name)
    if cov is None:
        return TSOLRow(
            name=name,
            category=_category_of(name),
            has_registry_entry=False,
            math_semantics="-",
            shape_rule="-",
            dtype_layout_rule="-",
            vjp="-",
            jvp="-",
            sharding_rule="-",
            backend_kernel="-",
            lowering_rule="-",
            notes="MISSING from primitive_coverage.py",
        )
    s = cov.contract_status
    return TSOLRow(
        name=name,
        category=_category_of(name),
        has_registry_entry=True,
        math_semantics=s.get("math_semantics", "?"),
        shape_rule=s.get("shape_rule", "?"),
        dtype_layout_rule=s.get("dtype_layout_rule", "?"),
        vjp=s.get("vjp", "?"),
        jvp=s.get("jvp", "?"),
        sharding_rule=s.get("sharding_rule", "?"),
        backend_kernel=s.get("backend_kernel", "?"),
        lowering_rule=s.get("lowering_rule", "?"),
        notes=cov.notes or "",
    )


def collect_tsol_coverage() -> tuple[TSOLRow, ...]:
    """Walk the canonical TSOL op list + look up each in the
    primitive coverage registry.  Returns rows in spec-category
    order then alphabetical within each category."""
    from .primitive_coverage import all_primitive_coverages
    coverages = all_primitive_coverages()
    out: list[TSOLRow] = []
    # Walk in spec-category order rather than alphabetical so the
    # dashboard reads in the same flow as the spec doc.
    seen: set[str] = set()
    for cat, names in _TSOL_CATEGORIES:
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            out.append(_row_for(name, coverages))
    return tuple(out)


def coverage_summary() -> dict[str, dict[str, int]]:
    """Return per-axis ``{status: count}`` across the TSOL slice."""
    rows = collect_tsol_coverage()
    axes = (
        "math_semantics", "shape_rule", "dtype_layout_rule",
        "vjp", "jvp",
        "sharding_rule", "backend_kernel", "lowering_rule",
    )
    out: dict[str, dict[str, int]] = {a: {} for a in axes}
    out["__totals__"] = {
        "rows": len(rows),
        "has_registry_entry": sum(1 for r in rows if r.has_registry_entry),
    }
    for row in rows:
        for axis in axes:
            value = getattr(row, axis)
            out[axis][value] = out[axis].get(value, 0) + 1
    return out


# ─────────────────────────────────────────────────────────────────────────
# Dashboard render
# ─────────────────────────────────────────────────────────────────────────


def render_dashboard() -> str:
    """Render the TSOL coverage dashboard as Markdown text."""
    rows = collect_tsol_coverage()
    summary = coverage_summary()

    lines: list[str] = []
    lines.append("# TSOL Coverage Dashboard")
    lines.append("")
    lines.append(
        "Generated from `python/tessera/compiler/tsol_coverage.py`.  "
        "Don't edit by hand — regenerate via "
        "`python -c \"from tessera.compiler.tsol_coverage import "
        "render_dashboard; "
        "open('docs/audit/generated/tsol_coverage.md', 'w').write("
        "render_dashboard())\"`.  Drift gated by "
        "`tests/unit/test_tsol_coverage.py`."
    )
    lines.append("")
    lines.append(
        "Spec: `docs/operations/Tessera_Standard_Operations.md`.  "
        "Full primitive registry: "
        "`docs/audit/standalone_primitive_coverage.md`."
    )
    lines.append("")

    # ── Headline summary ───────────────────────────────────────────────
    totals = summary["__totals__"]
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- **{totals['rows']}** canonical TSOL ops in the spec catalog."
    )
    lines.append(
        f"- **{totals['has_registry_entry']}** of those have a "
        f"matching row in `primitive_coverage.py`."
    )
    missing = totals["rows"] - totals["has_registry_entry"]
    if missing:
        lines.append(
            f"- **{missing}** TSOL canonical name(s) MISSING from the "
            f"primitive registry — see per-op table for details."
        )
    lines.append("")

    # ── Per-axis summary ───────────────────────────────────────────────
    lines.append("## Per-axis status counts (TSOL slice only)")
    lines.append("")
    lines.append(
        "Counts below are restricted to the TSOL canonical names.  "
        "The full 432-primitive registry is summarised in "
        "`docs/audit/standalone_primitive_coverage.md`."
    )
    lines.append("")
    lines.append("| Axis | complete | partial | planned | N/A | other |")
    lines.append("|------|----------|---------|---------|-----|-------|")
    axis_order = (
        "math_semantics", "shape_rule", "dtype_layout_rule",
        "vjp", "jvp",
        "lowering_rule",
        "sharding_rule", "backend_kernel",
    )
    for axis in axis_order:
        counts = summary[axis]
        n_complete = counts.get("complete", 0)
        n_partial = counts.get("partial", 0)
        n_planned = counts.get("planned", 0)
        n_na = counts.get("not_applicable", 0)
        n_other = sum(
            v for k, v in counts.items()
            if k not in ("complete", "partial", "planned", "not_applicable")
        )
        lines.append(
            f"| `{axis}` | {n_complete:>3} | {n_partial:>3} | "
            f"{n_planned:>3} | {n_na:>3} | {n_other:>3} |"
        )
    lines.append("")

    # ── Per-op detail, grouped by spec category ────────────────────────
    lines.append("## Per-op coverage")
    lines.append("")
    lines.append(
        "Status legend: ✅ `complete`  • ◐ `partial`  • ◯ `planned`  "
        "• – `not_applicable`  • ? `unknown` / missing registry entry."
    )
    lines.append("")

    def _glyph(status: str) -> str:
        return {
            "complete": "✅",
            "partial": "◐",
            "planned": "◯",
            "not_applicable": "–",
        }.get(status, "?")

    seen_categories: set[str] = set()
    for cat, _names in _TSOL_CATEGORIES:
        if cat in seen_categories:
            continue
        seen_categories.add(cat)
        cat_rows = [r for r in rows if r.category == cat]
        if not cat_rows:
            continue
        lines.append(f"### {cat}")
        lines.append("")
        lines.append(
            "| Op | math | shape | dtype | vjp | jvp | lowering | "
            "sharding | backend |"
        )
        lines.append(
            "|----|------|-------|-------|-----|-----|----------|"
            "----------|---------|"
        )
        for r in cat_rows:
            lines.append(
                f"| `{r.name}` "
                f"| {_glyph(r.math_semantics)} "
                f"| {_glyph(r.shape_rule)} "
                f"| {_glyph(r.dtype_layout_rule)} "
                f"| {_glyph(r.vjp)} "
                f"| {_glyph(r.jvp)} "
                f"| {_glyph(r.lowering_rule)} "
                f"| {_glyph(r.sharding_rule)} "
                f"| {_glyph(r.backend_kernel)} |"
            )
        lines.append("")

    # ── Honest gaps section ────────────────────────────────────────────
    lines.append("## Notable gaps")
    lines.append("")
    notable = [
        r for r in rows
        if not r.has_registry_entry
        or r.vjp == "planned"
        or r.jvp == "planned"
    ]
    if not notable:
        lines.append(
            "_None today — every TSOL canonical op has a registry "
            "entry, a VJP (or N/A), and a JVP (or N/A)._"
        )
    else:
        for r in notable:
            bits: list[str] = []
            if not r.has_registry_entry:
                bits.append("**missing registry entry**")
            else:
                if r.vjp == "planned":
                    bits.append("vjp planned")
                if r.jvp == "planned":
                    bits.append("jvp planned")
            lines.append(f"- `{r.name}` — {', '.join(bits)}.")
    lines.append("")

    # ── Honest baseline about backend_kernel ──────────────────────────
    lines.append("## Backend kernel honest baseline")
    lines.append("")
    lines.append(
        "Per the registry's gating rule "
        "(`primitive_coverage.py` line 351-352), `backend_kernel = "
        "complete` requires every declared target to ship a real "
        "hardware kernel with numerical proof.  Today **zero** TSOL "
        "entries can claim `complete` because NVIDIA / ROCm / "
        "Tenstorrent Metalium proofs aren't available on this Mac.  "
        "See `docs/audit/phase_ghi_hardware_frontier.md` for the "
        "full hardware-gated punch list."
    )
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_dashboard(path: Path | None = None) -> Path:
    """Render and write the dashboard to disk.  Defaults to the
    canonical location at ``docs/audit/generated/tsol_coverage.md``."""
    target = path or _DASHBOARD_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_dashboard())
    return target


__all__ = [
    "TSOLRow",
    "all_tsol_op_names",
    "collect_tsol_coverage",
    "coverage_summary",
    "render_dashboard",
    "write_dashboard",
]
