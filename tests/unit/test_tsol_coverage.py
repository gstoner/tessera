"""TSOL-1 (2026-05-22) — Tessera Standard Operator Library drift gate.

Pins:

  * Every canonical TSOL op (per spec catalog) has a matching
    `PrimitiveCoverage` row.  No silent registry drops.
  * Generated dashboard at ``docs/audit/generated/tsol_coverage.md``
    matches what the registry would render today.
  * Floor counts per axis at the TSOL slice — the `complete` count
    never drops below the 2026-05-22 baseline.
  * The TSOL canonical-name list parses cleanly + doesn't shadow
    private / internal helper names.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler.primitive_coverage import all_primitive_coverages
from tessera.compiler.tsol_coverage import (
    TSOLRow,
    all_tsol_op_names,
    collect_tsol_coverage,
    coverage_summary,
    render_dashboard,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = REPO_ROOT / "docs" / "audit" / "generated" / "tsol_coverage.md"


# ─────────────────────────────────────────────────────────────────────────
# Structural sanity
# ─────────────────────────────────────────────────────────────────────────


def test_tsol_catalog_has_no_duplicates() -> None:
    """De-duplication is enforced inside ``all_tsol_op_names`` — this
    test just locks it as a contract."""
    names = all_tsol_op_names()
    assert len(names) == len(set(names)), "duplicate TSOL canonical names"


def test_tsol_catalog_count_at_or_above_baseline() -> None:
    """When the spec adds new canonical ops, the catalog count grows.
    Locked floor catches accidental deletions during a refactor."""
    names = all_tsol_op_names()
    assert len(names) >= 47, (
        f"TSOL catalog dropped below 47 ops ({len(names)}) — was "
        f"_TSOL_CATEGORIES accidentally truncated?"
    )


# ─────────────────────────────────────────────────────────────────────────
# Registry coverage
# ─────────────────────────────────────────────────────────────────────────


def test_every_tsol_op_has_registry_entry() -> None:
    """Every canonical TSOL op MUST have a `PrimitiveCoverage` row.
    A missing row means the spec promises something the audit
    surface can't track."""
    rows = collect_tsol_coverage()
    missing = [r.name for r in rows if not r.has_registry_entry]
    assert not missing, (
        f"TSOL canonical ops missing from primitive_coverage.py: "
        f"{missing}.  Either add a coverage entry or remove from the "
        f"TSOL catalog in tsol_coverage.py."
    )


@pytest.mark.parametrize("name", all_tsol_op_names())
def test_tsol_op_is_in_primitive_coverage(name: str) -> None:
    """One test per canonical op so pytest output shows exactly which
    op regressed when the registry drifts."""
    coverages = all_primitive_coverages()
    assert name in coverages, (
        f"TSOL canonical op {name!r} not found in "
        f"primitive_coverage registry"
    )


# ─────────────────────────────────────────────────────────────────────────
# Floor counts per axis at the TSOL slice
# ─────────────────────────────────────────────────────────────────────────

# Captured 2026-05-22 right after TSOL-1 landing.  These are the
# minimum acceptable `complete` counts across the 47-op TSOL slice.
# A regression below floor means a TSOL op silently lost a contract.
_AXIS_COMPLETE_FLOORS = {
    "math_semantics": 47,    # All 47 ops have explicit math contracts.
    "shape_rule": 47,        # Same — all 47 have shape contracts.
    "dtype_layout_rule": 47, # Same.
    "lowering_rule": 47,     # Same — all have Graph IR lowering.
    "vjp": 41,               # 41 have VJP; 6 are not_applicable.
    "jvp": 40,               # 40 have JVP; 7 are not_applicable.
    "sharding_rule": 31,     # 31 complete; 16 partial pending Phase G.
    # backend_kernel: 0 by registry design (hardware-gated for every
    # entry).  Not enforced here — see test_no_tsol_op_claims_backend_kernel_complete.
}


@pytest.mark.parametrize("axis,floor", sorted(_AXIS_COMPLETE_FLOORS.items()))
def test_tsol_axis_at_or_above_complete_floor(axis: str, floor: int) -> None:
    summary = coverage_summary()
    actual = summary[axis].get("complete", 0)
    assert actual >= floor, (
        f"TSOL `{axis}` complete count dropped below floor {floor} "
        f"(got {actual}).  A canonical op likely lost a contract — "
        f"check the per-op table in docs/audit/generated/tsol_coverage.md."
    )


def test_no_tsol_op_claims_backend_kernel_complete() -> None:
    """By registry design, backend_kernel = complete requires real
    GPU hardware proofs across every declared target — unavailable
    on this Mac.  Today no TSOL op claims `complete`.  When the first
    NVIDIA / ROCm proof lands (and an `execute_compare_fixture` is
    registered), update this test to expect the first complete entry."""
    summary = coverage_summary()
    complete = summary["backend_kernel"].get("complete", 0)
    assert complete == 0, (
        f"Unexpected backend_kernel = complete claim on TSOL surface: "
        f"got {complete} entries claiming hardware-verified status "
        f"without registered execute_compare fixtures.  See "
        f"docs/audit/backend/BACKEND_AUDIT.md."
    )


def test_no_tsol_op_has_planned_vjp_or_jvp() -> None:
    """The Sprint A long-tail closure (2026-05-11) eliminated all
    `planned` VJP/JVP entries across the registry.  TSOL ops should
    inherit that — every spec-listed op is either VJP-complete or
    not-applicable, never planned."""
    rows = collect_tsol_coverage()
    planned_vjp = [r.name for r in rows if r.vjp == "planned"]
    planned_jvp = [r.name for r in rows if r.jvp == "planned"]
    assert not planned_vjp, f"TSOL ops with planned VJP: {planned_vjp}"
    assert not planned_jvp, f"TSOL ops with planned JVP: {planned_jvp}"


# ─────────────────────────────────────────────────────────────────────────
# Dashboard drift gate
# ─────────────────────────────────────────────────────────────────────────


def test_dashboard_exists() -> None:
    assert DASHBOARD.exists(), (
        f"TSOL dashboard missing: {DASHBOARD.relative_to(REPO_ROOT)}.  "
        f"Regenerate via `tessera.compiler.tsol_coverage.write_dashboard()`."
    )


def test_dashboard_matches_live_registry() -> None:
    """The checked-in markdown must match what the registry would
    render right now.  When TSOL coverage advances, regenerate the
    dashboard or this test fails."""
    if not DASHBOARD.exists():
        pytest.skip("dashboard not yet generated")
    live = render_dashboard()
    on_disk = DASHBOARD.read_text()
    if live == on_disk:
        return
    # Provide a focused diff hint without dumping the whole file.
    live_lines = live.splitlines()
    disk_lines = on_disk.splitlines()
    first_diff = next(
        (i for i, (l, d) in enumerate(zip(live_lines, disk_lines))
         if l != d),
        min(len(live_lines), len(disk_lines)),
    )
    pytest.fail(
        f"TSOL dashboard drift at line {first_diff + 1}: "
        f"on-disk has {disk_lines[first_diff]!r}, live has "
        f"{live_lines[first_diff]!r}.  Regenerate the dashboard."
    )


def test_dashboard_pins_canonical_phrases() -> None:
    """Lock canonical phrases that downstream docs link to."""
    assert DASHBOARD.exists()
    text = DASHBOARD.read_text()
    for phrase in (
        "# TSOL Coverage Dashboard",
        "## Headline",
        "## Per-axis status counts",
        "## Per-op coverage",
        "## Backend kernel honest baseline",
        # Spec back-reference.
        "Tessera_Standard_Operations.md",
        # Cross-link to the current backend audit.
        "BACKEND_AUDIT.md",
    ):
        assert phrase in text, (
            f"TSOL dashboard missing canonical phrase {phrase!r}"
        )
