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

import re

from pathlib import Path

import pytest

from tessera.compiler.primitive_coverage import all_primitive_coverages
from tessera.compiler.tsol_coverage import (
    SPEC_CATALOG_BEGIN,
    SPEC_CATALOG_END,
    TSOLRow,
    all_tsol_op_names,
    collect_tsol_coverage,
    coverage_summary,
    render_dashboard,
    render_spec_catalog,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = REPO_ROOT / "docs" / "audit" / "generated" / "tsol_coverage.md"
SPEC = REPO_ROOT / "docs" / "operations" / "Tessera_Standard_Operations.md"


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


def test_normative_spec_catalog_matches_live_inventory() -> None:
    """The readable catalog in the normative spec is generated from the
    same inventory as the coverage dashboard."""
    text = SPEC.read_text()
    assert text.count(SPEC_CATALOG_BEGIN) == 1
    assert text.count(SPEC_CATALOG_END) == 1
    start = text.index(SPEC_CATALOG_BEGIN)
    end = text.index(SPEC_CATALOG_END) + len(SPEC_CATALOG_END)
    assert text[start:end] == render_spec_catalog(), (
        "TSOL normative catalog drifted from tsol_coverage.py; regenerate "
        "the marked catalog block."
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
    hardware proofs across every declared target. Exact-target proof
    may exist while this all-target aggregate remains incomplete."""
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


# ── Structural gates ──────────────────────────────────────────────────────
#
# The dashboard drift gate compares the generated file against
# `render_dashboard()`. That makes it blind by construction to any *constant*
# baked into the renderer: both sides carry the same wrong literal and agree.
# Three stale claims survived that way (a 432-entry registry that had grown to
# 482, a `primitive_coverage.py line 351-352` reference pointing at an
# unrelated table, and a hardcoded "zero"). These gates cover the class.


def test_dashboard_quotes_no_hardcoded_registry_size() -> None:
    """Any registry-size claim must be derived, so it cannot go stale.

    Catches the reintroduction of a literal like "the full 432-primitive
    registry" — self-consistent with the renderer, and wrong.
    """
    from tessera.compiler.primitive_coverage import all_primitive_coverages

    live = len(all_primitive_coverages())
    text = DASHBOARD.read_text(encoding="utf-8")
    assert f"{live}-primitive registry" in text, (
        "the dashboard no longer reports the live registry size; it must be "
        "derived from all_primitive_coverages(), never written as a literal")
    for stale in re.findall(r"(\d+)-primitive registry", text):
        assert int(stale) == live, (
            f"dashboard claims a {stale}-primitive registry but the live "
            f"registry has {live}; render this from the registry, not a literal")


def test_dashboard_cites_no_source_line_numbers() -> None:
    """Line-number citations rot on the next edit above them."""
    text = DASHBOARD.read_text(encoding="utf-8")
    offenders = re.findall(r"`?[\w/]+\.py`?\s+line[s]?\s+[\d-]+", text)
    assert not offenders, (
        f"dashboard cites source line numbers {offenders}; name the rule, "
        "function, or constant instead — line numbers go stale silently")


def test_backend_kernel_aggregate_claim_is_derived_not_asserted() -> None:
    """The "zero complete" sentence must track the registry.

    It is true today, which is exactly why it is dangerous: a literal stays
    "zero" on the day the first entry legitimately completes.
    """
    summary = coverage_summary()
    complete = summary["backend_kernel"].get("complete", 0)
    text = DASHBOARD.read_text(encoding="utf-8")
    expected = "**zero**" if complete == 0 else f"**{complete}**"
    assert f"Today {expected} of the" in text, (
        f"dashboard's backend_kernel aggregate disagrees with the registry "
        f"({complete} complete); it must be rendered from coverage_summary()")


def test_regeneration_instruction_names_the_owning_generator() -> None:
    """Following the dashboard's own instructions must not leave the CSV stale.

    `generated_docs` owns both the .md and the .csv, so a `python -c` snippet
    that writes only the .md leaves `check_generated_docs.sh` failing with no
    hint about why.
    """
    text = DASHBOARD.read_text(encoding="utf-8")
    assert "python -m tessera.compiler.generated_docs --write tsol_coverage" in text
    assert "python -c" not in text, (
        "the dashboard tells readers to regenerate with a python -c one-liner; "
        "that writes only the Markdown and leaves the CSV stale")


def test_dashboard_writer_pins_utf8() -> None:
    """The dashboard contains ✅ / ◐ / ◯; the platform locale must not decide."""
    source = (
        REPO_ROOT / "python/tessera/compiler/tsol_coverage.py"
    ).read_text(encoding="utf-8")
    assert 'target.write_text(render_dashboard(), encoding="utf-8")' in source, (
        "write_dashboard must pass encoding='utf-8' explicitly — the rendered "
        "glyphs are non-ASCII and would raise under a non-UTF-8 locale")
