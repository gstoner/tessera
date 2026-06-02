"""GA4 acceptance: GA primitives registered in primitive_coverage.

Sprint: GA4 (registry bookkeeping).
Roadmap: docs/audit/domain/DOMAIN_AUDIT.md § GA4

Each shipped GA op gets an entry in `primitive_coverage.py` under
`category="geometric_algebra"`. This lets the existing
`standalone_primitive_coverage.md` dashboard track the GA surface and
the per-axis hardening sweeps include it.
"""

from __future__ import annotations

import pytest

from tessera.compiler import primitive_coverage as pc


EXPECTED_GA_PRIMITIVES = {
    # GA3 — core multivector ops
    "clifford_geometric_product",
    "clifford_grade_projection",
    "clifford_wedge",
    "clifford_left_contraction",
    "clifford_inner",
    "clifford_reverse",
    "clifford_grade_involution",
    "clifford_conjugate",
    "clifford_norm",
    "clifford_exp",
    "clifford_log",
    "clifford_rotor_sandwich",
    # GA5 — differential-form calculus
    "clifford_hodge_star",
    "clifford_ext_deriv",
    "clifford_codiff",
    "clifford_vec_deriv",
    "clifford_integral",
}


def test_ga_primitives_registered_under_geometric_algebra_category() -> None:
    ga_entries = {
        e.name: e
        for e in pc.all_primitive_coverages().values()
        if e.category == "geometric_algebra"
    }
    assert EXPECTED_GA_PRIMITIVES.issubset(ga_entries.keys())
    for name in EXPECTED_GA_PRIMITIVES:
        assert ga_entries[name].status == "planned"
        assert ga_entries[name].category == "geometric_algebra"


def test_ga_entries_each_have_a_reference() -> None:
    """GA primitives are mathematically classical; each entry cites a
    standard reference (Hestenes, Doran & Lasenby, Frankel)."""
    ga_entries = [
        e for e in pc.all_primitive_coverages().values()
        if e.category == "geometric_algebra"
    ]
    for entry in ga_entries:
        assert len(entry.references) >= 1, f"{entry.name} has no references"


def test_ga_entries_contribute_to_dashboard_summary() -> None:
    """The dashboard's category enumeration includes 'geometric_algebra'."""
    all_categories = sorted({
        e.category for e in pc.all_primitive_coverages().values()
    })
    assert "geometric_algebra" in all_categories


def test_ga4_count_matches_shipped_surface() -> None:
    """Sanity: 12 GA3 ops + 5 GA5 ops = 17 primitives."""
    ga_entries = [
        e for e in pc.all_primitive_coverages().values()
        if e.category == "geometric_algebra"
    ]
    assert len(ga_entries) == 17


def test_ebm_partition_primitives_present_after_ebm3() -> None:
    """While we're auditing the registry, double-check EBM3 lands too."""
    names = {e.name for e in pc.all_primitive_coverages().values()}
    expected = {"ebm_partition_exact", "ebm_partition_monte_carlo", "ebm_partition_ais"}
    assert expected.issubset(names)


def test_no_unknown_category_collisions() -> None:
    """Categories used in `geometric_algebra` and `ebm` entries don't
    accidentally clash with existing categories used for other surfaces."""
    by_cat = {}
    for e in pc.all_primitive_coverages().values():
        by_cat.setdefault(e.category, []).append(e.name)
    # Every category has at least one entry.
    for cat, names in by_cat.items():
        assert len(names) >= 1, f"empty category {cat!r}"
    # geometric_algebra is solely populated by clifford_* primitives.
    for name in by_cat["geometric_algebra"]:
        assert name.startswith("clifford_"), (
            f"geometric_algebra category has non-clifford entry {name!r}"
        )
