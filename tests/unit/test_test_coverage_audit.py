"""Audit-D (2026-05-22) — drift gate for the test-coverage-by-op audit.

The ``primitive_coverage.py`` registry marks all 432 entries as
``tests=complete``.  That's a category-level rollup, not a per-op
proof.  This gate sits on top of
``python/tessera/compiler/test_coverage_audit.py`` and pins the
plausible numbers so we notice when:

  * A sweep accidentally deletes a chunk of tests (total floors).
  * A sentinel high-traffic op (matmul / flash_attn / softmax / gemm)
    silently loses coverage during a rewrite.
  * The generated dashboard at
    ``docs/audit/generated/test_coverage_by_op.md`` drifts out of
    sync with the live scan.

Honest scope note matches the module's: reference counts are not
the same as numerical-coverage quality.  The floors here are loose
on purpose — they catch regressions, not normal week-to-week churn.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler.test_coverage_audit import (
    OpTestCoverage,
    collect_op_test_coverage,
    coverage_summary,
    render_dashboard,
    thinly_tested_ops,
    top_tested_ops,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = REPO_ROOT / "docs" / "audit" / "generated" / "test_coverage_by_op.md"


# ─────────────────────────────────────────────────────────────────────────
# Headline floors
# ─────────────────────────────────────────────────────────────────────────


def test_total_op_count_floor() -> None:
    """The registry currently has 432 ops; floor at 400 to absorb the
    rare deletion but flag accidental large drops."""
    rows = collect_op_test_coverage()
    assert len(rows) >= 400, (
        f"Op count dropped below 400 (got {len(rows)}).  "
        f"Did the registry get partially cleared?"
    )


def test_total_reference_count_floor() -> None:
    """Locked at the Audit-D landing — total Python + lit refs across
    every op should stay near the 2,390 baseline.  Floor at 1,800
    absorbs ordinary churn but flags a sweep that accidentally
    deletes whole test files."""
    summary = coverage_summary()
    total = summary["total_python_refs"] + summary["total_lit_refs"]
    assert total >= 1800, (
        f"Total test references across all ops dropped below 1,800 "
        f"(got {total} = {summary['total_python_refs']} py + "
        f"{summary['total_lit_refs']} lit).  Did a test sweep get "
        f"deleted by accident?"
    )


def test_well_tested_op_count_floor() -> None:
    """At Audit-D landing, 41 ops have ≥10 references.  Floor at 30
    means most heavy hitters retained meaningful coverage."""
    summary = coverage_summary()
    assert summary["well_tested"] >= 30, (
        f"Well-tested op count (≥10 refs) dropped below 30 "
        f"(got {summary['well_tested']})."
    )


def test_ops_with_negative_tests_floor() -> None:
    """At Audit-D landing, 37 ops have at least one nearby
    `pytest.raises`.  Floor at 25 — negative-path coverage is the
    most valuable kind, so a regression here is worth flagging."""
    summary = coverage_summary()
    assert summary["with_negative_tests"] >= 25, (
        f"Ops with negative-test coverage dropped below 25 "
        f"(got {summary['with_negative_tests']}).  Did we lose a "
        f"`pytest.raises` block during a refactor?"
    )


# ─────────────────────────────────────────────────────────────────────────
# Sentinel ops — high-traffic primitives that must NOT silently
# lose coverage.
# ─────────────────────────────────────────────────────────────────────────


# (op_name, minimum_total_refs).  Floors are set to roughly half the
# current count — they catch real regressions while absorbing a 50%
# sweep without a doc update.
_SENTINEL_OP_FLOORS = (
    ("matmul",     200),    # currently 425
    ("flash_attn",  60),    # currently 141
    ("gemm",        50),    # currently 131
    ("softmax",     50),    # currently 121
    ("reduce",      50),    # currently 128
    ("mul",         50),    # currently 106
    ("add",         30),    # currently  72
    ("relu",        20),    # currently  52
    ("dropout",     15),    # currently  38
    ("gelu",        15),    # currently  38
    ("rope",        15),    # currently  37
    ("transpose",   15),    # currently  35
    ("layer_norm",  12),    # currently  32
    ("cast",        10),    # currently  28
)


@pytest.mark.parametrize(
    "op_name,floor", _SENTINEL_OP_FLOORS, ids=[s[0] for s in _SENTINEL_OP_FLOORS]
)
def test_sentinel_op_reference_count_floor(op_name: str, floor: int) -> None:
    """High-traffic ops must retain at least ~half their current
    test coverage.  A drop below floor means a major test file
    got accidentally truncated or renamed."""
    rows = {r.op_name: r for r in collect_op_test_coverage()}
    cov = rows.get(op_name)
    assert cov is not None, (
        f"Sentinel op {op_name!r} is missing from primitive_coverage.py.  "
        f"Was the registry refactored?"
    )
    assert cov.total_refs >= floor, (
        f"Sentinel op {op_name!r} test references dropped below floor "
        f"{floor} (got {cov.total_refs} = {cov.python_refs} py + "
        f"{cov.lit_refs} lit).  Investigate whether a test file was "
        f"deleted or renamed."
    )


# ─────────────────────────────────────────────────────────────────────────
# Matmul / flash_attn — the two flagship ops — must keep their
# negative-test coverage.
# ─────────────────────────────────────────────────────────────────────────


def test_matmul_has_negative_tests() -> None:
    rows = {r.op_name: r for r in collect_op_test_coverage()}
    matmul = rows["matmul"]
    assert matmul.negative_refs >= 5, (
        f"matmul lost negative-test coverage (got "
        f"{matmul.negative_refs} pytest.raises near references; "
        f"baseline ~18)."
    )


def test_gemm_has_negative_tests() -> None:
    rows = {r.op_name: r for r in collect_op_test_coverage()}
    gemm = rows["gemm"]
    assert gemm.negative_refs >= 5, (
        f"gemm lost negative-test coverage (got {gemm.negative_refs} "
        f"pytest.raises near references; baseline ~16)."
    )


# ─────────────────────────────────────────────────────────────────────────
# Thin-coverage envelope — the thinly-tested set should NOT explode.
# ─────────────────────────────────────────────────────────────────────────


def test_thinly_tested_count_does_not_balloon() -> None:
    """At Audit-D landing, 241/432 ops have ≤1 reference.  Many are
    legitimate (alias rollups, S15 data combinators tested elsewhere,
    optimizer / schedule helpers under conformance smoke tests, etc.).
    Cap at 300 — going above that means we've added a wave of
    unexposed primitives without writing tests."""
    summary = coverage_summary()
    assert summary["thinly_tested"] <= 300, (
        f"Thinly-tested op count rose above 300 "
        f"(got {summary['thinly_tested']}).  A wave of unexposed "
        f"primitives was added without tests."
    )


# ─────────────────────────────────────────────────────────────────────────
# Top-20 invariant
# ─────────────────────────────────────────────────────────────────────────


def test_matmul_is_the_most_tested_op() -> None:
    """matmul is the universal proof anchor across CPU/GPU/lit suites.
    If anything else ever leads the leaderboard, investigate — that's
    not a regression, but it does mean the testing surface shifted
    significantly."""
    top = top_tested_ops(1)
    assert top[0].op_name == "matmul", (
        f"Top-tested op is {top[0].op_name!r}, not 'matmul' "
        f"(refs={top[0].total_refs}).  This is unusual — verify the "
        f"matmul test suite is intact."
    )


# ─────────────────────────────────────────────────────────────────────────
# Dashboard drift gate
# ─────────────────────────────────────────────────────────────────────────


def test_dashboard_exists() -> None:
    assert DASHBOARD.exists(), (
        f"Generated dashboard missing: "
        f"{DASHBOARD.relative_to(REPO_ROOT)}.  Regenerate via "
        f"`tessera.compiler.test_coverage_audit.write_dashboard()`."
    )


def test_dashboard_pins_canonical_phrases() -> None:
    """The dashboard structure (headings) is part of the contract —
    consumers (and future docs) link into these sections."""
    if not DASHBOARD.exists():
        pytest.skip("dashboard not generated yet")
    text = DASHBOARD.read_text()
    for phrase in (
        "# Test Coverage by Op Family",
        "## Headline",
        "## Top 20 most-tested ops",
        "## Thinly-tested ops",
        "Honest scope note",
    ):
        assert phrase in text, (
            f"Test-coverage dashboard missing canonical phrase {phrase!r}"
        )


def test_dashboard_includes_matmul_sentinel() -> None:
    """The top section must mention matmul — it's the single most
    visible op in the audit and its omission means the renderer
    broke."""
    if not DASHBOARD.exists():
        pytest.skip("dashboard not generated yet")
    text = DASHBOARD.read_text()
    assert "`matmul`" in text, (
        "Dashboard rendered without the matmul row — top-20 table broke."
    )


def test_dashboard_headline_numbers_match_live_scan() -> None:
    """The on-disk dashboard's headline numbers should match the
    live scan within reasonable tolerance (allowing for the rare
    case where a developer adds a test but hasn't regenerated the
    dashboard yet)."""
    if not DASHBOARD.exists():
        pytest.skip("dashboard not generated yet")
    text = DASHBOARD.read_text()
    summary = coverage_summary()
    # The total ops number is the most stable — it changes only when
    # the registry changes.  Require it to be in the dashboard.
    assert f"**{summary['total_ops']}** ops" in text, (
        f"Dashboard headline ops count is stale.  Live scan says "
        f"{summary['total_ops']} ops; regenerate via "
        f"`tessera.compiler.test_coverage_audit.write_dashboard()`."
    )
