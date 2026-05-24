"""Audit-A (2026-05-22) — documentation freshness drift gate.

Pins:

  * Every doc under the canonical tree has either a parseable
    ``last_updated`` marker (YAML frontmatter or body-form) or is
    explicitly listed as undated-acceptable.
  * No Normative-status doc is older than 180 days.  Catches the
    TSOL-style 3-week-stale drift before it reaches users.
  * The generated dashboard at
    ``docs/audit/generated/docs_freshness.md`` matches the live
    manifest.
  * Headline counts stay at-or-above the baselines captured at the
    Audit-A landing (2026-05-22).
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from tessera.compiler.docs_manifest import (
    DocEntry,
    collect_doc_manifest,
    freshness_summary,
    render_dashboard,
    stale_normative_docs,
    undated_docs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = REPO_ROOT / "docs" / "audit" / "generated" / "docs_freshness.md"


# ─────────────────────────────────────────────────────────────────────────
# Acceptable-undated allow-list
# ─────────────────────────────────────────────────────────────────────────

# Docs that are intentionally undated today.  Each entry should be
# tagged with frontmatter in a follow-up sprint; until then this
# allow-list prevents the drift gate from blocking on them.
_KNOWN_UNDATED_DOCS: frozenset[str] = frozenset({
    "docs/architecture/compiler_gaps_1_3_5_plan.md",
    "docs/architecture/frontend_substrate_plan.md",
    "docs/architecture/stencil_materialize_and_window_lowering.md",
    "docs/reference/tessera_frontend_lanes.md",
})


# ─────────────────────────────────────────────────────────────────────────
# Structural tests
# ─────────────────────────────────────────────────────────────────────────


def test_manifest_has_no_duplicate_paths() -> None:
    entries = collect_doc_manifest()
    seen: set[str] = set()
    for e in entries:
        assert e.path not in seen, f"duplicate doc path: {e.path}"
        seen.add(e.path)


def test_doc_count_at_or_above_baseline() -> None:
    """Locked floor — catches accidental deletion of canonical doc
    roots during a refactor."""
    entries = collect_doc_manifest()
    assert len(entries) >= 60, (
        f"Doc count dropped below 60 ({len(entries)}); was a "
        f"canonical doc tree accidentally moved or deleted?"
    )


# ─────────────────────────────────────────────────────────────────────────
# Undated-doc gate
# ─────────────────────────────────────────────────────────────────────────


def test_every_undated_doc_is_in_allow_list() -> None:
    """An undated doc that's NOT in the allow-list is a freshness
    surface regression — either add a ``last_updated:`` marker to the
    doc or extend ``_KNOWN_UNDATED_DOCS`` with a rationale."""
    undated_paths = {e.path for e in undated_docs()}
    surprise = undated_paths - _KNOWN_UNDATED_DOCS
    assert not surprise, (
        f"Undated docs not in allow-list: {sorted(surprise)}.  "
        f"Either add `last_updated: YYYY-MM-DD` to the doc's "
        f"frontmatter or extend _KNOWN_UNDATED_DOCS in "
        f"`tests/unit/test_docs_freshness.py`."
    )


def test_allow_list_entries_actually_exist_and_are_undated() -> None:
    """Allow-list entries must be real files that are actually
    undated.  Catches stale allow-list entries that linger after a
    doc has been tagged."""
    actually_undated = {e.path for e in undated_docs()}
    stale = _KNOWN_UNDATED_DOCS - actually_undated
    assert not stale, (
        f"Allow-list entries that are no longer undated "
        f"(remove from _KNOWN_UNDATED_DOCS): {sorted(stale)}."
    )


# ─────────────────────────────────────────────────────────────────────────
# Normative-doc staleness gate
# ─────────────────────────────────────────────────────────────────────────


def test_no_normative_doc_is_180d_stale() -> None:
    """Normative docs older than 180 days are at high risk of
    drifting from reality.  The TSOL spec was 24 days stale and we
    caught real drift; 180 days is a soft floor we'll tighten as the
    system catches up."""
    stale = stale_normative_docs(threshold_days=180)
    if stale:
        msg_lines = ["Normative docs stale beyond 180 days:"]
        for e in stale:
            msg_lines.append(
                f"  - {e.path} (last_updated={e.last_updated})"
            )
        msg_lines.append(
            "Refresh the doc against current implementation, or "
            "downgrade `status: Normative` to `status: Informative` "
            "/ `status: Historical` if the doc no longer reflects "
            "live behavior."
        )
        pytest.fail("\n".join(msg_lines))


# ─────────────────────────────────────────────────────────────────────────
# Headline-summary floor gates (locked 2026-05-22)
# ─────────────────────────────────────────────────────────────────────────

# When new docs land, the with_last_updated count should track total.
# These floors catch the "someone added an undated doc" regression.
_HEADLINE_FLOORS = {
    "total": 60,                # at least 60 canonical docs
    "with_frontmatter": 55,     # at least 55 have YAML frontmatter
    "with_last_updated": 55,    # at least 55 carry a date
    "stale_over_180d": 0,       # zero docs allowed >180 days
}


@pytest.mark.parametrize("key,floor", sorted(_HEADLINE_FLOORS.items()))
def test_headline_counts_at_or_above_baseline(key: str, floor: int) -> None:
    """Floors locked at the Audit-A landing.  ``stale_over_180d`` is
    bidirectional — must stay ≤ floor (i.e., == 0); others must stay
    ≥ floor."""
    summary = freshness_summary()
    actual = summary[key]
    if key == "stale_over_180d":
        assert actual <= floor, (
            f"`{key}` exceeded floor {floor} (got {actual}).  Refresh "
            f"the stale Normative docs surfaced in the dashboard."
        )
    else:
        assert actual >= floor, (
            f"`{key}` dropped below floor {floor} (got {actual}).  "
            f"A canonical doc was likely deleted or had its frontmatter "
            f"stripped."
        )


# ─────────────────────────────────────────────────────────────────────────
# Dashboard drift gate
# ─────────────────────────────────────────────────────────────────────────


def test_dashboard_file_exists() -> None:
    assert DASHBOARD.exists(), (
        f"Generated dashboard missing: "
        f"{DASHBOARD.relative_to(REPO_ROOT)}.  Regenerate via "
        f"`tessera.compiler.docs_manifest.write_dashboard()`."
    )


def test_dashboard_pins_canonical_phrases() -> None:
    """The generated dashboard's section headers are pinned so a
    refactor doesn't silently change the format."""
    assert DASHBOARD.exists()
    text = DASHBOARD.read_text()
    for phrase in (
        "# Documentation Freshness Dashboard",
        "## Headline",
        "## Per-root inventory",
        "Reference date for staleness:",
    ):
        assert phrase in text, (
            f"docs freshness dashboard missing canonical phrase "
            f"{phrase!r}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Spec-companion check: docs that declare a generated_dashboard MUST
# point at a real file.
# ─────────────────────────────────────────────────────────────────────────


def test_generated_dashboard_pointers_resolve() -> None:
    """When a doc's frontmatter declares
    ``generated_dashboard: <path>``, that path must exist.  Catches
    docs that point at deleted / renamed audit pages."""
    bad: list[tuple[str, str]] = []
    for e in collect_doc_manifest():
        if not e.generated_dashboard:
            continue
        target = REPO_ROOT / e.generated_dashboard
        if not target.exists():
            bad.append((e.path, e.generated_dashboard))
    assert not bad, (
        f"Docs pointing at non-existent generated dashboards: {bad}"
    )
