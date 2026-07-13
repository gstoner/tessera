"""Visual-complex status-document routing checks.

The public status card must remain a compact entry point to generated
evidence. The dated milestone belongs under ``docs/audit/domain`` and must
identify itself as historical rather than presenting a second live inventory.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

PUBLIC_STATUS = REPO_ROOT / "docs" / "status" / "visual_complex.md"
STATUS_INDEX = REPO_ROOT / "docs" / "status" / "README.md"
MILESTONE = (
    REPO_ROOT / "docs" / "audit" / "domain" / "archive"
    / "visual_complex_milestone_2026_05.md"
)
SUPPORT_TABLE = (
    REPO_ROOT / "docs" / "audit" / "generated" / "support_table.md"
)


def test_public_status_doc_exists() -> None:
    assert PUBLIC_STATUS.exists(), (
        f"public status doc missing: {PUBLIC_STATUS}"
    )


def test_public_status_doc_names_required_surfaces() -> None:
    text = PUBLIC_STATUS.read_text()
    # Surface coverage — these must be named so external readers can find it.
    for token in (
        "tessera.complex",
        "@complex_jit",
        "@analytic",
        "mobius",
        "stereographic",
        "Wirtinger",
    ):
        assert token in text, f"missing required token {token!r}"


def test_public_status_doc_routes_to_generated_evidence() -> None:
    """The status card must not become a second mutable target inventory."""
    text = PUBLIC_STATUS.read_text()
    for token in (
        "support_table.md",
        "runtime_execution_matrix.md",
        "visual_complex_fused",
        "separate facts",
    ):
        assert token in text, f"missing evidence-routing token {token!r}"


def test_public_status_doc_cross_links_milestone_audit() -> None:
    text = PUBLIC_STATUS.read_text()
    assert "visual_complex_milestone_2026_05.md" in text


def test_milestone_doc_redirects_external_readers_to_public_doc() -> None:
    text = MILESTONE.read_text()
    # The redirect banner must appear near the frontmatter and title.
    head = "\n".join(text.splitlines()[:20])
    assert "visual_complex.md" in head, (
        "milestone doc must direct external readers to the public status"
    )
    assert "historical" in head.lower(), (
        "milestone doc must declare itself historical"
    )


def test_status_index_routes_to_both_status_cards() -> None:
    text = STATUS_INDEX.read_text()
    assert "ga_ebm.md" in text
    assert "visual_complex.md" in text
    assert "release_gates.md" in text


def test_support_table_includes_visual_complex_rows() -> None:
    """The generated support table must surface the visual_complex
    family.  This pins one of the M7-shipped guarantees."""
    text = SUPPORT_TABLE.read_text()
    # Row count belongs to the generated table, so only require the family.
    occurrences = text.count("visual_complex")
    assert occurrences >= 20, (
        f"expected at least 20 visual_complex references in support_table; "
        f"found {occurrences}"
    )
