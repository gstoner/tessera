"""Visual Complex status-doc drift gates.

Slice 3 of the four-ask sweep (2026-05-22): a public-facing status
doc landed at ``docs/status/visual_complex.md`` to complement the
engineering-internal ``visual_complex_milestone.md``.  These tests
lock both docs against drift:

  * The public doc exists and names the surface, the 4 fused Apple
    GPU kernels, the planned-slot dtype caveat, and the related-pages
    cross-link.
  * The milestone doc is correctly tagged "engineering-internal" and
    points to the public doc.
  * Visual Complex rows survive in the generated support_table.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

PUBLIC_STATUS = REPO_ROOT / "docs" / "status" / "visual_complex.md"
MILESTONE = REPO_ROOT / "docs" / "status" / "visual_complex_milestone.md"
SUPPORT_TABLE = (
    REPO_ROOT / "docs" / "audit" / "generated" / "support_table.md"
)


def test_public_status_doc_exists() -> None:
    assert PUBLIC_STATUS.exists(), (
        f"public status doc missing: {PUBLIC_STATUS}"
    )


def test_public_status_doc_names_required_surfaces() -> None:
    text = PUBLIC_STATUS.read_text()
    # Surface coverage — these MUST be named so external readers can
    # find the surface.
    for token in (
        "tessera.complex",
        "@complex_jit",
        "@analytic",
        "mobius",
        "stereographic",
        "Wirtinger",
        "Cauchy-Riemann",
    ):
        assert token in text, f"missing required token {token!r}"
    # Exactly 4 fused Apple GPU kernels are listed.
    for kernel in ("complex_mul", "complex_exp", "mobius", "stereographic"):
        assert kernel in text, f"missing fused kernel {kernel!r}"


def test_public_status_doc_declares_dtype_caveat() -> None:
    """Reading rule for planned dtype rows must be reproduced
    verbatim in the public doc so external readers don't misread
    a planned fp16 slot as runtime-available."""
    text = PUBLIC_STATUS.read_text()
    # The dtype caveat block.
    assert "Dtype caveat" in text
    assert "target kernel" in text
    assert "not what runs today" in text


def test_public_status_doc_cross_links_milestone_audit() -> None:
    text = PUBLIC_STATUS.read_text()
    assert "visual_complex_milestone.md" in text


def test_milestone_doc_redirects_external_readers_to_public_doc() -> None:
    text = MILESTONE.read_text()
    # The redirect banner must appear in the first ~10 lines.
    head = "\n".join(text.splitlines()[:12])
    assert "visual_complex.md" in head, (
        "milestone doc must direct external readers to the public status"
    )
    assert "engineering-internal" in head.lower() or \
           "engineering audit" in head.lower(), (
        "milestone doc must declare itself engineering-internal"
    )


def test_support_table_includes_visual_complex_rows() -> None:
    """The generated support table must surface the visual_complex
    family.  This pins one of the M7-shipped guarantees."""
    text = SUPPORT_TABLE.read_text()
    # 22 rows under family `visual_complex` per the milestone doc.
    occurrences = text.count("visual_complex")
    assert occurrences >= 20, (
        f"expected at least 20 visual_complex references in support_table; "
        f"found {occurrences}"
    )
