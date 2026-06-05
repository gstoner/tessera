"""Arch-2 (2026-05-22) — MLIR verifier coverage drift gate.

Pins:
  * Known sprint deliverables (V1 / V4b / V6c / V8) stay at
    ``impl_status == "real"``.
  * No op is in the ``absent`` bucket (``hasVerifier = 1;`` without an
    implementing ``verify()`` body) — that's always a build-broken
    regression.
  * The generated dashboard at ``docs/audit/generated/verifier_coverage.md``
    is consistent with the live registry render.
  * Summary counts stay at-or-above floor values so regressions in the
    ``real`` bucket fail fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler.verifier_coverage import (
    VerifierEntry,
    CSV_COLUMNS,
    collect_verifier_coverage,
    coverage_summary,
    render_csv,
    render_dashboard,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_DASHBOARD = REPO_ROOT / "docs" / "audit" / "generated" / "verifier_coverage.csv"
MD_DASHBOARD = REPO_ROOT / "docs" / "audit" / "generated" / "verifier_coverage.md"


# ─────────────────────────────────────────────────────────────────────────
# Locked sentinels — these ops MUST stay at impl_status == "real".
# Each entry was landed in a specific MLIR verifier sprint and a
# regression would break the corresponding lit fixture.
# ─────────────────────────────────────────────────────────────────────────

_LOCKED_REAL_OPS = (
    # V1 (2026-05-22): first batch of verifiers added.
    "TransposeOp",
    "LayerNormOp",
    "MoeDispatchOp",
    # V4b (2026-05-22): long-tail per-op verifiers.
    "CastOp",
    "SoftmaxOp",
    "RopeOp",
    "DropoutOp",
    # V6a (2026-05-22): ReshapeOp registered with element-count-preserving check.
    "ReshapeOp",
    # V3 (2026-05-22): target-aware head_dim ceiling on FlashAttnOp.
    "FlashAttnOp",
    # Pre-V1 already-real verifiers (regression gates).
    "MatmulOp",
    "Conv2DNHWCOp",
    "FusedEpilogueOp",
    "AttnLocalWindow2DOp",
    # V6c (2026-05-22): per-SM tile_q / tile_kv ceilings on FA-4 op.
    "ScaledDotProductOp",
    # V8 (2026-05-22): Queue dialect verifier closure.
    "CreateOp",
    "PushOp",
    "PopOp",
    # Sprint B (2026-06-04): attention-family + KV/MoE verifier hardening.
    "MultiHeadAttentionOp",
    "GQAAttentionOp",
    "MQAAttentionOp",
    "MLADecodeOp",
    "MLADecodeFusedOp",
    "LinearAttnOp",
    "LinearAttnStateOp",
    "PowerAttnOp",
    "LightningAttentionOp",
    "GatedAttentionOp",
    "AttnSlidingWindowOp",
    "AttnCompressedBlocksOp",
    "AttnTopKBlocksOp",
    "DeepSeekSparseAttentionOp",
    "MoeCombineOp",
    "KVCacheAppendOp",
    "KVCachePruneOp",
)


@pytest.fixture(scope="module")
def coverage_entries() -> tuple[VerifierEntry, ...]:
    return collect_verifier_coverage()


@pytest.mark.parametrize("op_class", _LOCKED_REAL_OPS)
def test_locked_real_op_stays_real(coverage_entries, op_class: str) -> None:
    """A sprint deliverable's verifier status must remain `real`."""
    matches = [e for e in coverage_entries if e.op_class == op_class]
    assert matches, (
        f"locked op {op_class!r} disappeared from ODS — a TD edit "
        f"likely deleted the `def Tessera_{op_class}` declaration"
    )
    assert len(matches) == 1, (
        f"locked op {op_class!r} declared in multiple TD files "
        f"(ambiguous); review the dialect surface"
    )
    entry = matches[0]
    assert entry.impl_status == "real", (
        f"locked op {op_class!r} regressed from `real` to "
        f"`{entry.impl_status}` — either hasVerifier was removed or the "
        f"verify() body was reduced to a trivial stub.  Restore the "
        f"sprint deliverable or document the rollback in SHAPE_SYSTEM.md."
    )


# ─────────────────────────────────────────────────────────────────────────
# Structural invariants — apply across the whole coverage set.
# ─────────────────────────────────────────────────────────────────────────


def test_no_absent_verifiers(coverage_entries) -> None:
    """``absent`` means ``hasVerifier = 1;`` was declared in TD but no
    matching ``OpName::verify()`` body exists.  That's a build break in
    waiting — should be zero across the project."""
    absents = [e.op_class for e in coverage_entries if e.impl_status == "absent"]
    assert not absents, (
        f"Ops with hasVerifier=1 but no verify() body found: {absents}.  "
        f"This is a build error in waiting.  Either implement the verifier "
        f"or remove `let hasVerifier = 1;` from the TD."
    )


def test_summary_floor_counts(coverage_entries) -> None:
    """Lock floor values so a regression in the `real` bucket fails
    fast.  These numbers were captured 2026-05-22 right after V8.

    NOTE: floor is one-sided — `real` count is allowed to grow.  When
    a future sprint flips more ops to `real`, just update the floor.
    """
    summary = coverage_summary()
    assert summary["real"] >= 19, (
        f"Real-verifier count regressed below 19 (got {summary['real']}).  "
        f"A sprint deliverable likely lost its verify() body."
    )
    assert summary["absent"] == 0, (
        f"Absent count must stay at 0; got {summary['absent']}"
    )
    # Total is informational — the parser pulls from a manually-listed
    # set of TD files, so the count grows when new dialects are added.
    assert summary["total"] >= 100, (
        f"Coverage total dropped below 100 ops (got {summary['total']}).  "
        f"Either a dialect TD was deleted or the parser is missing it."
    )


def test_no_duplicate_op_class_names(coverage_entries) -> None:
    """Two TD files defining ops with the same C++ class name would
    cause linker collisions and mask a verifier status check."""
    seen: dict[str, Path] = {}
    for entry in coverage_entries:
        if entry.op_class in seen:
            pytest.fail(
                f"op_class {entry.op_class!r} declared in both "
                f"{seen[entry.op_class]} and {entry.td_file}"
            )
        seen[entry.op_class] = entry.td_file


# ─────────────────────────────────────────────────────────────────────────
# Generated dashboard ↔ live registry drift gate
#
# The CSV is the canonical, machine-readable artifact and the only thing
# we byte-compare.  The Markdown companion is checked only for existence
# + canonical heading phrases, so cosmetic formatting never reds CI.
# Regenerate both with:
#   `python -m tessera.compiler.audit verifier_coverage --write`
# ─────────────────────────────────────────────────────────────────────────


def test_csv_dashboard_exists() -> None:
    assert CSV_DASHBOARD.exists(), (
        f"Generated CSV missing: {CSV_DASHBOARD.relative_to(REPO_ROOT)}.  "
        f"Regenerate via "
        f"`python -m tessera.compiler.audit verifier_coverage --write`."
    )


def test_csv_matches_live_registry() -> None:
    """The checked-in CSV must match what the registry renders right
    now.  When a sprint changes the ODS surface, regenerate the CSV or
    this test fails."""
    if not CSV_DASHBOARD.exists():
        pytest.skip("CSV not generated yet")
    live = render_csv()
    on_disk = CSV_DASHBOARD.read_text()
    if live != on_disk:
        live_lines = live.splitlines()
        disk_lines = on_disk.splitlines()
        first_diff = next(
            (i for i, (l, d) in enumerate(zip(live_lines, disk_lines)) if l != d),
            min(len(live_lines), len(disk_lines)),
        )
        pytest.fail(
            f"Verifier coverage CSV drift at line {first_diff + 1}: "
            f"on-disk has {disk_lines[first_diff]!r}, live has "
            f"{live_lines[first_diff]!r}.  Regenerate with "
            f"`python -m tessera.compiler.audit verifier_coverage --write`."
        )


def test_csv_header_is_stable() -> None:
    """Downstream tooling parses the CSV by header name; the column
    order is an append-only contract."""
    first_line = render_csv().splitlines()[0]
    assert first_line == ",".join(CSV_COLUMNS)


def test_markdown_companion_exists_with_canonical_phrases() -> None:
    """The human-readable Markdown is regenerated alongside the CSV but
    is NOT byte-gated — we only require it to exist and carry the
    canonical headings downstream docs link to."""
    assert MD_DASHBOARD.exists(), (
        f"Markdown companion missing: {MD_DASHBOARD.relative_to(REPO_ROOT)}."
    )
    text = render_dashboard()  # render fresh; on-disk MD is not byte-gated
    for phrase in (
        "# MLIR Verifier Coverage Dashboard",
        "| Status | Count | Meaning |",
        "`real`",
        "`trivial_stub`",
        "`absent`",
        "`no_verifier`",
        "## Per-dialect details",
    ):
        assert phrase in text, (
            f"dashboard missing canonical phrase {phrase!r} — "
            f"a refactor changed the render format"
        )
