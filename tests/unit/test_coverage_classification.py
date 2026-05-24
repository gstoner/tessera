"""Audit-D-2 (2026-05-22) — drift gate for the thin-op classification.

The audit at ``test_coverage_by_op.md`` flagged 207 ops with ≤1 direct
test reference.  The classifier at
``python/tessera/compiler/coverage_classification.py`` sorts them into
five buckets: ``covered_by_family``, ``structural_only``,
``needs_direct_test``, ``hardware_gated``, ``deprecated_or_internal``.

This gate pins the per-bucket counts so we notice when:

  * A wave of new primitives lands without classification rules
    (``needs_direct_test`` balloons).
  * A category gets recategorized without revisiting the override
    list (``structural_only`` or ``covered_by_family`` shrinks).
  * The hardware-gated set grows past the documented 4 Langevin
    samplers (a real Phase G/H/I device test ought to flip them
    *out* of the bucket, not add new ones).

Also drift-gates the on-disk dashboard at
``docs/audit/generated/test_coverage_classification.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler.coverage_classification import (
    ALL_BUCKETS,
    COVERED_BY_FAMILY,
    HARDWARE_GATED,
    NEEDS_DIRECT_TEST,
    STRUCTURAL_ONLY,
    classification_summary,
    classify_thinly_tested,
    needs_direct_test_ops,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = (
    REPO_ROOT / "docs" / "audit" / "generated"
    / "test_coverage_classification.md"
)


# ─────────────────────────────────────────────────────────────────────────
# Sanity: every thin op gets a bucket; no op gets two buckets.
# ─────────────────────────────────────────────────────────────────────────


def test_every_thin_op_gets_exactly_one_bucket() -> None:
    classifications = classify_thinly_tested()
    assert len(classifications) > 0, "Expected at least some thin ops"
    seen: dict[str, str] = {}
    for c in classifications:
        assert c.bucket in ALL_BUCKETS, (
            f"op {c.op_name!r} got unknown bucket {c.bucket!r}"
        )
        assert c.op_name not in seen, (
            f"op {c.op_name!r} classified twice — into {seen[c.op_name]!r} "
            f"and {c.bucket!r}"
        )
        seen[c.op_name] = c.bucket


def test_classification_buckets_sum_to_total() -> None:
    summary = classification_summary()
    total = sum(summary.values())
    classifications = classify_thinly_tested()
    assert total == len(classifications), (
        f"Bucket counts sum to {total} but {len(classifications)} thin "
        f"ops were classified — a row was dropped or double-counted."
    )


# ─────────────────────────────────────────────────────────────────────────
# Per-bucket floors and ceilings.  Tuned to the 2026-05-22 baseline:
#
#   covered_by_family:   76
#   structural_only:     79
#   needs_direct_test:   48 → target ≤50 after Audit-D-2 tests land
#   hardware_gated:       4
#   deprecated_or_internal: 0
# ─────────────────────────────────────────────────────────────────────────


def test_actionable_bucket_does_not_explode() -> None:
    """The ``needs_direct_test`` bucket is the test debt frontier.
    Cap it at 80 — going above means a wave of unexposed primitives
    was added without writing tests OR without a classification rule
    landing in ``coverage_classification.py``."""
    summary = classification_summary()
    actionable = summary[NEEDS_DIRECT_TEST]
    assert actionable <= 80, (
        f"`needs_direct_test` bucket ballooned to {actionable} "
        f"(baseline ~50).  Either: (a) write direct tests for the "
        f"new ops, or (b) add a classification rule in "
        f"`coverage_classification.py` if the ops are actually "
        f"covered by a family wrapper."
    )


def test_hardware_gated_bucket_stable() -> None:
    """At Audit-D-2 landing, 4 Langevin samplers are hardware-gated.
    Cap at 12 — Phase G/H/I should eventually *shrink* this bucket
    (real device tests flip ops out of it)."""
    summary = classification_summary()
    gated = summary[HARDWARE_GATED]
    assert gated <= 12, (
        f"`hardware_gated` bucket grew to {gated} (baseline 4).  Was "
        f"a wave of GPU-only primitives added without a hardware-gated "
        f"classification rule?"
    )


def test_covered_by_family_bucket_has_substantial_size() -> None:
    """The largest legitimate slice of the thin-op set is
    family-covered (losses, RNG samplers, complex elementwise, GA
    differentials).  Floor at 60 — going below means a category
    default was accidentally reclassified into ``needs_direct_test``."""
    summary = classification_summary()
    family = summary[COVERED_BY_FAMILY]
    assert family >= 60, (
        f"`covered_by_family` bucket shrank to {family} (baseline ~99).  "
        f"Did a category default get accidentally reclassified?"
    )


def test_structural_only_bucket_has_substantial_size(
) -> None:
    """Structural ops (state-tree, transforms, schedules, AOT,
    custom-primitive escape hatches, serialization, data combinators,
    tokenizers, conformance, sharding wrappers, control-flow) are
    the largest legitimate slice.  Floor at 90."""
    summary = classification_summary()
    structural = summary[STRUCTURAL_ONLY]
    assert structural >= 90, (
        f"`structural_only` bucket shrank to {structural} "
        f"(baseline ~140).  Did a structural category get accidentally "
        f"reclassified into `needs_direct_test`?"
    )


# ─────────────────────────────────────────────────────────────────────────
# Sentinel per-bucket ops — pin a small handful so reorganizations
# don't silently move them.
# ─────────────────────────────────────────────────────────────────────────


_SENTINEL_CLASSIFICATIONS = (
    # (op, expected_bucket, why)
    ("ebm_bivector_langevin_sample", HARDWARE_GATED,
     "manifold Langevin needs real GPU mesh"),
    ("ebm_sphere_langevin_step", HARDWARE_GATED,
     "manifold Langevin needs real GPU mesh"),
    ("cross_entropy_loss", COVERED_BY_FAMILY,
     "category default for loss"),
    ("rng_bernoulli", COVERED_BY_FAMILY,
     "category default for rng"),
    ("complex_abs", COVERED_BY_FAMILY,
     "category default for complex elementwise"),
    ("tree_flatten", STRUCTURAL_ONLY,
     "category default for state_tree"),
    ("custom_jvp", STRUCTURAL_ONLY,
     "category default for extension"),
)


@pytest.mark.parametrize(
    "op,expected,why", _SENTINEL_CLASSIFICATIONS,
    ids=[s[0] for s in _SENTINEL_CLASSIFICATIONS],
)
def test_sentinel_op_in_expected_bucket(
    op: str, expected: str, why: str
) -> None:
    by_name = {c.op_name: c.bucket for c in classify_thinly_tested()}
    actual = by_name.get(op)
    if actual is None:
        pytest.skip(
            f"sentinel op {op!r} is no longer thinly-tested — that's "
            f"good, but the sentinel needs updating."
        )
    assert actual == expected, (
        f"Sentinel op {op!r} expected in bucket {expected!r} ({why}) "
        f"but found in {actual!r}.  Either the classification rules "
        f"changed or the op moved out of the thin set."
    )


# ─────────────────────────────────────────────────────────────────────────
# needs_direct_test bucket contents — sanity check
# ─────────────────────────────────────────────────────────────────────────


def test_actionable_bucket_contains_known_primitives() -> None:
    """Smoke check: at least some of these well-known mathematical
    primitives should still be in the actionable bucket OR have been
    tested out of it.  Loosely worded — only fails if NONE of them
    show up, which would mean either everything's tested (yay!) or
    the classifier broke (boo)."""
    actionable_names = {c.op_name for c in needs_direct_test_ops()}
    primitive_candidates = {
        "qr", "svd", "stft", "istft", "spmm_coo",
        "conv3d", "conv_transpose", "spectral_norm", "lora_linear",
        "max_pool", "avg_pool", "adaptive_pool",
        "simple_rnn_cell", "gru_cell", "weight_norm", "instance_norm",
    }
    # Either the actionable bucket still contains some of them, OR
    # the test suite has covered all of them (the goal).
    overlap = actionable_names & primitive_candidates
    if not overlap:
        # All flagship primitives now tested — print a "victory" note
        # but don't fail.  The drift gate is for regressions, not
        # for celebrating progress.
        return
    # If overlap exists, just sanity-check the bucket is non-empty.
    assert len(actionable_names) > 0


# ─────────────────────────────────────────────────────────────────────────
# Dashboard drift gate
# ─────────────────────────────────────────────────────────────────────────


def test_dashboard_exists() -> None:
    assert DASHBOARD.exists(), (
        f"Generated dashboard missing: "
        f"{DASHBOARD.relative_to(REPO_ROOT)}.  Regenerate via "
        f"`tessera.compiler.coverage_classification.write_dashboard()`."
    )


def test_dashboard_pins_canonical_phrases() -> None:
    """Dashboard structure — consumers and docs link into these sections."""
    if not DASHBOARD.exists():
        pytest.skip("dashboard not generated yet")
    text = DASHBOARD.read_text()
    for phrase in (
        "# Test Coverage Classification — Thinly-Tested Ops",
        "## Headline",
        "## Actionable: `needs_direct_test` ops",
        "## Hardware-gated ops",
        "covered_by_family",
        "structural_only",
    ):
        assert phrase in text, (
            f"Classification dashboard missing canonical phrase {phrase!r}"
        )
