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
    _CATEGORY_DEFAULT_BUCKET,
    _NAME_OVERRIDES,
    classification_summary,
    classify_op,
    classify_thinly_tested,
    needs_direct_test_ops,
)
from tessera.compiler.primitive_coverage import all_primitive_coverages
from tessera.compiler.test_coverage_audit import OpTestCoverage


REPO_ROOT = Path(__file__).resolve().parents[2]
# The classification dashboard was merged (2026-06-04) into the
# registry-managed ``test_coverage.md`` (the classification section is
# rendered after the by-op section); these phrase/exists checks hold
# against the merged doc.
DASHBOARD = (
    REPO_ROOT / "docs" / "audit" / "generated" / "test_coverage.md"
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
    Post Audit-D-3, the baseline is **0** — every primitive flagged
    by the audit has at least one direct numerical test in
    ``test_thin_op_direct_coverage.py`` or
    ``test_thin_op_direct_coverage_extra.py``.

    Cap at 25 to absorb new primitives landing between audit cycles
    without immediately blocking the build; any growth here is real
    test debt that should be addressed (either by writing a direct
    test or by adding a classification override saying why the op
    doesn't need one)."""
    summary = classification_summary()
    actionable = summary[NEEDS_DIRECT_TEST]
    assert actionable <= 25, (
        f"`needs_direct_test` bucket grew to {actionable} (baseline "
        f"is 0 as of Audit-D-3).  Either: (a) write a direct test in "
        f"`tests/unit/test_thin_op_direct_coverage_extra.py`, or "
        f"(b) add a classification rule in "
        f"`coverage_classification.py` if the new ops are actually "
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
    differentials).  The bucket *legitimately shrinks* as primitives gain
    direct device tests + real backend coverage — exactly like the
    ``hardware_gated`` bucket (the x86 AVX-512 elementwise lanes,
    2026-06-26, moved reduce/unary/binary/compare ops out as they earned
    real `x86:fused` slots with on-device fixtures).  The guard exists to
    catch an *accidental* default reclassification into
    ``needs_direct_test`` — so the real signal is ``needs_direct_test``
    staying small (asserted separately), not this floor.  Floor at 30.

    2026-06-28: lowered 50 -> 30. The S-series device-lane push (P2-P6:
    softcap/atan2/popcount/lgamma/digamma, lamb/muon/nesterov, group/instance/
    weight norm, complex arithmetic, Philox RNG) earned direct device tests for
    many family-covered ops, legitimately shrinking the bucket to 47 while
    ``needs_direct_test`` stayed at 2 — exactly the expected drift. More P7-P14
    device lanes will shrink it further; the guard tracks ``needs_direct_test``."""
    summary = classification_summary()
    family = summary[COVERED_BY_FAMILY]
    assert family >= 30, (
        f"`covered_by_family` bucket shrank to {family} (baseline ~99, "
        f"floor 30).  Did a category default get accidentally reclassified? "
        f"(Check that `needs_direct_test` stayed small — a shrink driven by "
        f"ops gaining direct device tests is expected.)"
    )


def test_structural_only_bucket_has_substantial_size(
) -> None:
    """Structural ops (state-tree, transforms, schedules, AOT,
    custom-primitive escape hatches, serialization, data combinators,
    tokenizers, conformance, sharding wrappers, control-flow) are
    the largest legitimate slice.  Floor at 85 — the bucket shrinks as
    formerly-structural ops (e.g. the EBM energy/step-compute ops) gain
    direct device tests, which is the desired direction; `needs_direct_test`
    staying small is the real guard."""
    summary = classification_summary()
    structural = summary[STRUCTURAL_ONLY]
    assert structural >= 85, (
        f"`structural_only` bucket shrank to {structural} "
        f"(baseline ~140).  Did a structural category get accidentally "
        f"reclassified into `needs_direct_test`?"
    )


# ─────────────────────────────────────────────────────────────────────────
# Classifier LOGIC tests — exercise classify_op's rules directly instead of
# pinning hand-picked example ops.
#
# The old approach pinned specific ops (e.g. complex_abs, tree_flatten) to an
# expected bucket and looked them up in the *thinly-tested* set.  That decays:
# the moment an example op accumulates >1 test reference it leaves the thin
# set and the assertion self-skips with "sentinel needs updating" — a test
# that quietly stops testing.  classify_op's bucket is a pure function of
# op_name (name override > registry category default > structural fallback;
# the coverage arg is only attached to the result), so we test the *rules*
# across the whole live registry.  These never decay.
# ─────────────────────────────────────────────────────────────────────────


def _dummy_coverage(op_name: str) -> OpTestCoverage:
    """A content-free coverage record — classify_op buckets by op_name only,
    so the counts here don't affect the result."""
    return OpTestCoverage(op_name, 0, 0, 0, (), ())


def test_category_default_table_is_well_formed() -> None:
    # every declared category routes to a real bucket
    for category, bucket in _CATEGORY_DEFAULT_BUCKET.items():
        assert bucket in ALL_BUCKETS, (
            f"category {category!r} maps to unknown bucket {bucket!r}"
        )


def test_name_override_table_is_well_formed() -> None:
    for op, (bucket, reason) in _NAME_OVERRIDES.items():
        assert bucket in ALL_BUCKETS, f"override {op!r} → bad bucket {bucket!r}"
        assert reason, f"override {op!r} has an empty reason"


def test_every_category_routes_to_its_declared_default() -> None:
    # For each declared category, sample a LIVE registry op of that category
    # (one without a per-name override) and verify classify_op routes it to the
    # category default.  Sampling dynamically asks "does the rule still fire",
    # never "is this specific example op still thin" — so it cannot decay.
    covs = all_primitive_coverages()
    mismatches: list[tuple[str, str, str, str]] = []
    sampled = 0
    for category, expected in _CATEGORY_DEFAULT_BUCKET.items():
        rep = next(
            (name for name, c in sorted(covs.items())
             if c.category == category and name not in _NAME_OVERRIDES),
            None,
        )
        if rep is None:
            continue  # declared category with no override-free op: nothing to route
        sampled += 1
        bucket = classify_op(rep, _dummy_coverage(rep)).bucket
        if bucket != expected:
            mismatches.append((category, rep, bucket, expected))
    assert not mismatches, f"category routing broken: {mismatches}"
    assert sampled > 0, "no categories had a sampleable op"


def test_name_overrides_take_precedence_over_category_default() -> None:
    checked = 0
    for op, (bucket, _reason) in _NAME_OVERRIDES.items():
        result = classify_op(op, _dummy_coverage(op))
        assert result.bucket == bucket, (
            f"override op {op!r} classified as {result.bucket!r}, "
            f"expected override bucket {bucket!r}"
        )
        checked += 1
    assert checked > 0, "no name overrides to verify"


def test_unknown_op_falls_back_to_structural_only() -> None:
    name = "__definitely_not_a_real_op__"
    result = classify_op(name, _dummy_coverage(name))
    assert result.bucket == STRUCTURAL_ONLY
    assert "unclassified" in result.reason


def test_classify_op_is_total_and_deterministic_over_registry() -> None:
    # The real safety invariant the sentinels gestured at: EVERY registry op
    # classifies to exactly one valid bucket, reproducibly.
    covs = all_primitive_coverages()
    assert covs, "registry is empty"
    for name in covs:
        a = classify_op(name, _dummy_coverage(name)).bucket
        b = classify_op(name, _dummy_coverage(name)).bucket
        assert a in ALL_BUCKETS, f"op {name!r} → invalid bucket {a!r}"
        assert a == b, f"op {name!r} classification is non-deterministic"


def test_known_bucket_examples_still_route_as_documented() -> None:
    # A few representative ops, asserted through classify_op directly (NOT via
    # the thin-set lookup) so they never self-skip.  These document intent;
    # if the registry category for one changes, update the expectation here.
    covs = all_primitive_coverages()
    expectations = {
        "cross_entropy_loss": COVERED_BY_FAMILY,   # loss family
        "kv_cache_append": NEEDS_DIRECT_TEST,      # state_update (this session)
    }
    for op, expected in expectations.items():
        if op not in covs:
            continue
        assert classify_op(op, _dummy_coverage(op)).bucket == expected, (
            f"{op!r} no longer routes to {expected!r}"
        )


# ─────────────────────────────────────────────────────────────────────────
# needs_direct_test bucket contents — sanity check
# ─────────────────────────────────────────────────────────────────────────


def test_flagship_primitives_have_left_actionable_bucket() -> None:
    """As of Audit-D-3, every flagship primitive (qr, svd, pooling,
    norms, …) has direct numerical tests and has exited the actionable
    bucket.  Lock that in — if any of them returns to the bucket, a
    test file was deleted or a regression broke the op."""
    actionable_names = {c.op_name for c in needs_direct_test_ops()}
    flagships = {
        "qr", "svd", "stft", "istft", "spmm_coo",
        "conv3d", "conv_transpose", "spectral_norm", "lora_linear",
        "max_pool", "avg_pool", "adaptive_pool", "min_pool",
        "simple_rnn_cell", "gru_cell", "weight_norm", "instance_norm",
        "psum", "pmean", "pmax", "pmin",
        "ppo_policy_loss", "grpo_policy_loss", "cispo_policy_loss",
        "lamb", "muon", "nesterov",
        "quantize_int8", "dequantize_int8", "fake_quantize",
        "memory_write", "moe_combine", "log_softmax", "sigmoid_safe",
    }
    regressions = actionable_names & flagships
    assert not regressions, (
        f"Flagship primitives back in the actionable bucket: "
        f"{sorted(regressions)}.  Either a direct-coverage test was "
        f"deleted, or the op broke and the test no longer references "
        f"it via a recognized module pattern."
    )


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
