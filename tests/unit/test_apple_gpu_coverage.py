"""Phase B — guard for the Apple GPU numpy-lane displacement worklist.

Pins the invariant that guided Phase C (the pointwise-DAG fusion vocabulary is
fully GPU-covered, so single-op elementwise displacement is already complete) and
the structural soundness of the worklist classifier.
"""

from __future__ import annotations

from tessera.compiler import apple_gpu_coverage as cov
from tessera.compiler import fusion as F


def test_pointwise_vocab_is_fully_gpu_covered():
    """Every op in the pointwise-DAG fusion vocabulary has a single-op GPU lane —
    the empirical fact from Phase B: there is no elementwise single-op numpy to
    displace; the win is multi-op DAG fusion + the non-elementwise tail."""
    report = cov.numpy_lane_worklist()
    assert report.pointwise_vocab_covered, (
        "a POINTWISE_OPS vocab op lost its GPU lane: "
        + ", ".join(k for k in F.POINTWISE_OPS if not cov.has_gpu_lane(k))
    )


def test_worklist_is_wellformed_and_nonempty():
    report = cov.numpy_lane_worklist()
    assert report.total > 0
    assert report.covered + report.numpy_only_count == report.total
    # Some ops genuinely have no GPU lane yet (linalg/spectral/complex tail).
    assert report.numpy_only_count > 0
    # Categories partition the numpy-only set exactly.
    flat = [op for ops in report.by_category.values() for op in ops]
    assert sorted(flat) == list(report.numpy_only)


def test_softcap_was_displaced_off_the_numpy_lane():
    """softcap (the Gemma logit soft-cap cap*tanh(x/cap)) was the one genuinely
    numpy-only real-valued elementwise op; it now rides a GPU compose lane
    (div-scalar -> tanh -> mul-scalar), so it must no longer appear in the
    numpy-only worklist. The inverse of the original Phase-C progress marker."""
    report = cov.numpy_lane_worklist()
    assert "softcap" not in report.numpy_only
    assert cov.has_gpu_lane("softcap")


def test_render_report_runs():
    text = cov.render_report()
    assert "numpy-only" in text and "pointwise-vocab" in text


def test_displacement_disposition_classifies_the_real_gap():
    """Phase 2 (2026-06-17): the numpy-only tail is mostly NOT a real GPU
    execution gap. The disposition map must separate the genuine target
    (real_gap_structural — layout/indexing ops that demote a mixed program off
    metal_runtime) from the no-gap buckets (host_utility optimizers/RNG,
    distributed collectives) and the hard-kernel tail."""
    # The headline buckets exist and the structural categories are the real gap.
    assert cov.disposition_for("layout_transform") == "real_gap_structural"
    assert cov.disposition_for("indexing") == "real_gap_structural"
    # Optimizers run host-side on pytrees — no single-op GPU gap.
    assert cov.disposition_for("functional_optimizer_step") == "host_utility"
    # Collectives are multi-rank, not a single-device GPU concern.
    assert cov.disposition_for("collective") == "distributed"
    # Packed-format quant / sparse genuinely need dedicated kernels.
    assert cov.disposition_for("quantize") == "hard_kernel"
    # An unknown category is honestly unclassified (not silently bucketed).
    assert cov.disposition_for("a_brand_new_category") == "unclassified"

    text = cov.render_report()
    assert "displacement disposition" in text
    assert "real_gap_structural" in text
