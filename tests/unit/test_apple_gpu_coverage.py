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
