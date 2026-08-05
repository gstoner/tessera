from __future__ import annotations

from benchmarks.rocm.benchmark_rocm_e2e_real_matmul import decide
from benchmarks.x86.benchmark_x86_e2e_real_matmul import _summarize, verdict


def _x86_row(*, correct: bool, non_regression: bool) -> dict:
    return {
        "correctness": {"passed": correct},
        "timing": {"non_regression_10pct": non_regression},
    }


def test_x86_e2e_real4_requires_correctness_and_the_existing_ratchet() -> None:
    assert verdict([_x86_row(correct=True, non_regression=True)]) == "promote"
    assert verdict([_x86_row(correct=True, non_regression=False)]) == "retain"
    assert verdict([_x86_row(correct=False, non_regression=True)]) == "reject"
    summary = _summarize([1.0, 1.0, 1.0], [1.09, 1.09, 1.09])
    assert summary["non_regression_10pct"] is True
    assert summary["scheduled_over_production"] == 1.09


def test_rocm_e2e_real4_never_promotes_host_wall_evidence() -> None:
    assert decide(correct=True, tflops=12.5, direct_ratio=0.90, selector_eligible=False) == "retain"
    assert decide(correct=True, tflops=12.5, direct_ratio=0.90, selector_eligible=True) == "promote"
    assert decide(correct=True, tflops=8.01, direct_ratio=1.0, selector_eligible=True) == "reject"
    assert decide(correct=False, tflops=20.0, direct_ratio=1.0, selector_eligible=True) == "reject"
