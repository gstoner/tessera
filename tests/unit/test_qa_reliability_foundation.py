"""
Phase 6 — QA/reliability guide and helper contracts.
"""

from pathlib import Path

import numpy as np
import pytest

from tessera.testing import (
    PerformanceExpectation,
    assert_close_to_reference,
    assert_deterministic,
    assert_finite,
)


ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "guides" / "Tessera_QA_Reliability_Guide.md"


def test_qa_guide_exists_and_covers_required_behaviors():
    text = GUIDE.read_text(encoding="utf-8")
    required = [
        "Correctness",
        "Numerical Stability",
        "Determinism Testing",
        "Fault Tolerance",
        "Performance Consistency",
        "Distributed QA",
        "schedule artifact",
    ]
    for term in required:
        assert term in text


def test_docs_map_links_qa_guide():
    text = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert "docs/guides/Tessera_QA_Reliability_Guide.md" in text


def test_assert_close_to_reference_accepts_matching_arrays():
    actual = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert_close_to_reference(actual, ref, rtol=1e-6, atol=1e-7)


def test_assert_close_to_reference_reports_shape_mismatch():
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_close_to_reference(
            np.zeros((2, 2)),
            np.zeros((4,)),
            rtol=1e-6,
            atol=1e-7,
            name="matmul",
        )


def test_assert_finite_rejects_nan_and_inf():
    assert_finite(np.array([0.0, 1.0]))
    with pytest.raises(AssertionError, match="non-finite"):
        assert_finite(np.array([0.0, np.inf, np.nan]), name="softmax")


def test_assert_deterministic_compares_repeated_runs():
    def stable():
        return np.array([1.0, 2.0])

    assert_deterministic(stable, runs=3)

    counter = {"n": 0}

    def unstable():
        counter["n"] += 1
        return np.array([counter["n"]], dtype=np.float32)

    with pytest.raises(AssertionError, match="differed"):
        assert_deterministic(unstable, runs=2)


def test_performance_expectation_validates_thresholds():
    expect = PerformanceExpectation(
        name="matmul_4096_bf16",
        latency_ms_max=2.5,
        tflops_min=150.0,
        bandwidth_gbps_min=800.0,
    )
    expect.validate({"latency_ms": 2.1, "tflops": 171.0, "bandwidth_gbps": 900.0})

    with pytest.raises(AssertionError, match="performance expectation failed"):
        expect.validate({"latency_ms": 2.9, "tflops": 171.0, "bandwidth_gbps": 900.0})
