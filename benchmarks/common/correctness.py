"""Correctness helpers shared by benchmark suites."""

from __future__ import annotations

import numpy as np

from .artifact_schema import Correctness


def max_abs_error(actual, expected) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.size == 0 and expected.size == 0:
        return 0.0
    return float(np.max(np.abs(actual - expected)))


def relative_error(actual, expected, eps: float = 1e-12) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    numerator = float(np.linalg.norm(actual - expected))
    denominator = float(np.linalg.norm(expected)) + eps
    return numerator / denominator


def within_tolerance(actual, expected, tolerance: float) -> bool:
    return max_abs_error(actual, expected) <= tolerance


def correctness_report(actual, expected, tolerance: float) -> Correctness:
    max_err = max_abs_error(actual, expected)
    rel_err = relative_error(actual, expected)
    return Correctness(
        max_error=max_err,
        relative_error=rel_err,
        tolerance=tolerance,
        passed=max_err <= tolerance,
    )
