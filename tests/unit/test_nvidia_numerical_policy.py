"""Host-free contract tests for the shared NVIDIA numerical policy."""
from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia_numerics import assert_matches, tolerance


@pytest.mark.parametrize(
    "dtype",
    [
        "fp64", "f32", "f16", "bf16", "tf32", "fp8_e4m3", "fp8_e5m2",
        "fp6_e2m3", "fp6_e3m2", "fp4_e2m1",
    ],
)
def test_lossy_storage_has_a_nonzero_near_zero_budget(dtype):
    policy = tolerance(dtype)
    assert policy.atol > 0
    assert policy.exact is False


@pytest.mark.parametrize("dtype", ["int8", "nvfp4"])
def test_exact_storage_remains_exact(dtype):
    policy = tolerance(dtype)
    assert policy.exact is True
    with pytest.raises(AssertionError):
        assert_matches(np.array([1]), np.array([2]), dtype)


def test_policy_preserves_nonfinite_semantics():
    assert_matches(np.array([np.nan, np.inf, -np.inf], np.float32),
                   np.array([np.nan, np.inf, -np.inf], np.float32), "f16")


def test_long_reduction_scales_its_budget():
    assert tolerance("f16", reduction_length=1024).atol > tolerance("f16").atol


def test_approximate_cuda_math_rejects_a_zero_near_zero_budget():
    assert tolerance("f32", math_route="ptx_ex2_approx_f32").atol > 0
    with pytest.raises(ValueError, match="requires nonzero near-zero atol"):
        tolerance("int8", math_route="ptx_ex2_approx_f32")
