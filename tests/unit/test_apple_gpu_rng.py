"""GPU-native RNG lane (opt-in) — MPSMatrixRandomPhilox uniform / normal fills.

Philox-family but NOT bit-identical to Tessera's CPU Philox (tessera.rng), so this
is a deliberately separate opt-in surface — never wired into the deterministic
samplers (Decision #18). These tests check distribution + shape + seed
determinism, which hold on both the GPU path and the numpy fallback.
"""

import numpy as np
import pytest

import tessera.runtime as R


def _on_metal() -> bool:
    return R.DeviceTensor.is_metal()


def test_uniform_range_and_mean():
    out, ran = R.apple_gpu_random_uniform((1000, 1000), np, seed=42, low=-1.0, high=3.0)
    assert out.shape == (1000, 1000) and out.dtype == np.float32
    assert out.min() >= -1.0 and out.max() < 3.0 + 1e-3
    assert abs(float(out.mean()) - 1.0) < 0.02  # midpoint of [-1, 3)
    if _on_metal():
        assert ran is True


def test_normal_mean_std():
    out, ran = R.apple_gpu_random_normal((1000, 1000), np, seed=7, mean=2.0, std=0.5)
    assert out.shape == (1000, 1000)
    assert abs(float(out.mean()) - 2.0) < 0.02
    assert abs(float(out.std()) - 0.5) < 0.02
    if _on_metal():
        assert ran is True


def test_same_seed_is_deterministic():
    a, _ = R.apple_gpu_random_uniform(4096, np, seed=123)
    b, _ = R.apple_gpu_random_uniform(4096, np, seed=123)
    np.testing.assert_array_equal(a, b)


def test_different_seed_differs():
    a, _ = R.apple_gpu_random_uniform(4096, np, seed=1)
    b, _ = R.apple_gpu_random_uniform(4096, np, seed=2)
    assert not np.array_equal(a, b)


def test_scalar_and_tuple_shapes():
    a, _ = R.apple_gpu_random_normal(16, np, seed=0)
    assert a.shape == (16,)
    b, _ = R.apple_gpu_random_uniform((2, 3, 4), np, seed=0)
    assert b.shape == (2, 3, 4)


@pytest.mark.skipif(True, reason="documents intent, not a wiring assertion")
def test_not_wired_into_tessera_rng():
    # The GPU RNG is deliberately NOT the tessera.rng stream — kept as a doc note.
    pass
