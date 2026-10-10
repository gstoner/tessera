"""Host-independent quality-oracle contracts for the three-format benchmark."""
import ml_dtypes
import numpy as np
import pytest

from benchmarks.rocm.benchmark_gfx1201_three_formats import (
    quantize_folded, quantize_fp8, verify,
)


@pytest.mark.parametrize("e8m0", [False, True])
def test_fp8_zero_inputs_and_standard_scale_reconstruction(e8m0):
    a = np.zeros((65, 128), np.float32)
    b = np.zeros((128, 3), np.float32)
    qa, qb, scales, da, db = quantize_fp8(a, b, 32, 1, e8m0=e8m0)
    assert not np.any(qa.astype(np.float32))
    assert not np.any(qb.astype(np.float32))
    assert not np.any(da) and not np.any(db)
    if e8m0:
        assert np.all(scales[0] == 0) and np.all(scales[1] == 0)
        # Standard E8M0 code zero decodes to 2^-127, never a zero scale.
        assert np.all(scales[0].view(ml_dtypes.float8_e8m0fnu).astype(np.float32) == 2.**-127)


def test_fp4_ties_round_to_even_before_fold():
    a = np.ones((65, 128), np.float32)
    sequence = np.array([.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.], np.float32)
    b = np.tile(sequence, 16)[:, None]
    b = np.concatenate([b, -b], axis=1)
    _, _, folded, _, exact, physical = quantize_folded(a, b)
    expected = np.tile(np.array([0, 1, 1, 2, 2, 4, 4, 6.]), 16)
    np.testing.assert_array_equal(exact[:, 0], expected)
    np.testing.assert_array_equal(exact[:, 1], -expected)
    np.testing.assert_array_equal(physical, exact)
    assert folded.lossless


def test_fold_error_is_distinct_from_fp4_quantization():
    a = np.ones((65, 128), np.float32)
    b = np.concatenate([np.full((32, 2), 6.),
                        np.full((96, 2), 6. * 2.**-14)], axis=0).astype(np.float32)
    _, _, folded, _, exact, physical = quantize_folded(a, b)
    np.testing.assert_array_equal(exact, b)
    assert not folded.lossless
    assert folded.inexact_value_count == 192
    assert np.linalg.norm(physical - exact) > 0


def test_native_oracle_allows_store_rounding_and_detects_corrupted_result():
    rng = np.random.default_rng(32)
    a = rng.normal(size=(65, 128))
    b = rng.normal(size=(128, 7))
    ideal, magnitude = a @ b, np.abs(a) @ np.abs(b)
    stored = ideal.astype(np.float32).astype(ml_dtypes.bfloat16)
    assert verify(stored, ideal, magnitude, 128)["violations"] == 0
    stored[0, 0] = 1000
    with pytest.raises(AssertionError, match="forward bound"):
        verify(stored, ideal, magnitude, 128)
