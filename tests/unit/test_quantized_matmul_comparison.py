"""Host-free gates for the measured, owning-Mac benchmark."""
import numpy as np
import pytest

from benchmarks.apple_gpu.compare_quantized_matmul import pack_values, verify


def test_fp4_packing_keeps_adjacent_nibbles_and_exact_reference():
    packed, decoded = pack_values(np.array([[0, .5, 1, 1.5, -2, -3, -4, -6]], np.float32), 'fp4_e2m1')
    np.testing.assert_array_equal(packed, [[0x10, 0x32, 0xDC, 0xFE]])
    np.testing.assert_array_equal(decoded, [[0, .5, 1, 1.5, -2, -3, -4, -6]])


def test_gate_rejects_wrong_result_and_nonfinite():
    a = np.eye(4, dtype=np.float16)
    assert verify(a, a, a)['correctness_passed']
    for result in (a + 1, np.full((4, 4), np.nan), np.full((4, 4), np.inf)):
        with pytest.raises(ValueError, match='correctness gate'):
            verify(result, a, a)
