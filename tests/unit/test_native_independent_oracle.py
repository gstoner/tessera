"""The norm oracle carries FP32 error through discontinuous storage rounding."""
import numpy as np
import pytest
from tests.device.nvidia.test_native_tensor_independent_shapes import roots, check


def test_known_fp32_halfway_result_is_valid_but_wrong_norm_is_not():
    a, b = roots("norm_lhs", "f16", 65, 47, 64, seed=23403)
    wide = a.astype(np.float64)
    normalized = wide / np.sqrt(np.mean(wide * wide, axis=-1, keepdims=True) + 1e-5)
    stored = normalized.astype(np.float16)
    # Retained exact-device witness: the FP32 norm lands at this f16 midpoint.
    # Its storage rounding differs from rounding the unrounded FP64 norm.
    assert stored[11, 33] == np.float16(-0.316162109375)
    stored[11, 33] = np.float16(np.float32(-0.3160400390625))
    assert stored[11, 33] == np.float16(-0.31591796875)
    result = (stored.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
    check("norm_lhs", result, (a, b))
    stored[11, 33] = np.float16(-0.30)
    wrong = (stored.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
    with pytest.raises(AssertionError):
        check("norm_lhs", wrong, (a, b))
