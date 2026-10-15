"""Source-preserving NVFP4 import validation and independent bit oracles."""
import ml_dtypes
import numpy as np
import pytest

from tessera.compiler.graph_ir import _infer_result_types, tensor_ir_type
from tessera.compiler.nvfp4_expansion import (
    nvfp4_expansion_policy,
    reference_nvfp4_expand,
)


def inputs(k=48):
    codes = np.tile(np.arange(0, 256, 17, dtype=np.uint8), (7, (k // 2 + 15) // 16))
    codes = codes[:, :k // 2].copy()
    scale = np.arange(7 * (k // 16), dtype=np.float32).reshape(7, k // 16) / 4
    scales = scale.astype(ml_dtypes.float8_e4m3fn)
    globals_ = np.array([0.000123456789, 0.007654321], np.float64)
    return codes, scales, globals_


@pytest.mark.parametrize("k", [16, 32, 48, 80])
def test_exact_codes_and_projection_scale_rounding(k):
    codes, scales, globals_ = inputs(k)
    values, plane = reference_nvfp4_expand(
        codes, scales, globals_, row_offsets=[0, 3, 7],
        numeric_policy=nvfp4_expansion_policy())
    levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float64)
    nibble = np.stack((codes & 15, codes >> 4), axis=-1).reshape(7, k)
    expected = np.copysign(levels[nibble & 7], np.where(nibble & 8, -1., 1.))
    expected_bits = expected.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    np.testing.assert_array_equal(values.view(np.uint8), expected_bits)
    repeated_globals = np.array([globals_[0]] * 3 + [globals_[1]] * 4)
    expected_plane = (scales.astype(np.float64) * repeated_globals[:, None]).astype(np.float32).T
    np.testing.assert_array_equal(plane.view(np.uint32), expected_plane.copy().view(np.uint32))


def test_all_codes_and_signed_zero():
    codes = np.arange(0, 256, 17, dtype=np.uint8)[None, :]
    scales = np.array([[-0., 1.]], dtype=ml_dtypes.float8_e4m3fn)
    values, plane = reference_nvfp4_expand(
        codes, scales, np.array([1.], np.float64), row_offsets=[0, 1],
        numeric_policy=nvfp4_expansion_policy())
    assert values.view(np.uint8)[0, 0] == 0
    assert values.view(np.uint8)[0, 16] == 128
    assert plane.view(np.uint32)[0, 0] == 0x80000000


@pytest.mark.parametrize("kind", ["negative_scale", "nan_scale", "zero_global", "nan_global",
                                  "overflow", "underflow", "policy", "offsets", "dtype"])
def test_invalid_imports_rejected(kind):
    codes, scales, globals_ = inputs()
    offsets = [0, 3, 7]
    policy = nvfp4_expansion_policy()
    if kind == "negative_scale":
        scales[0, 0] = -1
    elif kind == "nan_scale":
        scales[0, 0] = np.nan
    elif kind == "zero_global":
        globals_[0] = 0
    elif kind == "nan_global":
        globals_[0] = np.nan
    elif kind == "overflow":
        globals_[0] = np.finfo(np.float64).max
    elif kind == "underflow":
        globals_[0] = np.nextafter(0., 1.)
    elif kind == "policy":
        policy["scale_rounding"] = "truncate"
    elif kind == "offsets":
        offsets = [0, True, 7]
    elif kind == "dtype":
        codes = codes.astype(np.int8)
    with pytest.raises(ValueError):
        reference_nvfp4_expand(codes, scales, globals_, row_offsets=offsets, numeric_policy=policy)


@pytest.mark.parametrize("k", [16, 48, 80])
def test_frontend_infers_both_results(k):
    results = _infer_result_types("tessera.nvfp4_expand", [
        tensor_ir_type((7, k // 2), "uint8"),
        tensor_ir_type((7, k // 16), "fp8_e4m3"),
        tensor_ir_type((2,), "fp64")], {})
    assert [(tuple(map(int, t.shape)), t.dtype) for t in results] == [
        ((7, k), "fp8_e4m3"), ((k // 16, 7), "fp32")]
