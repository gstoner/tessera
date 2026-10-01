import ml_dtypes
import numpy as np
import pytest

from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler import rocm_nvfp4_ingest as ingest


def _projection(name="w", global_scale=1.0, scale_values=(1.0, 0.5), codes=None):
    codes = (
        np.full((2, 32), 2, dtype=np.uint8) if codes is None
        else np.asarray(codes, dtype=np.uint8)
    )
    scales = np.asarray(
        np.broadcast_to(np.asarray(scale_values, np.float32), (codes.shape[0], 2)),
        dtype=ml_dtypes.float8_e4m3fn,
    )
    return ingest.NVFP4Projection(
        name, mx.pack_e2m1_codes(codes), scales, global_scale
    ), codes


def test_requantization_preserves_e2m1_payload_and_declares_one_lossy_step():
    projection, codes = _projection()
    packed, exponents, metadata = ingest.requantize_nvfp4_projection(projection)
    np.testing.assert_array_equal(packed, projection.packed_codes)
    assert exponents.shape == (1, 2)
    assert metadata.relative_rms_error >= 0.0
    assert metadata.sqnr_db is not None
    assert metadata.as_dict()["lossy_steps"] == [ingest.LOSSY_STEP]
    np.testing.assert_array_equal(mx.unpack_e2m1_codes(packed), codes)
    assert not np.array_equal(
        mx.exact_weights(codes, exponents),
        ingest._E2M1[codes] * projection.e4m3_scales.astype(np.float32).repeat(16, axis=1),
    )


def test_block_exponent_compares_no_clip_with_one_binade_finer_by_squared_error():
    codes = np.ones((1, 32), dtype=np.uint8)
    codes[:, 16:] = 7
    scales = np.asarray([[1.0, 0.5]], dtype=ml_dtypes.float8_e4m3fn)
    projection = ingest.NVFP4Projection(
        "weighted", mx.pack_e2m1_codes(codes), scales, 1.0
    )
    _, exponents, _ = ingest.requantize_nvfp4_projection(projection)
    # The high magnitude values belong to the finer-scale half, so the
    # E8M0 2^-1 candidate has lower decoded-weight SSE than the no-clip 2^0.
    assert int(exponents[0, 0]) == 126


def test_merged_gate_up_keeps_distinct_projection_global_scales():
    gate, _ = _projection("gate", 1.0, (1.0, 1.0))
    up, _ = _projection("up", 4.0, (1.0, 1.0))
    merged = ingest.ingest_nvfp4_projections((gate, up))
    assert merged.projection_names == ("gate", "up")
    assert merged.row_offsets == (0, 2, 4)
    assert [m.global_scale for m in merged.metadata] == [1.0, 4.0]
    assert merged.numeric_policy()["lossy_steps"] == [ingest.LOSSY_STEP]
    decoded = mx.exact_weights(
        mx.unpack_e2m1_codes(merged.packed_codes), merged.scale_exponents
    )
    np.testing.assert_array_equal(decoded[:2], 1.0)
    np.testing.assert_array_equal(decoded[2:], 4.0)


def test_zero_scale_blocks_remain_zero_after_ingest():
    codes = np.full((1, 32), 7, dtype=np.uint8)
    projection = ingest.NVFP4Projection(
        "zero", mx.pack_e2m1_codes(codes),
        np.zeros((1, 2), dtype=ml_dtypes.float8_e4m3fn), 3.0,
    )
    _, exponents, metadata = ingest.requantize_nvfp4_projection(projection)
    assert int(exponents[0, 0]) == 0
    assert metadata.relative_rms_error == 0.0
    np.testing.assert_array_equal(mx.exact_weights(codes, exponents), 0.0)


@pytest.mark.parametrize(
    "projection,error",
    [
        (ingest.NVFP4Projection("bad", np.zeros((1, 16), np.uint8),
                                np.ones((1, 2), np.uint8), 1.0), TypeError),
        (ingest.NVFP4Projection("bad", np.zeros((1, 16), np.uint8),
                                np.ones((1, 3), dtype=ml_dtypes.float8_e4m3fn),
                                1.0), ValueError),
        (ingest.NVFP4Projection("bad", np.zeros((1, 16), np.uint8),
                                np.ones((1, 2), dtype=ml_dtypes.float8_e4m3fn),
                                0.0), ValueError),
    ],
)
def test_ingest_rejects_ambiguous_or_malformed_checkpoint_contract(projection, error):
    with pytest.raises(error):
        ingest.requantize_nvfp4_projection(projection)


def test_merged_projections_require_unique_names_and_matching_k():
    first, _ = _projection("same")
    duplicate, _ = _projection("same", 2.0)
    with pytest.raises(ValueError, match="unique"):
        ingest.ingest_nvfp4_projections((first, duplicate))
    other_codes = np.full((2, 64), 1, dtype=np.uint8)
    other = ingest.NVFP4Projection(
        "other", mx.pack_e2m1_codes(other_codes),
        np.ones((2, 4), dtype=ml_dtypes.float8_e4m3fn), 1.0,
    )
    with pytest.raises(ValueError, match="matching K"):
        ingest.ingest_nvfp4_projections((first, other))


def test_nonzero_source_scale_below_e8m0_range_is_not_silently_zeroed():
    codes = np.full((1, 32), 7, dtype=np.uint8)
    projection = ingest.NVFP4Projection(
        "underflow", mx.pack_e2m1_codes(codes),
        np.ones((1, 2), dtype=ml_dtypes.float8_e4m3fn), 1.0e-100,
    )
    with pytest.raises(ValueError, match="representable E8M0 range"):
        ingest.requantize_nvfp4_projection(projection)
