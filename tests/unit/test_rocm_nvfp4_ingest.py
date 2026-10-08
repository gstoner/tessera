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


def test_requantization_jointly_selects_codes_and_scale_for_lower_weight_sse():
    projection, codes = _projection()
    packed, exponents, metadata = ingest.requantize_nvfp4_projection(projection)
    destination_codes = mx.unpack_e2m1_codes(packed)
    assert exponents.shape == (1, 2)
    assert metadata.relative_rms_error >= 0.0
    assert metadata.sqnr_db is None or np.isfinite(metadata.sqnr_db)
    assert metadata.as_dict()["lossy_steps"] == [ingest.LOSSY_STEP]
    assert ingest.LOSSY_STEP in ingest.MXFP4IngestedWeights(
        packed, exponents, ("w",), (0, 2), (metadata,)
    ).numeric_policy()["lossy_steps"]
    assert not np.array_equal(destination_codes, codes)
    source = ingest._E2M1[codes] * projection.e4m3_scales.astype(
        np.float32
    ).repeat(16, axis=1)
    destination = mx.exact_weights(destination_codes, exponents)
    preserve_code_baseline = mx.exact_weights(codes, exponents)
    optimized_sse = np.square(source - destination).sum()
    preserve_sse = np.square(source - preserve_code_baseline).sum()
    assert optimized_sse < preserve_sse
    assert metadata.relative_rms_error == pytest.approx(
        np.sqrt(optimized_sse / np.square(source).sum())
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


def test_block_exponent_uses_code_energy_weighted_mean_scale():
    # The largest scale belongs to zero codes, so choosing relative to the
    # maximum wastes precision on the active values in the other half block.
    codes = np.zeros((1, 32), dtype=np.uint8)
    codes[:, :16] = 7
    scales = np.asarray([[0.25, 3.0]], dtype=ml_dtypes.float8_e4m3fn)
    projection = ingest.NVFP4Projection(
        "skewed", mx.pack_e2m1_codes(codes), scales, 1.0
    )
    _, exponents, metadata = ingest.requantize_nvfp4_projection(projection)

    selected = np.ldexp(1.0, int(exponents[0, 0]) - 127)
    codes_f32 = ingest._E2M1[codes]
    source = codes_f32 * scales.astype(np.float32).repeat(16, axis=1)
    selected_error = np.square(codes_f32 * selected - source).sum()
    old_error = np.square(codes_f32 * 0.5 - source).sum()
    assert selected == 0.25
    assert selected_error < old_error
    assert metadata.relative_rms_error == 0.0


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


def _scalar_reference_projection(projection):
    n, k, packed, scales, global_scale = ingest._validate_projection(projection)
    codes = mx.unpack_e2m1_codes(packed)
    values = ingest._E2M1[codes]
    source = values * scales.repeat(16, axis=1)
    dst = np.empty_like(codes)
    exponents = np.empty((k // 32, n), dtype=np.uint8)
    for row in range(n):
        for group in range(k // 32):
            start = group * 32
            pair = scales[row, group * 2:group * 2 + 2]
            seed = ingest._choose_e8m0_exponent(
                values[row, start:start + 32], pair
            )
            block_codes, exponent = ingest._requantize_e2m1_block(
                source[row, start:start + 32], seed
            )
            dst[row, start:start + 32] = block_codes
            exponents[group, row] = 0 if not np.any(pair) else exponent + 127
    reconstructed = mx.exact_weights(dst, exponents)
    signal = float(np.square(source.astype(np.float64)).sum())
    error = float(np.square(
        source.astype(np.float64) - reconstructed.astype(np.float64)
    ).sum())
    relative_rms = np.sqrt(error / signal) if signal else 0.0
    return mx.pack_e2m1_codes(dst), exponents, relative_rms


def test_vectorized_ingest_matches_scalar_requantization_reference():
    rng = np.random.default_rng(0x1201_873)
    codes = rng.integers(0, 16, size=(11, 160), dtype=np.uint8)
    codes[0, :] = 0
    codes[1, :32] = 8
    scales = np.asarray(
        rng.choice(
            np.asarray([0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0], np.float32),
            size=(11, 10),
        ),
        dtype=ml_dtypes.float8_e4m3fn,
    )
    scales[0] = np.asarray([0.5, 1.0] * 5, dtype=ml_dtypes.float8_e4m3fn)
    scales[1, :2] = 0.0
    projection = ingest.NVFP4Projection(
        "vectorized-reference",
        mx.pack_e2m1_codes(codes),
        scales,
        0.75,
    )

    expected_packed, expected_exponents, expected_rms = (
        _scalar_reference_projection(projection)
    )
    actual_packed, actual_exponents, metadata = (
        ingest.requantize_nvfp4_projection(projection)
    )
    np.testing.assert_array_equal(actual_packed, expected_packed)
    np.testing.assert_array_equal(actual_exponents, expected_exponents)
    assert metadata.relative_rms_error == pytest.approx(expected_rms, abs=1e-15)


def test_joint_requantization_matches_exhaustive_e8m0_search():
    rng = np.random.default_rng(0x1201_20261002)
    positive_e4m3 = np.arange(1, 127, dtype=np.uint8).view(
        ml_dtypes.float8_e4m3fn
    )
    scale_values = positive_e4m3[
        rng.integers(0, len(positive_e4m3), size=(128, 2))
    ]
    codes = rng.integers(0, 16, size=(128, 32), dtype=np.uint8)
    codes[:, 0] = 7
    global_scale = 0.03125
    projection = ingest.NVFP4Projection(
        "exhaustive-scale-search",
        mx.pack_e2m1_codes(codes),
        scale_values,
        global_scale,
    )

    packed, exponents, _ = ingest.requantize_nvfp4_projection(projection)
    source_codes = mx.unpack_e2m1_codes(packed)
    source = (
        ingest._E2M1[codes].astype(np.float64)
        * scale_values.astype(np.float64).repeat(16, axis=1)
        * global_scale
    )
    actual = mx.exact_weights(source_codes, exponents).astype(np.float64)
    actual_error = np.square(source - actual).sum(axis=1)

    # Exhaustively test every finite E8M0 exponent while selecting the nearest
    # signed E2M1 code at that scale. The production bounded search must reach
    # the same global SSE minimum over all 254 representable destination scales.
    levels = ingest._E2M1[:8].astype(np.float64)
    midpoints = (levels[:-1] + levels[1:]) * 0.5
    best_error = np.full(source.shape[0], np.inf, dtype=np.float64)
    for exponent in range(-126, 128):
        scale = np.exp2(float(exponent))
        magnitude_codes = np.searchsorted(
            midpoints, np.abs(source) / scale, side="left"
        )
        candidate_codes = magnitude_codes.astype(np.uint8)
        candidate_codes |= np.where(source < 0.0, np.uint8(8), np.uint8(0))
        decoded = ingest._E2M1[candidate_codes] * scale
        best_error = np.minimum(
            best_error, np.square(source - decoded).sum(axis=1)
        )

    np.testing.assert_allclose(actual_error, best_error, rtol=1e-12, atol=1e-12)
