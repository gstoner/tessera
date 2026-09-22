"""Host-free gates for the planned gfx1201 MXFP4/W4A8 physical route."""

import numpy as np
import pytest

from tessera.compiler import rocm_mxfp4 as mx


def test_physical_contract_keeps_scale_group_independent_of_schedule_k() -> None:
    contract = mx.MXFP4PhysicalContract()
    assert contract.scale_group_k == 32
    assert contract.instruction_k == 16
    assert mx.numeric_policy("exact_per_block")["scale_application"] == "fp32_partial_per_32_k"
    # A future macro K=64/128 contains multiple format scale groups; neither
    # macro blocking nor kUnroll may rewrite this format semantic.
    assert 64 % contract.scale_group_k == 0
    assert 128 % contract.scale_group_k == 0


def test_pack_uses_low_nibble_for_even_k_and_round_trips() -> None:
    codes = np.arange(16, dtype=np.uint8).reshape(1, 16)
    packed = mx.pack_e2m1_codes(codes)
    np.testing.assert_array_equal(
        packed, [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]]
    )
    np.testing.assert_array_equal(mx.unpack_e2m1_codes(packed), codes)


def test_physical_payload_rejects_values_that_would_wrap_in_uint8() -> None:
    with pytest.raises(ValueError, match=r"\[0,15\]"):
        mx.pack_e2m1_codes(np.asarray([[0, 16]], dtype=np.int16))
    with pytest.raises(TypeError, match="integer element type"):
        mx.pack_e2m1_codes(np.asarray([[0.0, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match=r"\[0,254\]"):
        mx.exact_weights(
            np.zeros((1, 32), dtype=np.uint8),
            np.asarray([[255]], dtype=np.uint16),
        )


def test_fragment_order_is_a_byte_permutation_with_an_inverse() -> None:
    packed = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    permuted = mx.to_fragment_order(packed)
    assert permuted.shape == packed.shape
    np.testing.assert_array_equal(np.sort(permuted.reshape(-1)), np.sort(packed.reshape(-1)))
    np.testing.assert_array_equal(mx.from_fragment_order(permuted), packed)


def test_versioned_layout_registry_and_load_time_conversion_are_total() -> None:
    packed = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16)
    fragment = mx.convert_weight_layout(
        packed,
        source=mx.MXFP4_CHECKPOINT_LAYOUT_V1,
        destination=mx.MXFP4_GFX12_FRAGMENT_LAYOUT_V1,
    )
    assert mx.mxfp4_weight_layout(
        mx.MXFP4_GFX12_FRAGMENT_LAYOUT_V1
    ).lane_read == "one_contiguous_uint32_per_lane_per_n16_k16_step"
    np.testing.assert_array_equal(
        mx.convert_weight_layout(
            fragment,
            source=mx.MXFP4_GFX12_FRAGMENT_LAYOUT_V1,
            destination=mx.MXFP4_CHECKPOINT_LAYOUT_V1,
        ),
        packed,
    )
    np.testing.assert_array_equal(
        mx.convert_weight_layout(
            packed,
            source=mx.MXFP4_CHECKPOINT_LAYOUT_V1,
            destination=mx.MXFP4_TRANSPOSED_LAYOUT_V1,
        ),
        packed.T,
    )
    with pytest.raises(ValueError, match="recognized but has no proved"):
        mx.convert_weight_layout(
            packed,
            source=mx.MXFP4_CHECKPOINT_LAYOUT_V1,
            destination=mx.MXFP4_AITER_SHUFFLED_LAYOUT_V1,
        )


def test_exact_decode_uses_k_group_n_scale_order() -> None:
    codes = np.ones((2, 64), dtype=np.uint8)  # E2M1 0.5
    scales = np.asarray([[127, 128], [129, 130]], dtype=np.uint8)  # [K/32,N]
    got = mx.exact_weights(codes, scales)
    np.testing.assert_array_equal(got[0, :32], np.full(32, 0.5, np.float32))
    np.testing.assert_array_equal(got[0, 32:], np.full(32, 2.0, np.float32))
    np.testing.assert_array_equal(got[1, :32], np.full(32, 1.0, np.float32))
    np.testing.assert_array_equal(got[1, 32:], np.full(32, 4.0, np.float32))


def test_row_reference_fold_is_exact_through_delta_eight() -> None:
    # Row reference 135, then deltas 0..8 across nine scale groups.  Code 1 is
    # magnitude 0.5, so delta 8 reaches E4M3's smallest subnormal (2^-9).
    scales = np.arange(135, 126, -1, dtype=np.uint8)[:, None]
    codes = np.ones((1, scales.shape[0] * 32), dtype=np.uint8)
    folded = mx.fold_to_row_reference(codes, scales, allow_approximate=True)
    assert folded.lossless
    assert folded.inexact_value_count == 0
    assert folded.max_normalized_abs_error == 0.0
    assert folded.max_normalized_relative_error == 0.0
    assert int(folded.exponent_delta.max()) == 8
    np.testing.assert_array_equal(mx.folded_weights(folded), mx.exact_weights(codes, scales))


def test_row_reference_fold_reports_rounding_beyond_exact_range() -> None:
    # Delta 9 shifts magnitude 0.5 below E4M3's smallest subnormal.  The exact
    # per-block route remains non-zero; the fast fold must disclose the loss.
    codes = np.ones((1, 64), dtype=np.uint8)
    scales = np.asarray([[135], [126]], dtype=np.uint8)
    folded = mx.fold_to_row_reference(codes, scales, allow_approximate=True)
    assert not folded.lossless
    assert folded.inexact_value_count == 32
    assert folded.max_normalized_abs_error > 0.0
    assert folded.max_normalized_relative_error == 1.0
    assert folded.approximate_policy == "explicit_allow"
    assert int(folded.exponent_delta.max()) == 9
    assert np.any(mx.folded_weights(folded) != mx.exact_weights(codes, scales))
    policy = mx.numeric_policy("folded_row_reference")
    assert policy["lossless_for_all_inputs"] is False
    assert policy["fold_exact_max_exponent_delta"] == 8


def test_zero_codes_remain_exact_even_at_large_exponent_delta() -> None:
    codes = np.zeros((1, 64), dtype=np.uint8)
    scales = np.asarray([[140], [120]], dtype=np.uint8)
    folded = mx.fold_to_row_reference(codes, scales, allow_approximate=True)
    assert folded.lossless
    np.testing.assert_array_equal(mx.folded_weights(folded), 0.0)


def test_reserved_zero_scale_block_stays_zero_during_row_fold() -> None:
    # E8M0 code zero means the whole K32 block is zero, regardless of its E2M1
    # payload. It must not be treated as exponent zero relative to row_ref=1.
    codes = np.ones((1, 64), dtype=np.uint8)
    scales = np.asarray([[0], [1]], dtype=np.uint8)
    folded = mx.fold_to_row_reference(codes, scales, allow_approximate=True)
    reconstructed = mx.folded_weights(folded)
    assert folded.lossless
    np.testing.assert_array_equal(reconstructed, mx.exact_weights(codes, scales))
    np.testing.assert_array_equal(reconstructed[:, :32], 0.0)
    assert np.any(reconstructed[:, 32:] != 0.0)


def test_row_reference_fold_requires_explicit_approximate_policy() -> None:
    codes = np.ones((1, 32), dtype=np.uint8)
    scales = np.asarray([[127]], dtype=np.uint8)
    with pytest.raises(ValueError, match="explicit approximate numerical policy"):
        mx.fold_to_row_reference(codes, scales)
