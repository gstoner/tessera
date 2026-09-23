"""Model-load contract for the non-executable packed-folded gfx1201 candidate."""
from __future__ import annotations

import ast
import re

import ml_dtypes
import numpy as np
import pytest

from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_folded import prepare_folded_weights
from tessera.compiler.rocm_mxfp4_packed_folded import (
    PACKED_FOLDED_PHYSICAL_V1,
    PACKED_FOLDED_SCALE_PLANE_V1,
    PackedFoldedPayload,
    author_packed_folded_graph,
    emit_mxfp4_packed_folded_prefill_hip,
    folded_oracle_from_packed,
    prepare_packed_folded_payload,
)


@pytest.mark.parametrize("shape", [(48, 64), (80, 128)])
def test_packed_folded_payload_preserves_all_codes_and_declared_oracle(
    shape: tuple[int, int],
) -> None:
    n, k = shape
    codes = np.resize(np.arange(16, dtype=np.uint8), (n, k))
    scales = np.full((k // 32, n), 127, dtype=np.uint8)
    scales[0, 0::3] = 0  # reserved zero block, not exponent zero
    scales[-1, 1::3] = 116  # explicit lossy-fold cases
    checkpoint = mx.pack_e2m1_codes(codes)
    expected = prepare_folded_weights(checkpoint, scales, allow_approximate=True)
    payload = prepare_packed_folded_payload(
        checkpoint, scales, allow_approximate=True,
    )
    assert payload.shape == (n, k)
    assert payload.weight_bytes.shape == (n, k // 2)
    assert payload.scale_plane.shape == (k // 32 + 1, n)
    assert np.array_equal(payload.scale_plane[:-1], scales)
    assert np.array_equal(payload.scale_plane[-1], expected.row_reference)
    assert np.array_equal(
        mx.from_fragment_order(payload.weight_bytes), checkpoint,
    )
    assert not payload.weight_bytes.flags.writeable
    assert not payload.scale_plane.flags.writeable
    oracle = folded_oracle_from_packed(payload)
    np.testing.assert_array_equal(oracle.weight_bytes, expected.weight_bytes)
    np.testing.assert_array_equal(oracle.row_reference, expected.row_reference)
    assert oracle.lossless == payload.lossless == expected.lossless
    assert oracle.inexact_value_count == payload.inexact_value_count
    receipt = payload.receipt()
    assert receipt["weight_layout"] == mx.MXFP4_GFX12_FRAGMENT_LAYOUT_V1
    assert receipt["scale_layout"] == PACKED_FOLDED_SCALE_PLANE_V1
    assert receipt["execution_state"] == "artifact_only"
    assert len(receipt["weight_sha256"]) == len(receipt["scale_plane_sha256"]) == 64
    graph = author_packed_folded_graph(65, payload)
    assert PACKED_FOLDED_PHYSICAL_V1 in graph
    assert f"tensor<{n}x{k // 2}xui8>" in graph
    assert f"tensor<{k // 32 + 1}x{n}xui8>" in graph


def test_packed_folded_payload_refuses_policy_and_bad_reference_plane() -> None:
    checkpoint = mx.pack_e2m1_codes(np.ones((48, 64), dtype=np.uint8))
    scales = np.full((2, 48), 127, dtype=np.uint8)
    with pytest.raises(ValueError, match="explicit approximate"):
        prepare_packed_folded_payload(checkpoint, scales)
    payload = prepare_packed_folded_payload(
        checkpoint, scales, allow_approximate=True,
    )
    bad = payload.scale_plane.copy()
    bad[-1, 0] -= 1
    with pytest.raises(ValueError, match="row reference"):
        PackedFoldedPayload(
            payload.weight_bytes, bad, payload.lossless,
            payload.inexact_value_count, payload.max_normalized_abs_error,
            payload.max_normalized_relative_error,
        )
    with pytest.raises(ValueError, match="fragment-order"):
        PackedFoldedPayload(
            payload.weight_bytes, payload.scale_plane, payload.lossless,
            payload.inexact_value_count, payload.max_normalized_abs_error,
            payload.max_normalized_relative_error,
            weight_layout=mx.MXFP4_CHECKPOINT_LAYOUT_V1,
        )


def test_packed_folded_payload_rejects_fabricated_loss_receipt() -> None:
    codes = np.ones((48, 64), dtype=np.uint8)
    checkpoint = mx.pack_e2m1_codes(codes)
    scales = np.full((2, 48), 127, dtype=np.uint8)
    scales[0] = 116
    payload = prepare_packed_folded_payload(
        checkpoint, scales, allow_approximate=True,
    )
    assert not payload.lossless and payload.inexact_value_count > 0
    with pytest.raises(ValueError, match="loss metadata disagrees"):
        PackedFoldedPayload(
            payload.weight_bytes, payload.scale_plane,
            True, 0, 0.0, 0.0,
        )


def test_packed_folded_payload_canonicalizes_valid_integer_scales() -> None:
    checkpoint = mx.pack_e2m1_codes(np.ones((48, 64), dtype=np.uint8))
    scales = np.full((2, 48), 127, dtype=np.int32)
    payload = prepare_packed_folded_payload(
        checkpoint, scales, allow_approximate=True,
    )
    assert payload.scale_plane.dtype == np.uint8
    np.testing.assert_array_equal(payload.scale_plane[:-1], scales)


def test_packed_decode_table_matches_every_e2m1_code_and_delta() -> None:
    source = emit_mxfp4_packed_folded_prefill_hip()
    match = re.search(
        r"tessera_fold_e2m1_e4m3\[13\]\[16\] = (\{.*?\});",
        source, flags=re.DOTALL,
    )
    assert match is not None
    table = np.asarray(
        ast.literal_eval(match.group(1).replace("{", "[").replace("}", "]")),
        dtype=np.uint8,
    )
    assert table.shape == (13, 16)
    for delta in range(13):
        expected = (
            mx._E2M1 * np.exp2(np.float32(-delta))
        ).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
        np.testing.assert_array_equal(table[delta], expected)
    for delta in range(13, 255):
        expected = (
            mx._E2M1 * np.exp2(np.float32(-delta))
        ).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
        np.testing.assert_array_equal(
            expected, np.asarray([0] * 8 + [128] * 8, dtype=np.uint8),
        )
