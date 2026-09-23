"""Model-load contract for the manual packed-folded gfx1201 candidate."""
from __future__ import annotations

import ast
import re
from unittest.mock import patch

import ml_dtypes
import numpy as np
import pytest

from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler import rocm_mxfp4_packed_folded as packed_module
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


def test_factory_reuses_fold_but_direct_constructor_verifies_receipt() -> None:
    checkpoint = mx.pack_e2m1_codes(np.ones((48, 64), dtype=np.uint8))
    scales = np.full((2, 48), 127, dtype=np.uint8)
    with patch.object(
        packed_module, "prepare_folded_weights",
        wraps=packed_module.prepare_folded_weights,
    ) as fold:
        payload = prepare_packed_folded_payload(
            checkpoint, scales, allow_approximate=True,
        )
        assert fold.call_count == 1
        direct = PackedFoldedPayload(
            payload.weight_bytes, payload.scale_plane, payload.lossless,
            payload.inexact_value_count, payload.max_normalized_abs_error,
            payload.max_normalized_relative_error,
        )
        assert fold.call_count == 2
        assert direct.receipt() == payload.receipt()


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


def test_permute_decode_table_matches_scalar_oracle() -> None:
    source = emit_mxfp4_packed_folded_prefill_hip(permute_decode=True)
    match = re.search(
        r"tessera_fold_magnitudes\[13\]\[2\] = (\{.*?\});",
        source, flags=re.DOTALL,
    )
    assert match is not None
    words = [int(value, 16) for value in re.findall(r"0x([0-9a-f]+)u", match.group(1))]
    assert len(words) == 26
    scalar_match = re.search(
        r"tessera_fold_e2m1_e4m3\[13\]\[16\] = (\{.*?\});",
        source, flags=re.DOTALL,
    )
    assert scalar_match is not None
    scalar = np.asarray(
        ast.literal_eval(scalar_match.group(1).replace("{", "[").replace("}", "]")),
        dtype=np.uint8,
    )
    for delta in range(13):
        magnitudes = [
            (words[delta * 2 + code // 4] >> (8 * (code % 4))) & 255
            for code in range(8)
        ]
        for code in range(16):
            assert magnitudes[code & 7] | ((code & 8) << 4) == scalar[delta, code]
    assert "__builtin_amdgcn_perm" in source
    assert "tessera_fold_word_permute(word, delta, block_scale)" in source


def test_batched_packed_stage_issues_both_loads_before_decode() -> None:
    source = emit_mxfp4_packed_folded_prefill_hip(
        integer_decode=True, batched_loads=True,
    )
    loads = source.index("words[q] = reinterpret_cast<const unsigned int *>(B)")
    decode = source.index("const unsigned int word = words[q];")
    assert loads < decode
    assert source.count("#pragma unroll\n      for (int q = 0; q < 2; ++q)") == 2
    assert "__DECODE_EXPRESSION__" not in source


def test_batched_activation_stage_issues_all_loads_before_lds_stores() -> None:
    source = emit_mxfp4_packed_folded_prefill_hip(
        integer_decode=True, batched_a_loads=True,
    )
    assert source.index("staged_a[q] = *reinterpret_cast") < source.index(
        "sA + (slot / 4) * 80 + off) = staged_a[q]"
    )


def test_pair_scale_reuse_loads_one_scale_and_reference_for_two_k16_words() -> None:
    with pytest.raises(ValueError, match="requires batched B loads"):
        emit_mxfp4_packed_folded_prefill_hip(reuse_pair_scales=True)
    source = emit_mxfp4_packed_folded_prefill_hip(
        integer_decode=True, batched_loads=True, reuse_pair_scales=True,
    )
    assert source.count("const unsigned char block_scale = Ref[") == 1
    assert source.count("const unsigned char row_ref = Ref[") == 1
    assert source.index("words[q] = reinterpret_cast") < source.index(
        "const unsigned int word = words[q];"
    )


def test_vector_pair_stage_covers_each_fragment_slot_once() -> None:
    with pytest.raises(ValueError, match="require permute decode"):
        emit_mxfp4_packed_folded_prefill_hip(vector_pair_loads=True)
    source = emit_mxfp4_packed_folded_prefill_hip(
        permute_decode=True, vector_pair_loads=True,
    )
    assert "const int first_slot = tid * 2;" in source
    assert "const unsigned long long packed =" in source
    assert "const unsigned short block_scales =" in source
    assert "const unsigned short row_refs =" in source
    assert "__DECODE_EXPRESSION__" not in source
    slots = set()
    stores = set()
    for tid in range(256):
        first = tid * 2
        lane = first & 31
        tile = first >> 5
        n_tile, k_step = tile >> 2, tile & 3
        for q in range(2):
            slots.add((n_tile, k_step, lane + q))
            stores.add((n_tile * 16 + (lane & 15) + q,
                        k_step * 16 + (lane >> 4) * 8))
    assert len(slots) == 512
    assert len(stores) == 512
    assert slots == {(nt, ks, lane) for nt in range(4)
                     for ks in range(4) for lane in range(32)}


def test_packed_provenance_sync_keys_distinguish_ablation_variants() -> None:
    def key(**overrides: bool) -> str:
        flags = dict(vector_pair_loads=False, permute_decode=False,
                     batched_loads=False, batched_a_loads=False,
                     reuse_pair_scales=False, a_base_hoist=False,
                     a_offset32=False)
        flags.update(overrides)
        return packed_module._packed_sync_key(**flags)

    assert key() == "GFX1201-PACKED-FOLDED-DECODE-2026-09-23"
    assert key(batched_loads=True) == "GFX1201-PACKED-STAGING-ABLATION-2026-09-23"
    assert key(permute_decode=True) == "GFX1201-PACKED-PERMUTE-DECODE-2026-09-23"
    assert key(permute_decode=True, vector_pair_loads=True) == (
        "GFX1201-PACKED-VECTOR-PAIR-2026-09-23"
    )
    assert key(permute_decode=True, batched_loads=True, a_base_hoist=True) == (
        "GFX1201-PACKED-A-BASE-2026-09-23"
    )
    assert key(permute_decode=True, batched_loads=True, a_offset32=True) == (
        "GFX1201-PACKED-A-OFFSET32-2026-09-23"
    )


def test_hoisted_a_base_stage_preserves_clamped_row_and_lds_layout() -> None:
    with pytest.raises(ValueError, match="separate staging ablation"):
        emit_mxfp4_packed_folded_prefill_hip(
            permute_decode=True, a_base_hoist=True, vector_pair_loads=True,
        )
    source = emit_mxfp4_packed_folded_prefill_hip(
        permute_decode=True, batched_loads=True, a_base_hoist=True,
    )
    assert "const unsigned char *__restrict__ tile_a = A + m0 * K + kb;" in source
    assert "local_row < last_local_row ? local_row : last_local_row" in source
    assert "tile_a + safe_local_row * K + off" in source
    assert "sA + local_row * 80 + off" in source
    assert "A + safe * K + kb + off" not in source


def test_bounded_a_offset32_stage_only_narrows_lane_local_arithmetic() -> None:
    limit = packed_module._MAX_A_OFFSET32_K
    assert limit * 255 + 48 <= (1 << 31) - 1
    assert (limit + 1) * 255 + 48 > (1 << 31) - 1
    packed_module._validate_a_offset32_k(limit)
    with pytest.raises(ValueError, match="32-bit A row offset"):
        packed_module._validate_a_offset32_k(limit + 1)
    with pytest.raises(ValueError, match="32-bit A row offset"):
        packed_module._validate_a_offset32_k(0)
    with pytest.raises(ValueError, match="separate staging ablation"):
        emit_mxfp4_packed_folded_prefill_hip(
            permute_decode=True, a_base_hoist=True, a_offset32=True,
        )
    source = emit_mxfp4_packed_folded_prefill_hip(
        permute_decode=True, batched_loads=True, a_offset32=True,
    )
    assert "const unsigned char *__restrict__ tile_a = A + m0 * K + kb;" in source
    assert "const int local_stride = (int)K;" in source
    assert "remaining_rows < 255 ? (int)remaining_rows : 255" in source
    assert "const int byte_offset = safe_local_row * local_stride + off;" in source
    assert "tile_a + byte_offset" in source
    assert "sA + local_row * 80 + off" in source
