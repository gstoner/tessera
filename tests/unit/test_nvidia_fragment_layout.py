from __future__ import annotations

import pytest

from tessera.compiler.nvidia_fragment_layout import PackedPair, sm120_m16n8k16_f16_pairs


def test_sm120_m16n8k16_f16_a_lane_map_matches_execution_oracle() -> None:
    assert sm120_m16n8k16_f16_pairs("a", 0) == (
        PackedPair((0, 0), (0, 1)), PackedPair((8, 0), (8, 1)),
        PackedPair((0, 8), (0, 9)), PackedPair((8, 8), (8, 9)),
    )
    assert sm120_m16n8k16_f16_pairs("a", 31) == (
        PackedPair((7, 6), (7, 7)), PackedPair((15, 6), (15, 7)),
        PackedPair((7, 14), (7, 15)), PackedPair((15, 14), (15, 15)),
    )


def test_sm120_m16n8k16_f16_b_lane_map_matches_execution_oracle() -> None:
    assert sm120_m16n8k16_f16_pairs("b", 0) == (
        PackedPair((0, 0), (1, 0)), PackedPair((8, 0), (9, 0)),
    )
    assert sm120_m16n8k16_f16_pairs("b", 31) == (
        PackedPair((6, 7), (7, 7)), PackedPair((14, 7), (15, 7)),
    )


@pytest.mark.parametrize("role,lane", [("acc", 0), ("a", -1), ("b", 32)])
def test_sm120_m16n8k16_f16_rejects_unproven_or_invalid_maps(
    role: str, lane: int
) -> None:
    with pytest.raises(ValueError):
        sm120_m16n8k16_f16_pairs(role, lane)  # type: ignore[arg-type]
