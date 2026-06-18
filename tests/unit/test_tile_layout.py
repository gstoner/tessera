"""Tests for tessera.compiler.tile_layout — Gluon-style typed tile layouts."""

from __future__ import annotations

import math

import pytest

from tessera.compiler.tile_layout import (
    BlockedLayout,
    ConvertLayout,
    LinearLayout,
    SliceLayout,
    convert_cost,
)


# ── BlockedLayout ─────────────────────────────────────────────────────────────


def test_blocked_block_shape_is_elementwise_product() -> None:
    bl = BlockedLayout(
        size_per_thread=(1, 4),
        threads_per_warp=(8, 4),
        warps_per_cta=(2, 2),
        order=(1, 0),
    )
    assert bl.block_shape == (1 * 8 * 2, 4 * 4 * 2)
    assert bl.block_shape == (16, 32)
    assert bl.rank == 2
    assert bl.warp_size == 32
    assert bl.warps == 4


def test_blocked_rank_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="share one rank"):
        BlockedLayout(
            size_per_thread=(1, 4),
            threads_per_warp=(8,),  # rank 1, mismatched
            warps_per_cta=(2, 2),
            order=(1, 0),
        )


def test_blocked_order_must_be_permutation() -> None:
    with pytest.raises(ValueError, match="permutation"):
        BlockedLayout(
            size_per_thread=(1, 4),
            threads_per_warp=(8, 4),
            warps_per_cta=(2, 2),
            order=(0, 0),  # not a permutation
        )


def test_blocked_positive_components() -> None:
    with pytest.raises(ValueError, match="all-positive"):
        BlockedLayout(
            size_per_thread=(1, 0),
            threads_per_warp=(8, 4),
            warps_per_cta=(2, 2),
            order=(1, 0),
        )


def test_blocked_metadata_roundtrip_shape() -> None:
    bl = BlockedLayout((2, 2), (4, 8), (1, 2), (0, 1))
    md = bl.as_metadata_dict()
    assert md["kind"] == "blocked"
    assert md["block_shape"] == [2 * 4 * 1, 2 * 8 * 2]
    assert md["order"] == [0, 1]
    # all values JSON-able (lists / ints / str)
    assert all(isinstance(v, (list, int, str)) for v in md.values())


# ── SliceLayout ───────────────────────────────────────────────────────────────


def test_slice_removes_dim_from_block_shape() -> None:
    parent = BlockedLayout((1, 4), (8, 4), (2, 2), (1, 0))
    assert parent.block_shape == (16, 32)
    sl0 = SliceLayout(dim=0, parent=parent)
    sl1 = SliceLayout(dim=1, parent=parent)
    assert sl0.block_shape == (32,)
    assert sl1.block_shape == (16,)
    assert sl0.rank == 1


def test_slice_dim_out_of_range_raises() -> None:
    parent = BlockedLayout((1, 4), (8, 4), (2, 2), (1, 0))
    with pytest.raises(ValueError, match="out of range"):
        SliceLayout(dim=2, parent=parent)
    with pytest.raises(ValueError, match="out of range"):
        SliceLayout(dim=-1, parent=parent)


def test_slice_of_three_dim_parent() -> None:
    parent = BlockedLayout((1, 1, 2), (2, 4, 4), (1, 1, 1), (2, 1, 0))
    assert parent.block_shape == (2, 4, 8)
    sl = SliceLayout(dim=1, parent=parent)
    assert sl.block_shape == (2, 8)
    md = sl.as_metadata_dict()
    assert md["kind"] == "slice"
    assert md["dim"] == 1
    assert md["parent"]["kind"] == "blocked"
    assert md["block_shape"] == [2, 8]


# ── LinearLayout: construction + validity ─────────────────────────────────────


def _identity_4x4() -> LinearLayout:
    # shape (4, 4) = 16 elements = 4 bits. 2 reg bits along axis 0, 2 lane bits
    # along axis 1 — a bijective identity-ish distribution.
    return LinearLayout(
        reg_bases=((1, 0), (2, 0)),
        lane_bases=((0, 1), (0, 2)),
        warp_bases=(),
        block_bases=(),
        shape=(4, 4),
    )


def test_linear_basic_introspection() -> None:
    ll = _identity_4x4()
    assert ll.rank == 2
    assert ll.num_bits == 4
    assert ll.shape_bits == 4
    assert ll.block_shape == (4, 4)


def test_linear_is_valid_for_bijection() -> None:
    assert _identity_4x4().is_valid()


def test_linear_invalid_when_too_few_bits() -> None:
    ll = LinearLayout(
        reg_bases=((1, 0),),  # only 1 bit for a 16-element space
        lane_bases=(),
        warp_bases=(),
        block_bases=(),
        shape=(4, 4),
    )
    assert not ll.is_valid()


def test_linear_invalid_when_bases_collapse() -> None:
    # Two identical bases collapse two index bits onto one coordinate.
    ll = LinearLayout(
        reg_bases=((1, 0), (1, 0), (0, 1), (0, 2)),
        lane_bases=(),
        warp_bases=(),
        block_bases=(),
        shape=(4, 4),
    )
    assert ll.num_bits == ll.shape_bits  # right count
    assert not ll.is_valid()  # but linearly dependent


def test_linear_invalid_non_power_of_two_shape() -> None:
    ll = LinearLayout(
        reg_bases=((1,), (2,)),
        lane_bases=(),
        warp_bases=(),
        block_bases=(),
        shape=(6,),  # not a power of two
    )
    assert not ll.is_valid()


def test_linear_basis_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match="length 2"):
        LinearLayout(
            reg_bases=((1, 0, 0),),  # rank-3 vec but shape is rank 2
            lane_bases=(),
            warp_bases=(),
            block_bases=(),
            shape=(4, 4),
        )


def test_linear_empty_shape_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        LinearLayout((), (), (), (), shape=())


# ── LinearLayout: free metadata operations ────────────────────────────────────


def test_linear_permute_is_free_and_preserves_validity() -> None:
    ll = _identity_4x4()
    pl = ll.permute((1, 0))
    assert pl is not ll
    assert pl.shape == (4, 4)
    # axes swapped on each basis vector
    assert pl.reg_bases == ((0, 1), (0, 2))
    assert pl.lane_bases == ((1, 0), (2, 0))
    assert pl.is_valid()


def test_linear_transpose_equals_permute_rank2() -> None:
    ll = _identity_4x4()
    assert ll.transpose((1, 0)) == ll.permute((1, 0))


def test_linear_permute_rejects_non_permutation() -> None:
    ll = _identity_4x4()
    with pytest.raises(ValueError, match="permutation"):
        ll.permute((0, 0))


def test_linear_reshape_preserves_size_and_validity() -> None:
    ll = _identity_4x4()
    rl = ll.reshape((16,))
    assert rl is not ll
    assert rl.shape == (16,)
    assert rl.num_bits == 4
    assert rl.is_valid()


def test_linear_reshape_size_mismatch_raises() -> None:
    ll = _identity_4x4()
    with pytest.raises(ValueError, match="size must be preserved"):
        ll.reshape((4, 8))


def test_linear_split_and_join_roundtrip() -> None:
    ll = _identity_4x4()
    low, high = ll.split("reg_bases")
    assert len(low.reg_bases) == 1
    assert len(high.reg_bases) == 1
    rejoined = low.join(high, "reg_bases")
    assert rejoined.reg_bases == ll.reg_bases
    assert rejoined.is_valid()


def test_linear_split_empty_level_raises() -> None:
    ll = _identity_4x4()
    with pytest.raises(ValueError, match="no basis bits"):
        ll.split("warp_bases")


def test_linear_join_shape_mismatch_raises() -> None:
    a = _identity_4x4()
    b = LinearLayout((), (), (), (), shape=(2, 2))
    with pytest.raises(ValueError, match="shapes must match"):
        a.join(b)


def test_linear_metadata_roundtrip_shape() -> None:
    md = _identity_4x4().as_metadata_dict()
    assert md["kind"] == "linear"
    assert md["shape"] == [4, 4]
    assert md["num_bits"] == 4
    assert md["reg_bases"] == [[1, 0], [2, 0]]
    assert md["block_shape"] == [4, 4]


# ── cost model ────────────────────────────────────────────────────────────────


def test_convert_cost_zero_for_identity_blocked() -> None:
    bl = BlockedLayout((1, 4), (8, 4), (2, 2), (1, 0))
    assert convert_cost(bl, bl) == 0


def test_convert_cost_zero_for_pure_bit_permutation() -> None:
    # Same layout with reg bases listed in a different order == bit-permutation.
    a = LinearLayout(
        reg_bases=((1, 0), (2, 0)),
        lane_bases=((0, 1), (0, 2)),
        warp_bases=(),
        block_bases=(),
        shape=(4, 4),
    )
    b = LinearLayout(
        reg_bases=((2, 0), (1, 0)),  # reordered — free
        lane_bases=((0, 2), (0, 1)),
        warp_bases=(),
        block_bases=(),
        shape=(4, 4),
    )
    assert a != b
    assert convert_cost(a, b) == 0


def test_convert_cost_positive_for_genuinely_different_layout() -> None:
    # Different distribution: a register bit moves to a lane bit. Not a free
    # relabel — requires a shared-memory round-trip.
    a = LinearLayout(
        reg_bases=((1, 0), (2, 0)),
        lane_bases=((0, 1), (0, 2)),
        warp_bases=(),
        block_bases=(),
        shape=(4, 4),
    )
    b = LinearLayout(
        reg_bases=((1, 0),),
        lane_bases=((2, 0), (0, 1), (0, 2)),  # one reg bit became a lane bit
        warp_bases=(),
        block_bases=(),
        shape=(4, 4),
    )
    cost = convert_cost(a, b)
    assert cost == math.prod((4, 4))
    assert cost > 0


def test_convert_cost_positive_for_different_blocked() -> None:
    a = BlockedLayout((1, 4), (8, 4), (2, 2), (1, 0))
    b = BlockedLayout((4, 1), (4, 8), (2, 2), (0, 1))
    cost = convert_cost(a, b)
    assert cost == math.prod(b.block_shape)
    assert cost > 0


def test_convert_layout_dataclass() -> None:
    a = BlockedLayout((1, 4), (8, 4), (2, 2), (1, 0))
    cv_free = ConvertLayout(src=a, dst=a)
    assert cv_free.cost == 0
    assert cv_free.is_free

    b = BlockedLayout((4, 1), (4, 8), (2, 2), (0, 1))
    cv = ConvertLayout(src=a, dst=b)
    assert cv.cost == math.prod(b.block_shape)
    assert not cv.is_free
    md = cv.as_metadata_dict()
    assert md["kind"] == "convert_layout"
    assert md["src"]["kind"] == "blocked"
    assert md["dst"]["kind"] == "blocked"
    assert md["cost"] == cv.cost
    assert md["is_free"] is False


def test_convert_cost_bit_permutation_requires_same_shape() -> None:
    a = LinearLayout(((1, 0), (2, 0)), ((0, 1), (0, 2)), (), (), (4, 4))
    b = a.reshape((16,))
    # different shapes -> not a bit-permutation equivalence -> positive cost
    assert convert_cost(a, b) == 16
