"""Unit tests for the tail-KV split (flash-decoding) cost model (attn_split_kv.py).

Ported from the moonmath CDNA3 attention writeup: when the work-block grid does
not fill all CUs, split the stranded blocks' K/V range across idle CUs and merge
the partials with online-softmax rescale — but decline when the last wave is
already > 95% full or the sequence is too short to earn back the merge cost.
"""

from __future__ import annotations

import pytest

from tessera.compiler.attn_split_kv import (
    SplitKVPlan,
    TesseraSplitKVError,
    plan_split_kv,
)


# ── the split path ───────────────────────────────────────────────────────────


def test_underfilled_grid_splits() -> None:
    p = plan_split_kv(40, 304, kv_len=16384)
    assert p.is_split
    assert p.split_factor > 1
    assert p.occupancy_after > p.occupancy_before
    assert "split K/V" in p.reason


def test_split_improves_occupancy_monotonically() -> None:
    p = plan_split_kv(40, 304, kv_len=16384)
    # The chosen G genuinely raises last-wave occupancy.
    assert p.occupancy_after >= p.occupancy_before


def test_split_factor_bounded_by_kv_blocks() -> None:
    # kv_len 384 with kv_block 128 → only 3 KV blocks → G capped at 3.
    p = plan_split_kv(10, 304, kv_len=384, kv_block=128, min_kv_len=128)
    assert p.split_factor <= 3


def test_split_factor_bounded_by_max_split() -> None:
    p = plan_split_kv(10, 304, kv_len=1 << 20, max_split=4)
    assert p.split_factor <= 4


def test_merge_partials_and_effective_blocks() -> None:
    p = plan_split_kv(40, 304, kv_len=16384)
    assert p.effective_blocks == 40 * p.split_factor
    assert p.merge_partials == p.effective_blocks


# ── the decline paths ────────────────────────────────────────────────────────


def test_declines_when_last_wave_nearly_full() -> None:
    p = plan_split_kv(300, 304, kv_len=16384)
    assert not p.is_split
    assert p.split_factor == 1
    assert "full" in p.reason
    assert p.merge_partials == 0


def test_declines_when_grid_divides_evenly() -> None:
    p = plan_split_kv(608, 304, kv_len=16384)
    assert p.split_factor == 1
    assert p.occupancy_before == 1.0


def test_declines_for_short_sequence() -> None:
    p = plan_split_kv(40, 304, kv_len=256)
    assert p.split_factor == 1
    assert "min_kv_len" in p.reason


def test_declines_when_grid_exceeds_cu() -> None:
    p = plan_split_kv(310, 304, kv_len=16384)
    assert p.split_factor == 1
    assert ">= num_cu" in p.reason


def test_declines_when_under_two_kv_blocks() -> None:
    # kv_len 100 < one 128-block → cannot split, even past the min_kv_len gate.
    p = plan_split_kv(10, 304, kv_len=100, kv_block=128, min_kv_len=16)
    assert p.split_factor == 1


def test_custom_threshold_changes_decision() -> None:
    # 152/304 = 50% full.  A 0.40 threshold treats that as "full enough" and
    # declines; the default 0.95 threshold lets it through, and a 2-way split
    # fills the device exactly (152×2 = 304).
    strict = plan_split_kv(152, 304, kv_len=16384, occupancy_threshold=0.40)
    assert strict.split_factor == 1
    loose = plan_split_kv(152, 304, kv_len=16384, occupancy_threshold=0.95)
    assert loose.is_split
    assert loose.split_factor == 2
    assert loose.occupancy_after == 1.0


# ── metadata + validation ────────────────────────────────────────────────────


def test_metadata_round_trip() -> None:
    p = plan_split_kv(40, 304, kv_len=16384)
    md = p.as_metadata_dict()
    assert md["kind"] == "split_kv_plan"
    assert md["split_factor"] == p.split_factor
    assert md["effective_blocks"] == p.effective_blocks
    assert md["is_split"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(grid_blocks=0, num_cu=304, kv_len=16384),
        dict(grid_blocks=40, num_cu=0, kv_len=16384),
        dict(grid_blocks=40, num_cu=304, kv_len=0),
    ],
)
def test_rejects_nonpositive_core_args(kwargs: dict) -> None:
    with pytest.raises(TesseraSplitKVError, match="must all be positive"):
        plan_split_kv(**kwargs)


def test_rejects_bad_threshold() -> None:
    with pytest.raises(TesseraSplitKVError, match="occupancy_threshold"):
        plan_split_kv(40, 304, kv_len=16384, occupancy_threshold=1.5)


def test_plan_is_frozen_dataclass() -> None:
    p = plan_split_kv(40, 304, kv_len=16384)
    assert isinstance(p, SplitKVPlan)
    with pytest.raises(Exception):
        p.split_factor = 99  # type: ignore[misc]
