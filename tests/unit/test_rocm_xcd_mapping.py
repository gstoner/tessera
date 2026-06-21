"""Unit tests for chiplet/XCD-aware grid mapping (rocm_target.py).

Ported from the moonmath CDNA3 attention writeup's "head-first swizzle": pin all
of a head's Q-blocks to one XCD so its K/V stays resident in that XCD's L2 slice,
rather than scattering across dies under the default round-robin scheduler.
"""

from __future__ import annotations

import pytest

from tessera.compiler.rocm_target import (
    AMDArch,
    ROCmTargetProfile,
    TesseraROCmTargetError,
    head_first_xcd,
    naive_block_xcd,
    xcd_count,
)


# ── XCD topology table ───────────────────────────────────────────────────────


def test_xcd_counts() -> None:
    assert xcd_count(AMDArch.GFX_942) == 8     # MI300X
    assert xcd_count(AMDArch.GFX_940) == 6     # MI300A
    assert xcd_count(AMDArch.GFX_950) == 8     # CDNA 4
    assert xcd_count(AMDArch.GFX_90A) == 2     # MI250 — 2 GCDs


def test_rdna_is_monolithic() -> None:
    for arch in (AMDArch.GFX_1100, AMDArch.GFX_1151, AMDArch.GFX_1200):
        assert xcd_count(arch) == 1


def test_profile_topology_properties() -> None:
    assert ROCmTargetProfile(arch=AMDArch.GFX_942).num_xcds == 8
    assert ROCmTargetProfile(arch=AMDArch.GFX_942).is_multi_die is True
    assert ROCmTargetProfile(arch=AMDArch.GFX_1100).is_multi_die is False


# ── head-first residency property ────────────────────────────────────────────


def test_head_first_is_qblock_independent() -> None:
    # The whole point: every Q-block of a (batch, head) maps to the same XCD.
    nh, nx = 16, 8
    for batch in range(4):
        for head in range(nh):
            xcd = head_first_xcd(batch, head, num_heads=nh, num_xcds=nx)
            # head_first_xcd takes no q_block — it is constant for the pair.
            assert 0 <= xcd < nx


def test_head_first_pins_head_naive_scatters() -> None:
    nh, nx, qb = 16, 8, 4
    # Under head-first, a head's blocks share one XCD.
    hf = head_first_xcd(0, 3, num_heads=nh, num_xcds=nx)
    assert hf == 3
    # Under the default scheduler, the same head's blocks scatter across XCDs.
    naive = [
        naive_block_xcd(0, 3, q, num_heads=nh, q_blocks=qb, num_xcds=nx)
        for q in range(qb)
    ]
    assert len(set(naive)) > 1  # scattered — the residency problem


def test_head_first_distributes_pairs_across_xcds() -> None:
    # Consecutive (batch, head) pairs round-robin across all XCDs (balanced).
    nh, nx = 8, 8
    assigned = [
        head_first_xcd(b, h, num_heads=nh, num_xcds=nx)
        for b in range(2)
        for h in range(nh)
    ]
    # All XCDs are used.
    assert set(assigned) == set(range(nx))


def test_head_first_collapses_when_single_die() -> None:
    # On a monolithic part everything maps to the one die.
    for h in range(8):
        assert head_first_xcd(0, h, num_heads=8, num_xcds=1) == 0


# ── validation ───────────────────────────────────────────────────────────────


def test_head_first_rejects_bad_args() -> None:
    with pytest.raises(TesseraROCmTargetError, match="must be positive"):
        head_first_xcd(0, 0, num_heads=0, num_xcds=8)
    with pytest.raises(TesseraROCmTargetError, match="must be >= 0"):
        head_first_xcd(-1, 0, num_heads=8, num_xcds=8)
    with pytest.raises(TesseraROCmTargetError, match="out of range"):
        head_first_xcd(0, 8, num_heads=8, num_xcds=8)


def test_naive_block_xcd_validation() -> None:
    with pytest.raises(TesseraROCmTargetError, match="must be positive"):
        naive_block_xcd(0, 0, 0, num_heads=8, q_blocks=0, num_xcds=8)
    with pytest.raises(TesseraROCmTargetError, match="out of range"):
        naive_block_xcd(0, 0, 4, num_heads=8, q_blocks=4, num_xcds=8)
