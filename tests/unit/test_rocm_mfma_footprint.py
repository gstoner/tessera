"""Unit tests for the MFMA accumulator-footprint cost model (rocm_target.py).

Ported from the moonmath CDNA3 attention writeup, §"Matrix Core Selection":
16×16×16 is chosen over 32×32×8 despite identical FLOPs/cycle, purely because
its M×N accumulator is 4 fp32/lane vs 16 — freeing registers for deeper
prefetch rings + persistent Q.  These tests pin the footprint arithmetic and
the cheapest-shape selection that lever drives.
"""

from __future__ import annotations

import pytest

from tessera.compiler.rocm_target import (
    AMDArch,
    ROCmTargetProfile,
    TesseraROCmTargetError,
    cheapest_mfma_shape,
    mfma_accumulator_regs,
    rank_mfma_shapes_by_footprint,
)


# ── footprint arithmetic ─────────────────────────────────────────────────────


def test_footprint_matches_article_numbers() -> None:
    # The two numbers the article calls out explicitly (CDNA wave64).
    assert mfma_accumulator_regs((16, 16, 16, 1)) == 4
    assert mfma_accumulator_regs((32, 32, 8, 1)) == 16


def test_footprint_independent_of_k() -> None:
    # Accumulator is the M×N output tile — K does not change it.
    assert mfma_accumulator_regs((16, 16, 16, 1)) == mfma_accumulator_regs((16, 16, 32, 1))
    assert mfma_accumulator_regs((32, 32, 4, 1)) == mfma_accumulator_regs((32, 32, 16, 1))


def test_footprint_accepts_3tuple_wmma_shape() -> None:
    # WMMA shapes are 3-tuples; wave32 lanes=32.
    assert mfma_accumulator_regs((16, 16, 16), lanes=32) == 8


def test_footprint_rejects_indivisible_tile() -> None:
    with pytest.raises(TesseraROCmTargetError, match="does not divide evenly"):
        mfma_accumulator_regs((16, 17, 16, 1))


def test_footprint_rejects_malformed_shape() -> None:
    with pytest.raises(TesseraROCmTargetError, match="must be"):
        mfma_accumulator_regs((16,))
    with pytest.raises(TesseraROCmTargetError, match="positive"):
        mfma_accumulator_regs((0, 16, 16, 1))
    with pytest.raises(TesseraROCmTargetError, match="lanes must be positive"):
        mfma_accumulator_regs((16, 16, 16, 1), lanes=0)


# ── ranking + selection ──────────────────────────────────────────────────────


def test_ranking_orders_cheapest_first() -> None:
    ranked = rank_mfma_shapes_by_footprint(AMDArch.GFX_942)
    regs = [r for _, r in ranked]
    assert regs == sorted(regs), "ranking must be non-decreasing in footprint"
    # All 16×16 shapes (footprint 4) precede all 32×32 shapes (footprint 16).
    first_32 = next(i for i, (s, _) in enumerate(ranked) if s[0] == 32)
    assert all(s[0] == 16 for s, _ in ranked[:first_32])
    assert all(r == 4 for _, r in ranked[:first_32])


def test_ranking_tiebreak_prefers_denser_shape() -> None:
    # Within the footprint-4 tier, larger M*N*K (more work per issue) ranks first.
    ranked = rank_mfma_shapes_by_footprint(AMDArch.GFX_942)
    tier4 = [s for s, r in ranked if r == 4]
    densities = [s[0] * s[1] * s[2] for s in tier4]
    assert densities == sorted(densities, reverse=True)


def test_cheapest_is_low_footprint_shape() -> None:
    for arch in (AMDArch.GFX_942, AMDArch.GFX_950, AMDArch.GFX_90A):
        shape = cheapest_mfma_shape(arch)
        assert mfma_accumulator_regs(shape) == 4


def test_cheapest_bf16_reproduces_article_choice() -> None:
    # bf16 native form is k=16 → the article's 16×16×16 wins over the k=8
    # 32×32×8 form (which it would, since 4 regs < 16 regs).
    assert cheapest_mfma_shape(AMDArch.GFX_942, k=16) == (16, 16, 16, 1)
    # Cross-check the explicit pairwise comparison the article makes.
    assert mfma_accumulator_regs((16, 16, 16, 1)) < mfma_accumulator_regs((32, 32, 8, 1))


def test_cheapest_fp4_path_on_cdna4() -> None:
    # CDNA 4 fp4 lowers on the k=64 form; cheapest stays a 16×16 tile.
    assert cheapest_mfma_shape(AMDArch.GFX_950, k=64) == (16, 16, 64, 1)


def test_k_filter_rejects_absent_width() -> None:
    with pytest.raises(TesseraROCmTargetError, match="contraction width k="):
        cheapest_mfma_shape(AMDArch.GFX_942, k=999)


def test_cheapest_rejects_wmma_arch() -> None:
    for arch in (AMDArch.GFX_1100, AMDArch.GFX_1151,
                 AMDArch.GFX_1200, AMDArch.GFX_1201):
        with pytest.raises(TesseraROCmTargetError, match="no MFMA shapes"):
            cheapest_mfma_shape(arch)


def test_wmma_arch_ranking_is_empty() -> None:
    assert rank_mfma_shapes_by_footprint(AMDArch.GFX_1100) == []


# ── profile-method parity ────────────────────────────────────────────────────


def test_profile_methods_delegate() -> None:
    p = ROCmTargetProfile(arch=AMDArch.GFX_942)
    assert p.cheapest_mfma_shape(k=16) == cheapest_mfma_shape(AMDArch.GFX_942, k=16)
    assert p.mfma_shapes_by_footprint() == rank_mfma_shapes_by_footprint(AMDArch.GFX_942)
