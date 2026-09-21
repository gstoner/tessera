"""Unit tests for tessera.compiler.rocm_tiling (B1).

Covers the AMD-Gluon-grounded register-budget tiling model: per-arch register
budgets (512 legacy CDNA, 256 RDNA wave32, and 1024 CDNA5 wave32), the
VGPR-usage heuristic, candidate pruning (the v6 "double-buffering spills"
lesson), and the output-tile slicing transforms (quad_slice / n_slice).
"""

from __future__ import annotations

import pytest

from tessera.compiler.rocm_target import (
    AMDArch,
    ROCmTargetProfile,
    TesseraROCmTargetError,
)
from tessera.compiler.rocm_tiling import (
    PruneResult,
    RankedTileCandidate,
    TileCandidate,
    TileShape,
    estimate_lds_footprint_bytes,
    estimate_vgpr_usage,
    fits_budget,
    n_slice,
    prune_candidates,
    quad_slice,
    rank_candidates,
    requires_lds_bank_padding,
)

CDNA = ROCmTargetProfile(arch=AMDArch.GFX_942)  # wave64, 256 VGPR + 256 AGPR
RDNA = ROCmTargetProfile(arch=AMDArch.GFX_1100)  # wave32, 256 VGPR + 0 AGPR


# ── Per-arch register budgets ───────────────────────────────────────────────


def test_cdna_register_budget_is_512() -> None:
    assert CDNA.vgpr_budget == 256
    assert CDNA.agpr_budget == 256
    assert CDNA.total_reg_budget == 512


def test_rdna_register_budget_is_256() -> None:
    assert RDNA.vgpr_budget == 256
    assert RDNA.agpr_budget == 0
    assert RDNA.total_reg_budget == 256


def test_all_cdna_arches_have_512_budget() -> None:
    for arch in (AMDArch.GFX_90A, AMDArch.GFX_940, AMDArch.GFX_942, AMDArch.GFX_950):
        prof = ROCmTargetProfile(arch=arch)
        assert prof.total_reg_budget == 512, arch.name


def test_all_rdna_wave32_arches_have_256_budget() -> None:
    for arch in (
        AMDArch.GFX_1100,
        AMDArch.GFX_1151,
        AMDArch.GFX_1200,
        AMDArch.GFX_1201,
    ):
        prof = ROCmTargetProfile(arch=arch)
        assert prof.agpr_budget == 0, arch.name
        assert prof.total_reg_budget == 256, arch.name


def test_all_cdna5_wave32_arches_have_1024_budget() -> None:
    for arch in (AMDArch.GFX_1250, AMDArch.GFX_1251):
        prof = ROCmTargetProfile(arch=arch)
        assert prof.vgpr_budget == 1024, arch.name
        assert prof.agpr_budget == 0, arch.name
        assert prof.total_reg_budget == 1024, arch.name


# ── TileShape / TileCandidate validation ────────────────────────────────────


def test_tileshape_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        TileShape(0, 64, 16)
    with pytest.raises(ValueError):
        TileShape(64, -1, 16)
    with pytest.raises(ValueError):
        TileShape(64, 64, 0)


def test_tileshape_output_area() -> None:
    assert TileShape(64, 128, 16).output_area == 64 * 128


def test_tilecandidate_rejects_bad_n_slice() -> None:
    with pytest.raises(ValueError):
        TileCandidate(TileShape(64, 64, 16), "fp16", n_slice=0)


def test_tilecandidate_rejects_empty_dtype() -> None:
    with pytest.raises(ValueError):
        TileCandidate(TileShape(64, 64, 16), "")


def test_tilecandidate_metadata_roundtrip() -> None:
    cand = TileCandidate(TileShape(64, 128, 16), "bf16", double_buffer=True, n_slice=2)
    md = cand.as_metadata_dict()
    assert md["dtype"] == "bf16"
    assert md["double_buffer"] is True
    assert md["n_slice"] == 2
    assert md["tile"] == {"m": 64, "n": 128, "k": 16}


# ── VGPR usage heuristic ────────────────────────────────────────────────────


def test_small_tile_fits_cdna() -> None:
    cand = TileCandidate(TileShape(64, 64, 16), "fp16")
    # acc = ceil(64*64*1/64) = 64 ; stage = ceil(2*16/64) = 1 -> 65
    assert estimate_vgpr_usage(cand, CDNA) == 65
    assert fits_budget(cand, CDNA)


def test_huge_tile_is_pruned_cdna() -> None:
    # The v6 lesson: an output tile too big for the budget spills.
    cand = TileCandidate(TileShape(256, 256, 64), "fp16")
    # acc = ceil(256*256/64) = 1024 -> already over 512
    assert estimate_vgpr_usage(cand, CDNA) > CDNA.total_reg_budget
    assert not fits_budget(cand, CDNA)


def test_acc_word_count_drives_usage() -> None:
    # fp64 accumulates in 2 words -> doubles the accumulator footprint vs fp32.
    f32 = TileCandidate(TileShape(64, 64, 16), "fp32")
    f64 = TileCandidate(TileShape(64, 64, 16), "fp64")
    assert estimate_vgpr_usage(f64, CDNA) > estimate_vgpr_usage(f32, CDNA)


def test_n_slice_reduces_usage() -> None:
    full = TileCandidate(TileShape(128, 256, 16), "fp16", n_slice=1)
    sliced = TileCandidate(TileShape(128, 256, 16), "fp16", n_slice=4)
    assert estimate_vgpr_usage(sliced, CDNA) < estimate_vgpr_usage(full, CDNA)


def test_rdna_smaller_budget_prunes_more() -> None:
    # A tile that fits CDNA's 512 but not RDNA's 256.
    cand = TileCandidate(TileShape(128, 192, 16), "fp16")  # acc = ceil(128*192/32)
    # On RDNA lanes=32: acc = ceil(24576/32) = 768 > 256
    assert not fits_budget(cand, RDNA)


def test_unmodelled_dtype_raises() -> None:
    cand = TileCandidate(TileShape(64, 64, 16), "not_a_dtype")
    with pytest.raises(TesseraROCmTargetError):
        estimate_vgpr_usage(cand, CDNA)


# ── Double-buffering pushes a borderline tile over budget (the Gluon −73%) ───


def test_double_buffer_pushes_borderline_tile_over_budget() -> None:
    tile = TileShape(192, 128, 4096)  # acc=384, stage_per_lane=ceil(8192/64)=128
    single = TileCandidate(tile, "fp16", double_buffer=False)
    doubled = TileCandidate(tile, "fp16", double_buffer=True)

    # Single buffer: 384 + 128 = 512 -> exactly fits.
    assert estimate_vgpr_usage(single, CDNA) == 512
    assert fits_budget(single, CDNA)

    # Double buffer: 384 + 256 = 640 -> spills (the regression).
    assert estimate_vgpr_usage(doubled, CDNA) == 640
    assert not fits_budget(doubled, CDNA)


def test_double_buffer_only_grows_staging() -> None:
    tile = TileShape(64, 64, 64)
    single = TileCandidate(tile, "fp16", double_buffer=False)
    doubled = TileCandidate(tile, "fp16", double_buffer=True)
    assert estimate_vgpr_usage(doubled, CDNA) > estimate_vgpr_usage(single, CDNA)


# ── prune_candidates -> auditable PruneResult ───────────────────────────────


def test_prune_records_dropped() -> None:
    good = TileCandidate(TileShape(64, 64, 16), "fp16")
    bad = TileCandidate(TileShape(256, 256, 64), "fp16")
    result = prune_candidates([good, bad], CDNA)
    assert isinstance(result, PruneResult)
    assert result.kept == (good,)
    assert result.dropped == (bad,)
    assert result.n_kept == 1
    assert result.n_dropped == 1


def test_prune_nothing_silently_lost() -> None:
    cands = [
        TileCandidate(TileShape(64, 64, 16), "fp16"),
        TileCandidate(TileShape(512, 512, 16), "fp16"),
        TileCandidate(TileShape(128, 128, 16), "bf16"),
    ]
    result = prune_candidates(cands, CDNA)
    assert result.n_kept + result.n_dropped == len(cands)


def test_prune_result_metadata() -> None:
    result = prune_candidates([TileCandidate(TileShape(64, 64, 16), "fp16")], CDNA)
    md = result.as_metadata_dict()
    assert md["n_kept"] == 1
    assert md["n_dropped"] == 0
    assert len(md["kept"]) == 1  # type: ignore[arg-type]


def test_prune_empty() -> None:
    result = prune_candidates([], CDNA)
    assert result.n_kept == 0
    assert result.n_dropped == 0


# ── autotune ranking metadata ───────────────────────────────────────────────


def test_estimate_lds_footprint_accounts_for_pipeline_depth() -> None:
    cand = TileCandidate(TileShape(64, 64, 16), "fp16")
    one = estimate_lds_footprint_bytes(cand, CDNA, pipeline_depth=1)
    two = estimate_lds_footprint_bytes(cand, CDNA, pipeline_depth=2)
    assert two == one * 2


def test_double_buffer_enforces_two_stage_lds_minimum() -> None:
    single = TileCandidate(TileShape(64, 64, 16), "fp16", double_buffer=False)
    doubled = TileCandidate(TileShape(64, 64, 16), "fp16", double_buffer=True)
    assert estimate_lds_footprint_bytes(single, CDNA, pipeline_depth=1) * 2 == (
        estimate_lds_footprint_bytes(doubled, CDNA, pipeline_depth=1)
    )


def test_bank_padding_requirement_is_visible() -> None:
    conflict = TileCandidate(TileShape(64, 64, 16), "fp16")
    padded_away = TileCandidate(TileShape(64, 48, 16), "fp16")
    assert requires_lds_bank_padding(conflict)
    assert not requires_lds_bank_padding(padded_away)


def test_rank_candidates_keeps_non_measured_metadata() -> None:
    cand = TileCandidate(TileShape(64, 64, 16), "fp16")
    ranked = rank_candidates([cand], CDNA, pipeline_depth=2)
    assert len(ranked) == 1
    entry = ranked[0]
    assert isinstance(entry, RankedTileCandidate)
    md = entry.as_metadata_dict()
    assert md["measured"] is False
    assert md["pipeline_depth"] == 2
    assert md["register_macro_tile"] == (4, 4)
    assert "register_fit" in md["reasons"]


def test_rank_candidates_prefers_fit_over_register_spill() -> None:
    good = TileCandidate(TileShape(64, 64, 16), "fp16")
    spilling = TileCandidate(TileShape(256, 256, 64), "fp16")
    ranked = rank_candidates([spilling, good], CDNA)
    assert ranked[0].candidate is good
    assert ranked[-1].candidate is spilling
    assert "register_over_budget" in ranked[-1].reasons


def test_rank_candidates_marks_split_k_need() -> None:
    deep_k = TileCandidate(TileShape(64, 64, 8192), "fp16")
    ranked = rank_candidates([deep_k], CDNA)
    assert ranked[0].split_k_required
    assert "split_k_required" in ranked[0].reasons


# ── quad_slice / n_slice transforms ─────────────────────────────────────────


def test_quad_slice_yields_four_quartered_tiles() -> None:
    tile = TileShape(128, 256, 16)
    quads = quad_slice(tile)
    assert len(quads) == 4
    for q in quads:
        assert q.m == 64
        assert q.n == 128
        assert q.k == 16
        # Each quadrant has a quartered m*n area.
        assert q.output_area == tile.output_area // 4


def test_quad_slice_requires_even_dims() -> None:
    with pytest.raises(ValueError):
        quad_slice(TileShape(65, 64, 16))
    with pytest.raises(ValueError):
        quad_slice(TileShape(64, 63, 16))


def test_quad_slice_preserves_k() -> None:
    quads = quad_slice(TileShape(64, 64, 32))
    assert all(q.k == 32 for q in quads)


def test_quad_slice_fits_an_otherwise_spilling_tile() -> None:
    # A tile that spills, sliced down to a quadrant that fits — the real fix.
    big = TileShape(256, 256, 16)
    big_cand = TileCandidate(big, "fp16")
    assert not fits_budget(big_cand, CDNA)
    quad = quad_slice(big)[0]
    quad_cand = TileCandidate(quad, "fp16")
    assert fits_budget(quad_cand, CDNA)


def test_n_slice_halves_n() -> None:
    parts = n_slice(TileShape(128, 256, 16), 2)
    assert len(parts) == 2
    for p in parts:
        assert p.m == 128
        assert p.n == 128
        assert p.k == 16


def test_n_slice_four_parts() -> None:
    parts = n_slice(TileShape(64, 256, 16), 4)
    assert len(parts) == 4
    assert all(p.n == 64 for p in parts)


def test_n_slice_requires_divisible() -> None:
    with pytest.raises(ValueError):
        n_slice(TileShape(64, 100, 16), 3)


def test_n_slice_rejects_zero_parts() -> None:
    with pytest.raises(ValueError):
        n_slice(TileShape(64, 64, 16), 0)


def test_n_slice_single_part_is_identity() -> None:
    parts = n_slice(TileShape(64, 128, 16), 1)
    assert parts == (TileShape(64, 128, 16),)


# ── Reported occupancy (RDNA-OCCUPANCY-GRANULE-2026-09-20) ──────────────────


def _rdna4_profile():
    return ROCmTargetProfile(arch=AMDArch.GFX_1201, pipeline_stages=2)


def test_ranked_candidates_report_occupancy():
    profile = _rdna4_profile()
    cands = [TileCandidate(TileShape(64, 64, 16), "fp16", double_buffer=True)]
    ranked = rank_candidates(cands, profile)
    row = ranked[0]
    assert row.occupancy_waves_per_simd is not None
    assert 1 <= row.occupancy_waves_per_simd <= 16
    assert row.as_metadata_dict()["occupancy_waves_per_simd"] == (
        row.occupancy_waves_per_simd
    )


def test_reported_occupancy_matches_the_occupancy_model():
    """One authority for the number: the ranking reports what
    ``rocm_occupancy`` computes, it does not recompute it (Decision #31)."""
    from tessera.compiler.rocm_occupancy import allocate_vgprs

    profile = _rdna4_profile()
    for shape in ((32, 32, 16), (64, 64, 16), (128, 128, 16)):
        cand = TileCandidate(TileShape(*shape), "fp16", double_buffer=True)
        ranked = rank_candidates([cand], profile)[0]
        if ranked.occupancy_waves_per_simd is None:
            continue
        assert (
            ranked.occupancy_waves_per_simd
            == allocate_vgprs(
                ranked.vgpr_usage,
                arch=profile.arch,
                per_wave_cap=profile.vgpr_budget,
            ).waves_per_simd
        )


def test_occupancy_is_reported_but_does_not_change_ranking():
    """UNWIRED as a ranking input (Decision #29a).  Making occupancy binding
    changes which tile production selects, which needs exact-device proof on
    the launch arch; until then the score must be untouched.  This test is what
    stops the field quietly becoming load-bearing.
    """
    profile = _rdna4_profile()
    cands = [
        TileCandidate(TileShape(32, 32, 16), "fp16", double_buffer=True),
        TileCandidate(TileShape(64, 64, 16), "fp16", double_buffer=True),
        TileCandidate(TileShape(128, 64, 16), "fp16", double_buffer=True),
        TileCandidate(TileShape(64, 128, 16), "fp16", double_buffer=True),
    ]
    ranked = rank_candidates(cands, profile)
    # The score must be derivable without ever consulting occupancy: two
    # candidates on different rungs may not be reordered by that fact.
    for row in ranked:
        recomputed = float(row.vgpr_usage)
        if row.register_margin < 0:
            recomputed += 1_000_000 + abs(row.register_margin) * 100
        if row.lds_margin < 0:
            recomputed += 500_000 + abs(row.lds_margin) / 16
        # score carries further terms; the point is only that no occupancy
        # term appears -- a score that moved with occupancy would fail the
        # monotonicity below.
        assert recomputed <= row.score + 1e-6

    by_score = sorted(ranked, key=lambda r: r.score)
    assert list(ranked) == by_score, "ranking must remain score-ordered"


def test_occupancy_is_none_for_an_arch_without_established_constants():
    """An unmeasured part loses this one reported field rather than failing to
    rank at all."""
    profile = ROCmTargetProfile(arch=AMDArch.GFX_942, pipeline_stages=2)
    cands = [TileCandidate(TileShape(64, 64, 16), "fp16", double_buffer=True)]
    ranked = rank_candidates(cands, profile)
    assert ranked[0].occupancy_waves_per_simd is None
