"""Unit tests for LDS-budget-aware FlashAttention tile sizing (attn_lower.py).

Ported from the moonmath CDNA3 attention writeup ("3Q tiling"): the autotuner
should prune tilings against the per-arch LDS budget instead of sweeping blind.
CDNA 4's doubled LDS budget admits strictly more tilings.
"""

from __future__ import annotations

import pytest

from tessera.compiler.attn_lower import (
    FlashAttnLoweringConfig,
    TesseraAttnConfigError,
    candidate_configs,
    feasible_configs,
)
from tessera.compiler.rocm_target import AMDArch, ROCmTargetProfile


def _cfg(tile_kv: int = 64, stages: int = 2, tile_q: int = 64) -> FlashAttnLoweringConfig:
    return FlashAttnLoweringConfig(tile_q=tile_q, tile_kv=tile_kv, pipeline_stages=stages)


# ── lds_bytes estimate ───────────────────────────────────────────────────────


def test_lds_bytes_counts_k_and_v_double_buffered() -> None:
    # 2 stages × (K + V) × (64 × 128 × 2 bytes) = 65536.
    assert _cfg().lds_bytes(head_dim=128) == 65536


def test_lds_bytes_v_in_l1_drops_v_term() -> None:
    full = _cfg().lds_bytes(head_dim=128)
    no_v = _cfg().lds_bytes(head_dim=128, stage_v=False)
    assert no_v == full // 2  # exactly the K half remains


def test_lds_bytes_3q_adds_q_tile() -> None:
    base = _cfg().lds_bytes(head_dim=128, stage_v=False)
    with_q = _cfg().lds_bytes(head_dim=128, stage_v=False, stage_q=True)
    assert with_q == base + 64 * 128 * 2


def test_lds_bytes_scales_with_stages_and_tile() -> None:
    assert _cfg(stages=3).lds_bytes(head_dim=128) == _cfg(stages=2).lds_bytes(head_dim=128) // 2 * 3
    assert _cfg(tile_kv=128).lds_bytes(head_dim=128) == 2 * _cfg(tile_kv=64).lds_bytes(head_dim=128)


def test_lds_bytes_fp8_is_half_of_bf16() -> None:
    assert _cfg().lds_bytes(head_dim=128, dtype_bytes=1) == _cfg().lds_bytes(head_dim=128) // 2


def test_lds_bytes_rejects_bad_dims() -> None:
    with pytest.raises(TesseraAttnConfigError, match="head_dim must be positive"):
        _cfg().lds_bytes(head_dim=0)
    with pytest.raises(TesseraAttnConfigError, match="dtype_bytes must be positive"):
        _cfg().lds_bytes(head_dim=128, dtype_bytes=0)


# ── fits_lds feasibility ─────────────────────────────────────────────────────


def test_fits_lds_int_budget_boundary() -> None:
    c = _cfg()
    # Exactly 64 KiB usage fits a 64 KiB budget; one byte less does not.
    assert c.fits_lds(65536, head_dim=128)
    assert not c.fits_lds(65535, head_dim=128)


def test_fits_lds_accepts_profile() -> None:
    c = _cfg()
    g942 = ROCmTargetProfile(arch=AMDArch.GFX_942)   # 64 KiB
    g950 = ROCmTargetProfile(arch=AMDArch.GFX_950)   # 160 KiB
    assert c.fits_lds(g942, head_dim=128)
    # A 128 KiB tiling (tile_kv=128, 2 stages, K+V) overflows CDNA3's 64 KiB
    # but fits CDNA4's 160 KiB.
    big = _cfg(tile_kv=128, stages=2)
    assert big.lds_bytes(head_dim=128) == 131072
    assert not big.fits_lds(g942, head_dim=128)
    assert big.fits_lds(g950, head_dim=128)


def test_fits_lds_rejects_bad_budget() -> None:
    with pytest.raises(TesseraAttnConfigError, match="must be positive"):
        _cfg().fits_lds(0, head_dim=128)
    with pytest.raises(TesseraAttnConfigError, match="bool"):
        _cfg().fits_lds(True, head_dim=128)
    with pytest.raises(TesseraAttnConfigError, match="lds_capacity_bytes"):
        _cfg().fits_lds(object(), head_dim=128)


# ── candidate enumeration + pruning ──────────────────────────────────────────


def test_candidate_configs_is_full_grid() -> None:
    cands = candidate_configs()
    assert len(cands) == 4 * 4 * 2  # tile_q × tile_kv × stages
    # All are valid (power-of-two tiles enforced by the dataclass).
    assert all(isinstance(c, FlashAttnLoweringConfig) for c in cands)


def test_feasible_is_subset_and_all_fit() -> None:
    g942 = ROCmTargetProfile(arch=AMDArch.GFX_942)
    feas = feasible_configs(g942, head_dim=128)
    allc = candidate_configs()
    assert 0 < len(feas) < len(allc)
    assert all(c.fits_lds(g942, head_dim=128) for c in feas)


def test_cdna4_admits_more_configs_than_cdna3() -> None:
    g942 = ROCmTargetProfile(arch=AMDArch.GFX_942)
    g950 = ROCmTargetProfile(arch=AMDArch.GFX_950)
    n942 = len(feasible_configs(g942, head_dim=128))
    n950 = len(feasible_configs(g950, head_dim=128))
    assert n950 > n942


def test_feasible_preserves_candidate_order() -> None:
    g950 = ROCmTargetProfile(arch=AMDArch.GFX_950)
    feas = feasible_configs(g950, head_dim=128)
    ordered = [c for c in candidate_configs() if c in feas]
    assert feas == ordered


def test_feasible_v_in_l1_admits_more() -> None:
    # Moving V to L1 halves the K/V LDS term → strictly more tilings fit.
    g942 = ROCmTargetProfile(arch=AMDArch.GFX_942)
    both = len(feasible_configs(g942, head_dim=128, stage_v=True))
    v_l1 = len(feasible_configs(g942, head_dim=128, stage_v=False))
    assert v_l1 > both


def test_feasible_custom_candidate_pool() -> None:
    pool = [_cfg(tile_kv=32), _cfg(tile_kv=256, stages=3)]
    feas = feasible_configs(65536, head_dim=128, candidates=pool)
    assert _cfg(tile_kv=32) in feas
    assert _cfg(tile_kv=256, stages=3) not in feas
