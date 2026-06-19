"""Workstream C — attention lowering selector (IO-cost, not hard threshold).

Proves the selector ranks kernel variants by off-chip byte movement, respects
feasibility, and reproduces the old materialized→online→reference behavior as one
scored decision. Page-staging bytes from a Workstream-A PagedKVState enter the
score.

See docs/audit/roadmap/CONTRACT_PASS_PLAN.md (Workstream C).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.fusion import (
    SYNTH_MAX_N, SYNTH_MAX_D, AttentionRegion,
    attention_lowering_costs, select_attention_lowering, paged_stage_bytes,
    run_fused_attention)


# ── feasibility invariant ─────────────────────────────────────────────────────


@pytest.mark.parametrize("Nk", [16, 256, 1024, 1025, 4096, 16384])
@pytest.mark.parametrize("Dv", [32, 256, 320])
def test_selector_never_picks_infeasible(Nk, Dv):
    choice = select_attention_lowering(M=8, Nk=Nk, D=64, Dv=Dv)
    assert choice.feasible


# ── variant selection reproduces the old branch ──────────────────────────────


def test_small_nk_picks_materialized():
    c = select_attention_lowering(M=8, Nk=128, D=64, Dv=64)
    assert c.variant == "materialized"


def test_large_nk_small_dv_picks_online():
    c = select_attention_lowering(M=8, Nk=SYNTH_MAX_N + 1, D=64, Dv=64)
    assert c.variant == "online"


def test_large_nk_large_dv_falls_to_reference():
    c = select_attention_lowering(M=8, Nk=SYNTH_MAX_N + 1, D=512, Dv=SYNTH_MAX_D + 1)
    assert c.variant == "reference"


# ── cost-monotonicity: fused beats unfused on bytes; crossover is at the cap ──


def test_fused_has_fewer_bytes_than_reference():
    mat, online, ref = attention_lowering_costs(M=8, Nk=256, D=64, Dv=64)
    # The reference round-trips the score matrix through DRAM; fused keeps it
    # on-chip — so fused is strictly cheaper in bytes.
    assert mat.dram_bytes < ref.dram_bytes
    assert online.dram_bytes < ref.dram_bytes
    assert mat.dram_bytes == online.dram_bytes  # same external IO


def test_crossover_is_the_feasibility_cap_not_a_magic_constant():
    # At exactly the cap, materialized is still feasible & chosen; one past it,
    # the selector crosses to online — the crossover is the on-chip stack bound.
    at_cap = select_attention_lowering(M=8, Nk=SYNTH_MAX_N, D=64, Dv=64)
    past_cap = select_attention_lowering(M=8, Nk=SYNTH_MAX_N + 1, D=64, Dv=64)
    assert at_cap.variant == "materialized"
    assert past_cap.variant == "online"


def test_reference_always_feasible():
    _, _, ref = attention_lowering_costs(M=1, Nk=10**6, D=64, Dv=4096)
    assert ref.feasible and ref.variant == "reference"


# ── page-staging bytes enter the score (Workstream A linkage) ─────────────────


def test_stage_bytes_increase_cost():
    base = select_attention_lowering(M=8, Nk=256, D=64, Dv=64, stage_bytes=0)
    staged = select_attention_lowering(M=8, Nk=256, D=64, Dv=64, stage_bytes=1_000_000)
    assert staged.dram_bytes > base.dram_bytes
    assert staged.variant == base.variant  # feasibility unchanged


def test_paged_stage_bytes_zero_for_contiguous():
    from tessera.cache import KVCacheHandle
    h = KVCacheHandle(num_heads=2, head_dim=4, max_seq=64, page_size=8)
    h.append(np.zeros((10, 2, 4), np.float32), np.zeros((10, 2, 4), np.float32))
    assert paged_stage_bytes(h, list(range(10))) == 0  # all resident


def test_paged_stage_bytes_positive_for_cold_tiered():
    from tessera.cache import TieredKVCache
    c = TieredKVCache(num_heads=2, head_dim=4, max_seq=64, page_size=4,
                      resident_capacity=1)
    c.write(np.zeros((12, 2, 4), np.float32), np.zeros((12, 2, 4), np.float32))
    # Nothing staged yet → every touched page is cold → positive staging cost.
    assert paged_stage_bytes(c, list(range(12))) > 0


# ── behavior preservation: selector-driven dispatch stays numerically correct ─


@pytest.mark.parametrize("Nk", [64, 256])
def test_run_fused_attention_matches_reference(Nk):
    rng = np.random.default_rng(0)
    M, D, Dv = 8, 32, 32
    Q = rng.standard_normal((M, D)).astype(np.float32)
    K = rng.standard_normal((Nk, D)).astype(np.float32)
    V = rng.standard_normal((Nk, Dv)).astype(np.float32)
    region = AttentionRegion(scale=1.0 / np.sqrt(D))
    out, _ = run_fused_attention(region, Q, K, V)
    np.testing.assert_allclose(out, region.reference(Q, K, V), rtol=1e-4, atol=1e-4)
