"""NVIDIA/AMD matmul optimization ladder + split-K semantics-preserving rewrite.

Split-K is the executable rung — its correctness is provable here (CPU/numpy and,
via the ``matmul`` hook, any backend) before any NVIDIA/AMD run. The ladder guards
keep the bring-up sequence honest (every rung mapped to a target + owning pass).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.matmul_opt_ladder import (
    MATMUL_OPT_LADDER,
    OptTarget,
    SplitKConfig,
    ladder_for_target,
    register_tile_intensity,
    render_ladder_markdown,
    split_k_matmul,
    split_k_profitable,
    verify_split_k_equivalence,
)


# ── split-K is semantics-preserving (the executable oracle) ──────────────────


@pytest.mark.parametrize("shape", [(8, 8, 16), (32, 16, 64), (4, 12, 30), (16, 16, 17)])
@pytest.mark.parametrize("splits", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("reduce", ["tree", "atomic"])
def test_split_k_equals_dense(shape, splits, reduce):
    m, n, k = shape
    rng = np.random.default_rng(splits * 100 + m)
    A = rng.standard_normal((m, k))
    B = rng.standard_normal((k, n))
    verdict = verify_split_k_equivalence(A, B, SplitKConfig(splits, reduce))
    assert verdict.is_equivalent, verdict.detail
    assert verdict.max_rel_err <= 1e-9


def test_split_k_over_fragmented_clamped_to_k():
    # splits > K must not produce empty partitions or drop terms.
    A = np.arange(2 * 5, dtype=np.float64).reshape(2, 5)
    B = np.arange(5 * 3, dtype=np.float64).reshape(5, 3)
    got = split_k_matmul(A, B, SplitKConfig(splits=99))
    np.testing.assert_allclose(got, A @ B, rtol=1e-12, atol=1e-12)


def test_split_k_matmul_routes_through_a_backend_hook():
    # The matmul hook lets partials run on a real backend; a correct hook keeps
    # the rewrite equivalent.
    A = np.random.default_rng(0).standard_normal((6, 10))
    B = np.random.default_rng(1).standard_normal((10, 4))
    calls = {"n": 0}

    def counting_mm(a, b):
        calls["n"] += 1
        return np.asarray(a) @ np.asarray(b)

    got = split_k_matmul(A, B, SplitKConfig(splits=4), matmul=counting_mm)
    assert calls["n"] == 4
    np.testing.assert_allclose(got, A @ B, rtol=1e-9, atol=1e-9)


def test_split_k_rejects_bad_config():
    with pytest.raises(ValueError):
        SplitKConfig(splits=0)
    with pytest.raises(ValueError):
        SplitKConfig(splits=2, reduce="bogus")
    with pytest.raises(ValueError):
        split_k_matmul(np.zeros((2, 3)), np.zeros((4, 5)), SplitKConfig(2))  # K mismatch


# ── planning models ──────────────────────────────────────────────────────────


def test_split_k_profitable_skinny_vs_saturated():
    # Skinny output grid (few tiles) + many SMs + large K → profitable.
    assert split_k_profitable(128, 128, 16384, tile_m=128, tile_n=128,
                              sm_count=128, splits=8)
    # A large output grid already saturates the device → not profitable.
    assert not split_k_profitable(8192, 8192, 1024, tile_m=128, tile_n=128,
                                 sm_count=128, splits=8)
    # splits < 2 is never split-K.
    assert not split_k_profitable(128, 128, 16384, tile_m=128, tile_n=128,
                                 sm_count=128, splits=1)


def test_register_tile_intensity_grows_with_tile():
    # The register-tiling lever: a coarse tile has higher arithmetic intensity.
    one = register_tile_intensity(1, 1)
    coarse = register_tile_intensity(8, 8)
    assert coarse > one
    # 26×4 (blog's tile) beats 1×1 substantially.
    assert register_tile_intensity(26, 4) > 3 * one


# ── ladder integrity (keeps the bring-up sequence honest) ────────────────────


def test_ladder_rungs_are_well_formed():
    assert len(MATMUL_OPT_LADDER) >= 8
    keys = [t.key for t in MATMUL_OPT_LADDER]
    assert len(keys) == len(set(keys)), "duplicate ladder keys"
    for t in MATMUL_OPT_LADDER:
        assert t.title and t.owner and t.blog_speedup and t.mechanism
        assert isinstance(t.targets, OptTarget)


def test_split_k_and_tensor_core_marked_verifiable_now():
    by_key = {t.key: t for t in MATMUL_OPT_LADDER}
    # split-K is a semantics-preserving rewrite proven here.
    assert by_key["split_k"].verifiable_now
    # warp specialization / TMA need real silicon to measure.
    assert not by_key["warp_specialization"].verifiable_now
    assert not by_key["async_copy_tma"].verifiable_now


def test_ladder_for_target_filters_amd_specific():
    amd = {t.key for t in ladder_for_target("amd")}
    # AMD gets the BOTH rungs (split-K, register tiling) but NOT NVIDIA-only ones.
    assert "split_k" in amd and "register_tile" in amd
    assert "warp_specialization" not in amd  # mbarrier ring is NVIDIA-only
    assert "async_copy_tma" not in amd
    nvidia = {t.key for t in ladder_for_target(OptTarget.NVIDIA)}
    assert "warp_specialization" in nvidia


def test_render_markdown_lists_all_rungs():
    md = render_ladder_markdown()
    for t in MATMUL_OPT_LADDER:
        assert t.title in md
