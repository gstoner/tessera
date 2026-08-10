from __future__ import annotations

import pytest

from tessera.compiler.autotune_v2 import GEMMWorkload, TuningConfig
from tessera.compiler.reuse_distance_cost import (
    CacheLevel,
    dtype_storage_bits,
    estimate_gemm_reuse_distance,
    estimate_gemm_reuse_distance_hierarchy,
)


def _estimate(
    config: TuningConfig,
    *,
    bandwidth: float = 64.0,
    cache_bytes: int = 64 * 1024,
):
    return estimate_gemm_reuse_distance(
        GEMMWorkload(1024, 2048, 256, dtype="fp16"),
        config,
        peak_tflops=1000.0,
        dram_bw_gbps=bandwidth,
        cache_bytes=cache_bytes,
    )


def test_memory_bandwidth_is_a_real_latency_term() -> None:
    cfg = TuningConfig(64, 64, 32)
    slow = _estimate(cfg, bandwidth=32.0)
    fast = _estimate(cfg, bandwidth=256.0)

    assert slow.memory_ms > fast.memory_ms
    assert slow.latency_ms > fast.latency_ms
    assert slow.dram_bytes == fast.dram_bytes


def test_cache_capacity_changes_reuse_and_dram_traffic() -> None:
    cfg = TuningConfig(64, 64, 32)
    none = _estimate(cfg, cache_bytes=0)
    roomy = _estimate(cfg, cache_bytes=8 * 1024 * 1024)

    assert none.cache_hit_rate == 0.0
    assert roomy.cache_hit_rate > none.cache_hit_rate
    assert roomy.dram_bytes < none.dram_bytes


def test_rasterization_changes_symbolic_reuse_distance() -> None:
    row = _estimate(TuningConfig(64, 64, 32, raster_order="row_major"))
    grouped = _estimate(
        TuningConfig(
            64,
            64,
            32,
            raster_order="grouped_m",
            raster_group=8,
        )
    )

    assert grouped.cache_hit_rate != row.cache_hit_rate
    assert grouped.dram_bytes != row.dram_bytes


def test_warp_and_stage_have_no_hand_fitted_latency_bonus() -> None:
    shallow = _estimate(TuningConfig(64, 64, 32, num_warps=1, num_stages=1))
    deep = _estimate(TuningConfig(64, 64, 32, num_warps=8, num_stages=4))

    assert shallow.latency_ms == deep.latency_ms
    assert shallow.dram_bytes == deep.dram_bytes


def test_subbyte_storage_widths_are_physical_not_rounded_to_fp16() -> None:
    assert dtype_storage_bits("fp4_e2m1") == 4
    assert dtype_storage_bits("fp6_e3m2") == 6
    with pytest.raises(ValueError, match="physical storage width"):
        dtype_storage_bits("complex64")


def test_sampling_is_deterministic_and_visible() -> None:
    workload = GEMMWorkload(16384, 16384, 4096, dtype="bf16")
    cfg = TuningConfig(32, 32, 32)
    first = estimate_gemm_reuse_distance(
        workload,
        cfg,
        peak_tflops=312.0,
        dram_bw_gbps=2039.0,
        cache_bytes=40 * 1024 * 1024,
        max_output_tiles=64,
        max_k_samples=4,
    )
    second = estimate_gemm_reuse_distance(
        workload,
        cfg,
        peak_tflops=312.0,
        dram_bw_gbps=2039.0,
        cache_bytes=40 * 1024 * 1024,
        max_output_tiles=64,
        max_k_samples=4,
    )

    assert first == second
    assert not first.exact
    assert first.sampled_output_tiles == 64
    assert first.sampled_k_tiles == 4


def test_cpu_hierarchy_reports_level_provenance_and_dram_traffic() -> None:
    workload = GEMMWorkload(256, 256, 256, dtype="fp32")
    config = TuningConfig(16, 64, 32)
    estimate = estimate_gemm_reuse_distance_hierarchy(
        workload,
        config,
        peak_tflops=1.6,
        dram_bw_gbps=80.0,
        cache_levels=(
            CacheLevel("L1D", 32 * 1024, 600.0),
            CacheLevel("L2", 1024 * 1024, 300.0),
            CacheLevel("L3", 32 * 1024 * 1024, 150.0),
        ),
    )

    assert estimate.method == "tile_reuse_distance_cpu_hierarchy_v1"
    assert [level.name for level in estimate.levels] == ["L1D", "L2", "L3"]
    assert estimate.dram_bytes > 0
    assert estimate.memory_ms >= estimate.dram_ms
    assert estimate.levels[1].hit_rate > 0.0


def test_cpu_hierarchy_rejects_unordered_or_guessed_levels() -> None:
    workload = GEMMWorkload(64, 64, 64, dtype="fp32")
    config = TuningConfig(16, 16, 16)
    with pytest.raises(ValueError, match="non-decreasing"):
        estimate_gemm_reuse_distance_hierarchy(
            workload,
            config,
            peak_tflops=1.0,
            dram_bw_gbps=1.0,
            cache_levels=(
                CacheLevel("L2", 1024, 10.0),
                CacheLevel("L1D", 512, 20.0),
            ),
        )
