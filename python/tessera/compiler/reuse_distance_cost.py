"""Tile-granular reuse-distance cost model for GEMM schedule pruning.

The model deliberately predicts only facts supported by compiler-visible
inputs:

* the symbolic order of output tiles, including block rasterization;
* the A/B tile identities touched by each reduction step;
* physical storage bytes for the workload dtype;
* a fully-associative LRU cache with an explicitly supplied capacity;
* target peak throughput and DRAM bandwidth.

It contains no fitted tile, warp, stage, or architecture coefficients.  The
result is therefore suitable for pruning obviously poor schedules, not for
promoting a near-neighbour over measured exact-device evidence.

Large problems use a deterministic prefix of the output-tile stream and a
uniform sample of the reduction axis.  Representative tile weights preserve
the unsampled K working-set and traffic scale.  The result records whether it
was exact so callers cannot mistake a bounded search-time estimate for a full
trace.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .tile_rasterization import remap

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .autotune_v2 import GEMMWorkload, TuningConfig


_DTYPE_BITS = {
    "bf16": 16,
    "fp16": 16,
    "fp32": 32,
    "fp8": 8,
    "fp8_e4m3": 8,
    "fp8_e5m2": 8,
    "fp6_e2m3": 6,
    "fp6_e3m2": 6,
    "fp4_e2m1": 4,
    "nvfp4": 4,
    "int8": 8,
}


@dataclass(frozen=True)
class ReuseDistanceEstimate:
    """One deterministic analytical estimate.

    ``cache_hits`` and ``cache_misses`` count sampled symbolic A/B tile
    accesses. ``dram_bytes`` is extrapolated to the full problem and includes
    output stores.  ``latency_ms`` is the ordinary roofline maximum of the
    compute and memory terms; no empirical overlap coefficient is introduced.
    """

    latency_ms: float
    compute_ms: float
    memory_ms: float
    dram_bytes: float
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    sampled_output_tiles: int
    total_output_tiles: int
    sampled_k_tiles: int
    total_k_tiles: int
    exact: bool
    method: str = "tile_reuse_distance_lru_v1"


@dataclass(frozen=True)
class CacheLevel:
    """One measured CPU cache level used by the hierarchy model."""

    name: str
    capacity_bytes: int
    bandwidth_gbps: float


@dataclass(frozen=True)
class CacheLevelEstimate:
    name: str
    capacity_bytes: int
    hits: int
    misses: int
    hit_rate: float
    transfer_bytes: float
    transfer_ms: float


@dataclass(frozen=True)
class HierarchicalReuseDistanceEstimate:
    """Serial cache-hierarchy estimate for CPU schedule ranking.

    This deliberately remains an evidence/pruning model.  It uses measured
    capacities and bandwidths but does not pretend to model Zen replacement,
    prefetch, store-forwarding, or cache-set conflicts.
    """

    latency_ms: float
    compute_ms: float
    memory_ms: float
    dram_bytes: float
    dram_ms: float
    levels: tuple[CacheLevelEstimate, ...]
    sampled_output_tiles: int
    total_output_tiles: int
    sampled_k_tiles: int
    total_k_tiles: int
    exact: bool
    method: str = "tile_reuse_distance_cpu_hierarchy_v1"


def dtype_storage_bits(dtype: str) -> int:
    """Canonical physical storage width used by the tile trace."""

    try:
        return _DTYPE_BITS[dtype]
    except KeyError:
        raise ValueError(
            f"reuse-distance model has no physical storage width for {dtype!r}"
        ) from None


def estimate_gemm_reuse_distance(
    workload: "GEMMWorkload",
    config: "TuningConfig",
    *,
    peak_tflops: float,
    dram_bw_gbps: float,
    cache_bytes: int,
    max_output_tiles: int = 4096,
    max_k_samples: int = 16,
) -> ReuseDistanceEstimate:
    """Estimate GEMM latency from a symbolic tile trace and explicit hardware.

    A cache capacity of zero is a valid conservative boundary: every A/B tile
    access misses. Missing capacity must be represented explicitly by the
    caller rather than replaced with a guessed architecture default.
    """

    if peak_tflops <= 0.0:
        raise ValueError("peak_tflops must be > 0")
    if dram_bw_gbps <= 0.0:
        raise ValueError("dram_bw_gbps must be > 0")
    if cache_bytes < 0:
        raise ValueError("cache_bytes must be >= 0")
    if max_output_tiles < 1 or max_k_samples < 1:
        raise ValueError("sampling bounds must be >= 1")

    grid_m = _ceil_div(workload.M, config.tile_m)
    grid_n = _ceil_div(workload.N, config.tile_n)
    grid_k = _ceil_div(workload.K, config.tile_k)
    total_output_tiles = grid_m * grid_n
    sampled_output_tiles = min(total_output_tiles, max_output_tiles)
    k_indices = _uniform_indices(grid_k, max_k_samples)
    k_weight = grid_k / len(k_indices)
    output_weight = total_output_tiles / sampled_output_tiles
    storage_bits = dtype_storage_bits(workload.dtype)

    cache: OrderedDict[tuple[str, int, int], float] = OrderedDict()
    resident_bytes = 0.0
    hits = 0
    misses = 0
    sampled_miss_bytes = 0.0

    for bid in range(sampled_output_tiles):
        m_tile, n_tile = remap(
            config.raster_order,
            bid,
            grid_m,
            grid_n,
            group=config.raster_group,
        )
        m_extent = min(config.tile_m, workload.M - m_tile * config.tile_m)
        n_extent = min(config.tile_n, workload.N - n_tile * config.tile_n)
        for k_tile in k_indices:
            k_extent = min(
                config.tile_k, workload.K - k_tile * config.tile_k
            )
            # Each sampled K tile represents a uniform group of real K tiles.
            # Scaling its cache footprint preserves the unsampled working set.
            a_bytes = _packed_bytes(m_extent * k_extent, storage_bits) * k_weight
            b_bytes = _packed_bytes(k_extent * n_extent, storage_bits) * k_weight
            for key, size in (
                (("A", m_tile, k_tile), a_bytes),
                (("B", k_tile, n_tile), b_bytes),
            ):
                if key in cache:
                    hits += 1
                    cache.move_to_end(key)
                    continue
                misses += 1
                sampled_miss_bytes += size
                if cache_bytes == 0 or size > cache_bytes:
                    continue
                cache[key] = size
                resident_bytes += size
                while resident_bytes > cache_bytes and cache:
                    _, evicted = cache.popitem(last=False)
                    resident_bytes -= evicted

    input_dram_bytes = sampled_miss_bytes * output_weight
    output_dram_bytes = _packed_bytes(workload.M * workload.N, storage_bits)
    dram_bytes = input_dram_bytes + output_dram_bytes
    compute_ms = workload.flops() / (peak_tflops * 1e12) * 1_000.0
    memory_ms = dram_bytes / (dram_bw_gbps * 1e9) * 1_000.0
    accesses = hits + misses
    hit_rate = hits / accesses if accesses else 0.0
    return ReuseDistanceEstimate(
        latency_ms=max(compute_ms, memory_ms),
        compute_ms=compute_ms,
        memory_ms=memory_ms,
        dram_bytes=dram_bytes,
        cache_hits=hits,
        cache_misses=misses,
        cache_hit_rate=hit_rate,
        sampled_output_tiles=sampled_output_tiles,
        total_output_tiles=total_output_tiles,
        sampled_k_tiles=len(k_indices),
        total_k_tiles=grid_k,
        exact=(
            sampled_output_tiles == total_output_tiles
            and len(k_indices) == grid_k
        ),
    )


def estimate_gemm_reuse_distance_hierarchy(
    workload: "GEMMWorkload",
    config: "TuningConfig",
    *,
    peak_tflops: float,
    dram_bw_gbps: float,
    cache_levels: tuple[CacheLevel, ...],
    max_output_tiles: int = 4096,
    max_k_samples: int = 16,
) -> HierarchicalReuseDistanceEstimate:
    """Estimate a tiled GEMM against an explicit L1/L2/L3 hierarchy.

    Caches are modeled as independent fully-associative LRUs ordered nearest
    to farthest.  A hit promotes the symbolic tile into nearer levels; a DRAM
    miss fills every level that can hold it.  Transfer time is conservatively
    serialized across traversed links.  All omissions are explicit—there are
    no fitted architecture coefficients.
    """

    if peak_tflops <= 0.0 or dram_bw_gbps <= 0.0:
        raise ValueError("peak_tflops and dram_bw_gbps must be > 0")
    if not cache_levels:
        raise ValueError("CPU hierarchy model requires at least one cache level")
    previous_capacity = -1
    for level in cache_levels:
        if not level.name or level.capacity_bytes < 0 or level.bandwidth_gbps <= 0.0:
            raise ValueError("cache levels require a name, non-negative capacity, and bandwidth")
        if level.capacity_bytes < previous_capacity:
            raise ValueError("cache levels must be ordered by non-decreasing capacity")
        previous_capacity = level.capacity_bytes
    if max_output_tiles < 1 or max_k_samples < 1:
        raise ValueError("sampling bounds must be >= 1")

    grid_m = _ceil_div(workload.M, config.tile_m)
    grid_n = _ceil_div(workload.N, config.tile_n)
    grid_k = _ceil_div(workload.K, config.tile_k)
    total_output_tiles = grid_m * grid_n
    sampled_output_tiles = min(total_output_tiles, max_output_tiles)
    k_indices = _uniform_indices(grid_k, max_k_samples)
    k_weight = grid_k / len(k_indices)
    output_weight = total_output_tiles / sampled_output_tiles
    storage_bits = dtype_storage_bits(workload.dtype)

    caches: list[OrderedDict[tuple[str, int, int], float]] = [
        OrderedDict() for _ in cache_levels
    ]
    resident = [0.0 for _ in cache_levels]
    hits = [0 for _ in cache_levels]
    misses = [0 for _ in cache_levels]
    transfer = [0.0 for _ in cache_levels]
    sampled_dram_bytes = 0.0

    def insert(level_index: int, key: tuple[str, int, int], size: float) -> None:
        nonlocal caches, resident
        capacity = cache_levels[level_index].capacity_bytes
        if capacity == 0 or size > capacity:
            return
        cache = caches[level_index]
        if key in cache:
            cache.move_to_end(key)
            return
        cache[key] = size
        resident[level_index] += size
        while resident[level_index] > capacity and cache:
            _, evicted = cache.popitem(last=False)
            resident[level_index] -= evicted

    for bid in range(sampled_output_tiles):
        m_tile, n_tile = remap(
            config.raster_order,
            bid,
            grid_m,
            grid_n,
            group=config.raster_group,
        )
        m_extent = min(config.tile_m, workload.M - m_tile * config.tile_m)
        n_extent = min(config.tile_n, workload.N - n_tile * config.tile_n)
        for k_tile in k_indices:
            k_extent = min(config.tile_k, workload.K - k_tile * config.tile_k)
            accesses = (
                (("A", m_tile, k_tile),
                 _packed_bytes(m_extent * k_extent, storage_bits) * k_weight),
                (("B", k_tile, n_tile),
                 _packed_bytes(k_extent * n_extent, storage_bits) * k_weight),
            )
            for key, size in accesses:
                hit_level: int | None = None
                for index, cache in enumerate(caches):
                    if key in cache:
                        hits[index] += 1
                        cache.move_to_end(key)
                        hit_level = index
                        break
                    misses[index] += 1
                traversed = len(cache_levels) if hit_level is None else hit_level
                for index in range(traversed):
                    transfer[index] += size
                    insert(index, key, size)
                if hit_level is None:
                    sampled_dram_bytes += size
                    for index in range(len(cache_levels)):
                        insert(index, key, size)

    output_bytes = _packed_bytes(workload.M * workload.N, storage_bits)
    dram_bytes = sampled_dram_bytes * output_weight + output_bytes
    dram_ms = dram_bytes / (dram_bw_gbps * 1e9) * 1_000.0
    level_estimates = []
    transfer_ms_total = 0.0
    for index, level in enumerate(cache_levels):
        bytes_at_level = transfer[index] * output_weight + output_bytes
        transfer_ms = bytes_at_level / (level.bandwidth_gbps * 1e9) * 1_000.0
        transfer_ms_total += transfer_ms
        access_count = hits[index] + misses[index]
        level_estimates.append(
            CacheLevelEstimate(
                name=level.name,
                capacity_bytes=level.capacity_bytes,
                hits=hits[index],
                misses=misses[index],
                hit_rate=hits[index] / access_count if access_count else 0.0,
                transfer_bytes=bytes_at_level,
                transfer_ms=transfer_ms,
            )
        )
    compute_ms = workload.flops() / (peak_tflops * 1e12) * 1_000.0
    memory_ms = transfer_ms_total + dram_ms
    return HierarchicalReuseDistanceEstimate(
        latency_ms=max(compute_ms, memory_ms),
        compute_ms=compute_ms,
        memory_ms=memory_ms,
        dram_bytes=dram_bytes,
        dram_ms=dram_ms,
        levels=tuple(level_estimates),
        sampled_output_tiles=sampled_output_tiles,
        total_output_tiles=total_output_tiles,
        sampled_k_tiles=len(k_indices),
        total_k_tiles=grid_k,
        exact=(sampled_output_tiles == total_output_tiles and len(k_indices) == grid_k),
    )


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _packed_bytes(elements: int, bits: int) -> float:
    return float(math.ceil(elements * bits / 8))


def _uniform_indices(extent: int, limit: int) -> tuple[int, ...]:
    if extent <= limit:
        return tuple(range(extent))
    # Midpoints of equal-width bins avoid over-weighting either end and are
    # stable across runs/platforms.
    return tuple(
        min(extent - 1, int((index + 0.5) * extent / limit))
        for index in range(limit)
    )


__all__ = [
    "CacheLevel",
    "CacheLevelEstimate",
    "HierarchicalReuseDistanceEstimate",
    "ReuseDistanceEstimate",
    "dtype_storage_bits",
    "estimate_gemm_reuse_distance",
    "estimate_gemm_reuse_distance_hierarchy",
]
