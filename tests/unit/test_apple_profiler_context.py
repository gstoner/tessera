from __future__ import annotations

import pytest

from tessera.compiler.apple_profiler_context import (
    AppleProfilerContext,
    BANDWIDTH_BOUND,
    COMPUTE_BOUND,
    GPU_ACTIVE,
    IDLE,
    MEMORY_PRESSURED,
    THERMAL_THROTTLED,
    apple_profiler_context_contract,
    apple_unified_memory_bandwidth_ceiling_gbs,
    classify_apple_profiler_context,
)
from tessera.compiler.profiling_plan import HOST_CONTEXT, provider_capabilities


def test_apple_profiler_context_classifies_dominant_bottleneck() -> None:
    assert classify_apple_profiler_context(
        gpu_usage=0.95,
        bandwidth_gbs=360,
        achievable_bandwidth_gbs=400,
        memory_critical=True,
        thermal_throttling=True,
    ) == MEMORY_PRESSURED
    assert classify_apple_profiler_context(
        gpu_usage=0.95,
        bandwidth_gbs=360,
        achievable_bandwidth_gbs=400,
        thermal_throttling=True,
    ) == THERMAL_THROTTLED
    assert classify_apple_profiler_context(
        gpu_usage=0.20,
        bandwidth_gbs=360,
        achievable_bandwidth_gbs=400,
    ) == IDLE
    assert classify_apple_profiler_context(
        gpu_usage=0.60,
        bandwidth_gbs=340,
        achievable_bandwidth_gbs=400,
    ) == BANDWIDTH_BOUND
    assert classify_apple_profiler_context(
        gpu_usage=0.95,
        bandwidth_gbs=200,
        achievable_bandwidth_gbs=400,
    ) == COMPUTE_BOUND
    assert classify_apple_profiler_context(
        gpu_usage=0.60,
        bandwidth_gbs=200,
        achievable_bandwidth_gbs=400,
    ) == GPU_ACTIVE


def test_apple_profiler_context_serializes_with_classification() -> None:
    sample = AppleProfilerContext(
        gpu_usage=0.60,
        total_bandwidth_gbs=340,
        achievable_bandwidth_gbs=400,
        gpu_frequency_mhz=1296,
        gpu_power_watts=12.5,
        dram_power_watts=4.0,
    )

    payload = sample.to_dict()

    assert payload["schema"] == "tessera.apple_profiler_context.v1"
    assert payload["bottleneck"] == BANDWIDTH_BOUND
    assert payload["gpu_frequency_mhz"] == 1296


@pytest.mark.parametrize(
    ("chip", "p_cores", "expected"),
    [
        ("Apple M1", 4, 68.0),
        ("Apple M2 Pro", 8, 200.0),
        ("Apple M3 Max", 12, 400.0),
        ("Apple M3 Max", 10, 300.0),
        ("Apple M4 Max", 12, 546.0),
        ("Apple M4 Max", 10, 410.0),
        ("Intel Core i9", 8, 0.0),
    ],
)
def test_apple_bandwidth_ceiling_table(chip: str, p_cores: int, expected: float) -> None:
    assert apple_unified_memory_bandwidth_ceiling_gbs(chip, p_core_count=p_cores) == expected


def test_apple_context_provider_contract_and_plan_row() -> None:
    contract = apple_profiler_context_contract()
    host_context = {
        cap.feature: cap for cap in provider_capabilities("apple_gpu")
    }[HOST_CONTEXT]

    assert contract["provider"] == "apple-silicon-system-context"
    assert "total_bandwidth_gbs" in contract["signals"]
    assert "apple-silicon-system-context" in host_context.provider
    assert "include_ioreport" in host_context.controls
