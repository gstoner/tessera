from __future__ import annotations

import pytest

from tessera.compiler.accelerator_profiler_context import (
    AcceleratorProfilerContext,
    BANDWIDTH_BOUND,
    COMPUTE_BOUND,
    FABRIC_LIMITED,
    GPU_ACTIVE,
    IDLE,
    MEMORY_PRESSURED,
    POWER_CAPPED,
    RELIABILITY_RISK,
    THERMAL_THROTTLED,
    accelerator_profiler_context_contract,
    classify_accelerator_profiler_context,
)
from tessera.compiler.profiling_plan import HOST_CONTEXT, provider_capabilities


def test_accelerator_classifier_precedence() -> None:
    assert classify_accelerator_profiler_context(
        gpu_utilization=0.95,
        memory_utilization=0.95,
        memory_used_fraction=0.10,
        uncorrectable_ecc_errors=1,
    ) == RELIABILITY_RISK
    assert classify_accelerator_profiler_context(
        gpu_utilization=0.95,
        memory_utilization=0.95,
        memory_used_fraction=0.96,
    ) == MEMORY_PRESSURED
    assert classify_accelerator_profiler_context(
        gpu_utilization=0.95,
        memory_utilization=0.20,
        memory_used_fraction=0.10,
        temperature_c=96,
        temperature_limit_c=100,
    ) == THERMAL_THROTTLED
    assert classify_accelerator_profiler_context(
        gpu_utilization=0.95,
        memory_utilization=0.20,
        memory_used_fraction=0.10,
        power_watts=395,
        power_limit_watts=400,
    ) == POWER_CAPPED
    assert classify_accelerator_profiler_context(
        gpu_utilization=0.70,
        memory_utilization=0.20,
        memory_used_fraction=0.10,
        fabric_bandwidth_fraction=0.90,
    ) == FABRIC_LIMITED


@pytest.mark.parametrize(
    ("gpu", "mem", "used", "expected"),
    [
        (0.20, 0.95, 0.10, IDLE),
        (0.70, 0.90, 0.10, BANDWIDTH_BOUND),
        (0.95, 0.20, 0.10, COMPUTE_BOUND),
        (0.60, 0.20, 0.10, GPU_ACTIVE),
    ],
)
def test_accelerator_classifier_workload_shape(gpu: float, mem: float, used: float, expected: str) -> None:
    assert classify_accelerator_profiler_context(
        gpu_utilization=gpu,
        memory_utilization=mem,
        memory_used_fraction=used,
    ) == expected


def test_accelerator_context_serializes() -> None:
    sample = AcceleratorProfilerContext(
        vendor="nvidia",
        gpu_utilization=0.70,
        memory_utilization=0.90,
        memory_used_fraction=0.10,
        power_watts=300,
        power_limit_watts=700,
    )
    payload = sample.to_dict()

    assert payload["schema"] == "tessera.accelerator_profiler_context.v1"
    assert payload["vendor"] == "nvidia"
    assert payload["bottleneck"] == BANDWIDTH_BOUND


def test_accelerator_context_contracts_and_provider_rows() -> None:
    nvidia_contract = accelerator_profiler_context_contract("nvidia")
    rocm_contract = accelerator_profiler_context_contract("rocm")
    nvidia_host = {cap.feature: cap for cap in provider_capabilities("nvidia")}[HOST_CONTEXT]
    rocm_host = {cap.feature: cap for cap in provider_capabilities("rocm")}[HOST_CONTEXT]

    assert nvidia_contract["provider"] == "nvidia-system-context"
    assert "nvlink_bandwidth_fraction" in nvidia_contract["signals"]
    assert rocm_contract["provider"] == "rocm-system-context"
    assert "xgmi_bandwidth_fraction" in rocm_contract["signals"]
    assert "nvidia-system-context" in nvidia_host.provider
    assert "include_dcgm" in nvidia_host.controls
    assert "rocm-system-context" in rocm_host.provider
    assert "include_amd_smi" in rocm_host.controls
