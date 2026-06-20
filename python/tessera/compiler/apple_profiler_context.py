"""Apple Silicon profiler context helpers.

This module captures the hardware-free part of the Apple support idea from
SiliconScope: classify a workload using system-context telemetry such as GPU
activity, unified-memory bandwidth, memory pressure, and thermal throttling.
Native IOReport/SMC/HID sampling is intentionally out of scope here; Tessera can
feed these pure helpers from a future macOS collector without making compiler
tests depend on private Apple APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


APPLE_PROFILER_CONTEXT_SCHEMA_VERSION = "tessera.apple_profiler_context.v1"

IDLE = "idle"
GPU_ACTIVE = "gpu_active"
BANDWIDTH_BOUND = "bandwidth_bound"
COMPUTE_BOUND = "compute_bound"
THERMAL_THROTTLED = "thermal_throttled"
MEMORY_PRESSURED = "memory_pressured"

BOTTLENECKS: frozenset[str] = frozenset({
    IDLE,
    GPU_ACTIVE,
    BANDWIDTH_BOUND,
    COMPUTE_BOUND,
    THERMAL_THROTTLED,
    MEMORY_PRESSURED,
})


@dataclass(frozen=True)
class AppleProfilerContext:
    """One normalized Apple system-context sample for profiler correlation."""

    gpu_usage: float = 0.0
    total_bandwidth_gbs: float = 0.0
    achievable_bandwidth_gbs: float = 0.0
    memory_pressure: str = "normal"
    thermal_throttling: bool = False
    gpu_frequency_mhz: float | None = None
    gpu_power_watts: float | None = None
    dram_power_watts: float | None = None
    ane_power_watts: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.gpu_usage <= 1.0:
            raise ValueError("gpu_usage must be in [0, 1]")
        if self.total_bandwidth_gbs < 0:
            raise ValueError("total_bandwidth_gbs must be non-negative")
        if self.achievable_bandwidth_gbs < 0:
            raise ValueError("achievable_bandwidth_gbs must be non-negative")
        if self.memory_pressure not in {"normal", "elevated", "critical"}:
            raise ValueError("memory_pressure must be normal, elevated, or critical")

    def classify(self) -> str:
        return classify_apple_profiler_context(
            gpu_usage=self.gpu_usage,
            bandwidth_gbs=self.total_bandwidth_gbs,
            achievable_bandwidth_gbs=self.achievable_bandwidth_gbs,
            memory_critical=self.memory_pressure == "critical",
            thermal_throttling=self.thermal_throttling,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": APPLE_PROFILER_CONTEXT_SCHEMA_VERSION,
            "gpu_usage": self.gpu_usage,
            "total_bandwidth_gbs": self.total_bandwidth_gbs,
            "achievable_bandwidth_gbs": self.achievable_bandwidth_gbs,
            "memory_pressure": self.memory_pressure,
            "thermal_throttling": self.thermal_throttling,
            "gpu_frequency_mhz": self.gpu_frequency_mhz,
            "gpu_power_watts": self.gpu_power_watts,
            "dram_power_watts": self.dram_power_watts,
            "ane_power_watts": self.ane_power_watts,
            "bottleneck": self.classify(),
        }


def classify_apple_profiler_context(
    *,
    gpu_usage: float,
    bandwidth_gbs: float,
    achievable_bandwidth_gbs: float,
    memory_critical: bool = False,
    thermal_throttling: bool = False,
) -> str:
    """Classify the dominant Apple workload bottleneck from normalized inputs."""

    if not 0.0 <= gpu_usage <= 1.0:
        raise ValueError("gpu_usage must be in [0, 1]")
    if bandwidth_gbs < 0 or achievable_bandwidth_gbs < 0:
        raise ValueError("bandwidth inputs must be non-negative")
    if memory_critical:
        return MEMORY_PRESSURED
    if thermal_throttling:
        return THERMAL_THROTTLED
    if gpu_usage < 0.30:
        return IDLE
    bw_fraction = (
        bandwidth_gbs / achievable_bandwidth_gbs
        if achievable_bandwidth_gbs > 0
        else 0.0
    )
    if bw_fraction >= 0.85:
        return BANDWIDTH_BOUND
    if gpu_usage >= 0.90:
        return COMPUTE_BOUND
    return GPU_ACTIVE


def apple_unified_memory_bandwidth_ceiling_gbs(
    chip_name: str,
    *,
    p_core_count: int = 0,
) -> float:
    """Return the Apple Silicon unified-memory bandwidth ceiling in GB/s.

    Unknown chips return 0 so callers can fall back to an observed peak.
    """

    name = chip_name.lower()
    if "m1" in name:
        if "ultra" in name:
            return 800.0
        if "max" in name:
            return 400.0
        if "pro" in name:
            return 200.0
        return 68.0
    if "m2" in name:
        if "ultra" in name:
            return 800.0
        if "max" in name:
            return 400.0
        if "pro" in name:
            return 200.0
        return 100.0
    if "m3" in name:
        if "ultra" in name:
            return 800.0
        if "max" in name:
            return 400.0 if p_core_count >= 12 else 300.0
        if "pro" in name:
            return 150.0
        return 100.0
    if "m4" in name:
        if "max" in name:
            return 546.0 if p_core_count >= 12 else 410.0
        if "pro" in name:
            return 273.0
        return 120.0
    return 0.0


def apple_profiler_context_contract() -> dict[str, Any]:
    """Describe the future native collector fields this pure module expects."""

    return {
        "schema": APPLE_PROFILER_CONTEXT_SCHEMA_VERSION,
        "provider": "apple-silicon-system-context",
        "status": "planned",
        "collector": "sudoless IOReport/SMC/HID native helper",
        "signals": [
            "gpu_usage",
            "gpu_frequency_mhz",
            "total_bandwidth_gbs",
            "gpu_bandwidth_gbs",
            "cpu_bandwidth_gbs",
            "media_bandwidth_gbs",
            "gpu_power_watts",
            "dram_power_watts",
            "ane_power_watts_estimate",
            "memory_pressure",
            "thermal_throttling",
        ],
        "notes": (
            "Native sampling should stay separate from compiler tests because "
            "IOReport, SMC, and HID surfaces are private or host-specific."
        ),
    }


__all__ = [
    "APPLE_PROFILER_CONTEXT_SCHEMA_VERSION",
    "AppleProfilerContext",
    "BANDWIDTH_BOUND",
    "BOTTLENECKS",
    "COMPUTE_BOUND",
    "GPU_ACTIVE",
    "IDLE",
    "MEMORY_PRESSURED",
    "THERMAL_THROTTLED",
    "apple_profiler_context_contract",
    "apple_unified_memory_bandwidth_ceiling_gbs",
    "classify_apple_profiler_context",
]
