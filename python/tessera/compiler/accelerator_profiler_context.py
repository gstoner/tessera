"""NVIDIA and AMD system-context helpers for profiler correlation.

The SiliconScope lesson generalizes cleanly: keep high-frequency kernel traces
separate from lower-frequency system context, then correlate them in reports.
For NVIDIA the system-context provider is NVML/DCGM; for AMD it is AMD SMI/RDC.
This module contains only the hardware-free value model and classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ACCELERATOR_PROFILER_CONTEXT_SCHEMA_VERSION = "tessera.accelerator_profiler_context.v1"

IDLE = "idle"
GPU_ACTIVE = "gpu_active"
BANDWIDTH_BOUND = "bandwidth_bound"
COMPUTE_BOUND = "compute_bound"
THERMAL_THROTTLED = "thermal_throttled"
POWER_CAPPED = "power_capped"
MEMORY_PRESSURED = "memory_pressured"
FABRIC_LIMITED = "fabric_limited"
RELIABILITY_RISK = "reliability_risk"

ACCELERATOR_BOTTLENECKS: frozenset[str] = frozenset({
    IDLE,
    GPU_ACTIVE,
    BANDWIDTH_BOUND,
    COMPUTE_BOUND,
    THERMAL_THROTTLED,
    POWER_CAPPED,
    MEMORY_PRESSURED,
    FABRIC_LIMITED,
    RELIABILITY_RISK,
})

Vendor = Literal["nvidia", "rocm"]


@dataclass(frozen=True)
class AcceleratorProfilerContext:
    """Normalized system-context sample for NVIDIA or AMD GPU runs."""

    vendor: Vendor
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    memory_used_fraction: float = 0.0
    memory_bandwidth_fraction: float | None = None
    pcie_bandwidth_fraction: float | None = None
    fabric_bandwidth_fraction: float | None = None
    power_watts: float | None = None
    power_limit_watts: float | None = None
    temperature_c: float | None = None
    temperature_limit_c: float | None = None
    throttle_active: bool = False
    correctable_ecc_errors: int = 0
    uncorrectable_ecc_errors: int = 0
    xgmi_or_nvlink_replay_errors: int = 0

    def __post_init__(self) -> None:
        if self.vendor not in {"nvidia", "rocm"}:
            raise ValueError("vendor must be nvidia or rocm")
        _check_fraction("gpu_utilization", self.gpu_utilization)
        _check_fraction("memory_utilization", self.memory_utilization)
        _check_fraction("memory_used_fraction", self.memory_used_fraction)
        for name in ("memory_bandwidth_fraction", "pcie_bandwidth_fraction", "fabric_bandwidth_fraction"):
            value = getattr(self, name)
            if value is not None:
                _check_fraction(name, value)
        for name in ("power_watts", "power_limit_watts", "temperature_c", "temperature_limit_c"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.correctable_ecc_errors < 0 or self.uncorrectable_ecc_errors < 0:
            raise ValueError("ECC error counts must be non-negative")
        if self.xgmi_or_nvlink_replay_errors < 0:
            raise ValueError("fabric replay errors must be non-negative")

    def classify(self) -> str:
        return classify_accelerator_profiler_context(
            gpu_utilization=self.gpu_utilization,
            memory_utilization=self.memory_utilization,
            memory_used_fraction=self.memory_used_fraction,
            memory_bandwidth_fraction=self.memory_bandwidth_fraction,
            pcie_bandwidth_fraction=self.pcie_bandwidth_fraction,
            fabric_bandwidth_fraction=self.fabric_bandwidth_fraction,
            power_watts=self.power_watts,
            power_limit_watts=self.power_limit_watts,
            temperature_c=self.temperature_c,
            temperature_limit_c=self.temperature_limit_c,
            throttle_active=self.throttle_active,
            correctable_ecc_errors=self.correctable_ecc_errors,
            uncorrectable_ecc_errors=self.uncorrectable_ecc_errors,
            xgmi_or_nvlink_replay_errors=self.xgmi_or_nvlink_replay_errors,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ACCELERATOR_PROFILER_CONTEXT_SCHEMA_VERSION,
            "vendor": self.vendor,
            "gpu_utilization": self.gpu_utilization,
            "memory_utilization": self.memory_utilization,
            "memory_used_fraction": self.memory_used_fraction,
            "memory_bandwidth_fraction": self.memory_bandwidth_fraction,
            "pcie_bandwidth_fraction": self.pcie_bandwidth_fraction,
            "fabric_bandwidth_fraction": self.fabric_bandwidth_fraction,
            "power_watts": self.power_watts,
            "power_limit_watts": self.power_limit_watts,
            "temperature_c": self.temperature_c,
            "temperature_limit_c": self.temperature_limit_c,
            "throttle_active": self.throttle_active,
            "correctable_ecc_errors": self.correctable_ecc_errors,
            "uncorrectable_ecc_errors": self.uncorrectable_ecc_errors,
            "xgmi_or_nvlink_replay_errors": self.xgmi_or_nvlink_replay_errors,
            "bottleneck": self.classify(),
        }


def classify_accelerator_profiler_context(
    *,
    gpu_utilization: float,
    memory_utilization: float,
    memory_used_fraction: float,
    memory_bandwidth_fraction: float | None = None,
    pcie_bandwidth_fraction: float | None = None,
    fabric_bandwidth_fraction: float | None = None,
    power_watts: float | None = None,
    power_limit_watts: float | None = None,
    temperature_c: float | None = None,
    temperature_limit_c: float | None = None,
    throttle_active: bool = False,
    correctable_ecc_errors: int = 0,
    uncorrectable_ecc_errors: int = 0,
    xgmi_or_nvlink_replay_errors: int = 0,
) -> str:
    """Classify the dominant accelerator-system bottleneck from normalized inputs."""

    _check_fraction("gpu_utilization", gpu_utilization)
    _check_fraction("memory_utilization", memory_utilization)
    _check_fraction("memory_used_fraction", memory_used_fraction)
    for name, value in (
        ("memory_bandwidth_fraction", memory_bandwidth_fraction),
        ("pcie_bandwidth_fraction", pcie_bandwidth_fraction),
        ("fabric_bandwidth_fraction", fabric_bandwidth_fraction),
    ):
        if value is not None:
            _check_fraction(name, value)
    if uncorrectable_ecc_errors > 0 or xgmi_or_nvlink_replay_errors > 0:
        return RELIABILITY_RISK
    if memory_used_fraction >= 0.95:
        return MEMORY_PRESSURED
    temp_fraction = (
        temperature_c / temperature_limit_c
        if temperature_c is not None and temperature_limit_c and temperature_limit_c > 0
        else 0.0
    )
    if throttle_active or (temp_fraction >= 0.95 and gpu_utilization >= 0.30):
        return THERMAL_THROTTLED
    power_fraction = (
        power_watts / power_limit_watts
        if power_watts is not None and power_limit_watts and power_limit_watts > 0
        else 0.0
    )
    if power_fraction >= 0.95 and gpu_utilization >= 0.30:
        return POWER_CAPPED
    if fabric_bandwidth_fraction is not None and fabric_bandwidth_fraction >= 0.85:
        return FABRIC_LIMITED
    if pcie_bandwidth_fraction is not None and pcie_bandwidth_fraction >= 0.85:
        return FABRIC_LIMITED
    if gpu_utilization < 0.30:
        return IDLE
    effective_memory_fraction = (
        memory_bandwidth_fraction
        if memory_bandwidth_fraction is not None
        else memory_utilization
    )
    if effective_memory_fraction >= 0.85:
        return BANDWIDTH_BOUND
    if gpu_utilization >= 0.90:
        return COMPUTE_BOUND
    return GPU_ACTIVE


def accelerator_profiler_context_contract(vendor: Vendor) -> dict[str, Any]:
    if vendor == "nvidia":
        return {
            "schema": ACCELERATOR_PROFILER_CONTEXT_SCHEMA_VERSION,
            "provider": "nvidia-system-context",
            "status": "planned",
            "collector": "NVML/DCGM native helper",
            "signals": [
                "gpu_utilization",
                "memory_utilization",
                "memory_used_fraction",
                "power_watts",
                "power_limit_watts",
                "temperature_c",
                "temperature_limit_c",
                "throttle_reasons",
                "pcie_bandwidth_fraction",
                "nvlink_bandwidth_fraction",
                "ecc_errors",
            ],
            "trace_provider": "CUPTI remains authoritative for runtime/activity/counter proof.",
        }
    if vendor == "rocm":
        return {
            "schema": ACCELERATOR_PROFILER_CONTEXT_SCHEMA_VERSION,
            "provider": "rocm-system-context",
            "status": "planned",
            "collector": "AMD SMI/RDC native helper",
            "signals": [
                "gpu_utilization",
                "memory_utilization",
                "memory_used_fraction",
                "power_watts",
                "power_limit_watts",
                "temperature_c",
                "temperature_limit_c",
                "throttle_status",
                "pcie_bandwidth_fraction",
                "xgmi_bandwidth_fraction",
                "ras_errors",
            ],
            "trace_provider": "ROCprofiler-SDK remains authoritative for runtime/activity/counter proof.",
        }
    raise ValueError("vendor must be nvidia or rocm")


def _check_fraction(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


__all__ = [
    "ACCELERATOR_BOTTLENECKS",
    "ACCELERATOR_PROFILER_CONTEXT_SCHEMA_VERSION",
    "AcceleratorProfilerContext",
    "BANDWIDTH_BOUND",
    "COMPUTE_BOUND",
    "FABRIC_LIMITED",
    "GPU_ACTIVE",
    "IDLE",
    "MEMORY_PRESSURED",
    "POWER_CAPPED",
    "RELIABILITY_RISK",
    "THERMAL_THROTTLED",
    "accelerator_profiler_context_contract",
    "classify_accelerator_profiler_context",
]
