"""Best-effort profiler context collectors.

The collectors in this module are intentionally optional.  They either consume
mock/file samples for deterministic CI, or dynamically discover native
management libraries at runtime and return an honest unavailable artifact when a
library, device, or permission is missing.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Literal, Mapping, overload

from .accelerator_profiler_context import AcceleratorProfilerContext
from .apple_profiler_context import AppleProfilerContext, apple_unified_memory_bandwidth_ceiling_gbs
from .profiler_context import (
    PROFILER_CONTEXT_SCHEMA_VERSION,
    build_profiler_context_artifact,
    validate_profiler_context_artifact,
)


PROFILER_COLLECTOR_STATUS_MOCK = "mock"
PROFILER_COLLECTOR_STATUS_FILE = "file"
PROFILER_COLLECTOR_STATUS_MEASURED = "measured"
PROFILER_COLLECTOR_STATUS_UNAVAILABLE = "unavailable"
PROFILER_COLLECTOR_STATUS_HOST_METADATA = "host_metadata_only"


class NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def collect_profiler_context(
    provider: str,
    *,
    target: str | None = None,
    input_path: str | Path | None = None,
    device_index: int = 0,
) -> dict[str, Any]:
    """Collect or load a ``tessera.profiler_context.v1`` artifact."""

    normalized = provider.strip().lower().replace("-", "_")
    if input_path is not None:
        return load_context_samples_file(input_path, target=target or _target_for_provider(normalized))
    if normalized == "mock":
        return mock_profiler_context(target=target or "nvidia")
    if normalized == "nvidia":
        return sample_nvidia_nvml_context(device_index=device_index)
    if normalized == "rocm":
        return sample_rocm_amdsmi_context(device_index=device_index)
    if normalized == "apple":
        return sample_apple_system_context()
    raise ValueError("provider must be one of mock, nvidia, rocm, apple")


def load_context_samples_file(path: str | Path, *, target: str) -> dict[str, Any]:
    """Load either a full context artifact or raw sample/list JSON."""

    payload = json.loads(Path(path).read_text())
    if isinstance(payload, Mapping) and payload.get("schema") == PROFILER_CONTEXT_SCHEMA_VERSION:
        validate_profiler_context_artifact(payload)
        return dict(payload)
    samples = payload if isinstance(payload, list) else [payload]
    return build_profiler_context_artifact(
        target=target,
        samples=samples,
        source_status=PROFILER_COLLECTOR_STATUS_FILE,
        source=str(path),
    )


def mock_profiler_context(*, target: str = "nvidia") -> dict[str, Any]:
    normalized = target.strip().lower().replace("-", "_")
    if normalized in {"apple", "apple_gpu", "metal", "mps"}:
        sample = AppleProfilerContext(
            gpu_usage=0.62,
            total_bandwidth_gbs=340.0,
            achievable_bandwidth_gbs=400.0,
            gpu_frequency_mhz=1200.0,
            gpu_power_watts=14.0,
            dram_power_watts=5.0,
        ).to_dict()
        target_name = "apple_gpu"
    else:
        vendor: Literal["nvidia", "rocm"] = (
            "rocm"
            if normalized in {"rocm", "amd", "hip"} or normalized.startswith("gfx")
            else "nvidia"
        )
        sample = AcceleratorProfilerContext(
            vendor=vendor,
            gpu_utilization=0.72,
            memory_utilization=0.88,
            memory_used_fraction=0.40,
            memory_bandwidth_fraction=0.87,
            power_watts=260.0,
            power_limit_watts=400.0,
            temperature_c=62.0,
            temperature_limit_c=95.0,
        ).to_dict()
        target_name = vendor
    sample["metadata"] = {"collector": "mock"}
    return build_profiler_context_artifact(
        target=target_name,
        samples=(sample,),
        source_status=PROFILER_COLLECTOR_STATUS_MOCK,
    )


def sample_apple_system_context() -> dict[str, Any]:
    """Return host-safe Apple metadata without private IOReport/SMC/HID calls."""

    chip = _apple_chip_name()
    is_darwin = platform.system() == "Darwin"
    bandwidth = apple_unified_memory_bandwidth_ceiling_gbs(chip) if chip else 0.0
    sample = AppleProfilerContext(
        gpu_usage=0.0,
        total_bandwidth_gbs=0.0,
        achievable_bandwidth_gbs=bandwidth,
    ).to_dict()
    sample["metadata"] = {
        "collector": "apple-host-metadata",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip_name": chip,
        "native_helper": "IOReport/SMC/HID helper is compile-gated and not invoked here.",
    }
    return build_profiler_context_artifact(
        target="apple_gpu",
        samples=(sample,),
        provider="apple-silicon-system-context",
        source_status=(
            PROFILER_COLLECTOR_STATUS_HOST_METADATA
            if is_darwin
            else PROFILER_COLLECTOR_STATUS_UNAVAILABLE
        ),
    )


def sample_nvidia_nvml_context(*, device_index: int = 0, lib: Any | None = None) -> dict[str, Any]:
    try:
        library = lib if lib is not None else _load_nvml()
        sample = _sample_nvidia_nvml_from_library(library, device_index=device_index)
        status = PROFILER_COLLECTOR_STATUS_MEASURED
    except Exception as exc:
        sample = _unavailable_accelerator_sample("nvidia", "nvml", exc)
        status = PROFILER_COLLECTOR_STATUS_UNAVAILABLE
    return build_profiler_context_artifact(
        target="nvidia",
        samples=(sample,),
        provider="nvidia-system-context",
        source_status=status,
    )


def sample_rocm_amdsmi_context(*, device_index: int = 0, module: Any | None = None) -> dict[str, Any]:
    try:
        amdsmi = module if module is not None else importlib.import_module("amdsmi")
        sample = _sample_rocm_amdsmi_from_module(amdsmi, device_index=device_index)
        status = PROFILER_COLLECTOR_STATUS_MEASURED
    except Exception as exc:
        sample = _unavailable_accelerator_sample("rocm", "amdsmi", exc)
        status = PROFILER_COLLECTOR_STATUS_UNAVAILABLE
    return build_profiler_context_artifact(
        target="rocm",
        samples=(sample,),
        provider="rocm-system-context",
        source_status=status,
    )


def _sample_nvidia_nvml_from_library(lib: Any, *, device_index: int) -> dict[str, Any]:
    _nvml_call(lib, "nvmlInit_v2", fallback="nvmlInit")
    try:
        handle = ctypes.c_void_p()
        _nvml_call(
            lib,
            "nvmlDeviceGetHandleByIndex_v2",
            ctypes.c_uint(device_index),
            ctypes.byref(handle),
            fallback="nvmlDeviceGetHandleByIndex",
        )
        utilization = NvmlUtilization()
        memory = NvmlMemory()
        gpu_utilization = 0.0
        memory_utilization = 0.0
        memory_used_fraction = 0.0
        if _nvml_try(lib, "nvmlDeviceGetUtilizationRates", handle, ctypes.byref(utilization)):
            gpu_utilization = _pct(utilization.gpu)
            memory_utilization = _pct(utilization.memory)
        if _nvml_try(lib, "nvmlDeviceGetMemoryInfo", handle, ctypes.byref(memory)) and memory.total:
            memory_used_fraction = memory.used / memory.total
        power_watts = _nvml_uint(lib, "nvmlDeviceGetPowerUsage", handle, scale=1000.0)
        power_limit_watts = _nvml_uint(lib, "nvmlDeviceGetEnforcedPowerLimit", handle, scale=1000.0)
        temperature_c = _nvml_temperature(lib, handle)
        throttle_reasons = _nvml_ulonglong(lib, "nvmlDeviceGetCurrentClocksThrottleReasons", handle)
        correctable = _nvml_ecc_total(lib, handle, error_type=0)
        uncorrectable = _nvml_ecc_total(lib, handle, error_type=1)
        sample = AcceleratorProfilerContext(
            vendor="nvidia",
            gpu_utilization=gpu_utilization,
            memory_utilization=memory_utilization,
            memory_used_fraction=memory_used_fraction,
            power_watts=power_watts,
            power_limit_watts=power_limit_watts,
            temperature_c=temperature_c,
            throttle_active=bool(throttle_reasons),
            correctable_ecc_errors=correctable,
            uncorrectable_ecc_errors=uncorrectable,
        ).to_dict()
        sample["metadata"] = {
            "collector": "nvml",
            "device_index": device_index,
            "throttle_reasons": throttle_reasons,
        }
        return sample
    finally:
        _nvml_try(lib, "nvmlShutdown")


def _sample_rocm_amdsmi_from_module(amdsmi: Any, *, device_index: int) -> dict[str, Any]:
    _call_if_present(amdsmi, "amdsmi_init")
    try:
        handles = _call_first(amdsmi, ("amdsmi_get_processor_handles", "amdsmi_get_gpu_handles"))
        if handles is None:
            raise RuntimeError("AMD SMI did not return processor handles")
        handle = list(handles)[device_index]
        activity = _as_mapping(_call_first(amdsmi, ("amdsmi_get_gpu_activity", "amdsmi_get_gpu_utilization"), handle))
        vram = _as_mapping(_call_first(amdsmi, ("amdsmi_get_gpu_vram_usage", "amdsmi_get_gpu_memory_usage"), handle))
        power = _as_mapping(_call_first(amdsmi, ("amdsmi_get_power_info", "amdsmi_get_gpu_power_info"), handle))
        temp = _as_mapping(_call_first(amdsmi, ("amdsmi_get_temp_metric", "amdsmi_get_gpu_temperature"), handle))
        ras = _as_mapping(_call_first(amdsmi, ("amdsmi_get_gpu_total_ecc_count", "amdsmi_get_gpu_ras_error_count"), handle))
        used = _num(vram, "vram_used", "used", "used_vram", default=0.0)
        total = _num(vram, "vram_total", "total", "total_vram", default=0.0)
        sample = AcceleratorProfilerContext(
            vendor="rocm",
            gpu_utilization=_pct(_num(activity, "gfx_activity", "gpu_activity", "gpu_utilization", default=0.0)),
            memory_utilization=_pct(_num(activity, "umc_activity", "mem_activity", "memory_utilization", default=0.0)),
            memory_used_fraction=(used / total if total > 0 else 0.0),
            power_watts=_watts(_num(power, "socket_power", "average_socket_power", "power", default=None)),
            power_limit_watts=_watts(_num(power, "power_limit", "cap", "power_cap", default=None)),
            temperature_c=_temperature_c(_num(temp, "temperature", "edge", "junction", default=None)),
            correctable_ecc_errors=int(_num(ras, "correctable_count", "correctable", "ce_count", default=0)),
            uncorrectable_ecc_errors=int(_num(ras, "uncorrectable_count", "uncorrectable", "ue_count", default=0)),
            xgmi_or_nvlink_replay_errors=int(_num(ras, "xgmi_replay_count", "replay_count", default=0)),
        ).to_dict()
        sample["metadata"] = {"collector": "amdsmi", "device_index": device_index}
        return sample
    finally:
        _call_if_present(amdsmi, "amdsmi_shut_down", "amdsmi_shutdown")


def _load_nvml() -> Any:
    candidates = [
        ctypes.util.find_library("nvidia-ml"),
        "libnvidia-ml.so.1",
        "libnvidia-ml.so",
        "nvml.dll",
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise RuntimeError("NVML library not found")


def _nvml_call(lib: Any, name: str, *args: Any, fallback: str | None = None) -> int:
    fn = getattr(lib, name, None) or (getattr(lib, fallback, None) if fallback else None)
    if fn is None:
        raise RuntimeError(f"NVML symbol {name} not found")
    result = int(fn(*args))
    if result != 0:
        raise RuntimeError(f"NVML call {name} failed with status {result}")
    return result


def _nvml_try(lib: Any, name: str, *args: Any) -> bool:
    fn = getattr(lib, name, None)
    if fn is None:
        return False
    try:
        return int(fn(*args)) == 0
    except Exception:
        return False


def _nvml_uint(lib: Any, name: str, handle: ctypes.c_void_p, *, scale: float = 1.0) -> float | None:
    value = ctypes.c_uint()
    if _nvml_try(lib, name, handle, ctypes.byref(value)):
        return value.value / scale
    return None


def _nvml_ulonglong(lib: Any, name: str, handle: ctypes.c_void_p) -> int:
    value = ctypes.c_ulonglong()
    if _nvml_try(lib, name, handle, ctypes.byref(value)):
        return int(value.value)
    return 0


def _nvml_temperature(lib: Any, handle: ctypes.c_void_p) -> float | None:
    value = ctypes.c_uint()
    if _nvml_try(lib, "nvmlDeviceGetTemperature", handle, ctypes.c_uint(0), ctypes.byref(value)):
        return float(value.value)
    return None


def _nvml_ecc_total(lib: Any, handle: ctypes.c_void_p, *, error_type: int) -> int:
    value = ctypes.c_ulonglong()
    if _nvml_try(
        lib,
        "nvmlDeviceGetTotalEccErrors",
        handle,
        ctypes.c_uint(error_type),
        ctypes.c_uint(0),
        ctypes.byref(value),
    ):
        return int(value.value)
    return 0


def _call_if_present(obj: Any, *names: str) -> Any:
    for name in names:
        fn = getattr(obj, name, None)
        if fn is not None:
            return fn()
    return None


def _call_first(obj: Any, names: tuple[str, ...], *args: Any) -> Any:
    for name in names:
        fn = getattr(obj, name, None)
        if fn is None:
            continue
        try:
            return fn(*args)
        except TypeError:
            try:
                return fn()
            except TypeError:
                continue
    return None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "_asdict"):
        return value._asdict()
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


@overload
def _num(mapping: Mapping[str, Any], *keys: str, default: float) -> float: ...
@overload
def _num(mapping: Mapping[str, Any], *keys: str, default: None) -> float | None: ...
def _num(mapping: Mapping[str, Any], *keys: str, default: float | None) -> float | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            try:
                return float(mapping[key])
            except (TypeError, ValueError):
                return default
    return default


def _pct(value: float | int | None) -> float:
    if value is None:
        return 0.0
    value_f = float(value)
    if value_f > 1.0:
        value_f /= 100.0
    return min(1.0, max(0.0, value_f))


def _watts(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 1_000_000.0 if value > 10_000 else value


def _temperature_c(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 1000.0 if value > 1000 else value


def _unavailable_accelerator_sample(
    vendor: Literal["nvidia", "rocm"], collector: str, exc: Exception
) -> dict[str, Any]:
    sample = AcceleratorProfilerContext(vendor=vendor).to_dict()
    sample["metadata"] = {
        "collector": collector,
        "error": str(exc),
    }
    return sample


def _target_for_provider(provider: str) -> str:
    return {
        "apple": "apple_gpu",
        "nvidia": "nvidia",
        "rocm": "rocm",
        "mock": "nvidia",
    }.get(provider, provider)


def _apple_chip_name() -> str:
    if platform.system() != "Darwin":
        return ""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        return result.stdout.strip()
    except Exception:
        return ""


__all__ = [
    "PROFILER_COLLECTOR_STATUS_FILE",
    "PROFILER_COLLECTOR_STATUS_HOST_METADATA",
    "PROFILER_COLLECTOR_STATUS_MEASURED",
    "PROFILER_COLLECTOR_STATUS_MOCK",
    "PROFILER_COLLECTOR_STATUS_UNAVAILABLE",
    "collect_profiler_context",
    "load_context_samples_file",
    "mock_profiler_context",
    "sample_apple_system_context",
    "sample_nvidia_nvml_context",
    "sample_rocm_amdsmi_context",
]
