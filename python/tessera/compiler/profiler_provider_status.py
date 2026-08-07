"""Provider status helpers for Tessera profiler collectors."""

from __future__ import annotations

import ctypes.util
import importlib.util
import platform
from typing import Any, Literal, Mapping


PROFILER_PROVIDER_STATUS_SCHEMA_VERSION = "tessera.profiler_provider_status.v1"

ProviderStatus = Literal[
    "mock",
    "file",
    "host_metadata_only",
    "planned",
    "compiled_shell",
    "native_available",
    "native_failed",
    "unavailable",
]


def provider_status_artifact(
    *,
    provider: str,
    status: ProviderStatus,
    target: str | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": PROFILER_PROVIDER_STATUS_SCHEMA_VERSION,
        "provider": provider,
        "target": target or _target_for_provider(provider),
        "status": status,
        "diagnostics": dict(diagnostics or {}),
    }


def collect_provider_status(
    provider: str,
    *,
    native_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = provider.strip().lower().replace("-", "_")
    if normalized in {"apple", "metal", "apple_gpu"}:
        return _apple_status(native_proof=native_proof)
    if normalized in {"rocm", "amd", "hip", "rocprofiler"}:
        return _rocm_status(native_proof=native_proof)
    if normalized in {"nvidia", "cuda", "cupti"}:
        return _nvidia_status(native_proof=native_proof)
    if normalized in {"x86", "x86_64", "x86_avx512", "x86_perf"}:
        return _x86_status(native_proof=native_proof)
    if normalized == "cpu":
        return provider_status_artifact(
            provider="cpu",
            target="cpu",
            status="native_available",
            diagnostics={"runtime_trace": "tessera runtime callback spine"},
        )
    raise ValueError("provider must be one of apple, rocm, nvidia, x86, cpu")


def validate_provider_status_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != PROFILER_PROVIDER_STATUS_SCHEMA_VERSION:
        raise ValueError("unsupported provider status schema")
    if not isinstance(payload.get("provider"), str):
        raise ValueError("provider status requires provider")
    if payload.get("status") not in {
        "mock",
        "file",
        "host_metadata_only",
        "planned",
        "compiled_shell",
        "native_available",
        "native_failed",
        "unavailable",
    }:
        raise ValueError("unsupported provider status value")
    if not isinstance(payload.get("diagnostics", {}), Mapping):
        raise ValueError("provider status diagnostics must be a mapping")


def _apple_status(*, native_proof: Mapping[str, Any] | None = None) -> dict[str, Any]:
    is_darwin = platform.system() == "Darwin"
    proof = dict(native_proof or {})
    counter_proof = proof.get("counter_discovery")
    counter_discovery_available = isinstance(counter_proof, Mapping) and bool(
        counter_proof.get("counter_discovery_available")
    )
    command_buffer_proof = proof.get("command_buffer_timestamp")
    command_buffer_timestamp_available = isinstance(command_buffer_proof, Mapping) and bool(
        command_buffer_proof.get("timestamp_available")
    )
    proof_passed = bool(
        proof.get("metal_visible")
        and proof.get("fresh_process")
        and (counter_discovery_available or command_buffer_timestamp_available)
    )
    status: ProviderStatus
    if proof_passed:
        status = "native_available"
    else:
        status = "compiled_shell" if is_darwin else "planned"
    diagnostics: dict[str, Any] = {
        "platform": platform.platform(),
        "metal_framework": "compile-gated by TPROF_WITH_METAL",
        "native_proof_required": "fresh-process out-of-sandbox command-buffer/counter proof",
        "availability_rule": "apple remains compiled_shell until fresh-process Metal proof passes",
    }
    if proof:
        diagnostics["native_proof"] = proof
    return provider_status_artifact(
        provider="apple",
        target="apple_gpu",
        status=status,
        diagnostics=diagnostics,
    )


def _rocm_status(*, native_proof: Mapping[str, Any] | None = None) -> dict[str, Any]:
    has_amdsmi = importlib.util.find_spec("amdsmi") is not None
    proof = dict(native_proof or {})
    proof_passed = bool(
        proof.get("amd_gpu_visible")
        and proof.get("rocprofiler_sdk_visible")
        and proof.get("context_created")
        and proof.get("tool_registered")
        and (proof.get("hip_callback_seen") or proof.get("hsa_callback_seen"))
        and proof.get("dispatch_activity_seen")
    )
    status: ProviderStatus
    if proof_passed:
        status = "native_available"
    elif proof:
        status = "native_failed"
    else:
        status = "planned"
    diagnostics: dict[str, Any] = {
        "amdsmi_python": has_amdsmi,
        "rocprofiler_sdk": "compile-gated by TPROF_WITH_ROCPROFILER",
        "native_proof_required": "AMD GPU plus ROCprofiler-SDK HIP/HSA dispatch proof",
        "availability_rule": "ROCm remains planned/native_failed until ROCprofiler-SDK proof passes",
    }
    if proof:
        diagnostics["native_proof"] = proof
    return provider_status_artifact(
        provider="rocm",
        target="rocm",
        status=status,
        diagnostics=diagnostics,
    )


def _nvidia_status(*, native_proof: Mapping[str, Any] | None = None) -> dict[str, Any]:
    nvml = ctypes.util.find_library("nvidia-ml")
    cupti = ctypes.util.find_library("cupti")
    proof = dict(native_proof or {})
    proof_passed = bool(
        proof.get("nvidia_gpu_visible")
        and proof.get("cupti_visible")
        and proof.get("subscriber_created")
        and proof.get("callback_seen")
        and proof.get("activity_buffer_seen")
        and proof.get("activity_seen")
    )
    status: ProviderStatus
    if proof_passed:
        status = "native_available"
    elif proof:
        status = "native_failed"
    else:
        status = "planned"
    diagnostics: dict[str, Any] = {
        "nvml_library": nvml,
        "cupti_library": cupti,
        "cupti": "compile-gated by TPROF_WITH_CUPTI",
        "native_proof_required": "NVIDIA GPU plus CUPTI callback/activity proof",
        "availability_rule": "NVIDIA remains planned/native_failed until CUPTI callback/activity proof passes",
    }
    if proof:
        diagnostics["native_proof"] = proof
    return provider_status_artifact(
        provider="nvidia",
        target="nvidia",
        status=status,
        diagnostics=diagnostics,
    )


def _x86_status(*, native_proof: Mapping[str, Any] | None = None) -> dict[str, Any]:
    proof = dict(native_proof or {})
    proof_passed = bool(
        proof.get("x86_64")
        and proof.get("avx512_visible")
        and proof.get("monotonic_raw_valid")
        and proof.get("invariant_tsc")
        and proof.get("rdtscp_valid")
        and proof.get("affinity_stable")
        and proof.get("clock_agreement_valid")
        and proof.get("perf_event_open")
        and proof.get("timing_sample_valid")
        and proof.get("event_map_valid")
    )
    status: ProviderStatus
    if proof_passed:
        status = "native_available"
    elif proof:
        status = "native_failed"
    else:
        status = "planned"
    diagnostics: dict[str, Any] = {
        "platform": platform.platform(),
        "native_provider": "CLOCK_MONOTONIC_RAW + fenced RDTSCP + perf_event_open + exact event-map digest",
        "native_proof_required": (
            "exact x86/AVX-512 host, stable CPU affinity, clock agreement, readable perf task-clock sample, and event-map digest"
        ),
        "availability_rule": (
            "x86 remains planned/native_failed until the multi-clock timing artifact and perf-event proof pass"
        ),
    }
    if proof:
        diagnostics["native_proof"] = proof
    return provider_status_artifact(
        provider="x86",
        target="x86_avx512",
        status=status,
        diagnostics=diagnostics,
    )


def _target_for_provider(provider: str) -> str:
    normalized = provider.strip().lower().replace("-", "_")
    if normalized in {"apple", "metal", "apple_gpu"}:
        return "apple_gpu"
    if normalized in {"rocm", "amd", "hip", "rocprofiler"}:
        return "rocm"
    if normalized in {"nvidia", "cuda", "cupti"}:
        return "nvidia"
    if normalized in {"x86", "x86_64", "x86_avx512", "x86_perf"}:
        return "x86_avx512"
    return normalized


__all__ = [
    "PROFILER_PROVIDER_STATUS_SCHEMA_VERSION",
    "ProviderStatus",
    "collect_provider_status",
    "provider_status_artifact",
    "validate_provider_status_artifact",
]
