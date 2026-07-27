"""APPLE-COUNTER-1 — Apple's answer to the shared autotune evidence schema.

Cross-backend sync ``NVIDIA-TEST5-2026-07-16`` gave the shared autotune corpus
additive compiler/resource, cold/warm, cache, and two-run stability fields, and
recorded one Apple follow-up: populate the *same logical fields* from
Metal-native counters rather than leaving them blank or borrowing CUDA values.

This module is that mapping. Its whole design point is the distinction between
three different states a field can be in:

``measured``
    Metal genuinely reports it on this device — command-buffer kernel time,
    dispatch-boundary timestamp deltas, threads-per-threadgroup, execution
    width, maximum threads, static threadgroup memory, pipeline-cache size.
``unsupported_by_device``
    The API exists but this GPU family does not support it — for example
    dispatch-boundary counter sampling on a device whose counter sets omit
    ``MTLCommonCounterSetTimestamp``. Recording the limitation *is* evidence.
``not_measured``
    The device supports it and this run simply did not capture it — telemetry
    was disabled, or the caller did not request the sample. This is the only
    one of the four a later run can turn into ``measured``.
``no_public_api``
    Metal exposes no query at all. Register count, scratch bytes, spill count,
    and true occupancy are in this class. CUDA gets them from ``ptxas``; Metal
    has no public equivalent, so these must stay absent with a stated reason.

A missing field and a field that is absent *for a known reason* are not the
same evidence, and a selector must never treat the second as zero. That is why
nothing here defaults to 0 and why :func:`apple_autotune_evidence` refuses to
emit a value the capability bits do not support.

The capability bits mirror ``tessera_apple_gpu_profiling_capabilities`` in
``apple_gpu_runtime.mm``; the drift gate in
``tests/unit/test_apple_counter_evidence.py`` keeps the two in step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

# Bit positions of `tessera_apple_gpu_profiling_capabilities()`. Keep in sync
# with the comment block above that function in apple_gpu_runtime.mm.
CAP_PIPELINE_LIMITS = 1 << 0
CAP_TIMESTAMP_COUNTER_SET = 1 << 1
CAP_STATISTIC_COUNTER_SET = 1 << 2
CAP_STAGE_UTILIZATION_COUNTER_SET = 1 << 3
CAP_DISPATCH_BOUNDARY_SAMPLING = 1 << 4
CAP_STAGE_BOUNDARY_SAMPLING = 1 << 5
CAP_METAL4_TIMESTAMP_HEAP = 1 << 6


class AppleEvidenceState(str, Enum):
    """Why a shared-schema field holds the value it holds."""

    MEASURED = "measured"
    NOT_MEASURED = "not_measured"
    UNSUPPORTED_BY_DEVICE = "unsupported_by_device"
    NO_PUBLIC_API = "no_public_api"


# Fields the shared schema carries that public Metal cannot answer at all.
# CUDA sources these from ptxas; Metal has no equivalent query, so they are
# permanently absent rather than pending work.
NO_PUBLIC_METAL_QUERY = (
    "register_count",
    "scratch_bytes",
    "spill_count",
    "achieved_occupancy",
)


@dataclass(frozen=True)
class AppleEvidenceField:
    """One shared-schema field plus the reason it is present or absent."""

    value: Any
    state: AppleEvidenceState

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "state": self.state.value}


@dataclass(frozen=True)
class AppleProfilingCapabilities:
    """Decoded profiling-capability bits for one live Apple device."""

    raw: int

    @property
    def pipeline_limits(self) -> bool:
        return bool(self.raw & CAP_PIPELINE_LIMITS)

    @property
    def timestamp_counter_set(self) -> bool:
        return bool(self.raw & CAP_TIMESTAMP_COUNTER_SET)

    @property
    def dispatch_boundary_sampling(self) -> bool:
        return bool(self.raw & CAP_DISPATCH_BOUNDARY_SAMPLING)

    @property
    def stage_boundary_sampling(self) -> bool:
        return bool(self.raw & CAP_STAGE_BOUNDARY_SAMPLING)

    @property
    def metal4_timestamp_heap(self) -> bool:
        return bool(self.raw & CAP_METAL4_TIMESTAMP_HEAP)

    @property
    def counter_deltas_available(self) -> bool:
        """A timestamp delta needs both the counter set and a sampling point."""
        return self.timestamp_counter_set and self.dispatch_boundary_sampling

    def as_dict(self) -> dict[str, bool | int]:
        return {
            "raw": self.raw,
            "pipeline_limits": self.pipeline_limits,
            "timestamp_counter_set": self.timestamp_counter_set,
            "dispatch_boundary_sampling": self.dispatch_boundary_sampling,
            "stage_boundary_sampling": self.stage_boundary_sampling,
            "metal4_timestamp_heap": self.metal4_timestamp_heap,
        }


def probe_apple_profiling_capabilities() -> AppleProfilingCapabilities | None:
    """Read the live capability matrix, or ``None`` off a Metal host.

    ``None`` means "not probed", which is distinct from a probed zero: a zero
    raw value is a real device that reports no profiling capability at all.
    """
    import ctypes

    try:
        from tessera._apple_gpu_dispatch import apple_gpu_runtime

        runtime = apple_gpu_runtime()
    except Exception:
        return None
    probe = getattr(runtime, "tessera_apple_gpu_profiling_capabilities", None)
    if probe is None:
        return None
    probe.restype = ctypes.c_int32
    try:
        return AppleProfilingCapabilities(int(probe()))
    except OSError:
        return None


def apple_autotune_evidence(
    *,
    capabilities: AppleProfilingCapabilities,
    device_time_ns: int | None = None,
    counter_timestamp_delta: int | None = None,
    threadgroup_memory_bytes: int | None = None,
    execution_width: int | None = None,
    max_threads_per_threadgroup: int | None = None,
    pipeline_cache_size: int | None = None,
    cold_compile: bool | None = None,
    source_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Map measured Metal telemetry onto the shared autotune evidence fields.

    Every returned field carries its :class:`AppleEvidenceState`, so a corpus
    reader can tell a measured zero from an unavailable field. Passing a value
    the capability bits do not support is an error rather than a silent
    downgrade: a corpus that claims a counter delta on a device without counter
    sampling is corrupt evidence, not optimistic evidence.
    """
    if counter_timestamp_delta is not None and not capabilities.counter_deltas_available:
        raise ValueError(
            "APPLE_COUNTER_EVIDENCE_UNSUPPORTED: a counter timestamp delta was "
            "supplied for a device without timestamp counter sampling")
    if (execution_width is not None or max_threads_per_threadgroup is not None
            or threadgroup_memory_bytes is not None) and not capabilities.pipeline_limits:
        raise ValueError(
            "APPLE_COUNTER_EVIDENCE_UNSUPPORTED: pipeline resource limits were "
            "supplied for a device that reports no compiled-pipeline limits")

    def measured(value: Any) -> AppleEvidenceField:
        return AppleEvidenceField(value, AppleEvidenceState.MEASURED)

    def device_gap(supported: bool, value: Any) -> AppleEvidenceField:
        """Separate "this device cannot" from "this run did not capture it".

        Collapsing the two would make a corpus unimprovable: a reader could not
        tell which absent fields a better-instrumented rerun would fill in.
        """
        if not supported:
            return AppleEvidenceField(
                None, AppleEvidenceState.UNSUPPORTED_BY_DEVICE)
        if value is None:
            return AppleEvidenceField(None, AppleEvidenceState.NOT_MEASURED)
        return measured(value)

    evidence: dict[str, AppleEvidenceField] = {
        # Timing domains. Kernel time comes from the completed command buffer's
        # kernelStart/kernelEnd interval, which needs no counter support.
        "device_time_ns": (measured(device_time_ns) if device_time_ns is not None
                           else AppleEvidenceField(
                               None, AppleEvidenceState.NOT_MEASURED)),
        "counter_timestamp_delta": device_gap(
            capabilities.counter_deltas_available, counter_timestamp_delta),
        # Resource fields Metal does expose, via the compiled pipeline state.
        "static_threadgroup_memory_bytes": device_gap(
            capabilities.pipeline_limits, threadgroup_memory_bytes),
        "execution_width": device_gap(
            capabilities.pipeline_limits, execution_width),
        "max_threads_per_threadgroup": device_gap(
            capabilities.pipeline_limits, max_threads_per_threadgroup),
        # Cache and cold/warm state — the shared schema's compile-state fields.
        "pipeline_cache_size": (
            measured(pipeline_cache_size) if pipeline_cache_size is not None
            else AppleEvidenceField(None, AppleEvidenceState.NOT_MEASURED)),
        "cold_compile": (measured(cold_compile) if cold_compile is not None
                         else AppleEvidenceField(
                             None, AppleEvidenceState.NOT_MEASURED)),
        # Compiler/source fingerprint: Apple's analogue of the CUDA toolchain
        # fingerprint is the exact MSL source the pipeline was built from.
        "source_fingerprint": (
            measured(source_sha256) if source_sha256 is not None
            else AppleEvidenceField(None, AppleEvidenceState.NOT_MEASURED)),
    }
    # The permanent gaps, stated once, with a reason a reader can act on.
    for field in NO_PUBLIC_METAL_QUERY:
        evidence[field] = AppleEvidenceField(None, AppleEvidenceState.NO_PUBLIC_API)

    return {name: field.as_dict() for name, field in evidence.items()}


def evidence_from_dispatch_record(
    record: Any,
    capabilities: AppleProfilingCapabilities,
    *,
    cold_compile: bool | None = None,
    resources: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the shared evidence fields from an ``AppleTileDispatchRecord``.

    The record is the Tile lane's existing provenance object; this adapts it to
    the cross-backend schema without teaching the record about NVIDIA.
    """
    resource_map = resources if resources is not None else getattr(record, "resources", {}) or {}
    threadgroup_bytes = resource_map.get("total_threadgroup_bytes")
    return apple_autotune_evidence(
        capabilities=capabilities,
        device_time_ns=getattr(record, "device_time_ns", None),
        counter_timestamp_delta=getattr(record, "counter_timestamp_delta", None),
        threadgroup_memory_bytes=threadgroup_bytes if capabilities.pipeline_limits else None,
        execution_width=(resource_map.get("simdgroup_lanes")
                         if capabilities.pipeline_limits else None),
        pipeline_cache_size=getattr(record, "msl_pipeline_cache_size", None),
        cold_compile=cold_compile,
        source_sha256=getattr(record, "source_sha256", None),
    )


__all__ = [
    "AppleEvidenceField",
    "AppleEvidenceState",
    "AppleProfilingCapabilities",
    "CAP_DISPATCH_BOUNDARY_SAMPLING",
    "CAP_METAL4_TIMESTAMP_HEAP",
    "CAP_PIPELINE_LIMITS",
    "CAP_STAGE_BOUNDARY_SAMPLING",
    "CAP_STAGE_UTILIZATION_COUNTER_SET",
    "CAP_STATISTIC_COUNTER_SET",
    "CAP_TIMESTAMP_COUNTER_SET",
    "NO_PUBLIC_METAL_QUERY",
    "apple_autotune_evidence",
    "evidence_from_dispatch_record",
    "probe_apple_profiling_capabilities",
]
