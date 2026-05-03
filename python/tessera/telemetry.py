"""Shared telemetry event model for Tessera tools.

The schema here is intentionally small and JSON-native so profiler traces,
autotune logs, benchmark reports, and runtime adapters can share one shape
without coupling to a specific backend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


TELEMETRY_SCHEMA_VERSION = "tessera.telemetry.v1"


@dataclass(frozen=True)
class TelemetryEvent:
    """Portable metric event used across profiler, autotune, and benchmarks."""

    name: str
    source: str
    op: str | None = None
    shape: Sequence[int] | Mapping[str, int] | None = None
    dtype: str | None = None
    arch: str | None = None
    graph_hash: str | None = None
    schedule_hash: str | None = None
    kernel_id: str | None = None
    device: str | int | None = None
    stream: str | int | None = None
    rank: int | None = None
    latency_ms: float | None = None
    tflops: float | None = None
    bandwidth_gbps: float | None = None
    memory_bytes: float | None = None
    status: str = "ok"
    counters: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": TELEMETRY_SCHEMA_VERSION,
            "name": self.name,
            "source": self.source,
            "status": self.status,
            "timestamp": self.timestamp,
            "counters": dict(self.counters),
            "metadata": dict(self.metadata),
        }
        optional = {
            "op": self.op,
            "shape": _normalize_shape(self.shape),
            "dtype": self.dtype,
            "arch": self.arch,
            "graph_hash": self.graph_hash,
            "schedule_hash": self.schedule_hash,
            "kernel_id": self.kernel_id,
            "device": self.device,
            "stream": self.stream,
            "rank": self.rank,
            "latency_ms": self.latency_ms,
            "tflops": self.tflops,
            "bandwidth_gbps": self.bandwidth_gbps,
            "memory_bytes": self.memory_bytes,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        payload["bottleneck"] = classify_bottleneck(payload)
        return payload


TesseraMetricEvent = TelemetryEvent
TesseraTraceEvent = TelemetryEvent


def make_event(
    name: str,
    *,
    source: str,
    op: str | None = None,
    shape: Sequence[int] | Mapping[str, int] | None = None,
    dtype: str | None = None,
    arch: str | None = None,
    graph_hash: str | None = None,
    schedule_hash: str | None = None,
    kernel_id: str | None = None,
    device: str | int | None = None,
    stream: str | int | None = None,
    rank: int | None = None,
    latency_ms: float | None = None,
    tflops: float | None = None,
    bandwidth_gbps: float | None = None,
    memory_bytes: float | None = None,
    status: str = "ok",
    counters: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
    timestamp: float | None = None,
) -> dict[str, Any]:
    return TelemetryEvent(
        name=name,
        source=source,
        op=op,
        shape=shape,
        dtype=dtype,
        arch=arch,
        graph_hash=graph_hash,
        schedule_hash=schedule_hash,
        kernel_id=kernel_id,
        device=device,
        stream=stream,
        rank=rank,
        latency_ms=latency_ms,
        tflops=tflops,
        bandwidth_gbps=bandwidth_gbps,
        memory_bytes=memory_bytes,
        status=status,
        counters=dict(counters or {}),
        metadata=dict(metadata or {}),
        timestamp=time.time() if timestamp is None else timestamp,
    ).to_dict()


def classify_bottleneck(event: Mapping[str, Any]) -> str:
    """Return a user-facing bottleneck label from available metrics."""

    status = str(event.get("status", "ok"))
    if status not in ("ok", "executable", "cached"):
        return "failed_or_unmeasured"
    if event.get("op") in ("all_reduce", "all_gather", "reduce_scatter", "broadcast", "collective"):
        return "collective_or_overlap"
    metadata = event.get("metadata") if isinstance(event.get("metadata"), Mapping) else {}
    bound = event.get("roofline_bound") or event.get("bound") or metadata.get("roofline_bound")
    if bound == "compute":
        return "compute_bound"
    if bound == "memory":
        return "memory_bound"
    latency_ms = float(event.get("latency_ms") or 0.0)
    tflops = float(event.get("tflops") or 0.0)
    bandwidth = float(event.get("bandwidth_gbps") or 0.0)
    if latency_ms > 0 and tflops == 0.0 and bandwidth == 0.0:
        return "launch_or_runtime_overhead"
    return "unknown"


def telemetry_report(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create a compact summary over telemetry events."""

    counts: dict[str, int] = {}
    for event in events:
        label = str(event.get("bottleneck") or classify_bottleneck(event))
        counts[label] = counts.get(label, 0) + 1
    return {
        "schema": TELEMETRY_SCHEMA_VERSION,
        "event_count": len(events),
        "bottlenecks": counts,
    }


def _normalize_shape(shape: Sequence[int] | Mapping[str, int] | None) -> list[int] | dict[str, int] | None:
    if shape is None:
        return None
    if isinstance(shape, Mapping):
        return {str(key): int(value) for key, value in shape.items()}
    return [int(value) for value in shape]


__all__ = [
    "TELEMETRY_SCHEMA_VERSION",
    "TelemetryEvent",
    "TesseraMetricEvent",
    "TesseraTraceEvent",
    "classify_bottleneck",
    "make_event",
    "telemetry_report",
]
