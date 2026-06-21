"""Provider trace normalization for ROCprofiler, CUPTI, and Metal.

This module is the handoff between vendor SDK callbacks/activity records and
Tessera's portable trace-event world.  It deliberately accepts plain mappings so
unit tests and recorded fixtures can exercise correlation before the process
loads ROCprofiler-SDK, CUPTI, or Metal framework code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from .profiler_provider_status import validate_provider_status_artifact


PROVIDER_TRACE_SCHEMA_VERSION = "tessera.profiler_provider_trace.v1"

Provider = Literal["rocprofiler", "cupti", "metal"]
TraceKind = Literal[
    "runtime_api",
    "device_activity",
    "counter",
    "thread_trace",
    "command_buffer",
    "intra_kernel",
]


@dataclass(frozen=True)
class ProviderTraceRecord:
    provider: Provider
    kind: TraceKind
    name: str
    ts_us: float
    duration_us: float = 0.0
    correlation_id: str | int | None = None
    thread_id: str | int = 0
    value: float | None = None
    args: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "kind": self.kind,
            "name": self.name,
            "ts_us": self.ts_us,
            "duration_us": self.duration_us,
            "correlation_id": self.correlation_id,
            "thread_id": self.thread_id,
            "value": self.value,
            "args": dict(self.args or {}),
        }

    def to_trace_event(self) -> dict[str, Any]:
        args = {
            "provider": self.provider,
            "kind": self.kind,
            **dict(self.args or {}),
        }
        if self.correlation_id is not None:
            args["correlation_id"] = self.correlation_id
        if self.kind == "counter":
            args["value"] = float(self.value if self.value is not None else 0.0)
            return {
                "name": self.name,
                "cat": "counters",
                "ph": "C",
                "ts": self.ts_us,
                "pid": 0,
                "tid": self.thread_id,
                "args": args,
            }
        cat = {
            "runtime_api": "runtime_api",
            "device_activity": "device_activity",
            "thread_trace": "intra_kernel",
            "command_buffer": "device_activity",
            "intra_kernel": "intra_kernel",
        }.get(self.kind, self.kind)
        return {
            "name": self.name,
            "cat": cat,
            "ph": "X",
            "ts": self.ts_us,
            "dur": max(0.0, self.duration_us),
            "pid": 0,
            "tid": self.thread_id,
            "args": args,
        }


def build_provider_trace_artifact(
    *,
    provider: Provider,
    records: Iterable[ProviderTraceRecord | Mapping[str, Any]],
    source_status: str = "normalized",
    source: str | None = None,
    provider_statuses: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    normalized = [_coerce_record(provider, record) for record in records]
    status_payloads = [dict(status) for status in provider_statuses]
    for status in status_payloads:
        validate_provider_status_artifact(status)
    trace_events = [record.to_trace_event() for record in normalized]
    return {
        "schema": PROVIDER_TRACE_SCHEMA_VERSION,
        "provider": provider,
        "source_status": source_status,
        "source": source,
        "record_source": _record_source_from_status(source_status),
        "record_count": len(normalized),
        "provider_statuses": status_payloads,
        "summary": summarize_provider_trace_records(normalized),
        "records": [record.to_dict() for record in normalized],
        "traceEvents": trace_events,
    }


def summarize_provider_trace_records(
    records: Iterable[ProviderTraceRecord | Mapping[str, Any]],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    correlations: set[str] = set()
    launch_ids: set[str] = set()
    probe_names: set[str] = set()
    dropped_records = 0
    for raw in records:
        record = raw if isinstance(raw, ProviderTraceRecord) else _coerce_record(None, raw)
        counts[record.kind] = counts.get(record.kind, 0) + 1
        if record.correlation_id is not None:
            correlations.add(str(record.correlation_id))
        args = dict(record.args or {})
        if args.get("launch_id") is not None:
            launch_ids.add(str(args["launch_id"]))
        if args.get("probe_name") is not None:
            probe_names.add(str(args["probe_name"]))
        dropped_records += int(args.get("dropped_records") or 0)
    return {
        "kinds": dict(sorted(counts.items())),
        "correlation_count": len(correlations),
        "launch_count": len(launch_ids),
        "probe_count": len(probe_names),
        "dropped_records": dropped_records,
    }


def normalize_rocprofiler_api_record(raw: Mapping[str, Any]) -> ProviderTraceRecord:
    domain = str(_first(raw, "domain", "api_domain", default="hip"))
    phase = str(_first(raw, "phase", default="complete"))
    args = {
        "domain": domain,
        "phase": phase,
        "operation": _first(raw, "operation", "op", "api", default=None),
        "launch_id": _first(raw, "launch_id", default=None),
        "target": _first(raw, "target", default=None),
        "backend": _first(raw, "backend", default="rocm"),
        "stream_or_queue": _first(raw, "stream_or_queue", "queue_id", "stream_id", default=None),
        "fallback_reason": _first(raw, "fallback_reason", default=None),
        "sdk_version": _first(raw, "sdk_version", "rocprofiler_sdk_version", default=None),
        "context_id": _first(raw, "context_id", default=None),
        "callback_id": _first(raw, "callback_id", "callbackId", default=None),
        "dropped_records": _first(raw, "dropped_records", default=None),
    }
    return ProviderTraceRecord(
        provider="rocprofiler",
        kind="runtime_api",
        name=str(_first(raw, "name", "api", "operation", default="rocprofiler.api")),
        ts_us=_time_us(raw),
        duration_us=_duration_us(raw),
        correlation_id=_first(raw, "correlation_id", "correlationId", "corr_id", default=None),
        thread_id=_first(raw, "thread_id", "tid", default=0),
        args={k: v for k, v in args.items() if v is not None},
    )


def normalize_rocprofiler_activity_record(raw: Mapping[str, Any]) -> ProviderTraceRecord:
    activity = str(_first(raw, "activity", "kind", "record_kind", default="dispatch")).lower()
    is_memory = any(token in activity for token in ("copy", "memcpy", "memory", "memset"))
    args = {
        "activity": activity,
        "agent": _first(raw, "agent", "agent_id", default=None),
        "queue_id": _first(raw, "queue_id", "queue", default=None),
        "dispatch_id": _first(raw, "dispatch_id", default=None),
        "bytes": _first(raw, "bytes", "size", default=None),
        "launch_id": _first(raw, "launch_id", default=None),
        "probe_name": _first(raw, "probe_name", "probe", default=None),
        "target": _first(raw, "target", default=None),
        "backend": _first(raw, "backend", default="rocm"),
        "stream_or_queue": _first(raw, "stream_or_queue", "queue_id", "queue", default=None),
        "fallback_reason": _first(raw, "fallback_reason", default=None),
        "sdk_version": _first(raw, "sdk_version", "rocprofiler_sdk_version", default=None),
        "activity_service": _first(raw, "activity_service", "buffer_service", default=None),
        "buffer_bytes": _first(raw, "buffer_bytes", default=None),
        "dropped_records": _first(raw, "dropped_records", default=None),
    }
    return ProviderTraceRecord(
        provider="rocprofiler",
        kind="device_activity",
        name=str(_first(raw, "kernel_name", "name", "operation", default="rocprofiler.memcpy" if is_memory else "rocprofiler.dispatch")),
        ts_us=_time_us(raw),
        duration_us=_duration_us(raw),
        correlation_id=_first(raw, "correlation_id", "correlationId", "corr_id", default=None),
        thread_id=_first(raw, "queue_id", "tid", default=0),
        args={k: v for k, v in args.items() if v is not None},
    )


def normalize_rocprofiler_counter_record(raw: Mapping[str, Any]) -> ProviderTraceRecord:
    return ProviderTraceRecord(
        provider="rocprofiler",
        kind="counter",
        name=str(_first(raw, "metric", "counter", "name", default="rocprofiler.counter")),
        ts_us=_time_us(raw),
        value=float(_first(raw, "value", default=0.0) or 0.0),
        correlation_id=_first(raw, "correlation_id", "dispatch_id", default=None),
        thread_id=_first(raw, "agent_id", "tid", default=0),
        args={
            "unit": _first(raw, "unit", default=None),
            "agent": _first(raw, "agent", "agent_id", default=None),
            "launch_id": _first(raw, "launch_id", default=None),
            "probe_name": _first(raw, "probe_name", "probe", default=None),
            "target": _first(raw, "target", default=None),
            "backend": _first(raw, "backend", default="rocm"),
            "counter_request": _first(raw, "counter_request", "request", default=None),
            "counter_status": _first(raw, "counter_status", "request_status", default=None),
            "unsupported_counter": _first(raw, "unsupported_counter", default=None),
            "sdk_version": _first(raw, "sdk_version", "rocprofiler_sdk_version", default=None),
            "dropped_records": _first(raw, "dropped_records", default=None),
        },
    )


def normalize_rocprofiler_thread_trace_record(raw: Mapping[str, Any]) -> ProviderTraceRecord:
    return ProviderTraceRecord(
        provider="rocprofiler",
        kind="thread_trace",
        name=str(_first(raw, "kernel_name", "name", default="rocprofiler.thread_trace")),
        ts_us=_time_us(raw),
        duration_us=_duration_us(raw),
        correlation_id=_first(raw, "correlation_id", "dispatch_id", default=None),
        thread_id=_first(raw, "agent_id", "tid", default=0),
        args={
            "dispatch_id": _first(raw, "dispatch_id", default=None),
            "shader_engine_mask": _first(raw, "shader_engine_mask", default=None),
            "target_cu": _first(raw, "target_cu", default=None),
            "records": _first(raw, "records", "record_count", default=None),
            "launch_id": _first(raw, "launch_id", default=None),
            "probe_name": _first(raw, "probe_name", "probe", default=None),
            "target": _first(raw, "target", default=None),
            "backend": _first(raw, "backend", default="rocm"),
            "thread_trace_bytes": _first(raw, "thread_trace_bytes", "trace_bytes", default=None),
            "thread_trace_byte_limit": _first(raw, "thread_trace_byte_limit", default=None),
            "dropped_records": _first(raw, "dropped_records", default=None),
        },
    )


def normalize_compiler_probe_record(provider: Provider, raw: Mapping[str, Any]) -> ProviderTraceRecord:
    probe_name = str(_first(raw, "probe_name", "probe", "backend_correlation_key", default="compiler.probe"))
    target = _first(raw, "target", default=None)
    backend_default = {
        "cupti": "cuda",
        "metal": "metal",
        "rocprofiler": "rocm",
    }[provider]
    args = {
        "probe_name": probe_name,
        "source_op": _first(raw, "source_op", default=None),
        "region": _first(raw, "region", default=None),
        "schedule": _first(raw, "schedule", default=None),
        "metric": _first(raw, "metric", default=None),
        "aggregation": _first(raw, "aggregation", default=None),
        "payload_fields": _first(raw, "payload_fields", default=None),
        "backend_correlation_key": _first(raw, "backend_correlation_key", default=probe_name),
        "launch_id": _first(raw, "launch_id", default=None),
        "target": target,
        "backend": _first(raw, "backend", default=backend_default),
        "stream_or_queue": _first(raw, "stream_or_queue", "stream_id", "queue_id", default=None),
        "fallback_reason": _first(raw, "fallback_reason", default=None),
        "record_source": _first(raw, "record_source", default="compiler_probe"),
        "dropped_records": _first(raw, "dropped_records", default=None),
    }
    return ProviderTraceRecord(
        provider=provider,
        kind="intra_kernel",
        name=str(_first(raw, "name", "op_name", default=probe_name)),
        ts_us=_time_us(raw),
        duration_us=_duration_us(raw),
        correlation_id=_first(raw, "correlation_id", "launch_id", default=None),
        thread_id=_first(raw, "thread_id", "tid", "stream_id", "queue_id", default=0),
        args={k: v for k, v in args.items() if v is not None},
    )


def normalize_metal_command_buffer_record(raw: Mapping[str, Any]) -> ProviderTraceRecord:
    args = {
        "command_buffer": _first(raw, "command_buffer", "command_buffer_id", default=None),
        "kernel": _first(raw, "kernel", "probe", "label", default=None),
        "status": _first(raw, "status", default=None),
        "launch_id": _first(raw, "launch_id", default=None),
        "probe_name": _first(raw, "probe_name", "probe", default=None),
        "target": _first(raw, "target", default="apple_gpu"),
        "backend": _first(raw, "backend", default="metal"),
        "stream_or_queue": _first(raw, "stream_or_queue", "queue_id", default=None),
        "fallback_reason": _first(raw, "fallback_reason", default=None),
        "route": _first(raw, "route", "apple_route", "execution_route", default=None),
        "mps_route": _first(raw, "mps_route", default=None),
        "mtl4_route": _first(raw, "mtl4_route", default=None),
        "archive_cache_hit": _first(raw, "archive_cache_hit", "cache_hit", default=None),
        "mtl4_archive_path": _first(raw, "mtl4_archive_path", "archive_path", default=None),
        "tensor_phase": _first(raw, "tensor_phase", "phase", default=None),
        "tensor_bytes": _first(raw, "tensor_bytes", default=None),
        "resident_session_id": _first(raw, "resident_session_id", "session_id", default=None),
        "resident_session_phase": _first(raw, "resident_session_phase", default=None),
        "dropped_records": _first(raw, "dropped_records", default=None),
    }
    return ProviderTraceRecord(
        provider="metal",
        kind="command_buffer",
        name=str(_first(raw, "name", "label", "kernel", default="metal.command_buffer")),
        ts_us=_time_us(raw),
        duration_us=_duration_us(raw),
        correlation_id=_first(raw, "correlation_id", "command_buffer_id", default=None),
        thread_id=_first(raw, "queue_id", "tid", default=0),
        args={k: v for k, v in args.items() if v is not None},
    )


def normalize_metal_counter_record(raw: Mapping[str, Any]) -> ProviderTraceRecord:
    return ProviderTraceRecord(
        provider="metal",
        kind="counter",
        name=str(_first(raw, "counter", "metric", "name", default="metal.counter")),
        ts_us=_time_us(raw),
        value=float(_first(raw, "value", default=0.0) or 0.0),
        correlation_id=_first(raw, "correlation_id", "command_buffer_id", default=None),
        thread_id=_first(raw, "queue_id", "tid", default=0),
        args={
            "sample_index": _first(raw, "sample_index", default=None),
            "probe": _first(raw, "probe", "kernel", default=None),
            "launch_id": _first(raw, "launch_id", default=None),
            "probe_name": _first(raw, "probe_name", "probe", default=None),
            "target": _first(raw, "target", default="apple_gpu"),
            "backend": _first(raw, "backend", default="metal"),
            "counter_set": _first(raw, "counter_set", "set", default=None),
            "sample_buffer_id": _first(raw, "sample_buffer_id", "counter_sample_buffer", default=None),
            "command_buffer": _first(raw, "command_buffer", "command_buffer_id", default=None),
            "route": _first(raw, "route", "apple_route", "execution_route", default=None),
            "tensor_phase": _first(raw, "tensor_phase", "phase", default=None),
            "dropped_records": _first(raw, "dropped_records", default=None),
        },
    )


def normalize_cupti_callback_record(raw: Mapping[str, Any]) -> ProviderTraceRecord:
    domain = str(_first(raw, "domain", "callback_domain", default="runtime"))
    return ProviderTraceRecord(
        provider="cupti",
        kind="runtime_api",
        name=str(_first(raw, "name", "api", "cbid", default="cupti.callback")),
        ts_us=_time_us(raw),
        duration_us=_duration_us(raw),
        correlation_id=_first(raw, "correlation_id", "correlationId", default=None),
        thread_id=_first(raw, "thread_id", "tid", default=0),
        args={
            "domain": domain,
            "cbid": _first(raw, "cbid", default=None),
            "launch_id": _first(raw, "launch_id", default=None),
            "target": _first(raw, "target", default=None),
            "backend": _first(raw, "backend", default="cuda"),
            "stream_or_queue": _first(raw, "stream_or_queue", "stream_id", "streamId", default=None),
            "callback_id": _first(raw, "callback_id", "cbid", "callbackId", default=None),
            "context_id": _first(raw, "context_id", "contextId", default=None),
            "device_id": _first(raw, "device_id", "deviceId", default=None),
            "driver_version": _first(raw, "driver_version", default=None),
            "sdk_version": _first(raw, "sdk_version", "cupti_version", default=None),
            "fallback_reason": _first(raw, "fallback_reason", default=None),
            "dropped_records": _first(raw, "dropped_records", default=None),
        },
    )


def normalize_cupti_activity_record(raw: Mapping[str, Any]) -> ProviderTraceRecord:
    activity = str(_first(raw, "activity", "kind", default="kernel")).lower()
    args = {
        "activity": activity,
        "bytes": _first(raw, "bytes", default=None),
        "device_id": _first(raw, "device_id", "deviceId", default=None),
        "stream_id": _first(raw, "stream_id", "streamId", default=None),
        "launch_id": _first(raw, "launch_id", default=None),
        "probe_name": _first(raw, "probe_name", "probe", default=None),
        "target": _first(raw, "target", default=None),
        "backend": _first(raw, "backend", default="cuda"),
        "stream_or_queue": _first(raw, "stream_or_queue", "stream_id", "streamId", default=None),
        "context_id": _first(raw, "context_id", "contextId", default=None),
        "activity_buffer_id": _first(raw, "activity_buffer_id", "buffer_id", default=None),
        "activity_buffer_bytes": _first(raw, "activity_buffer_bytes", "buffer_bytes", default=None),
        "driver_version": _first(raw, "driver_version", default=None),
        "sdk_version": _first(raw, "sdk_version", "cupti_version", default=None),
        "unsupported_metric": _first(raw, "unsupported_metric", default=None),
        "fallback_reason": _first(raw, "fallback_reason", default=None),
        "dropped_records": _first(raw, "dropped_records", default=None),
    }
    return ProviderTraceRecord(
        provider="cupti",
        kind="device_activity",
        name=str(_first(raw, "kernel_name", "name", "activity", default="cupti.activity")),
        ts_us=_time_us(raw),
        duration_us=_duration_us(raw),
        correlation_id=_first(raw, "correlation_id", "correlationId", default=None),
        thread_id=_first(raw, "stream_id", "streamId", "tid", default=0),
        args={k: v for k, v in args.items() if v is not None},
    )


def records_from_raw(provider: Provider, raw_records: Sequence[Mapping[str, Any]]) -> tuple[ProviderTraceRecord, ...]:
    return tuple(_coerce_record(provider, record) for record in raw_records)


def load_provider_trace_input(path: str | Path, *, provider: Provider) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, Mapping) and payload.get("schema") == PROVIDER_TRACE_SCHEMA_VERSION:
        validate_provider_trace_artifact(payload)
        return dict(payload)
    raw_records = payload if isinstance(payload, list) else [payload]
    return build_provider_trace_artifact(
        provider=provider,
        records=records_from_raw(provider, raw_records),
        source_status="file",
        source=str(path),
    )


def write_provider_trace_artifact(payload: Mapping[str, Any], path: str | Path) -> Path:
    validate_provider_trace_artifact(payload)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out


def validate_provider_trace_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != PROVIDER_TRACE_SCHEMA_VERSION:
        raise ValueError("unsupported provider trace schema")
    records = payload.get("records")
    trace_events = payload.get("traceEvents")
    if not isinstance(records, list) or not isinstance(trace_events, list):
        raise ValueError("provider trace artifact requires records and traceEvents lists")
    if payload.get("record_count") != len(records):
        raise ValueError("record_count must match records length")
    provider_statuses = payload.get("provider_statuses", [])
    if provider_statuses is not None:
        if not isinstance(provider_statuses, list):
            raise ValueError("provider_statuses must be a list")
        for status in provider_statuses:
            if not isinstance(status, Mapping):
                raise ValueError("provider_statuses entries must be mappings")
            validate_provider_status_artifact(status)
    for event in trace_events:
        if not isinstance(event, Mapping):
            raise ValueError("provider trace events must be mappings")
        for key in ("name", "cat", "ph", "ts", "args"):
            if key not in event:
                raise ValueError(f"provider trace event missing {key!r}")
        args = event.get("args")
        if not isinstance(args, Mapping):
            raise ValueError("provider trace event args must be a mapping")
        if event.get("cat") in {"runtime_api", "device_activity", "counters", "intra_kernel"}:
            if "provider" not in args or "kind" not in args:
                raise ValueError("provider trace event args require provider and kind")


def _coerce_record(provider: Provider | None, raw: ProviderTraceRecord | Mapping[str, Any]) -> ProviderTraceRecord:
    if isinstance(raw, ProviderTraceRecord):
        return raw
    raw_provider = provider or raw.get("provider")
    if raw_provider not in {"rocprofiler", "cupti", "metal"}:
        raise ValueError("provider must be rocprofiler, cupti, or metal")
    if {"provider", "kind", "name", "ts_us", "duration_us", "args"} <= set(raw):
        return ProviderTraceRecord(
            provider=raw_provider,
            kind=str(raw["kind"]),  # type: ignore[arg-type]
            name=str(raw["name"]),
            ts_us=float(raw["ts_us"]),
            duration_us=float(raw.get("duration_us") or 0.0),
            correlation_id=raw.get("correlation_id"),
            thread_id=raw.get("thread_id", 0),
            value=(float(raw["value"]) if raw.get("value") is not None else None),
            args=dict(raw.get("args") or {}),
        )
    record_type = str(_first(raw, "record_type", "kind", "type", "activity", default="")).lower()
    if "probe" in record_type or str(raw.get("op_name", "")).endswith("profiler_probe"):
        return normalize_compiler_probe_record(raw_provider, raw)
    if raw_provider == "rocprofiler":
        if "thread" in record_type:
            return normalize_rocprofiler_thread_trace_record(raw)
        if "counter" in record_type or "metric" in raw:
            return normalize_rocprofiler_counter_record(raw)
        if any(token in record_type for token in ("dispatch", "activity", "memcpy", "copy", "memset", "kernel")):
            return normalize_rocprofiler_activity_record(raw)
        return normalize_rocprofiler_api_record(raw)
    if raw_provider == "metal":
        if "counter" in record_type or "counter" in raw or "metric" in raw:
            return normalize_metal_counter_record(raw)
        return normalize_metal_command_buffer_record(raw)
    if raw_provider == "cupti":
        if any(token in record_type for token in ("activity", "kernel", "memcpy", "memset")):
            return normalize_cupti_activity_record(raw)
        return normalize_cupti_callback_record(raw)
    raise AssertionError("unreachable provider")


def _first(raw: Mapping[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    return default


def _time_us(raw: Mapping[str, Any]) -> float:
    if "ts_us" in raw:
        return float(raw["ts_us"])
    if "timestamp_us" in raw:
        return float(raw["timestamp_us"])
    if "start_us" in raw:
        return float(raw["start_us"])
    if "begin_ns" in raw:
        return float(raw["begin_ns"]) / 1000.0
    if "start_ns" in raw:
        return float(raw["start_ns"]) / 1000.0
    return 0.0


def _duration_us(raw: Mapping[str, Any]) -> float:
    if "duration_us" in raw:
        return float(raw["duration_us"])
    if "dur_us" in raw:
        return float(raw["dur_us"])
    if "end_us" in raw and ("start_us" in raw or "ts_us" in raw):
        start = float(raw.get("start_us", raw.get("ts_us", 0.0)))
        return max(0.0, float(raw["end_us"]) - start)
    if "begin_ns" in raw and "end_ns" in raw:
        return max(0.0, (float(raw["end_ns"]) - float(raw["begin_ns"])) / 1000.0)
    if "start_ns" in raw and "end_ns" in raw:
        return max(0.0, (float(raw["end_ns"]) - float(raw["start_ns"])) / 1000.0)
    return 0.0


def _record_source_from_status(source_status: str) -> str:
    normalized = str(source_status).lower()
    if normalized in {"native", "fixture", "replay", "file", "mock"}:
        return normalized
    if "file" in normalized:
        return "file"
    if "replay" in normalized:
        return "replay"
    if "fixture" in normalized:
        return "fixture"
    if "native" in normalized or "measured" in normalized:
        return "native"
    return "unknown"


__all__ = [
    "PROVIDER_TRACE_SCHEMA_VERSION",
    "ProviderTraceRecord",
    "build_provider_trace_artifact",
    "load_provider_trace_input",
    "normalize_cupti_activity_record",
    "normalize_cupti_callback_record",
    "normalize_metal_command_buffer_record",
    "normalize_metal_counter_record",
    "normalize_compiler_probe_record",
    "normalize_rocprofiler_activity_record",
    "normalize_rocprofiler_api_record",
    "normalize_rocprofiler_counter_record",
    "normalize_rocprofiler_thread_trace_record",
    "records_from_raw",
    "summarize_provider_trace_records",
    "validate_provider_trace_artifact",
    "write_provider_trace_artifact",
]
