"""Merge Tessera runtime, provider, and context profiler artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .profiler_context import validate_profiler_context_artifact
from .profiler_provider_trace import validate_provider_trace_artifact


MERGED_PROFILER_TRACE_SCHEMA_VERSION = "tessera.merged_profiler_trace.v1"


def merge_profiler_traces(
    *,
    runtime_trace: Mapping[str, Any] | None = None,
    provider_traces: Sequence[Mapping[str, Any]] = (),
    context_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge runtime Trace Event JSON, provider traces, and context metadata."""

    trace_events: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    if runtime_trace is not None:
        events = list(runtime_trace.get("traceEvents", []))
        trace_events.extend(dict(event) for event in events)
        sources.append({"kind": "runtime_trace", "events": len(events)})
    for provider in provider_traces:
        validate_provider_trace_artifact(provider)
        events = list(provider.get("traceEvents", []))
        trace_events.extend(dict(event) for event in events)
        sources.append({
            "kind": "provider_trace",
            "provider": provider.get("provider"),
            "events": len(events),
            "records": provider.get("record_count", 0),
        })
    context_summary = None
    if context_artifact is not None:
        validate_profiler_context_artifact(context_artifact)
        context_summary = {
            "schema": context_artifact.get("schema"),
            "target": context_artifact.get("target"),
            "provider": context_artifact.get("provider"),
            "source_status": context_artifact.get("source_status"),
            "sample_count": context_artifact.get("sample_count"),
            "bottleneck_summary": context_artifact.get("bottleneck_summary", {}),
        }
        trace_events.append(_context_marker_event(context_summary))
        sources.append({
            "kind": "profiler_context",
            "provider": context_artifact.get("provider"),
            "samples": context_artifact.get("sample_count", 0),
        })
    trace_events.sort(key=lambda event: float(event.get("ts", 0.0)))
    return {
        "schema": MERGED_PROFILER_TRACE_SCHEMA_VERSION,
        "displayTimeUnit": "ns",
        "traceEvents": trace_events,
        "sources": sources,
        "context_summary": context_summary,
        "summary": _summarize_trace_events(trace_events),
    }


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_merged_profiler_trace(payload: Mapping[str, Any], path: str | Path) -> Path:
    validate_merged_profiler_trace(payload)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out


def validate_merged_profiler_trace(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != MERGED_PROFILER_TRACE_SCHEMA_VERSION:
        raise ValueError("unsupported merged profiler trace schema")
    if not isinstance(payload.get("traceEvents"), list):
        raise ValueError("merged profiler trace requires traceEvents list")
    if not isinstance(payload.get("summary"), Mapping):
        raise ValueError("merged profiler trace requires summary mapping")
    event_count = payload["summary"].get("event_count")
    if event_count != len(payload["traceEvents"]):
        raise ValueError("summary.event_count must match traceEvents length")


def _context_marker_event(summary: Mapping[str, Any]) -> dict[str, Any]:
    bottleneck = (summary.get("bottleneck_summary") or {}).get("dominant_bottleneck")
    return {
        "name": "profiler_context.summary",
        "cat": "host_context",
        "ph": "i",
        "s": "p",
        "ts": 0.0,
        "pid": 0,
        "tid": 0,
        "args": {
            "provider": summary.get("provider"),
            "source_status": summary.get("source_status"),
            "dominant_bottleneck": bottleneck,
            "sample_count": summary.get("sample_count"),
        },
    }


def _summarize_trace_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    categories: dict[str, int] = {}
    correlations: set[str] = set()
    for event in events:
        cat = str(event.get("cat", "unknown"))
        categories[cat] = categories.get(cat, 0) + 1
        args = event.get("args", {})
        if isinstance(args, Mapping) and args.get("correlation_id") is not None:
            correlations.add(str(args["correlation_id"]))
    return {
        "event_count": len(events),
        "categories": dict(sorted(categories.items())),
        "correlation_count": len(correlations),
    }


__all__ = [
    "MERGED_PROFILER_TRACE_SCHEMA_VERSION",
    "load_json",
    "merge_profiler_traces",
    "validate_merged_profiler_trace",
    "write_merged_profiler_trace",
]
