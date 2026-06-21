"""Portable profiler context artifacts.

Runtime traces answer "what happened when?" while system context answers "what
else was the accelerator doing?".  This module keeps that second answer in a
stable JSON artifact that reports and Model Analyzer runs can consume before
native vendor collectors are available on every machine.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


PROFILER_CONTEXT_SCHEMA_VERSION = "tessera.profiler_context.v1"


def build_profiler_context_artifact(
    *,
    target: str,
    samples: Iterable[Mapping[str, Any]],
    provider: str | None = None,
    source_status: str = "planned",
    source: str | None = None,
) -> dict[str, Any]:
    """Build a normalized profiler context artifact from backend samples."""

    normalized_samples = [_normalize_sample(sample) for sample in samples]
    summary = summarize_profiler_context(normalized_samples)
    return {
        "schema": PROFILER_CONTEXT_SCHEMA_VERSION,
        "target": str(target),
        "provider": provider or _infer_provider(normalized_samples, target),
        "source_status": str(source_status),
        "source": source,
        "sample_count": len(normalized_samples),
        "bottleneck_summary": summary,
        "samples": normalized_samples,
    }


def summarize_profiler_context(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize normalized context samples by bottleneck and provider."""

    bottlenecks: collections.Counter[str] = collections.Counter()
    providers: collections.Counter[str] = collections.Counter()
    for sample in samples:
        bottleneck = str(sample.get("bottleneck") or "unknown")
        provider = str(sample.get("provider") or "unknown")
        bottlenecks[bottleneck] += 1
        providers[provider] += 1
    total = sum(bottlenecks.values())
    dominant = bottlenecks.most_common(1)[0][0] if bottlenecks else None
    return {
        "total_samples": total,
        "dominant_bottleneck": dominant,
        "bottlenecks": dict(sorted(bottlenecks.items())),
        "providers": dict(sorted(providers.items())),
    }


def load_profiler_context_artifact(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    validate_profiler_context_artifact(payload)
    return payload


def write_profiler_context_artifact(payload: Mapping[str, Any], path: str | Path) -> Path:
    validate_profiler_context_artifact(payload)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out


def validate_profiler_context_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != PROFILER_CONTEXT_SCHEMA_VERSION:
        raise ValueError("unsupported profiler context artifact schema")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("profiler context artifact requires a samples list")
    summary = payload.get("bottleneck_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("profiler context artifact requires bottleneck_summary")
    sample_count = payload.get("sample_count")
    if sample_count != len(samples):
        raise ValueError("sample_count must match samples length")
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError("profiler context samples must be mappings")
        for key in ("provider", "bottleneck", "raw"):
            if key not in sample:
                raise ValueError(f"profiler context sample missing {key!r}")


def _normalize_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(sample)
    schema = str(raw.get("schema", ""))
    provider = str(raw.get("provider") or _provider_from_sample(raw, schema))
    bottleneck = str(raw.get("bottleneck") or "unknown")
    timestamp_us = raw.get("timestamp_us", raw.get("ts_us"))
    out = {
        "provider": provider,
        "bottleneck": bottleneck,
        "timestamp_us": timestamp_us,
        "raw": raw,
    }
    vendor = raw.get("vendor")
    if vendor is not None:
        out["vendor"] = vendor
    return out


def _provider_from_sample(sample: Mapping[str, Any], schema: str) -> str:
    if schema == "tessera.apple_profiler_context.v1":
        return "apple-silicon-system-context"
    if schema == "tessera.accelerator_profiler_context.v1":
        vendor = sample.get("vendor")
        if vendor == "nvidia":
            return "nvidia-system-context"
        if vendor == "rocm":
            return "rocm-system-context"
    return "unknown"


def _infer_provider(samples: Iterable[Mapping[str, Any]], target: str) -> str:
    providers = {str(sample.get("provider")) for sample in samples if sample.get("provider")}
    if len(providers) == 1:
        return next(iter(providers))
    if providers:
        return "mixed-system-context"
    normalized = str(target).lower().replace("-", "_")
    if normalized in {"apple", "apple_gpu", "metal", "mps"}:
        return "apple-silicon-system-context"
    if normalized in {"nvidia", "cuda"} or normalized.startswith("sm"):
        return "nvidia-system-context"
    if normalized in {"rocm", "hip", "amd"} or normalized.startswith("gfx"):
        return "rocm-system-context"
    return "unknown"


__all__ = [
    "PROFILER_CONTEXT_SCHEMA_VERSION",
    "build_profiler_context_artifact",
    "load_profiler_context_artifact",
    "summarize_profiler_context",
    "validate_profiler_context_artifact",
    "write_profiler_context_artifact",
]
