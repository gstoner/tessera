"""Runner-facing Tessera Model Analyzer manifest execution helpers.

The first runner is intentionally hardware-free. It consumes the manifest
emitted by ``profiling_plan.model_analyzer_manifest`` and produces a stable
result artifact over the requested search space. Callers may pass a measurement
function to provide real latency/memory data; otherwise trials are marked as
estimated/planned so the artifact remains honest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


MODEL_ANALYZER_RESULT_SCHEMA_VERSION = "tessera.compiler.model_analyzer_result.v1"


@dataclass(frozen=True)
class AnalyzerTrial:
    batch_size: int
    instance_count: int
    dynamic_batching: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "instance_count": self.instance_count,
            "dynamic_batching": self.dynamic_batching,
        }


MeasurementFn = Callable[[AnalyzerTrial, Mapping[str, Any]], Mapping[str, Any]]


def load_model_analyzer_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_model_analyzer_result(result: Mapping[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return out


def run_model_analyzer_manifest(
    manifest: Mapping[str, Any],
    *,
    measure: MeasurementFn | None = None,
) -> dict[str, Any]:
    """Run or estimate all configurations from a Model Analyzer manifest."""

    search = _required_mapping(manifest, "search")
    trials = list(_iter_trials(search))
    if not trials:
        raise ValueError("model analyzer manifest has an empty search space")

    results = [_evaluate_trial(trial, manifest, measure=measure) for trial in trials]
    objective = _required_mapping(manifest, "objective")
    primary = str(objective.get("primary", "latency_ms"))
    best = _select_best_trial(results, primary)
    return {
        "schema": MODEL_ANALYZER_RESULT_SCHEMA_VERSION,
        "manifest_schema": manifest.get("schema"),
        "target": manifest.get("target"),
        "model_name": manifest.get("model_name"),
        "runner": dict(_required_mapping(manifest, "runner")),
        "objective": dict(objective),
        "trial_count": len(results),
        "best": best,
        "trials": results,
        "notes": [
            "Default runner output is estimated unless a measurement function supplies measured=true.",
            "Native NVIDIA/ROCm/Apple GPU collectors must promote provider status before hardware claims.",
        ],
    }


def _iter_trials(search: Mapping[str, Any]) -> list[AnalyzerTrial]:
    batch_sizes = _positive_ints(search.get("batch_sizes", (1,)))
    instance_counts = _positive_ints(search.get("instance_counts", (1,)))
    dynamic_batching = _bools(search.get("dynamic_batching", (False,)))
    return [
        AnalyzerTrial(batch, instances, dynamic)
        for batch in batch_sizes
        for instances in instance_counts
        for dynamic in dynamic_batching
    ]


def _evaluate_trial(
    trial: AnalyzerTrial,
    manifest: Mapping[str, Any],
    *,
    measure: MeasurementFn | None,
) -> dict[str, Any]:
    if measure is not None:
        measured = dict(measure(trial, manifest))
        return _normalize_measurement(trial, measured, default_status="measured")
    return _estimate_trial(trial, manifest)


def _estimate_trial(trial: AnalyzerTrial, manifest: Mapping[str, Any]) -> dict[str, Any]:
    runner = _required_mapping(manifest, "runner")
    provider_status = str(runner.get("status", "planned"))
    target = str(manifest.get("target", "generic"))
    target_factor = {
        "cpu": 1.0,
        "nvidia": 0.28,
        "rocm": 0.34,
        "apple_gpu": 0.46,
    }.get(target, 1.2)
    dynamic_factor = 0.88 if trial.dynamic_batching else 1.0
    latency_ms = max(0.01, target_factor * trial.batch_size / trial.instance_count * dynamic_factor)
    throughput_qps = trial.batch_size * trial.instance_count * 1000.0 / latency_ms
    memory_bytes = int(64 * 1024 * 1024 * trial.batch_size * trial.instance_count)
    status = "estimated" if provider_status == "available" else f"{provider_status}_estimate"
    return _normalize_measurement(
        trial,
        {
            "latency_ms": latency_ms,
            "throughput_qps": throughput_qps,
            "memory_bytes": memory_bytes,
            "status": status,
            "measured": False,
        },
        default_status=status,
    )


def _normalize_measurement(
    trial: AnalyzerTrial,
    measurement: Mapping[str, Any],
    *,
    default_status: str,
) -> dict[str, Any]:
    latency_ms = float(measurement.get("latency_ms", 0.0))
    throughput_qps = float(measurement.get("throughput_qps", 0.0))
    memory_bytes = int(measurement.get("memory_bytes", 0))
    if latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")
    if throughput_qps < 0:
        raise ValueError("throughput_qps must be non-negative")
    if memory_bytes < 0:
        raise ValueError("memory_bytes must be non-negative")
    out = {
        **trial.to_dict(),
        "latency_ms": latency_ms,
        "throughput_qps": throughput_qps,
        "memory_bytes": memory_bytes,
        "status": str(measurement.get("status", default_status)),
        "measured": bool(measurement.get("measured", default_status == "measured")),
    }
    if "metadata" in measurement:
        out["metadata"] = dict(measurement["metadata"])
    return out


def _select_best_trial(results: Sequence[Mapping[str, Any]], objective: str) -> dict[str, Any]:
    # Bind the chosen row first, then dict() it: wrapping max()/min() directly in
    # dict() makes mypy infer the lambda's `row` as SupportsKeysAndGetItem (which
    # has no .get) via the dict() constructor overload — splitting the call keeps
    # `row` typed as the sequence's Mapping element.
    if objective == "throughput_qps":
        best = max(results, key=lambda row: float(row.get("throughput_qps", 0.0)))
    elif objective == "memory_bytes":
        best = min(results, key=lambda row: int(row.get("memory_bytes", 0)))
    else:
        best = min(results, key=lambda row: float(row.get("latency_ms", 0.0)))
    return dict(best)


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"model analyzer manifest missing mapping {key!r}")
    return value


def _positive_ints(value: Any) -> tuple[int, ...]:
    out = tuple(int(v) for v in value)
    if not out or any(v <= 0 for v in out):
        raise ValueError("search dimensions must contain positive integers")
    return out


def _bools(value: Any) -> tuple[bool, ...]:
    out = tuple(bool(v) for v in value)
    if not out:
        raise ValueError("dynamic_batching must be non-empty")
    return out


__all__ = [
    "AnalyzerTrial",
    "MODEL_ANALYZER_RESULT_SCHEMA_VERSION",
    "MeasurementFn",
    "load_model_analyzer_manifest",
    "run_model_analyzer_manifest",
    "write_model_analyzer_result",
]
