#!/usr/bin/env python3
"""Gate deterministic telemetry reports against a small JSON baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


def load_events(path: str | Path) -> list[Mapping[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if "telemetry_events" in payload:
        return list(payload["telemetry_events"])
    if "events" in payload:
        return list(payload["events"])
    raise ValueError(f"{path} does not contain telemetry_events or events")


def load_baseline(path: str | Path) -> Mapping[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate(events: list[Mapping[str, Any]], baseline: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    required_schema = baseline.get("schema", "tessera.telemetry.v1")
    max_latency_ms = float(baseline.get("max_latency_ms", float("inf")))
    min_tflops = float(baseline.get("min_tflops", 0.0))
    min_bandwidth_gbps = float(baseline.get("min_bandwidth_gbps", 0.0))
    allowed_statuses = set(baseline.get("allowed_statuses", ["ok", "executable", "cached"]))

    for index, event in enumerate(events):
        label = f"{event.get('name', 'event')}[{index}]"
        if event.get("schema") != required_schema:
            failures.append(f"{label}: schema={event.get('schema')!r}, want {required_schema!r}")
        status = str(event.get("status", ""))
        if status not in allowed_statuses:
            failures.append(f"{label}: status={status!r} not in {sorted(allowed_statuses)}")
        latency = event.get("latency_ms")
        if latency is not None and float(latency) > max_latency_ms:
            failures.append(f"{label}: latency_ms={latency} exceeds {max_latency_ms}")
        tflops = event.get("tflops")
        if tflops is not None and float(tflops) < min_tflops:
            failures.append(f"{label}: tflops={tflops} below {min_tflops}")
        bandwidth = event.get("bandwidth_gbps")
        if bandwidth is not None and float(bandwidth) < min_bandwidth_gbps:
            failures.append(f"{label}: bandwidth_gbps={bandwidth} below {min_bandwidth_gbps}")
    if len(events) < int(baseline.get("min_event_count", 1)):
        failures.append(f"event_count={len(events)} below {baseline.get('min_event_count')}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate telemetry against a performance baseline.")
    parser.add_argument("report", help="Benchmark/profiler/runtime JSON report")
    parser.add_argument("--baseline", required=True, help="Baseline JSON")
    args = parser.parse_args(argv)

    failures = evaluate(load_events(args.report), load_baseline(args.baseline))
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("Telemetry perf gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
