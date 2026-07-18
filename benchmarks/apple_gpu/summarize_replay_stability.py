"""Combine independent ReplaySSM reports into a retained stability ledger.

This is characterization, not a route promoter: the legacy Replay benchmark
does not interleave paired route blocks. The ledger records native, correct,
resource-complete timings and drift while declining a selector decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _drift(values: list[float]) -> float | None:
    if not values or min(values) <= 0:
        return None
    return (max(values) - min(values)) / min(values)


def summarize(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("ReplaySSM stability requires at least two reports")
    indexes: list[dict[tuple[str, str], dict[str, Any]]] = []
    keys: set[tuple[str, str]] = set()
    for report in reports:
        runs = report.get("runs")
        if not isinstance(runs, list):
            raise ValueError("each ReplaySSM report must contain a runs list")
        index: dict[tuple[str, str], dict[str, Any]] = {}
        for row in runs:
            if not isinstance(row, dict) or row.get("mode") not in {
                    "replay_fused", "replay_block"}:
                continue
            key = str(row.get("shape")), str(row.get("mode"))
            if key in index:
                raise ValueError(f"duplicate ReplaySSM row: {key!r}")
            index[key] = row
            keys.add(key)
        indexes.append(index)

    rows = []
    for shape, mode in sorted(keys):
        source = [index.get((shape, mode)) for index in indexes]
        present = all(row is not None for row in source)
        native = present and all(bool(row.get("native_dispatched")) for row in source)
        correct = present and all(bool(row.get("numerically_validated")) for row in source)
        resources = present and all(isinstance(row.get("resources"), dict)
                                    for row in source)
        coverage = [float(row.get("device_time_coverage", 0.0))
                    for row in source if row is not None]
        e2e = [float(row["latency_ms"]) for row in source if row is not None]
        device = [int(row["device_time_median_ns"]) for row in source
                  if row is not None and row.get("device_time_median_ns") is not None]
        complete_device = len(device) == len(reports) and all(
            value >= 0.9 for value in coverage)
        rows.append({
            "shape": shape,
            "mode": mode,
            "dtype": "f32",
            "report_count": len(reports),
            "native_dispatched_all_runs": native,
            "numerically_validated_all_runs": correct,
            "resource_evidence_all_runs": resources,
            "device_time_coverage": coverage,
            "end_to_end_per_token_ms": e2e,
            "device_per_token_ns": device,
            "end_to_end_cross_run_drift_fraction": _drift(e2e),
            "device_cross_run_drift_fraction": _drift(
                [float(value) for value in device]),
            "evidence_complete": native and correct and resources and complete_device,
            "resources": [row.get("resources") if row is not None else None
                          for row in source],
        })
    return {
        "schema_version": 1,
        "backend": "apple_gpu",
        "device": "apple7",
        "op": "ssm_replay_decode",
        "report_count": len(reports),
        "selector_status": "characterized_not_selector_eligible",
        "selector_reason": (
            "reports retain independent repeated medians but do not use "
            "interleaved paired route blocks"),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = summarize([
        json.loads(path.read_text(encoding="utf-8")) for path in args.reports])
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True),
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
