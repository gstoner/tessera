"""Fixed eight-process APPLE-ROUTE-1 experiment; never installs a route ledger.

Run with PYTHONPATH=python on the owning Mac. The plan is written before any
measurement. Failure leaves the partial experiment intact; there is no retry
or stop-on-promotion loop. Synthetic slowdown scenarios are analysis only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from tessera.compiler.apple_route_selector import aggregate_stable_route_reports


PLAN = {
    "run_count": 8, "reps": 5, "trials": 9, "profile": "extended",
    "first_seed": 1701, "process_timeout_seconds": 300,
    "synthetic_run_index": 0, "synthetic_candidate_time_factors": [1.5, 3.0],
}
INCUMBENTS = {
    "retune_grouped_gemm": "grouped_fused", "retune_moe_swiglu": "composed",
    "retune_reduce_sum": "mpsgraph", "retune_resident_kv_read": "resident_view",
    "retune_mla_decode": "explicit", "retune_replay_decode": "fused_block",
}


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def compare_reports(reports: list[dict[str, Any]], *,
                    incumbents: dict[str, str] = INCUMBENTS) -> dict[str, Any]:
    """Evaluate identical inputs, keeping counterfactual data out of ledgers."""
    if len(reports) != PLAN["run_count"]:
        raise ValueError("experiment requires exactly eight reports")
    scenarios = [("observed", reports)]
    for factor in PLAN["synthetic_candidate_time_factors"]:
        altered = copy.deepcopy(reports)
        for row in altered[PLAN["synthetic_run_index"]]["runs"]:
            incumbent = incumbents.get(row["op"])
            if incumbent is None or row["route"] == incumbent:
                continue
            telemetry = row["telemetry"]
            for key in ("end_to_end_median_ns", "device_time_median_ns"):
                if telemetry.get(key) is not None:
                    telemetry[key] *= factor
            for key in ("paired_trial_end_to_end_medians_ns",
                        "paired_trial_device_medians_ns"):
                if telemetry.get(key) is not None:
                    telemetry[key] = [value * factor for value in telemetry[key]]
        scenarios.append((f"synthetic_candidate_slowdown_{factor}x", altered))
    results = []
    for name, inputs in scenarios:
        policies = {
            estimator: aggregate_stable_route_reports(
                inputs, incumbent_routes=incumbents, cross_run_estimator=estimator)
            for estimator in ("mean_student_t", "median_order_statistic")
        }
        mean, median = (policies[key]["decisions"] for key in policies)
        changed = [
            {key: left[key] for key in ("op", "shape", "dtype", "device", "timing_domain")}
            for left, right in zip(mean, median, strict=True)
            if (left["selected_route"], left["status"]) !=
               (right["selected_route"], right["status"])
        ]
        results.append({"scenario": name, "synthetic": name != "observed",
                        "changed_decisions": changed, "policies": policies})
    return {"schema": "tessera.apple.cross-run-policy-comparison.v1",
            "promotion_allowed": False, "default_policy_changed": False,
            "scope": "policy_analysis_not_sealed_device_evidence",
            "plan": PLAN, "scenarios": results}


def collect(output: Path) -> None:
    library = Path(os.environ["TESSERA_APPLE_GPU_RUNTIME_LIB"]).resolve(strict=True)
    root = Path(__file__).resolve().parents[2]
    inputs = [Path(__file__).resolve(), library,
              root / "benchmarks/apple_gpu/benchmark_legacy_retune.py",
              root / "python/tessera/compiler/apple_route_selector.py",
              root / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm"]

    def fingerprints():
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}

    before = fingerprints()
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "plan.json", PLAN)
    _write(output / "inputs.json", before)
    reports, sources = [], []
    for index in range(PLAN["run_count"]):
        path = output / f"run-{index:02d}.json"
        # Each run gets fresh runtime state and its own warmup. No failed run
        # is silently replaced, and output directories cannot be reused.
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(index),
             "--output", str(path.resolve())], check=True,
            timeout=PLAN["process_timeout_seconds"])
        raw = path.read_bytes()
        reports.append(json.loads(raw))
        sources.append({"path": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
    if fingerprints() != before:
        raise RuntimeError("experiment source or runtime changed during collection")
    result = compare_reports(reports)
    result["source_reports"] = sources
    _write(output / "comparison.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=range(8), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is None:
        collect(args.output)
        return
    # Import the Metal recorder only inside the fresh measurement process.
    from benchmark_legacy_retune import run_report
    from tessera import runtime
    if not runtime.DeviceTensor.is_metal():
        raise RuntimeError("owning Apple Metal device is not visible")
    _write(args.output, run_report(
        reps=PLAN["reps"], trials=PLAN["trials"], profile=PLAN["profile"],
        seed=PLAN["first_seed"] + args.worker))


if __name__ == "__main__":
    main()
