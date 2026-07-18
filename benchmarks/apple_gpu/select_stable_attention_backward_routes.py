"""Build an evidence-gated Apple attention-backward route ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tessera.compiler.apple_attention_backward import ROUTES
from tessera.compiler.apple_route_selector import aggregate_stable_route_reports


def build_backward_ledger(
    reports: Sequence[Mapping[str, Any]], *, minimum_speedup: float = 0.05,
) -> dict[str, Any]:
    ledger = aggregate_stable_route_reports(
        reports,
        incumbent_routes={"flash_attn_bwd": "serial_recompute"},
        min_speedup=minimum_speedup,
    )
    proof_runs = []
    for index, report in enumerate(reports):
        rows = list(report.get("runs", []))
        dtypes = sorted({str(row.get("dtype")) for row in rows})
        routes = sorted({str(row.get("route")) for row in rows})
        dtype_route_pairs = sorted({
            (str(row.get("dtype")), str(row.get("route"))) for row in rows})
        semantic_rows = [row.get("semantics") for row in rows]
        proof_runs.append({
            "run": index + 1,
            "device": report.get("device"),
            "row_count": len(rows),
            "dtypes": dtypes,
            "routes": routes,
            "dtype_route_pairs": [list(pair) for pair in dtype_route_pairs],
            "all_native_correct_and_timed": bool(rows) and all(
                row.get("native_dispatched") is True
                and row.get("numerically_validated") is True
                and row.get("telemetry", {}).get("device_time_coverage") == 1.0
                for row in rows),
            "semantic_matrix_present": all(
                isinstance(value, Mapping) for value in semantic_rows),
        })
    required_dtypes = {"f32", "f16", "bf16"}
    required_routes = set(ROUTES)
    required_pairs = {
        (dtype, route) for dtype in required_dtypes for route in required_routes}
    complete = len(proof_runs) >= 2 and all(
        row["all_native_correct_and_timed"]
        and row["semantic_matrix_present"]
        and required_dtypes.issubset(row["dtypes"])
        and required_routes.issubset(row["routes"])
        and required_pairs.issubset({tuple(pair)
                                     for pair in row["dtype_route_pairs"]})
        for row in proof_runs)
    ledger["attention_backward_proof"] = {
        "native_storage_dtypes": sorted(required_dtypes),
        "gradient_accumulation_dtype": "f32",
        "candidate_routes": list(ROUTES),
        "atomic_determinism": "nondeterministic",
        "split_reduction_order": "fixed_two_partition",
        "runs": proof_runs,
        "complete": complete,
    }
    return ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--minimum-speedup", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.reports) < 2:
        parser.error("at least two independent reports are required")
    reports = [json.loads(path.read_text(encoding="utf-8"))
               for path in args.reports]
    ledger = build_backward_ledger(
        reports, minimum_speedup=args.minimum_speedup)
    args.output.write_text(
        json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
