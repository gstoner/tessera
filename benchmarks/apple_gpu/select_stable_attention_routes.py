"""Aggregate Apple attention reports and retain forward-proof evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tessera.compiler.apple_route_selector import aggregate_stable_route_reports


def build_attention_ledger(reports: Sequence[Mapping[str, Any]], *,
                           minimum_speedup: float = 0.05) -> dict[str, Any]:
    ledger = aggregate_stable_route_reports(
        reports,
        incumbent_routes={"flash_attn_mha": "online_msl_variant"},
        min_speedup=minimum_speedup,
    )
    proof_runs = []
    for index, report in enumerate(reports):
        variants = list(report.get("variant_coverage", []))
        resident = list(report.get("resident_comparison", []))
        proof_runs.append({
            "run": index + 1,
            "device": report.get("device"),
            "variant_coverage": variants,
            "resident_comparison": resident,
            "all_variants_native_and_correct": bool(variants) and all(
                row.get("native_dispatched") is True
                and row.get("numerically_validated") is True
                for row in variants),
            "all_resident_candidates_native_correct_and_timed": bool(resident)
            and all(
                row.get("native_dispatched") is True
                and row.get("numerically_validated") is True
                and row.get("device_time_coverage") == 1.0
                and isinstance(row.get("device_time_median_ns"), int)
                and row["device_time_median_ns"] > 0
                for row in resident),
        })
    ledger["attention_forward_proof"] = {
        "native_storage_dtypes": ["f32", "f16", "bf16"],
        "accumulation_dtype": "f32",
        "candidate_timing_domain": "device_input_host_output",
        "cooperative_candidate": "cooperative_simdgroup",
        "cooperative_matrix_status": "capability_unavailable",
        "runs": proof_runs,
        "complete": len(proof_runs) >= 2 and all(
            row["all_variants_native_and_correct"]
            and row["all_resident_candidates_native_correct_and_timed"]
            for row in proof_runs),
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
    ledger = build_attention_ledger(
        reports, minimum_speedup=args.minimum_speedup)
    args.output.write_text(
        json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
