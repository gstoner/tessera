"""Normalize Nsight Compute CSV exports into committed TEST-5 resource rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


_FIELDS = {
    "Registers Per Thread": "registers_per_thread",
    "Static Shared Memory Per Block": "static_shared_memory_bytes",
    "Dynamic Shared Memory Per Block": "dynamic_shared_memory_bytes",
    "Theoretical Occupancy": "theoretical_occupancy_pct",
    "Achieved Occupancy": "achieved_occupancy_pct",
    "Local Load Transactions": "local_load_transactions",
    "Local Store Transactions": "local_store_transactions",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": "local_load_transactions",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum": "local_store_transactions",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld": "local_load_transactions",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st": "local_store_transactions",
    # CUDA 13.3 full-set display names. Both local and shared counters are
    # retained because Blackwell can spill registers into either memory space.
    "Local Memory Spilling Requests": "local_spill_requests",
    "Shared Memory Spilling Requests": "shared_spill_requests",
}


def _number(text: str) -> float:
    return float(text.replace(",", ""))


def parse_csv(text: str) -> list[dict[str, Any]]:
    """Aggregate launch metrics by kernel with a reproducible fingerprint."""
    launches: dict[tuple[str, str], dict[str, Any]] = {}
    seen_spill_metrics: set[str] = set()
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        kernel = row.get("Kernel Name", "").strip()
        launch_id = row.get("ID", "").strip()
        metric = row.get("Metric Name", "").strip()
        value = row.get("Metric Value", "").strip()
        if not kernel or metric not in _FIELDS or not value:
            continue
        field = _FIELDS[metric]
        if field in {"local_load_transactions", "local_store_transactions",
                     "local_spill_requests", "shared_spill_requests"}:
            seen_spill_metrics.add(field)
        launch = launches.setdefault((kernel, launch_id), {
            "kernel": kernel, "cc": row.get("CC", ""),
            "block_size": row.get("Block Size", ""),
            "grid_size": row.get("Grid Size", ""),
        })
        launch[field] = _number(value)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for launch in launches.values():
        grouped[launch["kernel"]].append(launch)
    result: list[dict[str, Any]] = []
    for kernel, samples in sorted(grouped.items()):
        row: dict[str, Any] = {
            "kernel": kernel, "launches": len(samples),
            "cc": samples[0].get("cc", ""),
            "block_size": samples[0].get("block_size", ""),
            "grid_size": samples[0].get("grid_size", ""),
        }
        for field in _FIELDS.values():
            values = [float(s[field]) for s in samples if field in s]
            if values:
                value = statistics.median(values)
                row[field] = int(value) if value.is_integer() else round(value, 4)
        legacy_complete = {
            "local_load_transactions", "local_store_transactions"} <= seen_spill_metrics
        cuda_13_complete = {
            "local_spill_requests", "shared_spill_requests"} <= seen_spill_metrics
        complete = legacy_complete or cuda_13_complete
        row["spill_evidence_complete"] = complete
        row["spills_detected"] = (None if not complete else bool(
            row.get("local_load_transactions", 0) or
            row.get("local_store_transactions", 0) or
            row.get("local_spill_requests", 0) or
            row.get("shared_spill_requests", 0)))
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
        row["resource_fingerprint"] = "sha256:" + hashlib.sha256(
            canonical.encode()).hexdigest()
        result.append(row)
    return result


def export_report(report: Path, ncu: str = "/usr/local/cuda/bin/ncu") -> str:
    completed = subprocess.run(
        [ncu, "--import", str(report), "--page", "details", "--csv",
         "--print-units", "base"], check=True, capture_output=True, text=True)
    return completed.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--ncu", default="/usr/local/cuda/bin/ncu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = parse_csv(export_report(args.report, args.ncu))
    args.output.write_text(json.dumps({
        "schema": "tessera.nvidia.resources.v1", "source": args.report.name,
        "source_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "rows": rows,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
