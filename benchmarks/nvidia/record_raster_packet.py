"""Record a fail-closed SM120 raster-order decision packet.

This is intentionally a diagnostic recorder: it measures the same emitted
fused-GEMM implementation under its carried raster permutations, verifies each
result against the row-major output, and records CUDA-event plus host end-to-end
times separately.  It never changes selector state.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import subprocess
import time
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

_NCU_METRICS = (
    "lts__t_sector_hit_rate.pct",
    "dram__bytes.sum",
)
_ORDERS = ("row_major", "column_major", "grouped_m", "grouped_n")
_GROUPED_ORDERS = {"grouped_m", "grouped_n"}


def _candidate_pairs(orders: list[str] | tuple[str, ...],
                     groups: list[int] | tuple[int, ...]) -> list[tuple[str, int]]:
    return [(order, group)
            for order in orders
            for group in (groups if order in _GROUPED_ORDERS else (1,))]


def _shape(text: str) -> tuple[int, int, int]:
    parts = tuple(int(x) for x in text.split("x"))
    if len(parts) != 3 or min(parts) < 1:
        raise argparse.ArgumentTypeError("shapes must be positive MxNxK")
    return parts


def _ncu_number(text: str) -> float:
    """Parse an Nsight CSV base-unit value without accepting missing data."""
    return float(text.replace(",", "").replace("%", "").strip())


def parse_ncu_counters(text: str) -> dict[str, float]:
    """Return the fused-kernel L2-hit and DRAM counters from an NCU CSV run.

    NCU prepends status lines before its CSV header.  Selecting the header
    rather than assuming byte zero keeps this parser stable across the CUDA
    12/13 NCU formatting change.  The counter child emits exactly one target
    kernel; require both values so an incomplete profile cannot influence a
    selector decision.
    """
    lines = text.splitlines()
    start = next((index for index, line in enumerate(lines)
                  if line.startswith('"ID",') and '"Kernel Name"' in line), None)
    if start is None:
        raise RuntimeError("NCU did not produce a CSV metric table")
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    values: dict[str, list[float]] = {metric: [] for metric in _NCU_METRICS}
    for row in rows:
        if "tessera_nvidia_mma_fused_kernel" not in row.get("Kernel Name", ""):
            continue
        if "Metric Name" in row:
            metric = row.get("Metric Name", "").strip()
            value = row.get("Metric Value", "").strip()
            if metric in values and value:
                values[metric].append(_ncu_number(value))
        else:
            # ``ncu --import ... --page raw --csv`` has one wide row per
            # launch instead of one row per metric.
            for metric in values:
                value = row.get(metric, "").strip()
                if value:
                    values[metric].append(_ncu_number(value))
    missing = [metric for metric, samples in values.items() if not samples]
    if missing:
        raise RuntimeError(f"NCU raster counters missing: {', '.join(missing)}")
    # A single counter child owns one kernel launch.  Median makes the parser
    # robust to a future profiler replay that exposes more than one instance.
    return {
        "l2_sector_hit_rate_pct": statistics.median(values[_NCU_METRICS[0]]),
        "dram_bytes": statistics.median(values[_NCU_METRICS[1]]),
    }


def _ncu_report_counters(report: Path, *, ncu_bin: str) -> dict[str, float]:
    """Read completed NCU evidence instead of profiling a fresh JIT child.

    A report is captured separately with NCU's ``-o`` mode, after the emitted
    image has been made warm.  This avoids treating a stalled cold NVRTC child
    as either counter evidence or a performance result.
    """
    completed = subprocess.run(
        [ncu_bin, "--import", str(report), "--csv", "--page", "raw",
         "--print-units", "base"], check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"NCU raster report import failed ({report}):\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    return parse_ncu_counters(completed.stdout)


def _ncu_report_spec(text: str) -> tuple[tuple[int, int, int], str, int, Path]:
    """Parse ``MxNxK/order/group=/path/to/report.ncu-rep`` exactly once."""
    try:
        key, raw_path = text.split("=", 1)
        raw_shape, order, raw_group = key.split("/", 2)
        group = int(raw_group)
        if order not in _ORDERS or group < 1:
            raise ValueError(f"unknown raster candidate {order!r}/{group}")
        if order not in _GROUPED_ORDERS and group != 1:
            raise ValueError(f"{order} requires group 1")
        return _shape(raw_shape), order, group, Path(raw_path)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "NCU reports must be MxNxK/order/group=/path/report.ncu-rep") from exc


def _selector_decision(rows: list[dict[str, Any]]) -> dict[str, object]:
    """Make a conservative measured decision; absent counter proof retains row-major."""
    by_shape: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_shape.setdefault(tuple(row["shape"]), []).append(row)
    winners: list[str] = []
    improvements: list[float] = []
    for shape_rows in by_shape.values():
        incumbent = next(row for row in shape_rows
                         if row["raster_order"] == "row_major")
        winner = min(shape_rows, key=lambda row: float(row["device_median_ms"]))
        winners.append(f"{winner['raster_order']}:{winner['raster_group']}")
        improvements.append(
            (float(incumbent["device_median_ms"]) - float(winner["device_median_ms"]))
            / float(incumbent["device_median_ms"]))
    consensus = winners[0] if winners and len(set(winners)) == 1 else None
    counter_rows = [row for row in rows if "ncu" in row]
    promote = bool(consensus and not consensus.startswith("row_major:") and
                   min(improvements) >= .03 and counter_rows)
    return {
        "selected": consensus if promote else "row_major:1",
        "changed": promote,
        "reason": ("stable >=3% device-event winner with attached NCU evidence"
                   if promote else
                   "retain row-major: no stable >=3% all-shape winner with NCU evidence"),
        "per_shape_winners": winners,
        "per_shape_improvement_vs_row_major": improvements,
        "ncu_rows": len(counter_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", type=_shape,
                        default=((512, 512, 512), (128, 256, 64), (127, 259, 63)))
    parser.add_argument("--variants", nargs="+", choices=_ORDERS,
                        default=_ORDERS)
    parser.add_argument("--groups", nargs="+", type=int, default=(2, 4, 8))
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--ncu-report", action="append", type=_ncu_report_spec,
                        help="attach completed MxNxK/order/group=report.ncu-rep evidence")
    parser.add_argument("--ncu-unavailable-reason",
                        help="record why exact counter evidence could not be collected")
    parser.add_argument("--ncu-bin", default="/usr/local/cuda/bin/ncu")
    parser.add_argument("--profile-only", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.profile_only and args.output is None:
        parser.error("--output is required unless --profile-only is set")
    if not args.profile_only and (args.runs < 2 or args.reps < 1 or args.warmup < 0):
        raise ValueError("requires at least two runs, positive reps, and nonnegative warmup")
    if min(args.groups) < 1:
        raise ValueError("raster groups must be positive")
    candidates = _candidate_pairs(args.variants, args.groups)
    if args.profile_only and (len(args.shapes) != 1 or len(candidates) != 1):
        raise ValueError("--profile-only requires exactly one shape/order/group candidate")
    reports: dict[tuple[tuple[int, int, int], str, int], Path] = {}
    for shape, order, group, report in args.ncu_report or ():
        key = (shape, order, group)
        if key in reports:
            raise ValueError(f"duplicate NCU report for {shape}/{order}/{group}")
        if not report.is_file():
            raise ValueError(f"NCU report does not exist: {report}")
        reports[key] = report

    from tessera.compiler.emit.nvidia_cuda import _mma_fused_device_fn, _mma_fused_fn, _ptr

    rng = np.random.default_rng(20260730)
    rows: list[dict[str, Any]] = []
    for m, n, k in args.shapes:
        a = np.ascontiguousarray((rng.normal(size=(m, k)) * .1).astype(np.float16))
        b = np.ascontiguousarray((rng.normal(size=(k, n)) * .1).astype(np.float16))
        expected = None
        if not args.profile_only:
            # Keep the oracle independent of the native wrapper, whose output
            # array lifetime is intentionally not part of this recorder's contract.
            expected = (np.asarray(a, np.float32) @ np.asarray(b, np.float32)).copy()
            expected.setflags(write=False)
        for order, group in candidates:
            out = np.empty((m, n), np.float32)
            run = _mma_fused_fn(False, None, "f16", raster_order=order, raster_group=group)
            if run(_ptr(a), _ptr(b), None, _ptr(out), m, n, k) != 1:
                raise RuntimeError(f"{order}/{group} execution failed")
            if args.profile_only:
                continue
            assert expected is not None
            if out.shape != expected.shape:
                raise RuntimeError(
                    f"{order}/{group} changed output shape {out.shape}; "
                    f"oracle remains {expected.shape}"
                )
            if not np.isfinite(out).all():
                raise RuntimeError(f"{order}/{group} produced non-finite output")
            max_error = float(np.max(np.abs(out - expected)))
            tolerance = float(2e-2 + 2e-2 * np.max(np.abs(expected)))
            if max_error > tolerance:
                raise RuntimeError(
                    f"{order}/{group} max error {max_error} exceeds {tolerance}"
                )
            device = _mma_fused_device_fn(False, None, "f16", raster_order=order, raster_group=group)
            device_ms = [float(device(_ptr(a), _ptr(b), None, m, n, k, args.warmup, args.reps))
                         for _ in range(args.runs)]
            e2e_ms: list[float] = []
            for _ in range(args.runs):
                start = time.perf_counter_ns()
                if run(_ptr(a), _ptr(b), None, _ptr(out), m, n, k) != 1:
                    raise RuntimeError(f"{order}/{group} end-to-end execution failed")
                e2e_ms.append((time.perf_counter_ns() - start) / 1e6)
            row: dict[str, Any] = {"shape": [m, n, k], "raster_order": order,
                "raster_group": group, "device_ms": device_ms,
                "device_median_ms": statistics.median(device_ms),
                "end_to_end_ms": e2e_ms,
                "end_to_end_median_ms": statistics.median(e2e_ms),
                "correctness": "matches_f32_oracle", "max_abs_error": max_error}
            report = reports.get(((m, n, k), order, group))
            if report is not None:
                row["ncu"] = _ncu_report_counters(report, ncu_bin=args.ncu_bin)
                row["ncu_report"] = report.name
            rows.append(row)
    if args.profile_only:
        return 0
    assert args.output is not None
    device = subprocess.run([
        "nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version",
        "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip()
    args.output.write_text(json.dumps({"schema": "tessera.nvidia.raster.v3",
        "device": device, "runs": args.runs, "reps": args.reps,
        "warmup": args.warmup, "ncu_kernel_only": bool(reports),
        "ncu_metrics": list(_NCU_METRICS) if reports else [],
        "ncu_unavailable_reason": args.ncu_unavailable_reason,
        "selector_decision": _selector_decision(rows), "rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
