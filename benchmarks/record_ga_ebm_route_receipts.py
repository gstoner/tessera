#!/usr/bin/env python3
"""Record which route each GA/EBM composition benchmark's timed span took.

EVIDENCE-PACKET-1, sync ``EVIDENCE-PACKET-1-2026-09-27``. Runs the default
sweeps of ``clifford_core``, ``energy_core`` and ``visual_complex_core`` and
writes their rows -- each carrying ``route`` and ``route_receipts`` from
``tessera._route_receipts`` -- with the recording host and revision.

This is an attribution receipt, not a performance packet: the latencies are
host-wall composition timings and no row is promotion-eligible. What it
establishes is the route per public primitive call on *this* host; a route
observed on one box says nothing about another (a ``rocm`` receipt names the
lane, the ``host`` block names the machine).

    PYTHONPATH=python:. python3 benchmarks/record_ga_ebm_route_receipts.py \\
        --output benchmarks/baselines/ga_ebm_route_receipts_<date>/<host>.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "tessera.ga_ebm_route_receipts.v1"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _host() -> dict[str, object]:
    return {
        "node": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "source_commit": _git("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        # Recorded, never used to label a route: the ROCm lane's chip comes
        # from this host, and an unset variable is recorded as unset.
        "tessera_rocm_chip_env": os.environ.get("TESSERA_ROCM_CHIP"),
        "tessera_build_dir_env": os.environ.get("TESSERA_BUILD_DIR"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=2)
    args = parser.parse_args(argv)

    import benchmarks.clifford_core.core as clifford
    import benchmarks.energy_core.core as energy
    import benchmarks.visual_complex_core.core as visual

    suites = {
        "clifford_core": (clifford.CliffordCoreBenchmark, clifford.default_sweep()),
        "energy_core": (energy.EnergyCoreBenchmark, energy.default_sweep()),
        "visual_complex_core": (visual.VisualComplexCoreBenchmark, visual.default_sweep()),
    }
    rows: dict[str, list[dict[str, object]]] = {}
    incomplete = 0
    for name, (bench_cls, sweep) in suites.items():
        bench = bench_cls(warmup=args.warmup, reps=args.reps)
        rows[name] = [r.to_dict() for r in bench.run(sweep)]
        incomplete += sum(r["route_receipts"]["attribution"] != "complete" for r in rows[name])
    record = {
        "schema": SCHEMA,
        "work_item": "EVIDENCE-PACKET-1",
        "host": _host(),
        "promotion_eligible": False,
        "suites": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    for name, suite_rows in rows.items():
        routes = sorted({str(r["route"]) for r in suite_rows})
        print(f"{name}: routes={routes} devices={sorted({str(r['device']) for r in suite_rows})}")
    # An unattributed row is recorded, and reported as a failure.
    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
