#!/usr/bin/env python3
"""Collect native rocprofv3 PMCs for one side of a ROCM-6 A/B experiment."""

from __future__ import annotations

import argparse
import json

from tessera.compiler.rocm_profiler_experiment import collect_native_counters


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=("G6-A", "G6-B", "G6-C"))
    p.add_argument("variant", choices=("production", "candidate"))
    p.add_argument(
        "--native-counters", action="store_true",
        help="enable bare-metal rocprofv3 PMCs (unsupported under WSL)")
    p.add_argument("--counter", action="append", default=[],
                   help="native rocprofv3 PMC name; repeat for multiple PMCs")
    p.add_argument("--output-directory", required=True)
    p.add_argument("application", nargs=argparse.REMAINDER,
                   help="application command, after --")
    ns = p.parse_args()
    app = ns.application[1:] if ns.application[:1] == ["--"] else ns.application
    run = collect_native_counters(
        ns.experiment, ns.variant, app, counters=ns.counter,
        output_directory=ns.output_directory, enabled=ns.native_counters)
    print(json.dumps(run.as_metadata_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
