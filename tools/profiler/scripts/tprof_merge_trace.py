#!/usr/bin/env python3
"""Merge Tessera runtime, provider, and context profiler traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_python_path() -> Path:
    return Path(__file__).resolve().parents[3] / "python"


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(_repo_python_path()))
    from tessera.compiler.profiler_trace_merge import (
        load_json,
        merge_profiler_traces,
        write_merged_profiler_trace,
    )

    parser = argparse.ArgumentParser(
        prog="tprof-merge-trace",
        description="Merge runtime trace, provider trace, and context JSON artifacts.",
    )
    parser.add_argument("--runtime-trace", help="Chrome/Perfetto Trace Event JSON from tprof/runtime.")
    parser.add_argument(
        "--provider-trace",
        action="append",
        default=[],
        help="tessera.profiler_provider_trace.v1 JSON. Can be repeated.",
    )
    parser.add_argument("--context-json", help="tessera.profiler_context.v1 JSON.")
    parser.add_argument("--out", required=True, help="Merged Trace Event JSON output.")
    args = parser.parse_args(argv)

    try:
        payload = merge_profiler_traces(
            runtime_trace=load_json(args.runtime_trace) if args.runtime_trace else None,
            provider_traces=[load_json(path) for path in args.provider_trace],
            context_artifact=load_json(args.context_json) if args.context_json else None,
        )
    except Exception as exc:
        parser.error(str(exc))
    write_merged_profiler_trace(payload, args.out)
    if not args.provider_trace and not args.runtime_trace and not args.context_json:
        sys.stderr.write("warning: merged trace contains no input artifacts\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
