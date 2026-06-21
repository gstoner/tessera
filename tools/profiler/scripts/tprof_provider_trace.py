#!/usr/bin/env python3
"""Normalize vendor profiler records into Tessera provider trace JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_python_path() -> Path:
    return Path(__file__).resolve().parents[3] / "python"


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(_repo_python_path()))
    from tessera.compiler.profiler_provider_trace import (
        build_provider_trace_artifact,
        load_provider_trace_input,
        records_from_raw,
        write_provider_trace_artifact,
    )
    from tessera.compiler.profiler_provider_status import validate_provider_status_artifact

    parser = argparse.ArgumentParser(
        prog="tprof-provider-trace",
        description="Normalize ROCprofiler/CUPTI/Metal records to tessera.profiler_provider_trace.v1.",
    )
    parser.add_argument("--provider", choices=("rocprofiler", "cupti", "metal"), required=True)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Raw record/list JSON or provider trace artifact. Can be repeated.",
    )
    parser.add_argument("--out", help="Write full provider trace artifact JSON.")
    parser.add_argument(
        "--trace-out",
        help="Write only Chrome/Perfetto-compatible Trace Event JSON.",
    )
    parser.add_argument(
        "--provider-status",
        action="append",
        default=[],
        help="tessera.profiler_provider_status.v1 JSON to embed as a sidecar. Can be repeated.",
    )
    args = parser.parse_args(argv)

    try:
        inputs = [load_provider_trace_input(path, provider=args.provider) for path in args.input]
        provider_statuses = []
        for path in args.provider_status:
            status = json.loads(Path(path).read_text())
            validate_provider_status_artifact(status)
            provider_statuses.append(status)
    except Exception as exc:
        parser.error(str(exc))
    records = []
    for payload in inputs:
        records.extend(records_from_raw(args.provider, payload.get("records", [])))
    payload = build_provider_trace_artifact(
        provider=args.provider,
        records=records,
        source_status="file" if len(args.input) == 1 else "file_batch",
        source=",".join(args.input),
        provider_statuses=provider_statuses,
    )

    if args.out:
        write_provider_trace_artifact(payload, args.out)
    if args.trace_out:
        trace = {
            "displayTimeUnit": "ns",
            "traceEvents": payload["traceEvents"],
            "summary": payload["summary"],
        }
        out = Path(args.trace_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    if not args.out and not args.trace_out:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
