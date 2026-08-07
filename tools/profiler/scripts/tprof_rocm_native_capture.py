#!/usr/bin/env python3
"""Capture native ROCprofiler or fresh-process rtg_tracer evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_python_path() -> Path:
    return Path(__file__).resolve().parents[3] / "python"


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(_repo_python_path()))
    from tessera.compiler.profiler_rocm_native import (
        ROCmCaptureRequest,
        collect_rocprofv3,
        collect_rtg_tracer,
    )

    parser = argparse.ArgumentParser(prog="tprof-rocm-native-capture")
    parser.add_argument("--provider", choices=("rocprofiler", "rtg"), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rocprofv3", default="rocprofv3")
    parser.add_argument("--counter", action="append", default=[])
    parser.add_argument("--pc-sampling", action="store_true")
    parser.add_argument("--pc-sampling-method", choices=("host_trap", "stochastic"), default="host_trap")
    parser.add_argument("--pc-sampling-unit", choices=("time", "cycles", "instructions"), default="time")
    parser.add_argument("--pc-sampling-interval", type=int, default=1)
    parser.add_argument("--kernel-include-regex")
    parser.add_argument("--rtg-library", type=Path)
    parser.add_argument("--rtg-converter", type=Path)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--allow-unavailable", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("an application command is required after --")

    if args.provider == "rocprofiler":
        artifact = collect_rocprofv3(ROCmCaptureRequest(
            application=tuple(command),
            output_directory=args.output_directory,
            rocprofv3=args.rocprofv3,
            counters=tuple(args.counter),
            pc_sampling=args.pc_sampling,
            pc_sampling_method=args.pc_sampling_method,
            pc_sampling_unit=args.pc_sampling_unit,
            pc_sampling_interval=args.pc_sampling_interval,
            kernel_include_regex=args.kernel_include_regex,
            timeout_seconds=args.timeout,
        ))
    else:
        if args.rtg_library is None:
            parser.error("--rtg-library is required for the rtg provider")
        artifact = collect_rtg_tracer(
            application=command,
            output_directory=args.output_directory,
            rtg_library=args.rtg_library,
            timeout_seconds=args.timeout,
            converter=args.rtg_converter,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if artifact["status"] == "collected" or args.allow_unavailable:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
