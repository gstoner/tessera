#!/usr/bin/env python3
"""Validate and canonically serialize a Tessera multi-clock timing sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_python_path() -> Path:
    return Path(__file__).resolve().parents[3] / "python"


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(_repo_python_path()))
    from tessera.compiler.profiler_timing import (
        ProfilerTimingError,
        validate_timing_sample,
    )

    parser = argparse.ArgumentParser(
        prog="tprof-timing",
        description=("Validate a tessera.profiler_timing.v1 sample without substituting values between clock domains."),
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        validate_timing_sample(payload)
    except (OSError, json.JSONDecodeError, ProfilerTimingError) as exc:
        parser.error(str(exc))

    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(rendered)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
