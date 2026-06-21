#!/usr/bin/env python3
"""Print profiler provider status JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_python_path() -> Path:
    return Path(__file__).resolve().parents[3] / "python"


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(_repo_python_path()))
    from tessera.compiler.profiler_provider_status import collect_provider_status

    parser = argparse.ArgumentParser(
        prog="tprof-provider-status",
        description="Report Tessera profiler provider readiness.",
    )
    parser.add_argument("--provider", choices=("apple", "rocm", "nvidia", "cpu"), required=True)
    parser.add_argument("--format", choices=("json",), default="json")
    args = parser.parse_args(argv)

    payload = collect_provider_status(args.provider)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
