#!/usr/bin/env python3
"""Bind gfx1151 clocks, native profiler records, and instrumentation overhead."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(ROOT / "python"))
    from tessera.compiler.profiler_rocm_evidence import build_rocm_profiler_packet

    parser = argparse.ArgumentParser(prog="tprof-rocm-packet")
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--uninstrumented", type=Path, required=True)
    parser.add_argument("--instrumented", type=Path, required=True)
    parser.add_argument("--maximum-overhead", type=float, default=1.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    def load(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    packet = build_rocm_profiler_packet(
        timing=load(args.timing),
        capture=load(args.capture),
        uninstrumented=load(args.uninstrumented),
        instrumented=load(args.instrumented),
        source={"source_commit": commit, "worktree_dirty": dirty},
        maximum_instrumentation_overhead=args.maximum_overhead,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if packet["eligible_for_regression"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
