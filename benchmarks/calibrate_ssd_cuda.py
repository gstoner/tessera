#!/usr/bin/env python3
"""Bind existing clean/profiled SSD measurements to an exported Nsight capture."""

import argparse
import json
from pathlib import Path
from tessera.compiler.profiler_cuda_window import (
    build_cuda_window_calibration,
    read_nsys_kernels,
    read_nsys_device,
    capture_digest,
)


def main():
    p = argparse.ArgumentParser()
    for name in ("clean", "profiled", "sqlite", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--dirty", action="store_true")
    p.add_argument("--environment", choices=["bare_metal", "wsl2"], required=True)
    args = p.parse_args()
    result = build_cuda_window_calibration(
        clean=json.loads(args.clean.read_text()),
        profiled=json.loads(args.profiled.read_text()),
        kernels=read_nsys_kernels(args.sqlite),
        capture_device=read_nsys_device(args.sqlite),
        source=dict(source_commit=args.commit, worktree_dirty=args.dirty, execution_environment=args.environment),
        capture_sha256=capture_digest(args.sqlite),
        sample_id=args.sqlite.stem,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
