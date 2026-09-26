#!/usr/bin/env python3
"""Nine prespecified SSD process pairs, each with its own CUDA calibration.

Qualification is checked before measurement. --diagnostic permits dirty/WSL
runs but cannot alter the admission policy or turn them into promotion evidence.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from tessera.compiler.profiler_cuda_window import (
    build_cuda_window_calibration,
    read_nsys_device,
    read_nsys_kernels,
    capture_digest,
)
from tessera.compiler.ssd_performance import summarize


def require_eligible_host(source):
    reasons = []
    if source["worktree_dirty"]:
        reasons.append("source tree has uncommitted changes")
    # WSL2 is admissible since 2026-09-25 (owner, MASTER_AUDIT): the
    # calibration's activity-window / event agreement gate decides, not the
    # host environment.
    if source["execution_environment"] not in ("bare_metal", "wsl2"):
        reasons.append(f"unknown execution environment {source['execution_environment']!r}")
    if reasons:
        raise ValueError("; ".join(reasons))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compiler", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--shape", type=int, nargs=4, default=(512, 2, 32, 8))
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--diagnostic", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]

    def git(*cmd):
        return subprocess.check_output(["git", *cmd], cwd=root, text=True).strip()

    source = dict(
        source_commit=git("rev-parse", "HEAD"),
        worktree_dirty=bool(git("status", "--porcelain")),
        execution_environment="wsl2" if "microsoft" in platform.release().lower() else "bare_metal",
    )
    if not args.diagnostic:
        require_eligible_host(source)
    destination = args.output_dir.resolve()
    if destination.is_relative_to(root):
        raise ValueError("write measurement evidence outside the source checkout")
    destination.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "PYTHONPATH": str(root / "python")}

    def run(command):
        subprocess.run(command, cwd=root, env=env, check=True, timeout=600)

    base = [
        sys.executable,
        str(root / "benchmarks/record_ssd_gpu.py"),
        "--backend",
        "nvidia",
        "--compiler",
        str(args.compiler.resolve()),
        "--shape",
        *map(str, args.shape),
        "--chunk",
        str(args.chunk),
        "--profile",
    ]
    pairs, calibrations = [], []
    for index in range(9):
        pair, calibrated = {}, {}
        names = ("serial", "cooperative") if index % 2 == 0 else ("cooperative", "serial")
        for name in names:
            prefix = destination / f"{index}-{name}"
            clean, profiled = Path(str(prefix) + "-clean.json"), Path(str(prefix) + "-profiled.json")
            command = base + (["--cooperative"] if name == "cooperative" else [])
            run(command + ["--output", str(clean)])
            run(
                [
                    "nsys",
                    "profile",
                    "--trace=cuda",
                    "--sample=none",
                    "--cpuctxsw=none",
                    "-o",
                    str(prefix),
                    *command,
                    "--output",
                    str(profiled),
                ]
            )
            database = Path(str(prefix) + ".sqlite")
            run(["nsys", "export", "--type", "sqlite", "--output", str(database), str(prefix) + ".nsys-rep"])
            pair[name] = json.loads(clean.read_text())
            calibrated[name] = build_cuda_window_calibration(
                clean=pair[name],
                profiled=json.loads(profiled.read_text()),
                kernels=read_nsys_kernels(database),
                capture_device=read_nsys_device(database),
                source=source,
                sample_id=prefix.name,
                capture_sha256=capture_digest(database),
            )
        pairs.append(pair)
        calibrations.extend(calibrated[name] for name in ("serial", "cooperative"))
    # Fail if the source changed during measurement rather than misattribute it.
    if (
        git("rev-parse", "HEAD") != source["source_commit"]
        or bool(git("status", "--porcelain")) != source["worktree_dirty"]
    ):
        raise ValueError("source state changed while collecting evidence")
    report = dict(pairs=pairs, calibrations=calibrations, summary=summarize(pairs))
    (destination / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
