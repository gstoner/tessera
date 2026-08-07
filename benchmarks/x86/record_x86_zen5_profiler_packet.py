#!/usr/bin/env python3
"""Record the E2E-REAL-4 matmul comparison with Zen 5 profiler evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]


def _load_benchmark_module():
    path = ROOT / "benchmarks/x86/benchmark_x86_e2e_real_matmul.py"
    spec = importlib.util.spec_from_file_location("x86_e2e_real4_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load x86 E2E-REAL-4 benchmark")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cpu_identity() -> dict[str, object]:
    fields: dict[str, str] = {}
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() not in fields:
            fields[key.strip()] = value.strip()
    return {
        "vendor_id": fields.get("vendor_id", "unknown"),
        "cpu_family": int(fields["cpu family"]) if fields.get("cpu family", "").isdigit() else None,
        "model": int(fields["model"]) if fields.get("model", "").isdigit() else None,
        "model_name": fields.get("model name", "unknown"),
        "microcode": fields.get("microcode", "unknown"),
        "flags": sorted(fields.get("flags", "").split()),
    }


def _environment() -> dict[str, object]:
    version = Path("/proc/version").read_text(encoding="utf-8")
    lower = version.lower()
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    worktree_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "host": platform.platform(),
        "kernel": platform.release(),
        "wsl": "microsoft" in lower or "wsl" in lower,
        "virtualized": "microsoft" in lower or "wsl" in lower,
        "affinity": sorted(__import__("os").sched_getaffinity(0)),
        "source_commit": source_commit,
        "worktree_dirty": bool(worktree_status.strip()),
    }


def main() -> int:
    from tessera.compiler.profiler_x86_evidence import build_x86_profiler_packet

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tprof", type=Path, required=True)
    parser.add_argument("--sampling", type=Path)
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    benchmark_module = _load_benchmark_module()
    benchmark = benchmark_module.run(trials=args.trials, warmup=args.warmup)
    canonical = json.dumps(benchmark, sort_keys=True, separators=(",", ":"))
    benchmark["report_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    timing_process = subprocess.run(
        [str(args.tprof), "x86", "timing-status"],
        check=True,
        capture_output=True,
        text=True,
    )
    timing_status = json.loads(timing_process.stdout)
    sampling = json.loads(args.sampling.read_text(encoding="utf-8")) if args.sampling is not None else None
    packet = build_x86_profiler_packet(
        benchmark=benchmark,
        timing_status=timing_status,
        cpu=_cpu_identity(),
        environment=_environment(),
        sampling=sampling,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 1 if packet["verdict"] == "reject" else 0


if __name__ == "__main__":
    raise SystemExit(main())
