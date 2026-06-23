#!/usr/bin/env python3
"""Memory-aware unit-test runner — shard the suite with pytest-xdist, sized to
the box it runs on.

The full ``tests/unit`` suite is a single ~17-minute, single-process run. That's
wall-time, not a memory defect (it fits comfortably in 32 GB run solo). Sharding
with ``pytest-xdist`` (``-n N``) cuts wall-time, but more workers = higher peak
memory, so ``N`` must be sized to BOTH cores and RAM. This picks a sane ``N``
per host:

    N = min(cpu_cores, floor(usable_RAM_GB / per_worker_GB))

so it scales right across very different machines:

    * 32 GB / 10-core Mac           → mem cap ~12, CPU-bound → N = 10
    * 32 GB / 8-core NVIDIA box     → mem cap ~12, CPU-bound → N = 8
    * 128 GB / 16-core AMD box      → mem cap ~51, CPU-bound → N = 16
    * 2 TB / 128-core GPU server    → mem cap huge,  CPU-bound → N = 128

i.e. on normal machines it's CPU-bound (use all cores); the RAM cap only bites
on small-RAM / many-core hosts, preventing an OOM like the one that motivated
this (two ~10k-test runs + a C++ link at once on 32 GB).

Usage:
    python scripts/run_unit_tests.py [--dry-run] [pytest args...]

Knobs (env or flag):
    --per-worker-gb / TESSERA_TEST_MEM_PER_WORKER_GB   per-worker RAM budget (default 2.0)
    --reserve       / TESSERA_TEST_RAM_RESERVE_FRAC    fraction of RAM left for the OS (default 0.2)
    --workers N     / TESSERA_TEST_WORKERS             force N (skip auto-sizing)
    --dry-run                                          print the plan, run nothing

Defaults: targets ``tests/unit`` with ``-m "not slow"`` unless you pass your own
paths / ``-m``. Extra args pass straight through to pytest.

NOTE on isolation: xdist redistributes tests across workers, which changes
execution order. This suite has a few known cross-test global-state
sensitivities; if a sharded run shows a failure that a serial run doesn't,
re-run that test serially before trusting it (it's an isolation artifact, not a
regression). ``--dist loadfile`` keeps each file on one worker and can help.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys


def total_ram_bytes() -> int | None:
    """Total physical RAM, cross-platform. None if undetectable."""
    try:  # Linux + most Unix (works on macOS too on current versions)
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        pass
    try:  # macOS fallback
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
        return int(out.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:  # Linux /proc fallback
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def compute_workers(total_gb: float | None, cores: int,
                    per_worker_gb: float, reserve_frac: float) -> tuple[int, str]:
    """Return (N, why). N>=1. ``why`` notes whether cores or RAM bound it."""
    cores = max(1, cores)
    if total_gb is None:
        return min(cores, 4), "RAM undetectable → conservative min(cores, 4)"
    usable = total_gb * (1.0 - reserve_frac)
    mem_cap = max(1, int(usable // per_worker_gb))
    n = max(1, min(cores, mem_cap))
    why = (f"cores={cores}, RAM={total_gb:.0f}GB → usable {usable:.0f}GB / "
           f"{per_worker_gb:g}GB-per-worker = {mem_cap} mem-cap; "
           f"min(cores, mem-cap) = {n} "
           f"({'CPU-bound' if cores <= mem_cap else 'RAM-bound'})")
    return n, why


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v else default
    except ValueError:
        return default


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(add_help=True, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-worker-gb", type=float,
                    default=_env_float("TESSERA_TEST_MEM_PER_WORKER_GB", 2.0))
    ap.add_argument("--reserve", type=float,
                    default=_env_float("TESSERA_TEST_RAM_RESERVE_FRAC", 0.2))
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("TESSERA_TEST_WORKERS", "0") or 0))
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and exit (run nothing).")
    args, passthrough = ap.parse_known_args(argv)

    cores = os.cpu_count() or 1
    ram = total_ram_bytes()
    total_gb = (ram / 1024 ** 3) if ram else None

    if args.workers > 0:
        n, why = args.workers, f"forced via --workers/TESSERA_TEST_WORKERS = {args.workers}"
    else:
        n, why = compute_workers(total_gb, cores, args.per_worker_gb, args.reserve)

    have_xdist = importlib.util.find_spec("xdist") is not None

    # Default target + marker unless the caller supplied their own.
    pa = list(passthrough)
    has_path = any(not a.startswith("-") for a in pa)
    has_marker = "-m" in pa or any(a.startswith("-m") for a in pa)
    cmd = [sys.executable, "-m", "pytest"]
    if not has_path:
        cmd.append("tests/unit")
    if not has_marker:
        cmd += ["-m", "not slow"]
    cmd += pa

    if have_xdist and n > 1:
        cmd += ["-n", str(n)]
        sched = "sharded"
    else:
        sched = "serial"

    print(f"[run_unit_tests] {sched}: N={n}")
    print(f"[run_unit_tests] sizing: {why}")
    if not have_xdist:
        print("[run_unit_tests] pytest-xdist NOT installed — running serial. "
              "Install it to enable sharding: pip install pytest-xdist")
    print(f"[run_unit_tests] $ {' '.join(cmd)}")
    if args.dry_run:
        return 0

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
