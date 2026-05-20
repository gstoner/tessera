# SPDX-License-Identifier: MIT
"""Spectral correctness proof lane (Phase A3, 2026-05-20).

Wraps the C++ ``ts-spectral-correctness`` binary built by the spectral
solver (``src/solvers/spectral/benchmarks/correctness_microbench.cpp``)
in a Python harness that:

* Compiles a standalone build of the binary if one is not provided
  via ``--bin`` (no MLIR / LLVM build dependency — the binary is a
  self-contained ~150-line C++ correctness test).
* Runs it and scrapes the ``key=value`` output into the standard
  Tessera benchmark JSON schema.
* Emits a single JSON row per FFT size with ``execution_kind="proof"``
  and a pass / fail verdict per size.
* Exits non-zero if any size deviates more than the C++ tolerance
  (the binary's own exit code is honored).

Usage:

  PYTHONPATH=python python benchmarks/spectral/spectral_correctness.py \
      --output /tmp/tessera_spectral_correctness.json

  # With a pre-built binary (skip the build step):
  PYTHONPATH=python python benchmarks/spectral/spectral_correctness.py \
      --bin /path/to/ts-spectral-correctness \
      --output /tmp/tessera_spectral_correctness.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MICROBENCH_CPP = (
    REPO_ROOT
    / "src" / "solvers" / "spectral" / "benchmarks" / "correctness_microbench.cpp"
)


def build_microbench(workdir: Path) -> Path:
    """Build the C++ correctness binary in a tempdir with no MLIR deps.

    The microbench is intentionally self-contained (just <complex>,
    <random>, <vector>, <iostream> from the standard library) so we
    can build it without spinning up the full LLVM/MLIR toolchain.
    Returns the absolute path to the built binary.
    """
    if not MICROBENCH_CPP.exists():
        raise FileNotFoundError(
            f"correctness_microbench source missing at {MICROBENCH_CPP}"
        )
    out = workdir / "ts-spectral-correctness"
    cxx = os.environ.get("CXX", "c++")
    cmd = [
        cxx, "-std=c++17", "-O2", "-Wall", "-Wextra",
        "-o", str(out), str(MICROBENCH_CPP),
    ]
    subprocess.check_call(cmd)
    return out


def parse_microbench_output(text: str) -> dict[str, Any]:
    """Parse ``key=value`` lines emitted by the C++ binary into a dict.

    Per-size lines are collected into ``rows`` (list of dicts); other
    lines become top-level metadata keys.
    """
    meta: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    pair_re = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        pairs = pair_re.findall(line)
        if not pairs:
            continue
        kv = {k: v for k, v in pairs}
        if "size" in kv:
            row: dict[str, Any] = {}
            for k, v in kv.items():
                if k in {"size", "pass"}:
                    row[k] = int(v)
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
            rows.append(row)
        else:
            meta.update(kv)
    return {"metadata": meta, "rows": rows}


def make_benchmark_row(size_row: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """Map one size's correctness numbers into the standard benchmark
    row schema (matches what ``benchmarks/spectral/spectral_bench.py``
    writes, with ``execution_kind="proof"`` instead of ``reference``)."""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "op": "fft1d_correctness",
        "device": "cpu",
        "backend": "tessera-correctness",
        "dtype": meta.get("dtype", "complex64"),
        "shape": str(size_row["size"]),
        "batch": 1,
        "repeats": 1,
        "time_ms": 0.0,
        "gflops": 0.0,
        "gbs": 0.0,
        "ai": 0.0,
        "bytes": 0,
        "flops": 0,
        "err_rel": size_row["max_abs_err"],
        "compiler_path": "tessera-spectral",
        "runtime_status": "ready",
        "execution_kind": "proof",
        "artifact_hash": "",
        "reason": (
            f"max_abs_err={size_row['max_abs_err']:.3e} "
            f"mean_abs_err={size_row['mean_abs_err']:.3e} "
            f"rms={size_row['rms_err']:.3e} "
            f"tol={meta.get('abs_tol', 'unknown')}"
        ),
        "pass": bool(size_row["pass"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Spectral correctness proof lane — Stockham vs naive DFT, "
            "exits non-zero on correctness regression."
        )
    )
    ap.add_argument(
        "--bin",
        default=None,
        help=(
            "Path to a pre-built ``ts-spectral-correctness`` binary. "
            "If omitted, the script builds the microbench in a tempdir "
            "via $CXX (default ``c++``)."
        ),
    )
    ap.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path.  Default: write to stdout (no file)."
        ),
    )
    ap.add_argument(
        "--no-build",
        action="store_true",
        help="Do not attempt to build if --bin is missing; fail instead.",
    )
    args = ap.parse_args()

    if args.bin:
        binary = Path(args.bin).resolve()
        if not binary.exists():
            print(f"FAIL: --bin {binary} does not exist", file=sys.stderr)
            return 2
        cleanup: Path | None = None
    elif args.no_build:
        print("FAIL: --no-build set but --bin not provided", file=sys.stderr)
        return 2
    else:
        if not shutil.which(os.environ.get("CXX", "c++")):
            print(
                "FAIL: no C++ compiler on PATH; pass --bin or set $CXX",
                file=sys.stderr,
            )
            return 2
        cleanup = Path(tempfile.mkdtemp(prefix="tessera_spectral_correctness_"))
        binary = build_microbench(cleanup)

    try:
        run = subprocess.run(
            [str(binary)],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        # Don't reraise here — let the binary's exit code propagate.
        pass

    parsed = parse_microbench_output(run.stdout)
    rows = [make_benchmark_row(r, parsed["metadata"]) for r in parsed["rows"]]
    verdict = parsed["metadata"].get("verdict", "fail")
    payload = {
        "schema": "tessera.benchmark.v1",
        "lane": "spectral_correctness_proof",
        "verdict": verdict,
        "binary_exit_code": run.returncode,
        "metadata": parsed["metadata"],
        "rows": rows,
    }
    out_text = json.dumps(payload, indent=2, default=str)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out_text + "\n", encoding="utf-8")
        print(f"wrote {args.output} ({len(rows)} rows; verdict={verdict})")
    else:
        sys.stdout.write(out_text + "\n")

    if cleanup is not None:
        shutil.rmtree(cleanup, ignore_errors=True)

    return run.returncode


if __name__ == "__main__":
    sys.exit(main())
