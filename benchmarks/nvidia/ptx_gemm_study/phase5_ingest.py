#!/usr/bin/env python3
"""Create a fail-closed Phase-5 proposal from a completed study packet.

This tool deliberately does *not* edit ``target_perf.py``, ``mma_selector.py``,
or ``capabilities.py``.  A study needs clean event rows *and* matching NCU rows
before it may even propose a calibration.  Further, WSL2 is proposal-only under
``target_perf.apply_corpus``: its measurements cannot become selector authority.
The generated JSON states precisely what a later bare-metal rerun may promote.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from check_consistency import check

REQUIRED = ("cublaslt_fp16", "int8_ptx_mma_k32", "int4_ptx_mma_k64")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def proposal(results: list[dict], counters: list[dict], capability: dict) -> dict:
    errors = check(results, counters)
    if errors:
        raise ValueError("Phase-4 consistency gate is red: " + "; ".join(errors))
    ok = [row for row in results if row.get("status") == "OK"]
    by_kernel: dict[str, list[dict]] = {}
    for row in ok:
        by_kernel.setdefault(row["kernel"], []).append(row)
    missing = [name for name in REQUIRED if not by_kernel.get(name)]
    if missing:
        raise ValueError("required validated candidates absent: " + ", ".join(missing))
    counter_keys = {(row.get("kernel"), row.get("dtype"), row.get("n"))
                    for row in counters if row.get("ncu_duration_ms") is not None}
    for name in REQUIRED:
        for row in by_kernel[name]:
            key = (name, row.get("dtype"), row.get("n"))
            if key not in counter_keys:
                raise ValueError(f"missing NCU evidence for {key}")
    host = platform.platform()
    wsl = "microsoft" in platform.release().lower() or "wsl" in host.lower()
    return {
        "kind": "tessera_nvidia_mma_study_proposal",
        "version": 1,
        "device": capability["device"],
        "compute_capability": capability["cc"],
        "selector_eligible": not wsl,
        "promotion": "blocked_on_bare_metal" if wsl else "review_required",
        "reason": ("WSL2 measurements are not selector authority; rerun the "
                   "same green packet bare metal before calling target_perf.apply_corpus()."
                   if wsl else "Review the complete green packet before producing a calibration corpus."),
        "validated_candidates": {
            name: [{"n": row["n"], "dtype": row["dtype"],
                    "latency_ms": row["latency_ms"], "tflops": row["tflops"]}
                   for row in by_kernel[name]]
            for name in REQUIRED
        },
        "capability_probe": capability["variants"],
        "registry_actions": {
            "target_perf": "no static patch; a measured corpus is permitted only after bare-metal evidence",
            "mma_selector": "no selector change; choose only after the evidence becomes selector_eligible",
            "capabilities": "no status promotion; the probe/benchmark package is evidence, not a production ABI expansion",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("counters", type=Path)
    parser.add_argument("capability", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        value = proposal(_rows(args.results), _rows(args.counters),
                         json.loads(args.capability.read_text()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Phase 5 BLOCKED: {exc}", file=sys.stderr)
        return 1
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(f"Phase 5 proposal -> {args.output} ({value['promotion']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
