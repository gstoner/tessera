#!/usr/bin/env python3
"""Emit an honest x86 multi-clock/perf provider proof snapshot."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path


def _collect_provider_status(proof: dict[str, object]) -> dict[str, object]:
    module_path = Path(__file__).resolve().parents[3] / "python/tessera/compiler/profiler_provider_status.py"
    spec = importlib.util.spec_from_file_location("tessera_profiler_provider_status", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load profiler provider status module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.collect_provider_status("x86", native_proof=proof)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tprof-x86-native-smoke")
    parser.add_argument("--tprof", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-unavailable", action="store_true")
    args = parser.parse_args(argv)

    proof: dict[str, object]
    try:
        completed = subprocess.run(
            [str(args.tprof), "x86", "timing-status"],
            check=True,
            capture_output=True,
            text=True,
        )
        proof = json.loads(completed.stdout)
        proof["timing_sample_valid"] = bool(
            proof.get("monotonic_raw_valid")
            and proof.get("rdtscp_valid")
            and proof.get("affinity_stable")
            and proof.get("clock_agreement_valid")
            and proof.get("perf_sample_valid")
            and float(proof.get("calibrated_tsc_hz", 0)) > 0
        )
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        proof = {"probe_error": str(exc), "timing_sample_valid": False}

    payload = _collect_provider_status(proof)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload["status"] == "native_available" or args.allow_unavailable:
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
