#!/usr/bin/env python3
"""Fail-closed preflight for an ISA-preserving gfx1201 phase probe.

The existing in-kernel wall-clock trace changes the folded HSACO. A profiler
can leave that HSACO intact, but profiler availability is not phase or clock
proof. This preflight records exact-host access and explicitly refuses
attribution until PC samples and cross-CU clock calibration are validated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess

from tessera import runtime as rt


def probe(*, kfd: Path = Path("/dev/kfd")) -> dict[str, object]:
    arch = rt._rocm_live_arch()
    if arch != "gfx1201":
        raise RuntimeError(f"gfx1201 profiler preflight requires gfx1201, got {arch!r}")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("gfx1201 profiler preflight requires live HIP")
    from benchmarks.rocm.benchmark_gfx1201_mxfp4_production import (
        _selected_device_name,
    )

    device = _selected_device_name(hip)
    profiler = shutil.which("rocprofv3")
    available = shutil.which("rocprofv3-avail")
    refusal: list[str] = []
    if not profiler or not available:
        refusal.append("rocprofv3_tool_missing")
    if not kfd.exists():
        refusal.append("kfd_device_missing")
    elif not kfd.is_char_device():
        refusal.append("kfd_not_character_device")
    availability: dict[str, object] | None = None
    if not refusal and available:
        result = subprocess.run(
            [available, "list", "--pc-sampling"],
            capture_output=True, text=True, check=False, timeout=15,
        )
        availability = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if result.returncode != 0:
            refusal.append("pc_sampling_capability_query_failed")
        elif device not in result.stdout:
            refusal.append("selected_device_not_pc_sampling_capable")
    return {
        "schema": "tessera.rocm.gfx1201_phase_profiler_preflight.v1",
        "recorder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "host": socket.gethostname(),
        "architecture": arch,
        "device": device,
        "method": "external_rocprofv3_pc_sampling",
        "profiler": profiler,
        "availability_tool": available,
        "kfd_path": str(kfd),
        "availability": availability,
        "isa_preserving_phase_probe_available": not refusal,
        "cross_cu_clock_validated": False,
        "clock_read_cost_validated": False,
        "phase_attribution_admissible": False,
        "promotion_eligible": False,
        "refusal_reasons": refusal or ["pc_samples_and_clock_validation_not_recorded"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps(packet, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
