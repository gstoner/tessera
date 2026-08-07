#!/usr/bin/env python3
"""Normalize the native gfx1151 probe into the shared multi-clock schema."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _repo_python_path() -> Path:
    return Path(__file__).resolve().parents[3] / "python"


def _source_state() -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"source_commit": commit, "worktree_dirty": bool(status.strip())}


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(_repo_python_path()))
    from tessera.compiler.profiler_timing import (
        build_timing_sample,
        measured_clock,
        unavailable_clock,
    )

    parser = argparse.ArgumentParser(prog="tprof-rocm-timing")
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--artifact-digest", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--spin-iterations", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--native-capture",
        type=Path,
        help="Optional tessera.profiler_rocm_native_capture.v1 activity proof.",
    )
    parser.add_argument("--allow-unavailable", action="store_true")
    args = parser.parse_args(argv)

    completed = subprocess.run(
        [
            str(args.probe),
            "--repetitions",
            str(args.repetitions),
            "--spin-iterations",
            str(args.spin_iterations),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        parser.error(f"native ROCm probe did not emit JSON: {exc}")
    if raw.get("schema") != "tessera.rocm_clock_probe.v1":
        parser.error("native ROCm probe emitted an unsupported schema")
    if raw.get("architecture") != "gfx1151" or raw.get("exact_gfx1151") is not True:
        parser.error("native ROCm probe did not prove exact gfx1151")
    source_state = _source_state()
    capture = None
    activity_ns = None
    if args.native_capture is not None:
        from tessera.compiler.profiler_rocm_native import (
            profiler_activity_interval_ns,
            validate_rocm_native_capture,
        )

        capture = json.loads(args.native_capture.read_text(encoding="utf-8"))
        validate_rocm_native_capture(capture)
        activity_ns = profiler_activity_interval_ns(capture)

    host_provenance = {
        "batch_size": raw.get("repetitions"),
        "empty_launch_batch_ns": raw.get("empty_host_wall_ns"),
        "terminal_synchronization": "hipEventSynchronize(stop)",
    }
    host = (
        measured_clock(
            "host_wall_ns",
            source="steady_clock",
            value=raw["host_wall_ns"],
            provenance=host_provenance,
            eligible_for_regression=True,
        )
        if raw.get("host_wall_valid")
        else unavailable_clock(
            "host_wall_ns",
            source="steady_clock",
            reason="HOST_WALL_INVALID",
            provenance=host_provenance,
            raw_value=raw.get("host_wall_ns"),
        )
    )
    event_provenance = {
        "api": "hipEventElapsedTime",
        "empty_launch_batch_ms": raw.get("empty_hip_event_ms"),
    }
    event_ns = float(raw.get("hip_event_ms", 0.0)) * 1_000_000.0
    hip_event = (
        measured_clock(
            "hip_event_ns",
            source="hip_event",
            value=event_ns,
            provenance=event_provenance,
            eligible_for_regression=True,
        )
        if raw.get("hip_event_valid")
        else unavailable_clock(
            "hip_event_ns",
            source="hip_event",
            reason="HIP_EVENT_INVALID",
            provenance=event_provenance,
            raw_value=event_ns,
        )
    )
    device_provenance = {
        "builtin": "wall_clock64",
        "raw_ticks": raw.get("device_wall_clock_ticks"),
        "wall_clock_rate_khz": raw.get("wall_clock_rate_khz"),
        "grid_blocks": raw.get("blocks"),
        "block_threads": raw.get("threads"),
        "architecture": raw.get("architecture"),
    }
    calibrated_against = []
    if raw.get("hip_event_valid"):
        calibrated_against.append("hip_event_ns")
    if activity_ns is not None:
        calibrated_against.append("profiler_activity_ns")
    promotion = bool(
        not raw.get("wsl")
        and calibrated_against
        and not source_state["worktree_dirty"]
    )
    device_clock = (
        measured_clock(
            "device_wall_clock_ns",
            source="device_wall_clock",
            value=raw["device_wall_clock_ns"],
            provenance=device_provenance,
            instrumented=True,
            calibrated_against=calibrated_against,
            eligible_for_regression=True,
            eligible_for_promotion=promotion,
        )
        if raw.get("device_wall_clock_valid")
        else unavailable_clock(
            "device_wall_clock_ns",
            source="device_wall_clock",
            reason="DEVICE_WALL_CLOCK_INVALID",
            provenance=device_provenance,
            raw_value=raw.get("device_wall_clock_ticks"),
        )
    )
    profiler_activity = (
        measured_clock(
            "profiler_activity_ns",
            source="rocprofiler_activity",
            value=activity_ns,
            provenance={
                "provider": "rocprofiler",
                "capture_sha256": __import__("hashlib").sha256(
                    args.native_capture.read_bytes()
                ).hexdigest(),
                "dispatch_activity_seen": capture["proof"]["dispatch_activity_seen"],
            },
            eligible_for_regression=True,
            eligible_for_promotion=promotion,
            calibrated_against=("device_wall_clock_ns",),
        )
        if activity_ns is not None
        else unavailable_clock(
            "profiler_activity_ns",
            source="rocprofiler_activity",
            reason="ROCPROFILER_ACTIVITY_NOT_COLLECTED",
            provenance={"provider": "native_clock_probe"},
        )
    )
    sample = build_timing_sample(
        sample_id=args.sample_id,
        target="rocm_gfx1151",
        clocks={
            "host_wall_ns": host,
            "hip_event_ns": hip_event,
            "device_wall_clock_ns": device_clock,
            "profiler_activity_ns": profiler_activity,
        },
        artifact_digests={"instrumented_probe": args.artifact_digest},
        batch_size=args.repetitions,
        warm_state="warm",
        synchronization="one terminal hipEventSynchronize after asynchronous batch",
        execution_environment="wsl2" if raw.get("wsl") else "bare_metal",
        resources={
            "blocks": raw.get("blocks"),
            "threads": raw.get("threads"),
            "instrumented": True,
        },
        environment={
            "architecture": raw.get("architecture"),
            "device_name": raw.get("device_name"),
            "clock_agreement_valid": raw.get("clock_agreement_valid"),
            "probe_returncode": completed.returncode,
            "probe_error": raw.get("error"),
            "native_capture": str(args.native_capture) if args.native_capture else None,
            **source_state,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sample, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if raw.get("eligible_for_regression") or args.allow_unavailable:
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
