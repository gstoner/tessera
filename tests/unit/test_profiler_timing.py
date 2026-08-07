from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tessera.compiler.profiler_timing import (
    PROFILER_TIMING_SCHEMA_VERSION,
    ProfilerTimingError,
    build_timing_sample,
    measure_synchronized_host_batch,
    measured_clock,
    unavailable_clock,
    validate_timing_sample,
    wall_clock_ticks_to_ns,
)


ROOT = Path(__file__).resolve().parents[2]


def _rocm_clocks() -> dict[str, object]:
    return {
        "host_wall_ns": measured_clock("host_wall_ns", source="steady_clock", value=1200),
        "hip_event_ns": unavailable_clock("hip_event_ns", source="hip_event", reason="HIP_EVENT_UNAVAILABLE"),
        "device_wall_clock_ns": measured_clock(
            "device_wall_clock_ns",
            source="device_wall_clock",
            value=900,
            instrumented=True,
        ),
        "profiler_activity_ns": unavailable_clock(
            "profiler_activity_ns",
            source="rocprofiler_activity",
            reason="ROCPROFILER_UNAVAILABLE",
        ),
    }


def _sample(*, environment: str = "wsl2") -> dict[str, object]:
    return build_timing_sample(
        sample_id="sample-1",
        target="rocm_gfx1151",
        clocks=_rocm_clocks(),
        artifact_digests={"package": "sha256:abc"},
        batch_size=100,
        warm_state="warm",
        synchronization="one hipDeviceSynchronize after batch",
        execution_environment=environment,
    )


def test_rocm_sample_keeps_unavailable_clock_domains_explicit() -> None:
    payload = _sample()
    assert payload["schema"] == PROFILER_TIMING_SCHEMA_VERSION
    assert payload["clocks"]["hip_event_ns"]["value"] is None
    assert payload["clocks"]["hip_event_ns"]["reason"] == "HIP_EVENT_UNAVAILABLE"
    validate_timing_sample(payload)


def test_rejects_host_wall_substitution_into_hip_event_slot() -> None:
    payload = _sample()
    payload["clocks"]["hip_event_ns"]["source"] = "steady_clock"
    payload["clocks"]["hip_event_ns"]["valid"] = True
    payload["clocks"]["hip_event_ns"]["value"] = 1200
    payload["clocks"]["hip_event_ns"]["reason"] = None
    with pytest.raises(ProfilerTimingError, match="cannot contain source"):
        validate_timing_sample(payload)


def test_wsl_and_rtg_dispatch_are_never_promotion_evidence() -> None:
    payload = _sample()
    payload["clocks"]["device_wall_clock_ns"]["eligible_for_promotion"] = True
    payload["clocks"]["device_wall_clock_ns"]["calibrated_against"] = ["hip_event_ns"]
    with pytest.raises(ProfilerTimingError, match="WSL"):
        validate_timing_sample(payload)

    with pytest.raises(ProfilerTimingError, match="never promotion"):
        measured_clock(
            "profiler_activity_ns",
            source="rtg_hsa_dispatch",
            value=500,
            calibrated_against=("device_wall_clock_ns",),
            eligible_for_promotion=True,
        )


def test_tsc_requires_invariant_frequency_and_no_cpu_migration() -> None:
    with pytest.raises(ProfilerTimingError, match="migrated"):
        measured_clock(
            "tsc_cycles",
            source="rdtscp",
            value=100,
            provenance={
                "invariant_tsc": True,
                "logical_cpu_start": 1,
                "logical_cpu_end": 2,
                "calibrated_frequency_hz": 3_000_000_000,
            },
        )
    with pytest.raises(ProfilerTimingError, match="calibrated_frequency_hz"):
        measured_clock(
            "tsc_cycles",
            source="rdtscp",
            value=100,
            provenance={
                "invariant_tsc": True,
                "logical_cpu_start": 1,
                "logical_cpu_end": 1,
            },
        )


def test_device_ticks_and_host_batch_measurement() -> None:
    assert wall_clock_ticks_to_ns(10, 1000) == 10_000
    calls: list[str] = []
    ticks = iter((100, 310))
    result = measure_synchronized_host_batch(
        lambda: calls.append("submit"),
        lambda: calls.append("sync"),
        batch_size=3,
        clock_ns=lambda: next(ticks),
    )
    assert calls == ["submit", "submit", "submit", "sync"]
    assert result == {"raw_batch_ns": 210, "raw_per_launch_ns": 70, "batch_size": 3}


def test_timing_cli_round_trips_valid_sample(tmp_path: Path) -> None:
    source = tmp_path / "sample.json"
    output = tmp_path / "canonical.json"
    source.write_text(json.dumps(_sample()), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/profiler/scripts/tprof_timing.py"),
            str(source),
            "--out",
            str(output),
        ],
        check=True,
    )
    validate_timing_sample(json.loads(output.read_text(encoding="utf-8")))
