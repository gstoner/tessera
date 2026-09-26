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
    promotion_clock_slots,
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


def _witnessed_wsl_sample(*, device_ns: float = 900, event_ns: float = 920,
                          target: str = "rocm_gfx1151") -> dict[str, object]:
    clocks = _rocm_clocks()
    clocks["hip_event_ns"] = measured_clock("hip_event_ns", source="hip_event", value=event_ns)
    clocks["device_wall_clock_ns"] = measured_clock(
        "device_wall_clock_ns",
        source="device_wall_clock",
        value=device_ns,
        instrumented=True,
        calibrated_against=("hip_event_ns",),
        eligible_for_promotion=True,
    )
    return build_timing_sample(
        sample_id=f"sample-wsl-{device_ns}-{event_ns}",
        target=target,
        clocks=clocks,
        artifact_digests={"package": "sha256:abc"},
        batch_size=100,
        warm_state="warm",
        synchronization="hipEventSynchronize",
        execution_environment="wsl2",
    )


def test_wsl_device_clock_with_an_agreeing_in_sample_witness_is_admissible() -> None:
    """Owner direction 2026-09-25: the wall_clock64 method is performance
    evidence without KFD or bare metal (DEVICE-CLOCK-DISCIPLINE-2026-08-31)."""
    payload = _witnessed_wsl_sample(device_ns=900, event_ns=920)  # 2.2% apart
    assert payload["clocks"]["device_wall_clock_ns"]["eligible_for_promotion"] is True
    validate_timing_sample(payload)


def test_a_valid_witness_that_disagrees_is_refused() -> None:
    """Review of #854: 1 ns against a valid 1 s event must not pass."""
    with pytest.raises(ProfilerTimingError, match="disagree beyond 5%"):
        _witnessed_wsl_sample(device_ns=1, event_ns=1_000_000_000)
    # The band is the providers' relation, |witness - clock| / witness <= 5%.
    _witnessed_wsl_sample(device_ns=950, event_ns=1000)
    with pytest.raises(ProfilerTimingError, match="disagree"):
        _witnessed_wsl_sample(device_ns=949, event_ns=1000)


def test_wsl_promotion_still_needs_a_kernel_side_clock() -> None:
    payload = _witnessed_wsl_sample()
    payload["clocks"]["hip_event_ns"]["eligible_for_promotion"] = True
    payload["clocks"]["hip_event_ns"]["calibrated_against"] = ["device_wall_clock_ns"]
    with pytest.raises(ProfilerTimingError, match="only a kernel-side clock"):
        validate_timing_sample(payload)


def test_wsl_witness_must_be_valid_in_the_same_sample() -> None:
    payload = _witnessed_wsl_sample()
    payload["clocks"]["hip_event_ns"].update(
        valid=False, value=None, reason="HIP_EVENT_ZERO_DURATION",
        eligible_for_regression=False)
    with pytest.raises(ProfilerTimingError, match="no independent witness"):
        validate_timing_sample(payload)


def test_promotion_clocks_are_target_specific() -> None:
    """Review of #854: a ROCm sample cannot promote through an appended TSC,
    and a target without a kernel-side slot (NVIDIA today) has none."""
    assert promotion_clock_slots("rocm_gfx1151") == {"device_wall_clock_ns"}
    assert promotion_clock_slots("x86") == {"tsc_cycles"}
    assert promotion_clock_slots("nvidia_sm120") == frozenset()

    payload = _witnessed_wsl_sample()
    payload["clocks"]["device_wall_clock_ns"]["eligible_for_promotion"] = False
    payload["clocks"]["monotonic_raw_ns"] = measured_clock(
        "monotonic_raw_ns", source="clock_monotonic_raw", value=1000).to_dict()
    payload["clocks"]["tsc_cycles"] = measured_clock(
        "tsc_cycles", source="rdtscp", value=4000,
        provenance={"invariant_tsc": True, "logical_cpu_start": 0,
                    "logical_cpu_end": 0, "calibrated_frequency_hz": 4.0e9},
        calibrated_against=("monotonic_raw_ns",), eligible_for_promotion=True).to_dict()
    with pytest.raises(ProfilerTimingError, match="kernel-side clock of target 'rocm_gfx1151'"):
        validate_timing_sample(payload)


def _x86_wsl_sample(frequency_source: str | None) -> dict[str, object]:
    provenance = {"invariant_tsc": True, "logical_cpu_start": 3,
                  "logical_cpu_end": 3, "calibrated_frequency_hz": 4.0e9}
    if frequency_source is not None:
        provenance["frequency_source"] = frequency_source
    clocks = {
        "host_wall_ns": measured_clock("host_wall_ns", source="steady_clock", value=1010),
        "monotonic_raw_ns": measured_clock("monotonic_raw_ns", source="clock_monotonic_raw", value=1000),
        "tsc_cycles": measured_clock(
            "tsc_cycles", source="rdtscp", value=4000, provenance=provenance,
            calibrated_against=("monotonic_raw_ns",), eligible_for_promotion=True),
        "perf_task_clock_ns": unavailable_clock(
            "perf_task_clock_ns", source="perf_event_task_clock", reason="PERF_EVENT_DENIED"),
    }
    return build_timing_sample(
        sample_id="x86-wsl", target="x86", clocks=clocks,
        artifact_digests={"package": "sha256:abc"}, batch_size=10, warm_state="warm",
        synchronization="none", execution_environment="wsl2")


def test_x86_tsc_needs_an_independently_sourced_frequency() -> None:
    """Review: a TSC frequency derived from tsc/raw over the checked interval
    makes TSC-vs-raw agree by construction, so it cannot be the witness."""
    with pytest.raises(ProfilerTimingError, match="calibrated_frequency_hz from"):
        _x86_wsl_sample(frequency_source=None)
    with pytest.raises(ProfilerTimingError, match="calibrated_frequency_hz from"):
        _x86_wsl_sample(frequency_source="tsc_over_raw_same_interval")
    _x86_wsl_sample(frequency_source="cpuid_leaf_0x15")


def test_host_wall_is_never_a_witness_and_one_disagreeing_witness_refuses() -> None:
    """Review demo: device 10,000 ns, HIP event 20,000 (50% off), host wall
    10,100. Host wall agreeing must not rescue a disagreeing HIP event."""
    clocks = _rocm_clocks()
    clocks["host_wall_ns"] = measured_clock("host_wall_ns", source="steady_clock", value=10_100)
    clocks["hip_event_ns"] = measured_clock("hip_event_ns", source="hip_event", value=20_000)
    clocks["device_wall_clock_ns"] = measured_clock(
        "device_wall_clock_ns", source="device_wall_clock", value=10_000, instrumented=True,
        calibrated_against=("hip_event_ns", "host_wall_ns"), eligible_for_promotion=True)
    with pytest.raises(ProfilerTimingError, match="witnesses disagree"):
        build_timing_sample(
            sample_id="demo", target="rocm_gfx1151", clocks=clocks,
            artifact_digests={"package": "sha256:abc"}, batch_size=1, warm_state="warm",
            synchronization="hipEventSynchronize", execution_environment="wsl2")
    # Host wall alone is not a witness either.
    clocks["hip_event_ns"] = unavailable_clock("hip_event_ns", source="hip_event", reason="ZERO")
    with pytest.raises(ProfilerTimingError, match="no admissible witness"):
        build_timing_sample(
            sample_id="demo2", target="rocm_gfx1151", clocks=clocks,
            artifact_digests={"package": "sha256:abc"}, batch_size=1, warm_state="warm",
            synchronization="hipEventSynchronize", execution_environment="wsl2")


def test_wsl_is_matched_exactly_not_by_substring() -> None:
    from tessera.compiler.profiler_timing import is_wsl_environment
    assert is_wsl_environment("wsl2") and is_wsl_environment("WSL")
    assert not is_wsl_environment("bare_metal_not_wsl")
    assert not is_wsl_environment("bare_metal")
    # ...and an unrecognized environment gets no promotion at all.
    payload = _witnessed_wsl_sample()
    payload["execution_environment"] = "bare_metal_not_wsl"
    with pytest.raises(ProfilerTimingError, match="unknown execution environment"):
        validate_timing_sample(payload)


def test_bare_metal_rules_are_unchanged_by_the_wsl_admission() -> None:
    payload = _sample(environment="bare_metal")
    payload["clocks"]["device_wall_clock_ns"]["eligible_for_promotion"] = True
    payload["clocks"]["device_wall_clock_ns"]["calibrated_against"] = ["hip_event_ns"]
    validate_timing_sample(payload)
