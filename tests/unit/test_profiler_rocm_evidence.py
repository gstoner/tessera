from __future__ import annotations

from tessera.compiler.profiler_provider_trace import build_provider_trace_artifact
from tessera.compiler.profiler_rocm_evidence import (
    build_rocm_profiler_packet,
    validate_rocm_profiler_packet,
)
from tessera.compiler.profiler_timing import build_timing_sample, measured_clock


def _timing(environment: str = "bare_metal") -> dict:
    return build_timing_sample(
        sample_id="gfx1151-calibration",
        target="rocm_gfx1151",
        clocks={
            "host_wall_ns": measured_clock("host_wall_ns", source="steady_clock", value=12_000),
            "hip_event_ns": measured_clock("hip_event_ns", source="hip_event", value=10_100),
            "device_wall_clock_ns": measured_clock(
                "device_wall_clock_ns", source="device_wall_clock", value=10_000,
                instrumented=True, calibrated_against=("hip_event_ns",),
                eligible_for_promotion=environment == "bare_metal",
            ),
            "profiler_activity_ns": measured_clock(
                "profiler_activity_ns", source="rocprofiler_activity", value=10_050,
                calibrated_against=("device_wall_clock_ns",),
                eligible_for_promotion=environment == "bare_metal",
            ),
        },
        artifact_digests={"probe": "abc"},
        batch_size=10,
        warm_state="warm",
        synchronization="terminal sync",
        execution_environment=environment,
    )


def _capture() -> dict:
    trace = build_provider_trace_artifact(provider="rocprofiler", records=(), source_status="native")
    return {
        "schema": "tessera.profiler_rocm_native_capture.v1",
        "provider": "rocprofiler",
        "status": "collected",
        "reason": None,
        "fresh_process": True,
        "process": {"clean_exit": True},
        "proof": {
            "dispatch_activity_seen": True,
            "hip_callback_seen": True,
            "counter_records_seen": True,
            "pc_samples_seen": True,
        },
        "requested": {"counters": ["SQ_WAVES"], "pc_sampling": True},
        "provider_trace": trace,
        "eligible_for_promotion": False,
    }


def _image(duration: int, suffix: str) -> dict:
    return {
        "architecture": "gfx1151",
        "kernel_name": "tessera_application_kernel",
        "semantic_sha256": "semantic-artifact",
        "image_sha256": "image-" + suffix,
        "isa_sha256": "isa-" + suffix,
        "duration_ns": duration,
        "clock_source": "hip_event",
        "instrumented": suffix == "probe",
        "calibration_sample_id": "gfx1151-calibration",
        "resources": {
            "vgpr": 32, "sgpr": 24, "lds_bytes": 4096,
            "scratch_bytes": 0, "spills": 0,
        },
    }


def test_bare_metal_packet_promotes_only_with_all_native_proofs() -> None:
    packet = build_rocm_profiler_packet(
        timing=_timing(), capture=_capture(),
        uninstrumented=_image(10_000, "clean"),
        instrumented=_image(10_200, "probe"),
        source={"source_commit": "a" * 40, "worktree_dirty": False},
    )
    assert packet["eligible_for_promotion"] is True
    assert packet["instrumentation_comparison"]["duration_ratio"] == 1.02
    validate_rocm_profiler_packet(packet)


def test_wsl_packet_remains_retain_only() -> None:
    timing = _timing()
    timing["execution_environment"] = "wsl2"
    for clock in timing["clocks"].values():
        clock["eligible_for_promotion"] = False
    packet = build_rocm_profiler_packet(
        timing=timing, capture=_capture(),
        uninstrumented=_image(10_000, "clean"),
        instrumented=_image(10_900, "probe"),
        source={"source_commit": "b" * 40, "worktree_dirty": False},
    )
    assert packet["eligible_for_promotion"] is False
    assert "BARE_METAL_REQUIRED" in packet["ineligibility_reasons"]
    assert "INSTRUMENTATION_OVERHEAD_EXCEEDED" in packet["ineligibility_reasons"]
