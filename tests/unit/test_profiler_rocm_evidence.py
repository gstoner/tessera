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


def _wsl_witness_timing(device_ns: int = 10_000, event_ns: int = 10_100,
                        image_sha256: str = "image-clean") -> dict:
    """WSL, no KFD: the device clock is the promotion clock, the HIP event its
    agreeing witness, and the profiler slot is unavailable."""
    from tessera.compiler.profiler_timing import unavailable_clock
    return build_timing_sample(
        sample_id="gfx1151-calibration",
        target="rocm_gfx1151",
        clocks={
            "host_wall_ns": measured_clock("host_wall_ns", source="steady_clock", value=12_000),
            "hip_event_ns": measured_clock("hip_event_ns", source="hip_event", value=event_ns),
            "device_wall_clock_ns": measured_clock(
                "device_wall_clock_ns", source="device_wall_clock", value=device_ns,
                instrumented=True, calibrated_against=("hip_event_ns",),
                eligible_for_promotion=True,
            ),
            "profiler_activity_ns": unavailable_clock(
                "profiler_activity_ns", source="rocprofiler_activity",
                reason="ROCPROFILER_UNAVAILABLE_NO_KFD"),
        },
        artifact_digests={"application_image": image_sha256},
        batch_size=10,
        warm_state="warm",
        synchronization="hipEventSynchronize",
        execution_environment="wsl2",
    )


def _no_kfd_capture() -> dict:
    capture = _capture()
    capture["status"] = "blocked"
    capture["reason"] = "rocprofiler requires /dev/kfd; WSL2 exposes /dev/dxg only"
    capture["proof"] = {key: False for key in capture["proof"]}
    capture["requested"] = {"counters": [], "pc_sampling": False}
    return capture


def test_wsl_device_clock_witness_packet_promotes_with_profiler_gaps_as_diagnostics() -> None:
    """Owner direction 2026-09-25: no KFD, no bare metal; the device clock
    with an agreeing in-sample witness is the evidence."""
    packet = build_rocm_profiler_packet(
        timing=_wsl_witness_timing(), capture=_no_kfd_capture(),
        uninstrumented=_image(10_000, "clean"),
        instrumented=_image(10_200, "probe"),
        source={"source_commit": "c" * 40, "worktree_dirty": False},
    )
    assert packet["admission_route"] == "device_clock_witness"
    assert packet["eligible_for_promotion"] is True
    assert "BARE_METAL_REQUIRED" in packet["diagnostic_gaps"]
    assert "ROCPROFILER_DISPATCH_MISSING" in packet["diagnostic_gaps"]
    assert packet["ineligibility_reasons"] == []
    validate_rocm_profiler_packet(packet)


def test_the_witness_route_still_blocks_on_timing_defects() -> None:
    """Only environment reasons become diagnostics; overhead still blocks."""
    packet = build_rocm_profiler_packet(
        timing=_wsl_witness_timing(), capture=_no_kfd_capture(),
        uninstrumented=_image(10_000, "clean"),
        instrumented=_image(10_900, "probe"),
        source={"source_commit": "c" * 40, "worktree_dirty": False},
    )
    assert packet["ineligibility_reasons"] == ["INSTRUMENTATION_OVERHEAD_EXCEEDED"]
    assert packet["eligible_for_promotion"] is False


def test_a_forged_admission_route_is_rejected() -> None:
    import pytest
    from tessera.compiler.profiler_rocm_evidence import ROCmProfilerPacketError, _digest
    timing = _timing()
    timing["execution_environment"] = "wsl2"
    for clock in timing["clocks"].values():
        clock["eligible_for_promotion"] = False
    packet = build_rocm_profiler_packet(
        timing=timing, capture=_capture(),
        uninstrumented=_image(10_000, "clean"),
        instrumented=_image(10_200, "probe"),
        source={"source_commit": "d" * 40, "worktree_dirty": False},
    )
    assert packet["admission_route"] == "profiler_correlated"
    packet["admission_route"] = "device_clock_witness"
    packet["diagnostic_gaps"] = packet["ineligibility_reasons"] = []
    packet["eligible_for_promotion"] = True
    packet.pop("packet_sha256")
    packet["packet_sha256"] = _digest(packet)
    with pytest.raises(ROCmProfilerPacketError, match="admission route"):
        validate_rocm_profiler_packet(packet)


def test_the_witness_sample_must_name_the_image_it_calibrates() -> None:
    """Review: one genuine sample copied under new ids must not vouch for a
    different image."""
    packet = build_rocm_profiler_packet(
        timing=_wsl_witness_timing(image_sha256="some-other-image"),
        capture=_no_kfd_capture(),
        uninstrumented=_image(10_000, "clean"),
        instrumented=_image(10_200, "probe"),
        source={"source_commit": "c" * 40, "worktree_dirty": False},
    )
    assert packet["ineligibility_reasons"] == ["CALIBRATION_IMAGE_UNBOUND"]
    assert packet["eligible_for_promotion"] is False


def test_a_packet_that_drops_a_timing_blocker_is_rejected() -> None:
    """Review: the validator re-derives reasons, so a re-digested packet with
    INSTRUMENTATION_OVERHEAD_EXCEEDED removed does not validate."""
    import pytest
    from tessera.compiler.profiler_rocm_evidence import ROCmProfilerPacketError, _digest
    packet = build_rocm_profiler_packet(
        timing=_wsl_witness_timing(), capture=_no_kfd_capture(),
        uninstrumented=_image(10_000, "clean"),
        instrumented=_image(10_900, "probe"),
        source={"source_commit": "c" * 40, "worktree_dirty": False},
    )
    assert packet["ineligibility_reasons"] == ["INSTRUMENTATION_OVERHEAD_EXCEEDED"]
    packet["ineligibility_reasons"] = []
    packet["eligible_for_promotion"] = True
    packet["calibration_status"] = "promotable"
    packet.pop("packet_sha256")
    packet["packet_sha256"] = _digest(packet)
    with pytest.raises(ROCmProfilerPacketError, match="differ"):
        validate_rocm_profiler_packet(packet)


def test_an_instrumented_image_faster_than_its_clean_twin_blocks() -> None:
    """Two-sided gate: measured 2026-09-26, stamping split the gfx1151 serial
    SSD entry block and the 'instrumented twin' ran 2.4x faster -- a different
    program, whose clock says nothing about the clean image."""
    packet = build_rocm_profiler_packet(
        timing=_wsl_witness_timing(), capture=_no_kfd_capture(),
        uninstrumented=_image(10_000, "clean"),
        instrumented=_image(4_100, "probe"),
        source={"source_commit": "c" * 40, "worktree_dirty": False},
    )
    assert packet["ineligibility_reasons"] == ["INSTRUMENTATION_CHANGED_THE_KERNEL"]
    assert packet["eligible_for_promotion"] is False


def test_non_finite_durations_are_refused() -> None:
    """Review: NaN/inf compare False against both overhead bounds."""
    import pytest
    from tessera.compiler.profiler_rocm_evidence import ROCmProfilerPacketError
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ROCmProfilerPacketError, match="finite"):
            build_rocm_profiler_packet(
                timing=_wsl_witness_timing(), capture=_no_kfd_capture(),
                uninstrumented=_image(10_000, "clean"), instrumented=_image(bad, "probe"),
                source={"source_commit": "c" * 40, "worktree_dirty": False})
