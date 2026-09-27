"""NVIDIA sm_120 %globaltimer device-clock packets (NVIDIA-GLOBALTIMER-MARKER-2026-09-26).

Host-free: every rule the builder applies is re-derived by the validator, so a
stored packet cannot drop a blocker, relabel its chip, or claim a foreign
route. The device evidence itself lives in
``benchmarks/baselines/sm120_ssd_calibrated_pairs_20260926/``.
"""

from __future__ import annotations

import copy

import pytest

from tessera.compiler.profiler_nvidia_evidence import (
    NVIDIADeviceClockPacketError,
    build_nvidia_device_clock_packet,
    validate_nvidia_device_clock_packet,
)
from tessera.compiler.profiler_timing import (
    ProfilerTimingError, build_timing_sample, measured_clock, unavailable_clock)

_COMMIT = "a" * 40
_IMAGE = "f" * 64


def _timing(*, environment: str = "wsl2", event: float = 24_000.0, device: float = 23_900.0,
            architecture: str = "sm_120", target: str = "nvidia_sm120",
            digests: dict | None = None, name: str = "NVIDIA GeForce RTX 5070",
            windows_device: list | None = None, windows_event: list | None = None,
            launches: int = 1000) -> dict:
    return build_timing_sample(
        sample_id="sm120-serial-0",
        target=target,
        clocks={
            "host_wall_ns": measured_clock("host_wall_ns", source="perf_counter", value=30_000),
            "cuda_event_ns": measured_clock("cuda_event_ns", source="cuda_event", value=event),
            "device_wall_clock_ns": measured_clock(
                "device_wall_clock_ns", source="device_wall_clock", value=device,
                instrumented=True, calibrated_against=("cuda_event_ns",),
                eligible_for_promotion=True,
                provenance={"clock": "%globaltimer", "launches_per_window": 1000,
                            "per_window_ns": windows_device or [device] * 7}),
            "profiler_activity_ns": unavailable_clock(
                "profiler_activity_ns", source="cupti_activity", reason="NOT_CAPTURED"),
        },
        artifact_digests=digests if digests is not None else {"application_image": _IMAGE},
        batch_size=launches, warm_state="warm", synchronization="cuEventSynchronize",
        execution_environment=environment,
        environment={"run_id": "r0", "per_window_event_ns": windows_event or [event] * 7,
                     "device_identity": {"architecture": architecture, "name": name,
                                         "uuid": "cba12639821a7a104cd3f918f9c0a545"}},
    )


def _image(duration: float, *, instrumented: bool, architecture: str = "sm_120") -> dict:
    return {
        "architecture": architecture, "kernel_name": "ssd", "semantic_sha256": "s" * 64,
        "image_sha256": _IMAGE, "isa_sha256": "i" * 64, "duration_ns": duration,
        "clock_source": "cuda_event", "instrumented": instrumented,
        "calibration_sample_id": "sm120-serial-0", "resources": {"registers": 40},
    }


def _packet(**timing_kwargs) -> dict:
    return build_nvidia_device_clock_packet(
        timing=_timing(**timing_kwargs), uninstrumented=_image(23_950.0, instrumented=False),
        instrumented=_image(24_000.0, instrumented=True),
        source={"source_commit": _COMMIT, "worktree_dirty": False})


def test_an_agreeing_globaltimer_sample_is_promotable_on_wsl_and_bare_metal() -> None:
    for environment in ("wsl2", "bare_metal"):
        packet = _packet(environment=environment)
        assert packet["eligible_for_promotion"] is True, packet["ineligibility_reasons"]
        assert packet["architecture"] == "sm_120"
        assert packet["admission_route"] == "device_clock_witness"
        validate_nvidia_device_clock_packet(packet)


def test_disagreement_blocks_on_bare_metal_too() -> None:
    """WSL refuses the sample outright; bare metal builds it, and the packet's
    own derivation refuses it -- there is no profiler behind this route."""
    with pytest.raises(ProfilerTimingError, match="witnesses disagree"):
        _packet(device=20_000.0)
    packet = _packet(environment="bare_metal", device=20_000.0)
    assert "DEVICE_CLOCK_WITNESS_DISAGREES" in packet["ineligibility_reasons"]
    assert packet["eligible_for_promotion"] is False


def test_overhead_is_two_sided_and_the_image_must_be_bound() -> None:
    timing = _timing()
    slow = build_nvidia_device_clock_packet(
        timing=timing, uninstrumented=_image(20_000.0, instrumented=False),
        instrumented=_image(24_000.0, instrumented=True),
        source={"source_commit": _COMMIT, "worktree_dirty": False})
    assert "INSTRUMENTATION_OVERHEAD_EXCEEDED" in slow["ineligibility_reasons"]
    fast = build_nvidia_device_clock_packet(
        timing=timing, uninstrumented=_image(30_000.0, instrumented=False),
        instrumented=_image(24_000.0, instrumented=True),
        source={"source_commit": _COMMIT, "worktree_dirty": False})
    assert "INSTRUMENTATION_CHANGED_THE_KERNEL" in fast["ineligibility_reasons"]
    unbound = _packet(digests={"application_image": "0" * 64})
    assert "CALIBRATION_IMAGE_UNBOUND" in unbound["ineligibility_reasons"]


def test_a_dirty_or_unstated_tree_blocks() -> None:
    for source in ({"source_commit": _COMMIT, "worktree_dirty": True},
                   {"source_commit": _COMMIT}):
        packet = build_nvidia_device_clock_packet(
            timing=_timing(), uninstrumented=_image(23_950.0, instrumented=False),
            instrumented=_image(24_000.0, instrumented=True), source=source)
        assert "SOURCE_WORKTREE_DIRTY" in packet["ineligibility_reasons"]


def test_the_queried_device_outranks_the_label() -> None:
    with pytest.raises(NVIDIADeviceClockPacketError, match="queried at record time"):
        _packet(architecture="sm_89")
    with pytest.raises((NVIDIADeviceClockPacketError, ProfilerTimingError)):
        _packet(target="nvidia_sm121")


def test_the_validator_rederives_every_verdict() -> None:
    packet = _packet(environment="bare_metal", device=20_000.0)
    forged = copy.deepcopy(packet)
    forged["ineligibility_reasons"] = []
    forged["eligible_for_promotion"] = True
    with pytest.raises(NVIDIADeviceClockPacketError):
        validate_nvidia_device_clock_packet(forged)
    relabelled = copy.deepcopy(_packet())
    relabelled["architecture"] = "sm_121"
    with pytest.raises(NVIDIADeviceClockPacketError, match="claims"):
        validate_nvidia_device_clock_packet(relabelled)
    tampered = copy.deepcopy(_packet())
    tampered["timing"]["clocks"]["cuda_event_ns"]["value"] = 24_001.0
    with pytest.raises(NVIDIADeviceClockPacketError, match="digest"):
        validate_nvidia_device_clock_packet(tampered)


def _forged_short_window_timing() -> dict:
    """The reviewer's forgery: the committed sm_120 packet rebuilt at 10
    launches per window, per-window errors up to 18%, median error 3.5%."""
    event = [99_000.0] * 7
    errors = [0.035, 0.035, 0.035, 0.18, 0.18, -0.18, 0.0]
    device = [e * (1 - err) for e, err in zip(event, errors)]
    import statistics
    return _timing(event=statistics.median(event), device=statistics.median(device),
                   windows_device=device, windows_event=event, launches=10)


def test_a_short_window_packet_with_agreeing_medians_is_refused() -> None:
    timing = _forged_short_window_timing()
    # The median check alone admits it: 3.5% is inside the band.
    packet = build_nvidia_device_clock_packet(
        timing=timing, uninstrumented=_image(99_000.0, instrumented=False),
        instrumented=_image(99_000.0, instrumented=True),
        source={"source_commit": _COMMIT, "worktree_dirty": False})
    assert packet["eligible_for_promotion"] is False
    assert "DEVICE_CLOCK_WINDOW_TOO_SHORT" in packet["ineligibility_reasons"]
    assert "DEVICE_CLOCK_WINDOW_DISAGREES" in packet["ineligibility_reasons"]


def test_long_windows_still_refuse_one_disagreeing_window() -> None:
    event = [24_000.0] * 7
    device = [24_000.0 * 0.999] * 6 + [24_000.0 * 0.82]
    import statistics
    packet = build_nvidia_device_clock_packet(
        timing=_timing(event=statistics.median(event), device=statistics.median(device),
                       windows_device=device, windows_event=event),
        uninstrumented=_image(23_950.0, instrumented=False),
        instrumented=_image(24_000.0, instrumented=True),
        source={"source_commit": _COMMIT, "worktree_dirty": False})
    assert packet["ineligibility_reasons"] == ["DEVICE_CLOCK_WINDOW_DISAGREES"]


def test_windows_must_be_the_ones_the_medians_came_from() -> None:
    packet = build_nvidia_device_clock_packet(
        timing=_timing(windows_device=[30_000.0] * 7),
        uninstrumented=_image(23_950.0, instrumented=False),
        instrumented=_image(24_000.0, instrumented=True),
        source={"source_commit": _COMMIT, "worktree_dirty": False})
    assert "DEVICE_CLOCK_WINDOWS_UNBOUND" in packet["ineligibility_reasons"]


def test_an_unvalidated_part_of_the_same_compute_capability_is_refused() -> None:
    """The window validation came from an RTX 5070; cc 12.0 alone is not it."""
    packet = _packet(name="NVIDIA GeForce RTX 5070 Ti")
    assert packet["ineligibility_reasons"] == ["DEVICE_CLOCK_PART_UNVALIDATED"]


def test_witness_refusals_keep_their_own_codes() -> None:
    from tessera.compiler.profiler_timing import witness_refusal_codes
    assert witness_refusal_codes([
        "device_wall_clock_ns: no admissible witness (cuda_event_ns) is valid in the sample, "
        "so the device clock has no independent witness"]) == ["DEVICE_CLOCK_WITNESS_MISSING"]
    assert witness_refusal_codes([
        "device_wall_clock_ns: witnesses disagree beyond 5%: cuda_event_ns (9.0%)"]) == [
        "DEVICE_CLOCK_WITNESS_DISAGREES"]


def test_the_committed_sm120_packet_still_validates_under_the_window_rule() -> None:
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "benchmarks/baselines/sm120_ssd_calibrated_pairs_20260926"
    for path in sorted(root.glob("*-calibration.json")):
        payload = json.loads(path.read_text())
        validate_nvidia_device_clock_packet(payload)
        assert payload["eligible_for_promotion"] is True
