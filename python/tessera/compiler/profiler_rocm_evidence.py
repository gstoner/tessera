"""Promotion-gated ROCm (gfx1151 / gfx1201) profiler calibration evidence.

The architecture is never assumed: it is derived from the timing sample's
exact target (``rocm_<arch>``) and must agree with both image records. Evidence
for one architecture never admits another (gfx1151 and gfx1201 proofs do not
transfer; sync ``GFX1201-SSD-CALIBRATION-2026-09-26``).
"""

from __future__ import annotations

import hashlib
import math
import json
from typing import Any, Mapping

from .profiler_rocm_native import validate_rocm_native_capture
from .profiler_timing import is_wsl_environment, validate_timing_sample, wsl_promotion_refusals


ROCM_PROFILER_PACKET_SCHEMA_VERSION = "tessera.profiler_rocm_packet.v1"

#: Architectures with a validated device-clock marker and calibration route
#: (``native_device_clock.build_device_clock_marker``). Explicit, not a prefix
#: match: a new RDNA part is refused until it has its own proof.
ROCM_PROFILER_ARCHITECTURES: tuple[str, ...] = ("gfx1151", "gfx1201")


class ROCmProfilerPacketError(ValueError):
    """Raised when ROCm profiler evidence is contradictory."""


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _image_record(image: Mapping[str, Any], role: str) -> dict[str, Any]:
    required = (
        "architecture", "kernel_name", "semantic_sha256", "image_sha256",
        "isa_sha256", "duration_ns", "clock_source", "instrumented",
        "calibration_sample_id",
    )
    missing = [
        field for field in required
        if field not in image or image.get(field) is None or image.get(field) == ""
    ]
    if missing:
        raise ROCmProfilerPacketError(f"{role} image record is missing {missing}")
    duration = image["duration_ns"]
    try:
        duration = float(duration) if isinstance(duration, int) and not isinstance(duration, bool) else duration
    except OverflowError as exc:
        raise ROCmProfilerPacketError(f"{role} image duration must be finite and positive") from exc
    # NaN and inf compare False against every bound, so they would slip past
    # both sides of the overhead gate (review); require a finite value.
    if (not isinstance(duration, (int, float)) or isinstance(duration, bool)
            or not math.isfinite(duration) or duration <= 0):
        raise ROCmProfilerPacketError(f"{role} image duration must be finite and positive")
    resources = image.get("resources")
    if not isinstance(resources, Mapping):
        raise ROCmProfilerPacketError(f"{role} image requires resources")
    return dict(image)


#: Reasons that describe the *environment* (bare metal, a profiler that needs
#: KFD) rather than whether the timing is true. On the device-clock-witness
#: route they are recorded as diagnostic gaps instead of blocking promotion.
_ENVIRONMENT_REASONS = frozenset({
    "BARE_METAL_REQUIRED",
    "ROCPROFILER_CAPTURE_MISSING",
    "ROCPROFILER_DISPATCH_MISSING",
    "ROCPROFILER_RUNTIME_CALLBACK_MISSING",
    "ROCPROFILER_COUNTERS_MISSING",
    "ROCPROFILER_PC_SAMPLES_MISSING",
})
ROUTE_PROFILER = "profiler_correlated"
ROUTE_DEVICE_CLOCK = "device_clock_witness"


def _admission_route(timing: Mapping[str, Any]) -> str:
    """Which evidence route this timing sample can use.

    ``device_clock_witness`` is the owner-accepted non-profiler method
    (MASTER_AUDIT, 2026-09-25): on WSL, a promotion-eligible in-kernel device
    clock whose witness is valid and agrees in the same sample. Derived from
    the sample every time, never read from a stored field.
    """
    clocks = timing["clocks"]
    device = clocks.get("device_wall_clock_ns", {})
    if (
        is_wsl_environment(timing.get("execution_environment", ""))
        and device.get("valid") is True
        and device.get("eligible_for_promotion") is True
        and not wsl_promotion_refusals(str(timing["target"]), clocks)
    ):
        return ROUTE_DEVICE_CLOCK
    return ROUTE_PROFILER


def _timing_architecture(timing: Mapping[str, Any]) -> str:
    """The exact ROCm architecture a timing sample was taken on."""
    target = timing.get("target")
    arch = target[len("rocm_"):] if isinstance(target, str) and target.startswith("rocm_") else None
    if arch not in ROCM_PROFILER_ARCHITECTURES:
        raise ROCmProfilerPacketError(
            f"ROCm profiler packet requires exact timing on one of "
            f"{', '.join('rocm_' + a for a in ROCM_PROFILER_ARCHITECTURES)}; got {target!r}")
    assert isinstance(arch, str)
    return arch


def _check_pairing(timing: Mapping[str, Any], clean: Mapping[str, Any],
                   probe: Mapping[str, Any], maximum_instrumentation_overhead: Any) -> str:
    """The clean/probe pairing rules, run by the builder AND the validator, so a
    stored packet cannot be mutated past them (review). Returns the exact
    architecture the timing target and both images agree on."""
    if (not isinstance(maximum_instrumentation_overhead, (int, float))
            or isinstance(maximum_instrumentation_overhead, bool)
            or not math.isfinite(maximum_instrumentation_overhead)
            or not 1.0 <= maximum_instrumentation_overhead <= 2.0):
        raise ROCmProfilerPacketError(
            "instrumentation overhead limit must be finite and within [1.0, 2.0]")
    arch = _timing_architecture(timing)
    if clean["clock_source"] != probe["clock_source"]:
        raise ROCmProfilerPacketError("instrumentation comparison requires one timing domain")
    clock_slot = {
        "hip_event": "hip_event_ns",
        "device_wall_clock": "device_wall_clock_ns",
        "rocprofiler_activity": "profiler_activity_ns",
    }.get(str(clean["clock_source"]))
    if clock_slot is None or timing["clocks"][clock_slot].get("valid") is not True:
        raise ROCmProfilerPacketError("application timing source is not valid in the calibration sample")
    if clean["kernel_name"] != probe["kernel_name"]:
        raise ROCmProfilerPacketError("instrumentation comparison requires one application kernel")
    if clean["semantic_sha256"] != probe["semantic_sha256"]:
        raise ROCmProfilerPacketError("instrumentation comparison requires one semantic artifact")
    if clean["instrumented"] is not False or probe["instrumented"] is not True:
        raise ROCmProfilerPacketError("instrumentation roles are inconsistent")
    if clean["architecture"] != arch or probe["architecture"] != arch:
        raise ROCmProfilerPacketError(
            f"instrumentation comparison requires exact {arch} images matching the "
            f"timing target; got {clean['architecture']!r}/{probe['architecture']!r}")
    if (
        clean["calibration_sample_id"] != timing.get("sample_id")
        or probe["calibration_sample_id"] != timing.get("sample_id")
    ):
        raise ROCmProfilerPacketError("application images do not bind the timing calibration sample")
    return arch


def _derive_eligibility(
    *,
    timing: Mapping[str, Any],
    capture: Mapping[str, Any],
    clean: Mapping[str, Any],
    probe: Mapping[str, Any],
    source: Mapping[str, Any],
    maximum_instrumentation_overhead: float,
) -> tuple[float, list[str], str, list[str]]:
    """(overhead, blocking reasons, admission route, diagnostic gaps).

    The single derivation both the builder and the validator run, so a stored
    packet cannot drop a blocker, claim a route, or relabel gaps (review):
    the validator recomputes all four from the packet's own inputs.
    """
    overhead = float(probe["duration_ns"]) / float(clean["duration_ns"])
    if not math.isfinite(overhead) or overhead <= 0:
        raise ROCmProfilerPacketError("instrumentation duration ratio must be finite and positive")
    reasons: list[str] = []
    if timing.get("execution_environment") != "bare_metal":
        reasons.append("BARE_METAL_REQUIRED")
    clocks = timing["clocks"]
    device = clocks["device_wall_clock_ns"]
    hip = clocks["hip_event_ns"]
    activity = clocks["profiler_activity_ns"]
    if not device.get("valid"):
        reasons.append("DEVICE_WALL_CLOCK_INVALID")
    if not (hip.get("valid") or activity.get("valid")):
        reasons.append("INDEPENDENT_DEVICE_CLOCK_MISSING")
    if device.get("valid") and not set(device.get("calibrated_against", ())).intersection(
        {"hip_event_ns", "profiler_activity_ns"}
    ):
        reasons.append("DEVICE_WALL_CLOCK_UNCALIBRATED")
    proof = capture.get("proof", {})
    if capture.get("provider") != "rocprofiler" or capture.get("status") != "collected":
        reasons.append("ROCPROFILER_CAPTURE_MISSING")
    if not proof.get("dispatch_activity_seen"):
        reasons.append("ROCPROFILER_DISPATCH_MISSING")
    if not (proof.get("hip_callback_seen") or proof.get("hsa_callback_seen")):
        reasons.append("ROCPROFILER_RUNTIME_CALLBACK_MISSING")
    requested = capture.get("requested", {})
    if requested.get("counters") and not proof.get("counter_records_seen"):
        reasons.append("ROCPROFILER_COUNTERS_MISSING")
    if requested.get("pc_sampling") and not proof.get("pc_samples_seen"):
        reasons.append("ROCPROFILER_PC_SAMPLES_MISSING")
    if overhead > maximum_instrumentation_overhead:
        reasons.append("INSTRUMENTATION_OVERHEAD_EXCEEDED")
    elif overhead < 1.0 / maximum_instrumentation_overhead:
        # Two-sided: an instrumented image materially FASTER than its clean
        # twin is a different program, so its clock says nothing about the
        # clean image (measured: gfx1151 serial SSD, 2026-09-26).
        reasons.append("INSTRUMENTATION_CHANGED_THE_KERNEL")
    if source.get("worktree_dirty"):
        reasons.append("SOURCE_WORKTREE_DIRTY")
    route = _admission_route(timing)
    diagnostic_gaps: list[str] = []
    if route == ROUTE_DEVICE_CLOCK:
        # The witness sample must name the image it calibrates (review): one
        # genuine sample copied under new sample_ids must not vouch for others.
        digests = timing.get("artifact_digests", {})
        if clean.get("image_sha256") not in set(digests.values()):
            reasons.append("CALIBRATION_IMAGE_UNBOUND")
        diagnostic_gaps = [r for r in reasons if r in _ENVIRONMENT_REASONS]
        reasons = [r for r in reasons if r not in _ENVIRONMENT_REASONS]
    return overhead, reasons, route, diagnostic_gaps


def build_rocm_profiler_packet(
    *,
    timing: Mapping[str, Any],
    capture: Mapping[str, Any],
    uninstrumented: Mapping[str, Any],
    instrumented: Mapping[str, Any],
    source: Mapping[str, Any],
    maximum_instrumentation_overhead: float = 1.05,
) -> dict[str, Any]:
    validate_timing_sample(timing)
    validate_rocm_native_capture(capture)
    clean = _image_record(uninstrumented, "uninstrumented")
    probe = _image_record(instrumented, "instrumented")
    arch = _check_pairing(timing, clean, probe, maximum_instrumentation_overhead)
    source_commit = source.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ROCmProfilerPacketError("ROCm profiler packet requires full source commit")
    overhead, reasons, route, diagnostic_gaps = _derive_eligibility(
        timing=timing, capture=capture, clean=clean, probe=probe, source=source,
        maximum_instrumentation_overhead=maximum_instrumentation_overhead)
    device = timing["clocks"]["device_wall_clock_ns"]
    packet = {
        "schema": ROCM_PROFILER_PACKET_SCHEMA_VERSION,
        "work_item": "TPROF-ROCM-NATIVE-1",
        "architecture": arch,
        "source": dict(source),
        "timing": dict(timing),
        "timing_sha256": _digest(timing),
        "capture": dict(capture),
        "capture_sha256": _digest(capture),
        "instrumentation_comparison": {
            "uninstrumented": clean,
            "instrumented": probe,
            "duration_ratio": overhead,
            "maximum_duration_ratio": maximum_instrumentation_overhead,
            "resource_delta": {
                key: probe["resources"].get(key, 0) - clean["resources"].get(key, 0)
                for key in ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "spills")
                if isinstance(probe["resources"].get(key, 0), (int, float))
                and isinstance(clean["resources"].get(key, 0), (int, float))
            },
        },
        "eligible_for_regression": bool(device.get("valid")) and overhead > 0,
        "eligible_for_promotion": not reasons,
        "ineligibility_reasons": reasons,
        "admission_route": route,
        "diagnostic_gaps": diagnostic_gaps,
        "calibration_status": "promotable" if not reasons else "retain_only",
    }
    packet["packet_sha256"] = _digest(packet)
    validate_rocm_profiler_packet(packet)
    return packet


def validate_rocm_profiler_packet(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != ROCM_PROFILER_PACKET_SCHEMA_VERSION:
        raise ROCmProfilerPacketError("unsupported ROCm profiler packet schema")
    if payload.get("architecture") not in ROCM_PROFILER_ARCHITECTURES:
        raise ROCmProfilerPacketError(
            f"ROCm profiler packet requires one of {', '.join(ROCM_PROFILER_ARCHITECTURES)}")
    timing = payload.get("timing")
    capture = payload.get("capture")
    if not isinstance(timing, Mapping) or not isinstance(capture, Mapping):
        raise ROCmProfilerPacketError("ROCm profiler packet requires timing and capture")
    validate_timing_sample(timing)
    validate_rocm_native_capture(capture)
    comparison = payload.get("instrumentation_comparison")
    if not isinstance(comparison, Mapping):
        raise ROCmProfilerPacketError("ROCm profiler packet requires instrumentation comparison")
    clean = comparison.get("uninstrumented")
    probe = comparison.get("instrumented")
    if not isinstance(clean, Mapping) or not isinstance(probe, Mapping):
        raise ROCmProfilerPacketError("ROCm profiler packet requires paired image records")
    _image_record(clean, "uninstrumented")
    _image_record(probe, "instrumented")
    if clean.get("semantic_sha256") != probe.get("semantic_sha256"):
        raise ROCmProfilerPacketError("ROCm profiler packet semantic lineage mismatch")
    if clean.get("calibration_sample_id") != timing.get("sample_id"):
        raise ROCmProfilerPacketError("ROCm profiler packet calibration lineage mismatch")
    expected_ratio = float(probe["duration_ns"]) / float(clean["duration_ns"])
    ratio = comparison.get("duration_ratio")
    if (not isinstance(ratio, (int, float)) or not math.isfinite(float(ratio))
            or abs(float(ratio) - expected_ratio) > 1e-12):
        raise ROCmProfilerPacketError("ROCm instrumentation duration ratio mismatch")
    if _digest(timing) != payload.get("timing_sha256"):
        raise ROCmProfilerPacketError("ROCm timing digest mismatch")
    if _digest(capture) != payload.get("capture_sha256"):
        raise ROCmProfilerPacketError("ROCm capture digest mismatch")
    reasons = payload.get("ineligibility_reasons")
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        raise ROCmProfilerPacketError("invalid ROCm profiler ineligibility reasons")
    if payload.get("eligible_for_promotion") and reasons:
        raise ROCmProfilerPacketError("promotion-eligible ROCm packet has blockers")
    source = payload.get("source")
    maximum = comparison.get("maximum_duration_ratio")
    if not isinstance(source, Mapping):
        raise ROCmProfilerPacketError("ROCm profiler packet requires source")
    commit = source.get("source_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ROCmProfilerPacketError("ROCm profiler packet requires full source commit")
    arch = _check_pairing(timing, clean, probe, maximum)
    if payload.get("architecture") != arch:
        raise ROCmProfilerPacketError(
            f"ROCm packet claims architecture {payload.get('architecture')!r}, but its "
            f"timing and images are {arch!r}")
    assert isinstance(maximum, (int, float))  # _check_pairing refused anything else
    _, derived, route, gaps = _derive_eligibility(
        timing=timing, capture=capture, clean=clean, probe=probe, source=source,
        maximum_instrumentation_overhead=float(maximum))
    stored_route = payload.get("admission_route", ROUTE_PROFILER)
    if stored_route != route:
        raise ROCmProfilerPacketError(
            f"ROCm packet claims admission route {stored_route!r}, but its inputs "
            f"support {route!r}")
    if reasons != derived:
        raise ROCmProfilerPacketError(
            f"ROCm packet reasons {reasons} differ from those its inputs derive {derived}")
    if payload.get("diagnostic_gaps", []) != gaps:
        raise ROCmProfilerPacketError("ROCm diagnostic gaps differ from those its inputs derive")
    if bool(payload.get("eligible_for_promotion")) != (not derived):
        raise ROCmProfilerPacketError("ROCm promotion eligibility differs from its derived reasons")
    unsigned = dict(payload)
    packet_digest = unsigned.pop("packet_sha256", None)
    if _digest(unsigned) != packet_digest:
        raise ROCmProfilerPacketError("ROCm profiler packet digest mismatch")


__all__ = [
    "ROCM_PROFILER_ARCHITECTURES",
    "ROCM_PROFILER_PACKET_SCHEMA_VERSION",
    "ROUTE_DEVICE_CLOCK",
    "ROUTE_PROFILER",
    "ROCmProfilerPacketError",
    "build_rocm_profiler_packet",
    "validate_rocm_profiler_packet",
]
