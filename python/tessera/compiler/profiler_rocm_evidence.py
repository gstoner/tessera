"""Promotion-gated gfx1151 profiler calibration evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .profiler_rocm_native import validate_rocm_native_capture
from .profiler_timing import validate_timing_sample


ROCM_PROFILER_PACKET_SCHEMA_VERSION = "tessera.profiler_rocm_packet.v1"


class ROCmProfilerPacketError(ValueError):
    """Raised when gfx1151 profiler evidence is contradictory."""


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
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise ROCmProfilerPacketError(f"{role} image duration must be positive")
    resources = image.get("resources")
    if not isinstance(resources, Mapping):
        raise ROCmProfilerPacketError(f"{role} image requires resources")
    return dict(image)


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
    if timing.get("target") != "rocm_gfx1151":
        raise ROCmProfilerPacketError("ROCm profiler packet requires exact gfx1151 timing")
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
    if clean["architecture"] != "gfx1151" or probe["architecture"] != "gfx1151":
        raise ROCmProfilerPacketError("instrumentation comparison requires exact gfx1151 images")
    if (
        clean["calibration_sample_id"] != timing.get("sample_id")
        or probe["calibration_sample_id"] != timing.get("sample_id")
    ):
        raise ROCmProfilerPacketError("application images do not bind the timing calibration sample")
    if maximum_instrumentation_overhead < 1.0:
        raise ROCmProfilerPacketError("instrumentation overhead limit must be at least 1.0")
    overhead = float(probe["duration_ns"]) / float(clean["duration_ns"])
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
    if source.get("worktree_dirty"):
        reasons.append("SOURCE_WORKTREE_DIRTY")
    source_commit = source.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ROCmProfilerPacketError("ROCm profiler packet requires full source commit")
    packet = {
        "schema": ROCM_PROFILER_PACKET_SCHEMA_VERSION,
        "work_item": "TPROF-ROCM-NATIVE-1",
        "architecture": "gfx1151",
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
        "calibration_status": "promotable" if not reasons else "retain_only",
    }
    packet["packet_sha256"] = _digest(packet)
    validate_rocm_profiler_packet(packet)
    return packet


def validate_rocm_profiler_packet(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != ROCM_PROFILER_PACKET_SCHEMA_VERSION:
        raise ROCmProfilerPacketError("unsupported ROCm profiler packet schema")
    if payload.get("architecture") != "gfx1151":
        raise ROCmProfilerPacketError("ROCm profiler packet requires gfx1151")
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
    if not isinstance(ratio, (int, float)) or abs(float(ratio) - expected_ratio) > 1e-12:
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
    unsigned = dict(payload)
    packet_digest = unsigned.pop("packet_sha256", None)
    if _digest(unsigned) != packet_digest:
        raise ROCmProfilerPacketError("ROCm profiler packet digest mismatch")


__all__ = [
    "ROCM_PROFILER_PACKET_SCHEMA_VERSION",
    "ROCmProfilerPacketError",
    "build_rocm_profiler_packet",
    "validate_rocm_profiler_packet",
]
