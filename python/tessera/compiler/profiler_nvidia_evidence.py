"""NVIDIA sm_120 device-clock calibration evidence (``%globaltimer`` marker).

Sync ``NVIDIA-GLOBALTIMER-MARKER-2026-09-26`` (follows
``DEVICE-CLOCK-MARKER-2026-09-26``). The packet is the NVIDIA twin of the ROCm
device-clock-witness packet (:mod:`.profiler_rocm_evidence`): one process's
clean image, timed by CUDA events, calibrated by the compiler-built
``%globaltimer`` marker (:mod:`.native_device_clock`) bracketing the same
stream interval. It is **not** the Nsight activity-window calibration
(:mod:`.profiler_cuda_window`), which stays a separate, profiler-derived route.

What makes the packet promotable, derived from its own inputs every time (the
validator re-runs the derivation, so a stored packet cannot drop a blocker):

* the timing target is exactly ``nvidia_sm120`` and the device queried at
  record time agrees (``environment.device_identity.architecture``);
* the device clock is valid, instrumented, calibrated against the CUDA event
  and agrees with every admissible witness within the 5% band -- checked in
  **every** environment, not only WSL, because this route has no profiler
  behind it on bare metal either;
* **every stored window** is at least 1 ms long and agrees within the band on
  its own, and the stored medians are those windows' medians
  (``profiler_timing.device_clock_window_refusals``; medians alone admitted a
  10-launch forgery with per-window errors up to 18% -- review, 2026-09-26);
* the queried part is one the window validation was measured on
  (``NVIDIA_DEVICE_CLOCK_VALIDATED_PARTS``: the RTX 5070 only; another cc 12.0
  part needs its own probe);
* the marker-bracketed / plain duration ratio is inside the two-sided overhead
  band (the image is the same; the ratio bounds the markers' cost);
* the witness sample names the calibrated image's digest; the source tree was
  clean.

Measured validation behind admitting this route (The-Super-Bear, RTX 5070,
WSL2, driver 610.88, 2026-09-26): ``%globaltimer`` advances in exact 32 ns
steps; span and CUDA events differ by a roughly fixed ~10-16 us per window
(~46 us at most), so windows of about 1 ms and longer agree within the band
and shorter ones do not -- which is why the recorder's window length is a
recorded parameter.
Evidence: ``benchmarks/baselines/sm120_ssd_calibrated_pairs_20260926/``.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from .evidence_reasons import ReasonVocabulary
from .profiler_timing import (
    device_clock_window_refusals, validate_timing_sample, witness_refusal_codes,
    wsl_promotion_refusals)


NVIDIA_DEVICE_CLOCK_PACKET_SCHEMA_VERSION = "tessera.profiler_nvidia_device_clock_packet.v1"

#: Every packet-local tag an NVIDIA device-clock packet's
#: ``ineligibility_reasons`` may carry (X86-EVIDENCE-VOCAB-1's rule, applied
#: 2026-09-27). The witness and window refusals from `profiler_timing` and
#: ``DEVICE_CLOCK_PART_UNVALIDATED`` are registered diagnostics, named by their
#: ``pass_origin`` rather than redeclared here.
NVIDIA_DEVICE_CLOCK_REASONS: dict[str, str] = {
    "DEVICE_WALL_CLOCK_INVALID":
        "the %globaltimer device clock sample is not valid",
    "DEVICE_WALL_CLOCK_UNCALIBRATED":
        "the device clock is valid but was not calibrated against a CUDA event",
    "DEVICE_WALL_CLOCK_NOT_PROMOTION_ELIGIBLE":
        "the device clock record itself declines promotion eligibility",
    "INSTRUMENTATION_OVERHEAD_EXCEEDED":
        "the instrumented image is slower than its clean twin beyond the allowed ratio",
    "INSTRUMENTATION_CHANGED_THE_KERNEL":
        "the instrumented image is materially faster than its clean twin, so it is "
        "a different program and its clock says nothing about the clean one",
    "CALIBRATION_IMAGE_UNBOUND":
        "the timing sample's artifact digests do not name the calibrated clean image",
    "SOURCE_WORKTREE_DIRTY":
        "the measured tree is not recorded clean, so the result names no revision",
}
ROUTE_DEVICE_CLOCK = "device_clock_witness"

#: Architectures whose ``%globaltimer`` marker has exact-device validation, and
#: the timing target each one is recorded under. Explicit: a new part is
#: refused until it has its own proof (a 5070 Ti is also cc 12.0 but is not
#: the part this was measured on; its evidence would be its own packet).
NVIDIA_DEVICE_CLOCK_ARCHITECTURES: dict[str, str] = {"sm_120": "nvidia_sm120"}

#: The exact parts the marker's window-length validation was measured on, per
#: architecture. Bound to the part (the queried device name), not to the
#: compute capability: cc 12.0 spans the consumer Blackwell line, and the
#: per-window span/event offset that sets the minimum window is a property of
#: the part and its driver, measured here on one RTX 5070 (The-Super-Bear,
#: 2026-09-26). Not bound to the UUID: a second card of the same model has the
#: same counter and offset behaviour, and every packet re-checks per-window
#: agreement anyway, so a UUID pin would add no evidence. Another part (a 5070
#: Ti, 5080, 5090) needs its own probe before it is added here.
NVIDIA_DEVICE_CLOCK_VALIDATED_PARTS: dict[str, frozenset[str]] = {
    "sm_120": frozenset({"NVIDIA GeForce RTX 5070"}),
}


class NVIDIADeviceClockPacketError(ValueError):
    """Raised when NVIDIA device-clock evidence is contradictory."""


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


_IMAGE_FIELDS = (
    "architecture", "kernel_name", "semantic_sha256", "image_sha256",
    "isa_sha256", "duration_ns", "clock_source", "instrumented",
    "calibration_sample_id",
)


def _image_record(image: Mapping[str, Any], role: str) -> dict[str, Any]:
    missing = [f for f in _IMAGE_FIELDS if image.get(f) is None or image.get(f) == ""]
    if missing:
        raise NVIDIADeviceClockPacketError(f"{role} image record is missing {missing}")
    duration = image["duration_ns"]
    if (not isinstance(duration, (int, float)) or isinstance(duration, bool)
            or not math.isfinite(float(duration)) or duration <= 0):
        raise NVIDIADeviceClockPacketError(f"{role} image duration must be finite and positive")
    if not isinstance(image.get("resources"), Mapping):
        raise NVIDIADeviceClockPacketError(f"{role} image requires resources")
    return dict(image)


def _timing_architecture(timing: Mapping[str, Any]) -> str:
    target = timing.get("target")
    arch = next((a for a, t in NVIDIA_DEVICE_CLOCK_ARCHITECTURES.items() if t == target), None)
    if arch is None:
        raise NVIDIADeviceClockPacketError(
            f"NVIDIA device-clock packet requires exact timing on one of "
            f"{sorted(NVIDIA_DEVICE_CLOCK_ARCHITECTURES.values())}; got {target!r}")
    identity = (timing.get("environment") or {}).get("device_identity")
    queried = identity.get("architecture") if isinstance(identity, Mapping) else None
    # Unlike the older ROCm packets there is no pre-query history to honour:
    # every NVIDIA packet must carry the identity it measured (review of the
    # gfx1201 relabelling case).
    if queried != arch:
        raise NVIDIADeviceClockPacketError(
            f"timing target names {arch!r} but the device queried at record time was {queried!r}")
    return arch


def _check_pairing(timing: Mapping[str, Any], clean: Mapping[str, Any],
                   probe: Mapping[str, Any], maximum: Any) -> str:
    if (not isinstance(maximum, (int, float)) or isinstance(maximum, bool)
            or not math.isfinite(maximum) or not 1.0 <= maximum <= 2.0):
        raise NVIDIADeviceClockPacketError(
            "instrumentation overhead limit must be finite and within [1.0, 2.0]")
    arch = _timing_architecture(timing)
    if clean["clock_source"] != "cuda_event" or probe["clock_source"] != "cuda_event":
        raise NVIDIADeviceClockPacketError("the application duration must be the CUDA event")
    if timing["clocks"]["cuda_event_ns"].get("valid") is not True:
        raise NVIDIADeviceClockPacketError("CUDA event timing is not valid in the calibration sample")
    for field in ("kernel_name", "semantic_sha256", "image_sha256"):
        if clean[field] != probe[field]:
            raise NVIDIADeviceClockPacketError(
                f"instrumentation comparison requires one {field}: the marker brackets "
                "the unmodified image")
    if clean["instrumented"] is not False or probe["instrumented"] is not True:
        raise NVIDIADeviceClockPacketError("instrumentation roles are inconsistent")
    if clean["architecture"] != arch or probe["architecture"] != arch:
        raise NVIDIADeviceClockPacketError(
            f"image records must name {arch}; got {clean['architecture']!r}/{probe['architecture']!r}")
    if (clean["calibration_sample_id"] != timing.get("sample_id")
            or probe["calibration_sample_id"] != timing.get("sample_id")):
        raise NVIDIADeviceClockPacketError("application images do not bind the timing calibration sample")
    return arch


NVIDIA_REASON_VOCABULARY = ReasonVocabulary(
    "NVIDIA device-clock packet", NVIDIA_DEVICE_CLOCK_REASONS,
    registered_origins=(
        "tessera.compiler.profiler_nvidia_evidence",
        "tessera.compiler.profiler_timing.witness_refusal_codes",
        "tessera.compiler.profiler_timing.device_clock_window_refusals",
    ))


def _derive(*, timing: Mapping[str, Any], clean: Mapping[str, Any], probe: Mapping[str, Any],
            source: Mapping[str, Any], maximum: float) -> tuple[float, list[str]]:
    overhead = float(probe["duration_ns"]) / float(clean["duration_ns"])
    if not math.isfinite(overhead) or overhead <= 0:
        raise NVIDIADeviceClockPacketError("instrumentation duration ratio must be finite and positive")
    reasons: list[str] = []
    clocks = timing["clocks"]
    device = clocks["device_wall_clock_ns"]
    if device.get("valid") is not True:
        reasons.append("DEVICE_WALL_CLOCK_INVALID")
    elif "cuda_event_ns" not in device.get("calibrated_against", ()):
        reasons.append("DEVICE_WALL_CLOCK_UNCALIBRATED")
    if device.get("eligible_for_promotion") is not True:
        reasons.append("DEVICE_WALL_CLOCK_NOT_PROMOTION_ELIGIBLE")
    # The witness agreement rule is environment-independent here: bare metal
    # adds no profiler to this route, so the same in-sample agreement decides.
    # Each refusal keeps its own code (a missing witness is not a disagreement).
    reasons.extend(witness_refusal_codes(wsl_promotion_refusals(str(timing["target"]), clocks)))
    # Medians agreeing is not enough: every stored window must be long enough
    # and agree on its own (review: a 10-launch packet with per-window errors
    # up to 18% and a 3.5% median error was admitted).
    reasons.extend(device_clock_window_refusals(timing))
    identity = (timing.get("environment") or {}).get("device_identity") or {}
    arch = identity.get("architecture")
    if identity.get("name") not in NVIDIA_DEVICE_CLOCK_VALIDATED_PARTS.get(str(arch), frozenset()):
        reasons.append("DEVICE_CLOCK_PART_UNVALIDATED")
    if overhead > maximum:
        reasons.append("INSTRUMENTATION_OVERHEAD_EXCEEDED")
    elif overhead < 1.0 / maximum:
        reasons.append("INSTRUMENTATION_CHANGED_THE_KERNEL")
    if clean.get("image_sha256") not in set(timing.get("artifact_digests", {}).values()):
        reasons.append("CALIBRATION_IMAGE_UNBOUND")
    if source.get("worktree_dirty") is not False:
        reasons.append("SOURCE_WORKTREE_DIRTY")
    return overhead, reasons


def build_nvidia_device_clock_packet(
    *,
    timing: Mapping[str, Any],
    uninstrumented: Mapping[str, Any],
    instrumented: Mapping[str, Any],
    source: Mapping[str, Any],
    maximum_instrumentation_overhead: float = 1.05,
) -> dict[str, Any]:
    validate_timing_sample(timing)
    clean = _image_record(uninstrumented, "uninstrumented")
    probe = _image_record(instrumented, "instrumented")
    arch = _check_pairing(timing, clean, probe, maximum_instrumentation_overhead)
    commit = source.get("source_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise NVIDIADeviceClockPacketError("NVIDIA device-clock packet requires full source commit")
    overhead, reasons = _derive(timing=timing, clean=clean, probe=probe, source=source,
                                maximum=maximum_instrumentation_overhead)
    packet = {
        "schema": NVIDIA_DEVICE_CLOCK_PACKET_SCHEMA_VERSION,
        "work_item": "NVIDIA-GLOBALTIMER-MARKER-2026-09-26",
        "architecture": arch,
        "source": dict(source),
        "timing": dict(timing),
        "timing_sha256": _digest(timing),
        "instrumentation_comparison": {
            "uninstrumented": clean,
            "instrumented": probe,
            "duration_ratio": overhead,
            "maximum_duration_ratio": maximum_instrumentation_overhead,
        },
        "eligible_for_promotion": not reasons,
        "ineligibility_reasons": reasons,
        "admission_route": ROUTE_DEVICE_CLOCK,
        "calibration_status": "promotable" if not reasons else "retain_only",
    }
    packet["packet_sha256"] = _digest(packet)
    validate_nvidia_device_clock_packet(packet)
    return packet


def validate_nvidia_device_clock_packet(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != NVIDIA_DEVICE_CLOCK_PACKET_SCHEMA_VERSION:
        raise NVIDIADeviceClockPacketError("unsupported NVIDIA device-clock packet schema")
    timing = payload.get("timing")
    comparison = payload.get("instrumentation_comparison")
    source = payload.get("source")
    if not isinstance(timing, Mapping) or not isinstance(comparison, Mapping) \
            or not isinstance(source, Mapping):
        raise NVIDIADeviceClockPacketError("packet requires timing, instrumentation comparison and source")
    validate_timing_sample(timing)
    clean = _image_record(comparison.get("uninstrumented") or {}, "uninstrumented")
    probe = _image_record(comparison.get("instrumented") or {}, "instrumented")
    maximum = comparison.get("maximum_duration_ratio")
    arch = _check_pairing(timing, clean, probe, maximum)
    if payload.get("architecture") != arch:
        raise NVIDIADeviceClockPacketError(
            f"packet claims {payload.get('architecture')!r} but its timing and images are {arch!r}")
    commit = source.get("source_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise NVIDIADeviceClockPacketError("NVIDIA device-clock packet requires full source commit")
    if _digest(timing) != payload.get("timing_sha256"):
        raise NVIDIADeviceClockPacketError("NVIDIA timing digest mismatch")
    assert isinstance(maximum, (int, float))  # _check_pairing refused anything else
    overhead, derived = _derive(timing=timing, clean=clean, probe=probe, source=source,
                                maximum=float(maximum))
    ratio = comparison.get("duration_ratio")
    if not isinstance(ratio, (int, float)) or abs(float(ratio) - overhead) > 1e-12:
        raise NVIDIADeviceClockPacketError("instrumentation duration ratio mismatch")
    stored = payload.get("ineligibility_reasons")
    if not isinstance(stored, list) or not all(isinstance(r, str) for r in stored):
        raise NVIDIADeviceClockPacketError("invalid NVIDIA ineligibility reasons")
    NVIDIA_REASON_VOCABULARY.require_known(stored, NVIDIADeviceClockPacketError)
    if stored != derived:
        raise NVIDIADeviceClockPacketError(
            f"packet reasons {payload.get('ineligibility_reasons')} differ from those its "
            f"inputs derive {derived}")
    if bool(payload.get("eligible_for_promotion")) != (not derived):
        raise NVIDIADeviceClockPacketError("promotion eligibility differs from its derived reasons")
    if payload.get("admission_route") != ROUTE_DEVICE_CLOCK:
        raise NVIDIADeviceClockPacketError("NVIDIA device-clock packet names a foreign route")
    unsigned = dict(payload)
    stored = unsigned.pop("packet_sha256", None)
    if _digest(unsigned) != stored:
        raise NVIDIADeviceClockPacketError("NVIDIA device-clock packet digest mismatch")


__all__ = [
    "NVIDIADeviceClockPacketError",
    "NVIDIA_DEVICE_CLOCK_ARCHITECTURES",
    "NVIDIA_DEVICE_CLOCK_VALIDATED_PARTS",
    "NVIDIA_DEVICE_CLOCK_PACKET_SCHEMA_VERSION",
    "NVIDIA_DEVICE_CLOCK_REASONS",
    "NVIDIA_REASON_VOCABULARY",
    "ROUTE_DEVICE_CLOCK",
    "build_nvidia_device_clock_packet",
    "validate_nvidia_device_clock_packet",
]
