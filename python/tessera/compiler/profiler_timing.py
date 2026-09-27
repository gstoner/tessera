"""Typed multi-clock evidence for profiler and benchmark packets.

The contract deliberately keeps clock domains independent.  An unavailable HIP
event, perf event, or provider activity interval stays unavailable; callers must
never fill its slot with host-wall time merely to make a packet complete.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping


PROFILER_TIMING_SCHEMA_VERSION = "tessera.profiler_timing.v1"

ROCM_CLOCK_SLOTS = (
    "host_wall_ns",
    "hip_event_ns",
    "device_wall_clock_ns",
    "profiler_activity_ns",
)

X86_CLOCK_SLOTS = (
    "host_wall_ns",
    "monotonic_raw_ns",
    "tsc_cycles",
    "perf_task_clock_ns",
)

#: NVIDIA sm_120 (sync NVIDIA-GLOBALTIMER-MARKER-2026-09-26). The kernel-side
#: clock is ``%globaltimer`` read by the compiler-built marker
#: (``native_device_clock``), already in ns; its witness is the CUDA event
#: bracketing the same stream interval. ``profiler_activity_ns`` is the CUPTI /
#: Nsight activity window, recorded unavailable when not captured.
NVIDIA_CLOCK_SLOTS = (
    "host_wall_ns",
    "cuda_event_ns",
    "device_wall_clock_ns",
    "profiler_activity_ns",
)

_ALLOWED_SOURCES: dict[str, frozenset[str]] = {
    "host_wall_ns": frozenset({"steady_clock", "perf_counter"}),
    "hip_event_ns": frozenset({"hip_event"}),
    "cuda_event_ns": frozenset({"cuda_event"}),
    "device_wall_clock_ns": frozenset({"device_wall_clock"}),
    "profiler_activity_ns": frozenset({"rocprofiler_activity", "rtg_hsa_dispatch", "cupti_activity"}),
    "monotonic_raw_ns": frozenset({"clock_monotonic_raw"}),
    "tsc_cycles": frozenset({"rdtscp"}),
    "perf_task_clock_ns": frozenset({"perf_event_task_clock"}),
}


class ProfilerTimingError(ValueError):
    """Raised when a timing artifact could misrepresent measurement evidence."""


@dataclass(frozen=True)
class ClockRecord:
    """One measurement source in one timing domain."""

    slot: str
    source: str
    value: int | float | None
    unit: str = "ns"
    valid: bool = False
    reason: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    instrumented: bool = False
    calibrated_against: tuple[str, ...] = ()
    eligible_for_regression: bool = False
    eligible_for_promotion: bool = False
    raw_value: int | float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "source": self.source,
            "value": self.value,
            "unit": self.unit,
            "valid": self.valid,
            "reason": self.reason,
            "provenance": dict(self.provenance),
            "instrumented": self.instrumented,
            "calibrated_against": list(self.calibrated_against),
            "eligible_for_regression": self.eligible_for_regression,
            "eligible_for_promotion": self.eligible_for_promotion,
            "raw_value": self.raw_value,
        }


def unavailable_clock(
    slot: str,
    *,
    source: str,
    reason: str,
    unit: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    raw_value: int | float | None = None,
) -> ClockRecord:
    """Build an explicit unavailable record without fabricating zero evidence."""

    return ClockRecord(
        slot=slot,
        source=source,
        value=None,
        unit=unit or ("cycles" if slot == "tsc_cycles" else "ns"),
        valid=False,
        reason=reason,
        provenance=dict(provenance or {}),
        raw_value=raw_value,
    )


def measured_clock(
    slot: str,
    *,
    source: str,
    value: int | float,
    unit: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    instrumented: bool = False,
    calibrated_against: Iterable[str] = (),
    eligible_for_regression: bool = True,
    eligible_for_promotion: bool = False,
) -> ClockRecord:
    record = ClockRecord(
        slot=slot,
        source=source,
        value=value,
        unit=unit or ("cycles" if slot == "tsc_cycles" else "ns"),
        valid=True,
        provenance=dict(provenance or {}),
        instrumented=instrumented,
        calibrated_against=tuple(calibrated_against),
        eligible_for_regression=eligible_for_regression,
        eligible_for_promotion=eligible_for_promotion,
        raw_value=value,
    )
    validate_clock_record(record.to_dict())
    return record


#: Clocks measured on the device or core itself, independent of the host event
#: API. Under WSL these are the only slots that may carry promotion, and only
#: with a calibration witness that is valid in the same sample **and agrees
#: with it**: the in-kernel ``wall_clock64`` cross-checked against HIP events and
#: host wall agreed to four significant figures on gfx1151
#: (DEVICE-CLOCK-DISCIPLINE-2026-08-31), and the owner accepted that method as
#: performance evidence without a profiler or KFD (MASTER_AUDIT, 2026-09-25).
#: Host-wall, event and profiler slots stay regression-only under WSL: none of
#: them is independent of the virtualized host path on its own.
KERNEL_SIDE_CLOCKS = frozenset({"device_wall_clock_ns", "tsc_cycles"})

#: Relative agreement a witness must reach, ``|witness - clock| / witness``.
#: The same band and the same relation the native providers compute
#: (``rocm_timing_provider.hip`` device vs HIP event; ``tprof.cpp`` x86 clocks),
#: so a Python-side admission cannot be looser than the probe that fed it.
CLOCK_AGREEMENT_BAND = 0.05
#: Kernel clocksources Linux selects only when it trusts the TSC to be
#: synchronized across CPUs (`tsc`), or that derive time from a hypervisor-
#: synchronized TSC page (`hyperv_clocksource_tsc_page`, WSL2). Under one of
#: these a TSC delta read on two CPUs is meaningful; the 5% agreement with
#: CLOCK_MONOTONIC_RAW still has to hold, so a real skew is caught.
TSC_SYNCHRONIZED_CLOCKSOURCES = frozenset({"tsc", "hyperv_clocksource_tsc_page"})


#: The stable reason a recorder stamps when its WSL timing cannot promote.
#: Since 2026-09-25 the blocker is never "bare metal": it is the absence of a
#: kernel-side clock with an agreeing in-sample witness. Recorders that time
#: with events or host wall only use this so every refusal names that.
WSL_WITNESS_MISSING = "kernel_clock_witness_required"
WSL_WITNESS_MISSING_DETAIL = (
    "promotion needs a kernel-side device clock with an agreeing in-sample "
    "witness (MASTER_AUDIT 2026-09-25); this recorder times with events or "
    "host wall only"
)


#: Execution environments that are WSL. Matched exactly (review): a
#: substring test let strings such as ``bare_metal_not_wsl`` change routes.
WSL_ENVIRONMENTS = frozenset({"wsl", "wsl2"})

#: Per promotion clock, the witnesses that may vouch for it. Host wall is never
#: one: it rides the same virtualized host path, and a device clock that
#: agrees with host wall while its HIP event is 50% off must not promote
#: (review). These are the partners ``validate_clock_record`` already demands
#: for a promotion device clock.
_ADMISSIBLE_WITNESSES: dict[str, frozenset[str]] = {
    # Intersected with the target's own slots, so a ROCm sample cannot be
    # vouched for by a CUDA event or an NVIDIA one by a HIP event.
    "device_wall_clock_ns": frozenset({"hip_event_ns", "cuda_event_ns", "profiler_activity_ns"}),
    "tsc_cycles": frozenset({"monotonic_raw_ns"}),
}

#: Where a TSC frequency may come from. Deriving it from ``tsc / raw`` over the
#: very interval being checked makes TSC-vs-raw agree by construction (review
#: of the x86 route); the frequency must come from somewhere else.
INDEPENDENT_TSC_FREQUENCY_SOURCES = frozenset({
    "cpuid_leaf_0x15", "independent_calibration_interval",
})


def is_wsl_environment(environment: object) -> bool:
    return str(environment).strip().lower() in WSL_ENVIRONMENTS


def promotion_clock_slots(target: str) -> frozenset[str]:
    """Kernel-side clocks that may carry promotion for ``target``.

    Target-specific (review of #854): a ROCm sample may not promote through an
    appended ``tsc_cycles`` record, nor an x86 sample through a device clock.
    NVIDIA sm_120 gained its slot on 2026-09-26: the ``%globaltimer`` marker
    was validated on The-Super-Bear against CUDA events (sync
    NVIDIA-GLOBALTIMER-MARKER-2026-09-26); its Nsight activity-window
    calibration remains separately in ``profiler_cuda_window``. A target with
    no kernel-side slot in this schema has none here.
    """
    return KERNEL_SIDE_CLOCKS.intersection(expected_clock_slots(target))


def _clock_ns(record: Mapping[str, Any]) -> float | None:
    value = record.get("value")
    if record.get("valid") is not True or not isinstance(value, (int, float)):
        return None
    if record.get("unit") == "cycles":
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            return None
        hz = provenance.get("calibrated_frequency_hz")
        if (provenance.get("frequency_source") not in INDEPENDENT_TSC_FREQUENCY_SOURCES
                or not isinstance(hz, (int, float)) or isinstance(hz, bool) or hz <= 0):
            return None
        return float(value) * 1.0e9 / float(hz)
    return float(value)


def wsl_promotion_refusals(target: str, clocks: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Why each promotion-eligible slot in a WSL sample is not admissible.

    Empty means every promotion claim uses a kernel-side clock of this target
    whose admissible witnesses (never host wall) include at least one valid in
    the same sample, and **every** valid admissible witness it names agrees
    within :data:`CLOCK_AGREEMENT_BAND` -- one disagreeing witness refuses,
    because a clock two witnesses cannot agree on is not measured.
    """
    allowed = promotion_clock_slots(target)
    target_slots = set(expected_clock_slots(target))
    reasons: list[str] = []
    for slot, record in clocks.items():
        if not record.get("eligible_for_promotion"):
            continue
        if slot not in allowed:
            reasons.append(
                f"{slot}: under WSL only a kernel-side clock of target "
                f"{target!r} ({', '.join(sorted(allowed)) or 'none'}) may carry promotion")
            continue
        clock_ns = _clock_ns(record)
        if clock_ns is None or clock_ns <= 0:
            reasons.append(
                f"{slot}: value cannot be expressed in ns (a TSC needs a "
                f"calibrated_frequency_hz from {sorted(INDEPENDENT_TSC_FREQUENCY_SOURCES)})")
            continue
        witnesses = _ADMISSIBLE_WITNESSES.get(slot, frozenset()) & target_slots
        agreeing: list[str] = []
        disagreeing: list[str] = []
        for name in record.get("calibrated_against", ()):
            witness = clocks.get(name)
            if name not in witnesses or not isinstance(witness, Mapping):
                continue
            witness_ns = _clock_ns(witness)
            if witness_ns is None or witness_ns <= 0:
                continue
            error = abs(witness_ns - clock_ns) / witness_ns
            (agreeing if error <= CLOCK_AGREEMENT_BAND else disagreeing).append(
                f"{name} ({error:.1%})")
        if disagreeing:
            reasons.append(
                f"{slot}: witnesses disagree beyond {CLOCK_AGREEMENT_BAND:.0%}: "
                + ", ".join(disagreeing))
        elif not agreeing:
            reasons.append(
                f"{slot}: no admissible witness ({', '.join(sorted(witnesses)) or 'none'}) "
                "is valid in the sample, so the device clock has no independent witness")
    return reasons


#: NVIDIA targets whose device clock is validated. Exact names, not a prefix:
#: another compute capability gets no device-clock slot until its own proof.
NVIDIA_CLOCK_TARGETS = frozenset({"nvidia_sm120"})


#: Shortest bracketed window (ns, the whole window: per-launch value times the
#: launch count) a device-clock calibration may use, per exact target. The
#: span and the event interval differ by a roughly fixed per-window offset, so
#: agreement is a property of window LENGTH, and a window short enough for the
#: offset to dominate must not vouch for anything even if it happens to agree.
#:
#: * ``nvidia_sm120`` -- 1 ms. Measured on The-Super-Bear (RTX 5070, WSL2,
#:   driver 610.88; ``benchmarks/baselines/sm120_ssd_calibrated_pairs_20260926/
#:   diagnostics/globaltimer_marker_probe.json``): offset ~10-16 us (up to ~46
#:   us); every window of >= ~1 ms agreed within 5% (worst 3.2%), every
#:   configuration of <= ~0.36 ms had a window outside it.
#: * ``rocm_gfx1151`` / ``rocm_gfx1201`` -- 5 ms. gfx1201 showed a ~60 us offset
#:   and 1.6 ms windows disagreeing by up to 6.8%; the legacy gfx1151 packet's
#:   1.36 ms windows reached 9.2% per window. No clean boundary was probed on
#:   ROCm, so this is ~3x the length at which disagreement was seen, and the
#:   committed interleaved packets' shortest windows (13.4 / 15.8 ms) clear it.
#:
#: A target missing here has no minimum and therefore no window route.
MINIMUM_DEVICE_CLOCK_WINDOW_NS: dict[str, float] = {
    "nvidia_sm120": 1_000_000.0,
    "rocm_gfx1151": 5_000_000.0,
    "rocm_gfx1201": 5_000_000.0,
}

#: The witness slot whose per-window values a device-clock calibration stores
#: in ``environment.per_window_event_ns``, per target family.
_WINDOW_WITNESS = {"nvidia": "cuda_event_ns", "rocm": "hip_event_ns"}


def device_clock_window_refusals(timing: Mapping[str, Any]) -> list[str]:
    """Per-window reasons a marker-bracketed calibration cannot vouch for itself.

    A sample's median device clock agreeing with its median event is not
    enough (NVIDIA pre-PR review, 2026-09-26): a packet rebuilt at 10 launches
    per window had per-window errors up to 18% and a median error of 3.5%,
    and was admitted. This reads the per-window values the recorder stores
    (``clocks.device_wall_clock_ns.provenance.per_window_ns`` and
    ``environment.per_window_event_ns``, both per launch) and requires

    * both present, equal in length, finite and positive
      (``DEVICE_CLOCK_WINDOWS_MISSING``);
    * the stored medians to be the medians of those windows
      (``DEVICE_CLOCK_WINDOWS_UNBOUND``);
    * every window at least :data:`MINIMUM_DEVICE_CLOCK_WINDOW_NS` long
      (``DEVICE_CLOCK_WINDOW_TOO_SHORT``);
    * every window's device clock within :data:`CLOCK_AGREEMENT_BAND` of its
      event (``DEVICE_CLOCK_WINDOW_DISAGREES``).
    """
    import statistics

    target = str(timing.get("target", ""))
    minimum = MINIMUM_DEVICE_CLOCK_WINDOW_NS.get(target)
    witness = _WINDOW_WITNESS.get(target.split("_", 1)[0])
    clocks = timing.get("clocks") or {}
    device = clocks.get("device_wall_clock_ns") or {}
    per_device: Any = (device.get("provenance") or {}).get("per_window_ns")
    per_event: Any = (timing.get("environment") or {}).get("per_window_event_ns")
    launches = timing.get("batch_size")

    def usable(values: Any) -> bool:
        return (isinstance(values, (list, tuple)) and len(values) > 0
                and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(float(v)) and float(v) > 0 for v in values))

    if (minimum is None or witness is None or not usable(per_device) or not usable(per_event)
            or len(per_device) != len(per_event)
            or not isinstance(launches, int) or isinstance(launches, bool) or launches <= 0):
        return ["DEVICE_CLOCK_WINDOWS_MISSING"]
    per_device = [float(v) for v in per_device]
    per_event = [float(v) for v in per_event]
    reasons: list[str] = []
    stored_device = _clock_ns(device)
    stored_event = _clock_ns(clocks.get(witness) or {})
    if (stored_device is None or stored_event is None
            or not math.isclose(statistics.median(per_device), stored_device, rel_tol=1e-9)
            or not math.isclose(statistics.median(per_event), stored_event, rel_tol=1e-9)):
        reasons.append("DEVICE_CLOCK_WINDOWS_UNBOUND")
    if any(float(event) * launches < minimum for event in per_event):
        reasons.append("DEVICE_CLOCK_WINDOW_TOO_SHORT")
    if any(abs(float(event) - float(dev)) / float(event) > CLOCK_AGREEMENT_BAND
           for dev, event in zip(per_device, per_event)):
        reasons.append("DEVICE_CLOCK_WINDOW_DISAGREES")
    return reasons


def witness_refusal_codes(refusals: Iterable[str]) -> list[str]:
    """Stable reason codes for :func:`wsl_promotion_refusals` messages.

    Mapped precisely (review): a missing witness is not a disagreement.
    """
    codes: list[str] = []
    for text in refusals:
        if "witnesses disagree" in text:
            code = "DEVICE_CLOCK_WITNESS_DISAGREES"
        elif "no admissible witness" in text:
            code = "DEVICE_CLOCK_WITNESS_MISSING"
        elif "may carry promotion" in text:
            code = "DEVICE_CLOCK_SLOT_INADMISSIBLE"
        elif "cannot be expressed in ns" in text:
            code = "DEVICE_CLOCK_VALUE_UNUSABLE"
        else:
            raise ProfilerTimingError(f"unmapped witness refusal: {text!r}")
        if code not in codes:
            codes.append(code)
    return codes


def expected_clock_slots(target: str) -> tuple[str, ...]:
    normalized = target.strip().lower().replace("-", "_")
    if normalized.startswith("gfx") or normalized.startswith("rocm"):
        return ROCM_CLOCK_SLOTS
    if normalized in NVIDIA_CLOCK_TARGETS:
        return NVIDIA_CLOCK_SLOTS
    if normalized in {"x86", "x86_64", "x86_avx512"} or normalized.startswith("x86_"):
        return X86_CLOCK_SLOTS
    return ("host_wall_ns",)


def build_timing_sample(
    *,
    sample_id: str,
    target: str,
    clocks: Mapping[str, ClockRecord | Mapping[str, Any]],
    artifact_digests: Mapping[str, str],
    batch_size: int,
    warm_state: str,
    synchronization: str,
    execution_environment: str,
    resources: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": PROFILER_TIMING_SCHEMA_VERSION,
        "sample_id": sample_id,
        "target": target,
        "execution_environment": execution_environment,
        "batch_size": batch_size,
        "warm_state": warm_state,
        "synchronization": synchronization,
        "artifact_digests": dict(artifact_digests),
        "resources": dict(resources or {}),
        "environment": dict(environment or {}),
        "clocks": {
            slot: record.to_dict() if isinstance(record, ClockRecord) else dict(record)
            for slot, record in clocks.items()
        },
    }
    validate_timing_sample(payload)
    return payload


def validate_clock_record(record: Mapping[str, Any]) -> None:
    slot = record.get("slot")
    source = record.get("source")
    if slot not in _ALLOWED_SOURCES:
        raise ProfilerTimingError(f"unknown clock slot {slot!r}")
    if source not in _ALLOWED_SOURCES[slot]:
        raise ProfilerTimingError(f"clock slot {slot!r} cannot contain source {source!r}")
    unit = record.get("unit")
    expected_unit = "cycles" if slot == "tsc_cycles" else "ns"
    if unit != expected_unit:
        raise ProfilerTimingError(f"clock slot {slot!r} requires unit {expected_unit!r}")
    if not isinstance(record.get("provenance"), Mapping):
        raise ProfilerTimingError(f"clock slot {slot!r} requires provenance")
    calibrated = record.get("calibrated_against")
    if not isinstance(calibrated, (list, tuple)) or not all(isinstance(value, str) for value in calibrated):
        raise ProfilerTimingError(f"clock slot {slot!r} requires calibrated_against strings")

    valid = record.get("valid") is True
    value = record.get("value")
    if valid:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ProfilerTimingError(f"valid clock slot {slot!r} requires a value")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ProfilerTimingError(f"valid clock slot {slot!r} requires a positive finite value")
        if record.get("reason") not in {None, ""}:
            raise ProfilerTimingError(f"valid clock slot {slot!r} cannot carry an unavailable reason")
    else:
        if value is not None:
            raise ProfilerTimingError(
                f"invalid clock slot {slot!r} must use value=null; preserve raw input in raw_value"
            )
        if not isinstance(record.get("reason"), str) or not record.get("reason"):
            raise ProfilerTimingError(f"invalid clock slot {slot!r} requires a stable reason")
        if record.get("eligible_for_regression") or record.get("eligible_for_promotion"):
            raise ProfilerTimingError(f"invalid clock slot {slot!r} cannot be verdict-eligible")

    if record.get("eligible_for_promotion") and not calibrated:
        raise ProfilerTimingError(f"promotion clock slot {slot!r} requires calibration")
    if source == "rtg_hsa_dispatch" and record.get("eligible_for_promotion"):
        raise ProfilerTimingError("rtg_hsa_dispatch is never promotion-eligible")
    if (
        slot == "device_wall_clock_ns"
        and record.get("eligible_for_promotion")
        and not {"hip_event_ns", "cuda_event_ns", "profiler_activity_ns"}.intersection(calibrated)
    ):
        raise ProfilerTimingError(
            "promotion device_wall_clock_ns requires HIP-event, CUDA-event or profiler-activity calibration")
    if slot == "device_wall_clock_ns" and valid and not record.get("instrumented"):
        raise ProfilerTimingError("device_wall_clock_ns must be marked instrumented")

    provenance = record["provenance"]
    if slot == "tsc_cycles" and valid:
        if provenance.get("invariant_tsc") is not True:
            raise ProfilerTimingError("tsc_cycles requires invariant_tsc proof")
        if (provenance.get("logical_cpu_start") != provenance.get("logical_cpu_end")
                and provenance.get("clocksource") not in TSC_SYNCHRONIZED_CLOCKSOURCES):
            raise ProfilerTimingError(
                "tsc_cycles read on two CPUs is valid only under a TSC-synchronized "
                f"kernel clocksource ({', '.join(sorted(TSC_SYNCHRONIZED_CLOCKSOURCES))})")
        frequency_hz = provenance.get("calibrated_frequency_hz")
        if not isinstance(frequency_hz, (int, float)) or frequency_hz <= 0:
            raise ProfilerTimingError("tsc_cycles requires a positive calibrated_frequency_hz")


def validate_timing_sample(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != PROFILER_TIMING_SCHEMA_VERSION:
        raise ProfilerTimingError("unsupported profiler timing schema")
    if not isinstance(payload.get("sample_id"), str) or not payload.get("sample_id"):
        raise ProfilerTimingError("timing sample requires sample_id")
    target = payload.get("target")
    if not isinstance(target, str) or not target:
        raise ProfilerTimingError("timing sample requires target")
    if not isinstance(payload.get("batch_size"), int) or payload["batch_size"] <= 0:
        raise ProfilerTimingError("timing sample requires positive batch_size")
    if payload.get("warm_state") not in {"cold", "warm"}:
        raise ProfilerTimingError("warm_state must be cold or warm")
    if not isinstance(payload.get("synchronization"), str) or not payload.get("synchronization"):
        raise ProfilerTimingError("timing sample requires synchronization")
    if not isinstance(payload.get("artifact_digests"), Mapping) or not payload.get("artifact_digests"):
        raise ProfilerTimingError("timing sample requires artifact_digests")
    if not all(
        isinstance(name, str) and bool(name) and isinstance(digest, str) and bool(digest)
        for name, digest in payload["artifact_digests"].items()
    ):
        raise ProfilerTimingError("artifact_digests must contain non-empty strings")
    if not isinstance(payload.get("execution_environment"), str) or not payload.get("execution_environment"):
        raise ProfilerTimingError("timing sample requires execution_environment")
    for field_name in ("resources", "environment"):
        if not isinstance(payload.get(field_name), Mapping):
            raise ProfilerTimingError(f"timing sample requires {field_name} mapping")

    clocks = payload.get("clocks")
    if not isinstance(clocks, Mapping):
        raise ProfilerTimingError("timing sample requires clocks")
    expected = expected_clock_slots(target)
    missing = [slot for slot in expected if slot not in clocks]
    if missing:
        raise ProfilerTimingError(f"timing sample is missing clock slots: {missing}")
    for slot, record in clocks.items():
        if not isinstance(record, Mapping):
            raise ProfilerTimingError(f"clock slot {slot!r} must be an object")
        if record.get("slot") != slot:
            raise ProfilerTimingError(f"clock map key {slot!r} does not match record slot {record.get('slot')!r}")
        validate_clock_record(record)

    environment = str(payload.get("execution_environment", "")).strip().lower()
    if environment != "bare_metal" and not is_wsl_environment(environment):
        # Fail closed: only the two environments the producers emit carry
        # promotion rules; an unrecognized string gets no promotion at all.
        promoted = [slot for slot, record in clocks.items() if record.get("eligible_for_promotion")]
        if promoted:
            raise ProfilerTimingError(
                f"unknown execution environment {environment!r} cannot carry promotion: {promoted}")
    if is_wsl_environment(environment):
        # Was a blanket ban on WSL promotion. Replaced 2026-09-25 by the
        # independent-witness rule above; bare-metal rules are unchanged.
        refusals = wsl_promotion_refusals(str(target), clocks)
        if refusals:
            raise ProfilerTimingError(
                "WSL timing sample is not promotion-admissible: " + "; ".join(refusals))


def wall_clock_ticks_to_ns(ticks: int, wall_clock_rate_khz: int) -> int:
    if ticks < 0:
        raise ProfilerTimingError("wall-clock ticks must be non-negative")
    if wall_clock_rate_khz <= 0:
        raise ProfilerTimingError("wall-clock rate must be positive")
    return (ticks * 1_000_000) // wall_clock_rate_khz


def measure_synchronized_host_batch(
    submit: Callable[[], Any],
    synchronize: Callable[[], Any],
    *,
    batch_size: int,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, int]:
    """Measure asynchronous submissions followed by exactly one synchronization."""

    if batch_size <= 0:
        raise ProfilerTimingError("batch_size must be positive")
    start = clock_ns()
    for _ in range(batch_size):
        submit()
    synchronize()
    end = clock_ns()
    duration = end - start
    if duration <= 0:
        raise ProfilerTimingError("host batch clock did not advance")
    return {
        "raw_batch_ns": duration,
        "raw_per_launch_ns": duration // batch_size,
        "batch_size": batch_size,
    }


__all__ = [
    "ClockRecord",
    "MINIMUM_DEVICE_CLOCK_WINDOW_NS",
    "NVIDIA_CLOCK_SLOTS",
    "NVIDIA_CLOCK_TARGETS",
    "PROFILER_TIMING_SCHEMA_VERSION",
    "ProfilerTimingError",
    "ROCM_CLOCK_SLOTS",
    "X86_CLOCK_SLOTS",
    "build_timing_sample",
    "device_clock_window_refusals",
    "expected_clock_slots",
    "measure_synchronized_host_batch",
    "measured_clock",
    "unavailable_clock",
    "validate_clock_record",
    "validate_timing_sample",
    "wall_clock_ticks_to_ns",
    "witness_refusal_codes",
]
