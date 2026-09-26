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

_ALLOWED_SOURCES: dict[str, frozenset[str]] = {
    "host_wall_ns": frozenset({"steady_clock", "perf_counter"}),
    "hip_event_ns": frozenset({"hip_event"}),
    "device_wall_clock_ns": frozenset({"device_wall_clock"}),
    "profiler_activity_ns": frozenset({"rocprofiler_activity", "rtg_hsa_dispatch"}),
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
    "device_wall_clock_ns": frozenset({"hip_event_ns", "profiler_activity_ns"}),
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
    A target with no kernel-side slot in this schema (NVIDIA: its non-profiler
    ``%globaltimer`` witness is not implemented; its Nsight activity-window
    calibration lives in ``profiler_cuda_window``) has none here.
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


def expected_clock_slots(target: str) -> tuple[str, ...]:
    normalized = target.strip().lower().replace("-", "_")
    if normalized.startswith("gfx") or normalized.startswith("rocm"):
        return ROCM_CLOCK_SLOTS
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
        and not {"hip_event_ns", "profiler_activity_ns"}.intersection(calibrated)
    ):
        raise ProfilerTimingError("promotion device_wall_clock_ns requires HIP-event or profiler-activity calibration")
    if slot == "device_wall_clock_ns" and valid and not record.get("instrumented"):
        raise ProfilerTimingError("device_wall_clock_ns must be marked instrumented")

    provenance = record["provenance"]
    if slot == "tsc_cycles" and valid:
        if provenance.get("invariant_tsc") is not True:
            raise ProfilerTimingError("tsc_cycles requires invariant_tsc proof")
        if provenance.get("logical_cpu_start") != provenance.get("logical_cpu_end"):
            raise ProfilerTimingError("migrated tsc_cycles sample is invalid")
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
    "PROFILER_TIMING_SCHEMA_VERSION",
    "ProfilerTimingError",
    "ROCM_CLOCK_SLOTS",
    "X86_CLOCK_SLOTS",
    "build_timing_sample",
    "expected_clock_slots",
    "measure_synchronized_host_batch",
    "measured_clock",
    "unavailable_clock",
    "validate_clock_record",
    "validate_timing_sample",
    "wall_clock_ticks_to_ns",
]
