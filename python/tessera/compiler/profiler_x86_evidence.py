"""Exact-host eligibility for Zen 5 profiler benchmark packets."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from typing import Any, Mapping

from .profiler_symbol_sampling import validate_symbol_sampling_artifact


X86_PROFILER_PACKET_SCHEMA_VERSION = "tessera.profiler_x86_packet.v2"
#: Packets recorded before the admission route existed. Accepted for reading
#: only, under the subset rule in the validator; never promotion-eligible
#: unless their stored inputs derive no blocker, and never on tsc_witness.
X86_PROFILER_PACKET_SCHEMA_V1 = "tessera.profiler_x86_packet.v1"
#: E2E-REAL-4's non-regression ratchet: scheduled median <= production * this.
NON_REGRESSION_LIMIT = 1.10
ZEN5_MODEL_TOKEN = "AMD RYZEN AI MAX+ 395"


class X86ProfilerPacketError(ValueError):
    """Raised when a Zen 5 evidence packet is incomplete or contradictory."""


#: The x86 promotion-ineligibility vocabulary, declared in ONE place.
#:
#: These are not diagnostic codes and must never be registered in
#: `diagnostic_codes.py` -- they are the reasons a measured x86 result may not
#: be promoted, and they decide `eligible_for_promotion`. That makes them
#: semantic keys under Decision #21a, and an unknown one must fail CLOSED.
#:
#: Before 2026-09-20 nothing enumerated them: the producer appended eleven bare
#: string literals, the validator checked only that each was a `str`, and no
#: consumer could know the vocabulary was eleven items or that a twelfth had
#: appeared. A reader handling a subset silently treated an unknown reason as
#: NO reason -- which promotes a result that something declined to vouch for.
#: They were found only because the diagnostic-code shape scan happened to see
#: exactly one of them, by the accident of a concatenated colon.
#:
#: A tag may carry a `:detail` suffix (`TIMING_PROOF_INCOMPLETE:a,b`); the part
#: before the colon is the tag and must appear here.
X86_INELIGIBILITY_REASONS: dict[str, str] = {
    "CPU_NOT_EXACT_ZEN5":
        "host CPU is not the exact Zen 5 part the x86 lane is calibrated on",
    "VIRTUALIZED_HOST":
        "running under a hypervisor, so cycle counts are not the bare-metal ones",
    "WSL_CLOCK_DOMAIN":
        "WSL2 clock domain; wall-clock and device-clock latencies are not comparable",
    "SOURCE_WORKTREE_DIRTY":
        "the measured tree has uncommitted changes, so the result names no revision",
    "TIMING_PROOF_INCOMPLETE":
        "one or more required timing proofs are missing; the detail lists which",
    "SYMBOL_SAMPLING_MISSING":
        "no symbol-sampling artifact accompanies the measurement",
    "SYMBOL_SAMPLING_INVALID":
        "the symbol-sampling artifact failed its own validation",
    "IMAGE_BUILD_ID_MISSING":
        "the sampled image carries no build id, so samples cannot be attributed",
    "EVENT_MAP_MISSING":
        "no PMU event map, so counter names cannot be resolved",
    "EVENT_MAP_NOT_PROMOTABLE":
        "the PMU event map is present but not one promotion accepts",
    "SAMPLING_AFFINITY_NOT_PINNED":
        "sampling ran without pinned affinity, so samples may cross cores",
}


#: Clock proofs the tprof sleep probe reports as booleans. On the tsc_witness
#: route they are SUPERSEDED by the per-row timing sample, which re-derives
#: each of them from measured integers (invariant TSC, same logical CPU, a
#: calibration-interval frequency, TSC-vs-raw agreement over the measured
#: region). avx512_visible is not a clock proof and still blocks.
_SUPERSEDED_BY_WITNESS = frozenset({
    "monotonic_raw_valid", "rdtscp_valid", "invariant_tsc", "affinity_stable",
    "clock_agreement_valid", "perf_event_open", "perf_sample_valid",
})
#: Tags about the environment or the profiler, not about whether the timing is
#: true. On the tsc_witness route they are diagnostic gaps.
_ENVIRONMENT_TAGS = frozenset({
    "VIRTUALIZED_HOST", "WSL_CLOCK_DOMAIN", "SYMBOL_SAMPLING_MISSING",
    "SYMBOL_SAMPLING_INVALID", "IMAGE_BUILD_ID_MISSING", "EVENT_MAP_MISSING",
    "EVENT_MAP_NOT_PROMOTABLE", "SAMPLING_AFFINITY_NOT_PINNED",
})
X86_ROUTE_PROFILER = "profiler_correlated"
X86_ROUTE_TSC = "tsc_witness"


def benchmark_verdict(benchmark: Mapping[str, Any]) -> str:
    """E2E-REAL-4's verdict re-derived from the rows' own samples.

    ``reject`` when any row failed correctness; ``promote`` only when every
    row's scheduled median is within :data:`NON_REGRESSION_LIMIT` of its
    production median, computed here from the stored samples rather than
    read from the stored ``non_regression_10pct`` flag; else ``retain``.
    """
    rows = benchmark.get("rows")
    if not isinstance(rows, list) or not rows:
        raise X86ProfilerPacketError("benchmark has no rows to derive a verdict from")
    ratchet = benchmark.get("ratchet")
    if not isinstance(ratchet, Mapping) or ratchet.get("limit") != NON_REGRESSION_LIMIT:
        raise X86ProfilerPacketError(
            f"benchmark ratchet limit must be the E2E-REAL-4 policy {NON_REGRESSION_LIMIT}")
    regressed = False
    for row in rows:
        try:
            passed = row["correctness"]["passed"]
            production = [float(v) for v in row["timing"]["production_samples_ms"]]
            scheduled = [float(v) for v in row["timing"]["scheduled_samples_ms"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise X86ProfilerPacketError(f"benchmark row lacks correctness or samples: {exc}") from exc
        if passed is not True:
            return "reject"
        if not production or not scheduled or not all(
                math.isfinite(v) and v > 0 for v in production + scheduled):
            raise X86ProfilerPacketError("benchmark row samples must be finite and positive")
        within = statistics.median(scheduled) <= statistics.median(production) * NON_REGRESSION_LIMIT
        if row["timing"].get("non_regression_10pct") is not within:
            raise X86ProfilerPacketError(
                "a row's stored non_regression_10pct disagrees with its own samples")
        regressed = regressed or not within
    return "retain" if regressed else "promote"


def _witness_refusal(row: Mapping[str, Any]) -> str | None:
    """Why this row's TSC witness cannot carry the tsc_witness route, or None."""
    from .profiler_timing import (
        CLOCK_AGREEMENT_BAND, ProfilerTimingError, validate_timing_sample,
        wsl_promotion_refusals)
    from .profiler_x86_clock import verify_witness_sample
    sample = row.get("timing_witness")
    if not isinstance(sample, Mapping):
        return "row carries no timing witness"
    try:
        validate_timing_sample(sample)
    except ProfilerTimingError as exc:
        return f"witness invalid: {exc}"
    clocks = sample["clocks"]
    tsc = clocks.get("tsc_cycles", {})
    image = ((row.get("compile") or {}).get("digests") or {}).get("image")
    if sample.get("target") != "x86" or tsc.get("eligible_for_promotion") is not True:
        return "witness TSC is not promotion-eligible"
    refusals = wsl_promotion_refusals("x86", clocks)
    if refusals:
        return "; ".join(refusals)
    if image is None or sample.get("artifact_digests", {}).get("image") != image:
        return "witness does not name this row's image"
    inconsistent = verify_witness_sample(sample)
    if inconsistent:
        return inconsistent
    # Bind the witness to the numbers that decide the verdict: the per-launch
    # host-wall samples are sub-intervals of the witnessed region, so their
    # sum can neither exceed it nor fall short of it by more than the band.
    try:
        timing = row["timing"]
        sample_ns = 1e6 * (sum(float(v) for v in timing["production_samples_ms"])
                           + sum(float(v) for v in timing["scheduled_samples_ms"]))
    except (KeyError, TypeError, ValueError):
        return "row lacks the samples its witness must bind"
    tsc_ns = tsc["value"] * 1e9 / tsc["provenance"]["calibrated_frequency_hz"]
    if not (1.0 - CLOCK_AGREEMENT_BAND) * tsc_ns <= sample_ns <= tsc_ns * (1.0 + 1e-6):
        return (f"per-launch samples ({sample_ns:.0f} ns) are not bound to the witnessed "
                f"region ({tsc_ns:.0f} ns) within {CLOCK_AGREEMENT_BAND:.0%}")
    return None


def x86_admission_route(benchmark: Mapping[str, Any]) -> str:
    """``tsc_witness`` only when EVERY benchmark row's witness survives
    :func:`_witness_refusal`: a valid sample naming the row's image, whose
    TSC -- re-derived from its stored window with the frequency re-derived
    from its stored, separate calibration intervals on the same CPU --
    agrees with CLOCK_MONOTONIC_RAW, and whose region brackets the per-launch
    samples the verdict is computed from. The tprof booleans play no part.

    What it proves under WSL2 is bounded: the raw clock is itself derived from
    the TSC there, so agreement shows a stable TSC scale and binds the
    samples to it, not agreement with an independent oscillator.
    """
    rows = benchmark.get("rows")
    if not isinstance(rows, list) or not rows:
        return X86_ROUTE_PROFILER
    for row in rows:
        if not isinstance(row, Mapping) or _witness_refusal(row) is not None:
            return X86_ROUTE_PROFILER
    return X86_ROUTE_TSC


def _split_for_route(reasons: list[str], route: str) -> tuple[list[str], list[str]]:
    """(blocking reasons, diagnostic gaps) for a route."""
    if route != X86_ROUTE_TSC:
        return reasons, []
    blocking, gaps = [], []
    for reason in reasons:
        tag, _, detail = reason.partition(":")
        if tag in _ENVIRONMENT_TAGS:
            gaps.append(reason)
        elif tag == "TIMING_PROOF_INCOMPLETE":
            fields = [f for f in detail.split(",") if f]
            kept = [f for f in fields if f not in _SUPERSEDED_BY_WITNESS]
            dropped = [f for f in fields if f in _SUPERSEDED_BY_WITNESS]
            if kept:
                blocking.append("TIMING_PROOF_INCOMPLETE:" + ",".join(kept))
            if dropped:
                gaps.append("TIMING_PROOF_INCOMPLETE:" + ",".join(dropped))
        else:
            blocking.append(reason)
    return blocking, gaps


def x86_reason_tag(reason: str) -> str:
    """The tag half of a reason, discarding any `:detail` suffix."""
    return reason.split(":", 1)[0]


def digest_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def exact_zen5_cpu(cpu: Mapping[str, Any]) -> bool:
    return bool(
        cpu.get("vendor_id") == "AuthenticAMD"
        and cpu.get("cpu_family") == 26
        and ZEN5_MODEL_TOKEN in str(cpu.get("model_name", "")).upper()
        and "avx512f" in set(cpu.get("flags", []))
    )


def _derive_reasons(
    timing_status: Mapping[str, Any],
    cpu: Mapping[str, Any],
    environment: Mapping[str, Any],
    sampling: Mapping[str, Any] | None,
) -> list[str]:
    """Every ineligibility reason the packet's own inputs imply, before the
    route split. The validator calls this on the stored inputs, so a packet
    cannot drop or add a reason by editing its lists."""
    reasons: list[str] = []
    if not exact_zen5_cpu(cpu):
        reasons.append("CPU_NOT_EXACT_ZEN5")
    if environment.get("virtualized"):
        reasons.append("VIRTUALIZED_HOST")
    if environment.get("wsl"):
        reasons.append("WSL_CLOCK_DOMAIN")
    if environment.get("worktree_dirty"):
        reasons.append("SOURCE_WORKTREE_DIRTY")
    required_timing = (
        "avx512_visible",
        "monotonic_raw_valid",
        "rdtscp_valid",
        "invariant_tsc",
        "affinity_stable",
        "clock_agreement_valid",
        "perf_event_open",
        "perf_sample_valid",
    )
    missing_timing = [field for field in required_timing if timing_status.get(field) is not True]
    if missing_timing:
        reasons.append("TIMING_PROOF_INCOMPLETE:" + ",".join(missing_timing))
    if sampling is None:
        reasons.append("SYMBOL_SAMPLING_MISSING")
    else:
        validate_symbol_sampling_artifact(sampling)
        if not sampling.get("eligible_for_regression"):
            reasons.append("SYMBOL_SAMPLING_INVALID")
        if not sampling.get("build_id") or sampling.get("build_id") == "not-present":
            reasons.append("IMAGE_BUILD_ID_MISSING")
        event_map = sampling.get("event_map")
        if not isinstance(event_map, Mapping):
            reasons.append("EVENT_MAP_MISSING")
        elif not event_map.get("eligible_for_promotion"):
            reasons.append("EVENT_MAP_NOT_PROMOTABLE")
        affinity = sampling.get("affinity")
        if not isinstance(affinity, Mapping) or affinity.get("pinned") is not True:
            reasons.append("SAMPLING_AFFINITY_NOT_PINNED")
    return reasons


def _packet_verdict(benchmark_derived: str, blocking: list[str]) -> str:
    """A packet can never be stronger than its benchmark: a benchmark that
    rejects or retains yields that verdict whatever the timing route."""
    if benchmark_derived in ("reject", "retain"):
        return benchmark_derived
    return "retain" if blocking else "promote"


def build_x86_profiler_packet(
    *,
    benchmark: Mapping[str, Any],
    timing_status: Mapping[str, Any],
    cpu: Mapping[str, Any],
    environment: Mapping[str, Any],
    sampling: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_commit = environment.get("source_commit")
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise X86ProfilerPacketError("Zen 5 packet requires a full source commit")
    reasons = _derive_reasons(timing_status, cpu, environment, sampling)
    rows = benchmark.get("rows")
    if not isinstance(rows, list) or not rows:
        raise X86ProfilerPacketError("Zen 5 packet requires benchmark rows")
    shape_classes = {row.get("shape_class") for row in rows if isinstance(row, Mapping)}
    if not {"aligned", "ragged"}.issubset(shape_classes):
        raise X86ProfilerPacketError("Zen 5 packet requires aligned and ragged rows")
    if benchmark.get("architecture") != "zen5-avx512":
        raise X86ProfilerPacketError("benchmark is not the Zen 5 AVX-512 lane")
    route = x86_admission_route(benchmark)
    reasons, diagnostic_gaps = _split_for_route(reasons, route)
    derived = benchmark_verdict(benchmark)
    if benchmark.get("verdict") != derived:
        raise X86ProfilerPacketError(
            f"benchmark states verdict {benchmark.get('verdict')!r} but its rows derive {derived!r}")
    verdict = _packet_verdict(derived, reasons)
    packet = {
        "schema": X86_PROFILER_PACKET_SCHEMA_VERSION,
        "work_item": "TPROF-X86-TIME-1",
        "architecture": "zen5-avx512",
        "source_commit": source_commit,
        "cpu": dict(cpu),
        "environment": dict(environment),
        "benchmark": dict(benchmark),
        "benchmark_sha256": digest_json(benchmark),
        "timing_status": dict(timing_status),
        "sampling": dict(sampling) if sampling is not None else None,
        "eligible_for_promotion": not reasons and verdict == "promote",
        "ineligibility_reasons": reasons,
        "admission_route": route,
        "diagnostic_gaps": diagnostic_gaps,
        "verdict": verdict,
    }
    packet["packet_sha256"] = digest_json(packet)
    validate_x86_profiler_packet(packet)
    return packet


def validate_x86_profiler_packet(payload: Mapping[str, Any]) -> None:
    schema = payload.get("schema")
    if schema not in (X86_PROFILER_PACKET_SCHEMA_VERSION, X86_PROFILER_PACKET_SCHEMA_V1):
        raise X86ProfilerPacketError("unsupported x86 profiler packet schema")
    legacy = schema == X86_PROFILER_PACKET_SCHEMA_V1
    if legacy and ("admission_route" in payload or "diagnostic_gaps" in payload):
        raise X86ProfilerPacketError(
            "a v1 x86 packet predates admission routes; route-bearing packets are v2")
    if payload.get("architecture") != "zen5-avx512":
        raise X86ProfilerPacketError("x86 profiler packet requires zen5-avx512")
    if (
        not isinstance(payload.get("source_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", payload["source_commit"]) is None
    ):
        raise X86ProfilerPacketError("x86 profiler packet requires source commit")
    if not isinstance(payload.get("benchmark"), Mapping):
        raise X86ProfilerPacketError("x86 profiler packet requires benchmark")
    if digest_json(payload["benchmark"]) != payload.get("benchmark_sha256"):
        raise X86ProfilerPacketError("benchmark digest mismatch")
    if not isinstance(payload.get("timing_status"), Mapping):
        raise X86ProfilerPacketError("x86 profiler packet requires timing status")
    sampling = payload.get("sampling")
    if sampling is not None:
        if not isinstance(sampling, Mapping):
            raise X86ProfilerPacketError("x86 profiler sampling must be an object")
        validate_symbol_sampling_artifact(sampling)
    reasons = payload.get("ineligibility_reasons")
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        raise X86ProfilerPacketError("invalid ineligibility reasons")
    # Fail CLOSED on an unknown tag (Decision #21a). A reason decides whether a
    # measurement may be promoted, so a tag no consumer knows must stop the
    # packet rather than be carried past readers that will ignore it.
    unknown = sorted({x86_reason_tag(r) for r in reasons} - set(X86_INELIGIBILITY_REASONS))
    if unknown:
        raise X86ProfilerPacketError(
            f"unknown x86 ineligibility reason(s) {unknown}; declare them in "
            f"X86_INELIGIBILITY_REASONS with a meaning, or the packet's readers "
            f"will treat an unknown reason as no reason and promote a result "
            f"something declined to vouch for")
    if payload.get("eligible_for_promotion") and reasons:
        raise X86ProfilerPacketError("promotion-eligible packet has blockers")
    # Re-derive the route and the reason split from the stored benchmark (the
    # same digest-bound record); a stored claim is never trusted.
    route = x86_admission_route(payload["benchmark"])
    if legacy and route != X86_ROUTE_PROFILER:
        raise X86ProfilerPacketError("a v1 x86 packet cannot carry a tsc_witness benchmark")
    if not legacy and payload.get("admission_route") != route:
        raise X86ProfilerPacketError(
            f"x86 packet claims admission route {payload.get('admission_route')!r}, "
            f"but its benchmark supports {route!r}")
    gaps = payload.get("diagnostic_gaps", [])
    if not isinstance(gaps, list) or not all(isinstance(g, str) for g in gaps):
        raise X86ProfilerPacketError("invalid x86 diagnostic gaps")
    unknown_gaps = sorted({x86_reason_tag(g) for g in gaps} - set(X86_INELIGIBILITY_REASONS))
    if unknown_gaps:
        raise X86ProfilerPacketError(f"unknown x86 gap tag(s) {unknown_gaps}")
    for field_name in ("cpu", "environment"):
        if not isinstance(payload.get(field_name), Mapping):
            raise X86ProfilerPacketError(f"x86 profiler packet requires {field_name}")
    blocking, derived_gaps = _split_for_route(
        _derive_reasons(payload["timing_status"], payload["cpu"], payload["environment"], sampling),
        route)
    if not legacy:
        # Current schema: the lists are exactly what the inputs derive.
        if sorted(blocking) != sorted(reasons) or sorted(derived_gaps) != sorted(gaps):
            raise X86ProfilerPacketError(
                "x86 reasons/gaps differ from what the packet's stored inputs derive")
    else:
        # Packets recorded before a reason existed (e.g. the 2026-08-06 packet
        # predates EVENT_MAP_*/SAMPLING_AFFINITY_*) may omit newer reasons;
        # everything they DO state must still derive, and the derived blockers
        # below still forbid promotion.
        if not set(reasons) <= set(blocking) or gaps:
            raise X86ProfilerPacketError(
                "legacy x86 packet states reasons its stored inputs do not derive")
    if legacy:
        # v1 packets predate the row-sample re-derivation; their stored
        # benchmark verdict still caps the packet's.
        stated = payload["benchmark"].get("verdict")
        benchmark_derived = stated if stated in ("reject", "retain") else "promote"
    else:
        benchmark_derived = benchmark_verdict(payload["benchmark"])
        if payload["benchmark"].get("verdict") != benchmark_derived:
            raise X86ProfilerPacketError(
                "benchmark's stated verdict differs from what its rows derive")
    expected_verdict = _packet_verdict(benchmark_derived, blocking)
    if payload.get("verdict") != expected_verdict:
        raise X86ProfilerPacketError(
            f"x86 packet verdict {payload.get('verdict')!r} is not the derived {expected_verdict!r}")
    if payload.get("eligible_for_promotion") and payload.get("verdict") != "promote":
        raise X86ProfilerPacketError("promotion eligibility requires promote verdict")
    unsigned = dict(payload)
    packet_digest = unsigned.pop("packet_sha256", None)
    if digest_json(unsigned) != packet_digest:
        raise X86ProfilerPacketError("packet digest mismatch")


__all__ = [
    "X86_PROFILER_PACKET_SCHEMA_VERSION",
    "NON_REGRESSION_LIMIT",
    "X86_PROFILER_PACKET_SCHEMA_V1",
    "X86_ROUTE_PROFILER",
    "X86_ROUTE_TSC",
    "benchmark_verdict",
    "x86_admission_route",
    "X86ProfilerPacketError",
    "build_x86_profiler_packet",
    "digest_json",
    "exact_zen5_cpu",
    "validate_x86_profiler_packet",
]
