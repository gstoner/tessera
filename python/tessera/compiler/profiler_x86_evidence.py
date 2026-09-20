"""Exact-host eligibility for Zen 5 profiler benchmark packets."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from .profiler_symbol_sampling import validate_symbol_sampling_artifact


X86_PROFILER_PACKET_SCHEMA_VERSION = "tessera.profiler_x86_packet.v1"
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


def build_x86_profiler_packet(
    *,
    benchmark: Mapping[str, Any],
    timing_status: Mapping[str, Any],
    cpu: Mapping[str, Any],
    environment: Mapping[str, Any],
    sampling: Mapping[str, Any] | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not exact_zen5_cpu(cpu):
        reasons.append("CPU_NOT_EXACT_ZEN5")
    if environment.get("virtualized"):
        reasons.append("VIRTUALIZED_HOST")
    if environment.get("wsl"):
        reasons.append("WSL_CLOCK_DOMAIN")
    source_commit = environment.get("source_commit")
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise X86ProfilerPacketError("Zen 5 packet requires a full source commit")
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
    rows = benchmark.get("rows")
    if not isinstance(rows, list) or not rows:
        raise X86ProfilerPacketError("Zen 5 packet requires benchmark rows")
    shape_classes = {row.get("shape_class") for row in rows if isinstance(row, Mapping)}
    if not {"aligned", "ragged"}.issubset(shape_classes):
        raise X86ProfilerPacketError("Zen 5 packet requires aligned and ragged rows")
    if benchmark.get("architecture") != "zen5-avx512":
        raise X86ProfilerPacketError("benchmark is not the Zen 5 AVX-512 lane")
    if benchmark.get("verdict") == "reject":
        verdict = "reject"
    elif reasons:
        verdict = "retain"
    else:
        verdict = "promote"
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
        "verdict": verdict,
    }
    packet["packet_sha256"] = digest_json(packet)
    validate_x86_profiler_packet(packet)
    return packet


def validate_x86_profiler_packet(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != X86_PROFILER_PACKET_SCHEMA_VERSION:
        raise X86ProfilerPacketError("unsupported x86 profiler packet schema")
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
    if payload.get("eligible_for_promotion") and payload.get("verdict") != "promote":
        raise X86ProfilerPacketError("promotion eligibility requires promote verdict")
    unsigned = dict(payload)
    packet_digest = unsigned.pop("packet_sha256", None)
    if digest_json(unsigned) != packet_digest:
        raise X86ProfilerPacketError("packet digest mismatch")


__all__ = [
    "X86_PROFILER_PACKET_SCHEMA_VERSION",
    "X86ProfilerPacketError",
    "build_x86_profiler_packet",
    "digest_json",
    "exact_zen5_cpu",
    "validate_x86_profiler_packet",
]
