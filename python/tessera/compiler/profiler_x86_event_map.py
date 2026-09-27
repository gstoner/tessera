"""Exact-machine Linux PMU/event-map evidence for x86 profiler packets."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .evidence_reasons import ReasonVocabulary


X86_EVENT_MAP_SCHEMA_VERSION = "tessera.profiler_x86_event_map.v1"


class X86EventMapError(ValueError):
    """Raised when an x86 event catalog cannot support exact-host evidence."""


#: Why an event map may not back a promotable x86 packet. Declared 2026-09-27
#: (X86-EVIDENCE-VOCAB-1): the builder appended these as bare literals and the
#: validator only checked each was a `str`, so a stored map could carry any
#: reason -- or drop one -- and still validate. It now re-derives them.
X86_EVENT_MAP_REASONS: dict[str, str] = {
    "CPU_NOT_EXACT_ZEN5_FAMILY":
        "the catalog was read on a CPU that is not AMD family 26 with AVX-512",
    "VIRTUALIZED_HOST":
        "the catalog was read under a hypervisor, whose PMU may be filtered",
    "PERF_EVENT_CATALOG_UNAVAILABLE":
        "`perf` or its event listing was unavailable, so no event names resolve",
}
X86_EVENT_MAP_VOCABULARY = ReasonVocabulary("x86 event map", X86_EVENT_MAP_REASONS)


def _derive(cpu: Mapping[str, Any], environment: Mapping[str, Any],
            event_sources: Mapping[str, Any], perf: Mapping[str, Any]) -> tuple[bool, bool, list[str]]:
    """(catalog present, exact Zen 5 family, reasons) from the stored inputs."""
    catalog_present = bool(
        event_sources
        and perf.get("available")
        and perf.get("version_returncode") == 0
        and perf.get("list_returncode") == 0
        and perf.get("catalog_text")
    )
    exact_zen5 = bool(
        cpu.get("vendor_id") == "AuthenticAMD"
        and cpu.get("cpu_family") == 26
        and "avx512f" in set(cpu.get("flags", ()))
    )
    reasons: list[str] = []
    if not exact_zen5:
        reasons.append("CPU_NOT_EXACT_ZEN5_FAMILY")
    if environment.get("virtualized"):
        reasons.append("VIRTUALIZED_HOST")
    if not catalog_present:
        reasons.append("PERF_EVENT_CATALOG_UNAVAILABLE")
    return catalog_present, exact_zen5, reasons


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_x86_event_map(
    *, cpu: Mapping[str, Any], environment: Mapping[str, Any],
    event_sources: Mapping[str, Any], perf: Mapping[str, Any],
) -> dict[str, Any]:
    catalog_present, exact_zen5, reasons = _derive(cpu, environment, event_sources, perf)
    body = {
        "schema": X86_EVENT_MAP_SCHEMA_VERSION,
        "cpu": dict(cpu),
        "environment": dict(environment),
        "event_sources": dict(event_sources),
        "perf": dict(perf),
        "exact_zen5_family": exact_zen5,
        "eligible_for_collection": catalog_present,
        "eligible_for_promotion": not reasons,
        "ineligibility_reasons": reasons,
    }
    body["event_map_sha256"] = _digest(body)
    validate_x86_event_map(body)
    return body


def validate_x86_event_map(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != X86_EVENT_MAP_SCHEMA_VERSION:
        raise X86EventMapError("unsupported x86 event-map schema")
    for field in ("cpu", "environment", "event_sources", "perf"):
        if not isinstance(payload.get(field), Mapping):
            raise X86EventMapError(f"x86 event map requires {field}")
    reasons = payload.get("ineligibility_reasons")
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        raise X86EventMapError("x86 event map requires ineligibility reasons")
    X86_EVENT_MAP_VOCABULARY.require_known(reasons, X86EventMapError)
    catalog_present, exact_zen5, derived = _derive(
        payload["cpu"], payload["environment"], payload["event_sources"], payload["perf"])
    if reasons != derived:
        raise X86EventMapError(
            f"x86 event map states reasons {reasons}, but its stored inputs derive {derived}")
    if payload.get("exact_zen5_family") is not exact_zen5 \
            or payload.get("eligible_for_collection") is not catalog_present:
        raise X86EventMapError(
            "x86 event map's exact_zen5_family/eligible_for_collection differ from "
            "what its stored inputs derive")
    if payload.get("eligible_for_promotion") is not (not derived):
        raise X86EventMapError(
            "x86 event map's promotion eligibility differs from its derived reasons")
    unsigned = dict(payload)
    digest = unsigned.pop("event_map_sha256", None)
    if _digest(unsigned) != digest:
        raise X86EventMapError("x86 event-map digest mismatch")


__all__ = [
    "X86_EVENT_MAP_SCHEMA_VERSION",
    "X86EventMapError",
    "X86_EVENT_MAP_REASONS",
    "X86_EVENT_MAP_VOCABULARY",
    "build_x86_event_map",
    "validate_x86_event_map",
]
