"""Exact-machine Linux PMU/event-map evidence for x86 profiler packets."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


X86_EVENT_MAP_SCHEMA_VERSION = "tessera.profiler_x86_event_map.v1"


class X86EventMapError(ValueError):
    """Raised when an x86 event catalog cannot support exact-host evidence."""


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_x86_event_map(
    *, cpu: Mapping[str, Any], environment: Mapping[str, Any],
    event_sources: Mapping[str, Any], perf: Mapping[str, Any],
) -> dict[str, Any]:
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
    if payload.get("eligible_for_promotion") and reasons:
        raise X86EventMapError("promotion-eligible x86 event map has blockers")
    unsigned = dict(payload)
    digest = unsigned.pop("event_map_sha256", None)
    if _digest(unsigned) != digest:
        raise X86EventMapError("x86 event-map digest mismatch")


__all__ = [
    "X86_EVENT_MAP_SCHEMA_VERSION",
    "X86EventMapError",
    "build_x86_event_map",
    "validate_x86_event_map",
]
