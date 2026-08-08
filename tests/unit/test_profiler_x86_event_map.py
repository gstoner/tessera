from __future__ import annotations

import copy

import pytest

from tessera.compiler.profiler_x86_event_map import (
    X86EventMapError,
    build_x86_event_map,
    validate_x86_event_map,
)


def test_exact_bare_metal_zen5_event_map_is_digest_bound() -> None:
    artifact = build_x86_event_map(
        cpu={"vendor_id": "AuthenticAMD", "cpu_family": 26, "flags": ["avx512f"]},
        environment={"virtualized": False, "kernel": "test"},
        event_sources={"cpu": {"type": "4", "events": {"cycles": "event=0x76"}}},
        perf={
            "available": True,
            "version": "perf test",
            "version_returncode": 0,
            "list_returncode": 0,
            "catalog_text": "[]",
            "catalog_format": "json",
        },
    )
    assert artifact["eligible_for_promotion"] is True
    validate_x86_event_map(artifact)
    tampered = copy.deepcopy(artifact)
    tampered["event_sources"]["cpu"]["type"] = "5"
    with pytest.raises(X86EventMapError, match="digest"):
        validate_x86_event_map(tampered)


def test_wsl_or_missing_perf_event_map_fails_closed() -> None:
    artifact = build_x86_event_map(
        cpu={"vendor_id": "AuthenticAMD", "cpu_family": 26, "flags": ["avx512f"]},
        environment={"virtualized": True, "wsl": True},
        event_sources={},
        perf={"available": False},
    )
    assert artifact["eligible_for_collection"] is False
    assert artifact["eligible_for_promotion"] is False
    assert "VIRTUALIZED_HOST" in artifact["ineligibility_reasons"]
