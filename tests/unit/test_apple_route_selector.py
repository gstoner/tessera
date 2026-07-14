"""Unit coverage for evidence-gated Apple route selection."""

from __future__ import annotations

import json

from tessera.compiler.apple_route_selector import (
    AppleRouteMeasurement,
    ROUTE_REPORT_SCHEMA_VERSION,
    load_route_measurements,
    package_route_selected,
    select_route,
)


def _row(route: str, latency_ms: float, **extra: object) -> dict[str, object]:
    return {
        "op": "matmul_softmax", "shape": "64x64x64", "dtype": "f32",
        "device": "apple_silicon_metal", "route": route,
        "latency_ms": latency_ms, "native_dispatched": True,
        "numerically_validated": True, **extra,
    }


def test_select_route_promotes_only_a_faster_proven_candidate():
    rows = tuple(AppleRouteMeasurement.from_mapping(row) for row in (
        _row("live", 1.0), _row("package", 0.4), _row("mpsgraph", 0.6),
    ))
    assert all(rows)
    assert select_route(rows, op="matmul_softmax", shape="64x64x64",
                        dtype="f32", device="apple_silicon_metal",
                        incumbent_route="live") == "package"


def test_select_route_refuses_missing_incumbent_or_unproven_candidate():
    rows = tuple(AppleRouteMeasurement.from_mapping(row) for row in (
        _row("package", 0.1, native_dispatched=False),
        _row("mpsgraph", 0.2),
    ))
    assert all(rows)
    assert select_route(rows, op="matmul_softmax", shape="64x64x64",
                        dtype="f32", device="apple_silicon_metal",
                        incumbent_route="live") is None


def test_loader_requires_current_schema_and_complete_proof(tmp_path):
    report = tmp_path / "routes.json"
    report.write_text(json.dumps({
        "schema_version": ROUTE_REPORT_SCHEMA_VERSION,
        "runs": [_row("live", 1.0), _row("package", 0.5)],
    }))
    assert package_route_selected(report, op="matmul_softmax", shape="64x64x64")

    report.write_text(json.dumps({"runs": [_row("live", 1.0)]}))
    assert load_route_measurements(report) == ()
    assert not package_route_selected(report, op="matmul_softmax", shape="64x64x64")
