"""Unit coverage for evidence-gated Apple route selection."""

from __future__ import annotations

import json
from pathlib import Path

from tessera.compiler.apple_route_selector import (
    AppleRouteMeasurement,
    ROUTE_REPORT_SCHEMA_VERSION,
    aggregate_stable_route_reports,
    load_route_measurements,
    package_route_selected,
    production_route_for,
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


def _stable_row(route: str, e2e_ns: int, device_ns: int | None, *,
                device: str = "apple7", valid: bool = True) -> dict[str, object]:
    e2e_trials = [e2e_ns - 10, e2e_ns + 5, e2e_ns, e2e_ns + 10, e2e_ns - 5]
    device_trials = ([device_ns - 5, device_ns + 2, device_ns,
                      device_ns + 5, device_ns - 2]
                     if device_ns is not None else None)
    return {
        "op": "matmul", "shape": "64x64x64", "dtype": "f32",
        "device": device, "route": route, "reps": 30,
        "native_dispatched": True, "numerically_validated": valid,
        "telemetry": {
            "end_to_end_median_ns": e2e_ns,
            "device_time_median_ns": device_ns,
            "paired_trial_end_to_end_medians_ns": e2e_trials,
            "paired_trial_device_medians_ns": device_trials,
            "resources": {"api": route},
        },
    }


def _report(*rows: dict[str, object]) -> dict[str, object]:
    return {"schema_version": ROUTE_REPORT_SCHEMA_VERSION, "runs": list(rows)}


def test_stable_aggregation_promotes_only_a_two_run_per_domain_winner():
    reports = [
        _report(_stable_row("mps", 1000, 800),
                _stable_row("simdgroup_matrix", 850, 700)),
        _report(_stable_row("mps", 1050, 820),
                _stable_row("simdgroup_matrix", 880, 710)),
    ]
    ledger = aggregate_stable_route_reports(reports)
    decisions = {row["timing_domain"]: row for row in ledger["decisions"]}
    assert decisions["end_to_end"]["selected_route"] == "simdgroup_matrix"
    assert decisions["device"]["selected_route"] == "simdgroup_matrix"
    assert all(row["status"] == "promote_candidate" for row in decisions.values())


def test_stable_aggregation_retains_incumbent_for_mixed_or_unstable_wins():
    reports = [
        _report(_stable_row("mps", 1000, 800),
                _stable_row("simdgroup_matrix", 900, 700)),
        _report(_stable_row("mps", 1010, 810),
                _stable_row("simdgroup_matrix", 1100, 705)),
    ]
    ledger = aggregate_stable_route_reports(reports)
    decisions = {row["timing_domain"]: row for row in ledger["decisions"]}
    assert decisions["end_to_end"]["selected_route"] == "mps"
    assert decisions["end_to_end"]["status"] == "retain_incumbent"
    assert decisions["device"]["selected_route"] == "simdgroup_matrix"


def test_stable_aggregation_marks_missing_device_timing_insufficient():
    reports = [
        _report(_stable_row("mps", 1000, None),
                _stable_row("simdgroup_matrix", 800, 600)),
        _report(_stable_row("mps", 1010, None),
                _stable_row("simdgroup_matrix", 810, 610)),
    ]
    ledger = aggregate_stable_route_reports(reports)
    device = next(row for row in ledger["decisions"]
                  if row["timing_domain"] == "device")
    assert device["selected_route"] is None
    assert device["status"] == "insufficient_evidence"


def test_paired_comparison_survives_absolute_clock_drift():
    reports = [
        _report(_stable_row("mps", 1000, 800),
                _stable_row("simdgroup_matrix", 850, 680)),
        _report(_stable_row("mps", 1400, 1120),
                _stable_row("simdgroup_matrix", 1190, 952)),
    ]
    ledger = aggregate_stable_route_reports(reports)
    assert all(row["selected_route"] == "simdgroup_matrix"
               for row in ledger["decisions"])
    assert all(not row["route_evidence"]["mps"]["absolute_time_stable"]
               for row in ledger["decisions"])


def test_production_promotions_are_exact_device_shape_and_domain():
    assert production_route_for(
        op="softmax", shape="128x257", dtype="f32", device="apple7",
        incumbent_route="msl") == "mpsgraph"
    assert production_route_for(
        op="softmax", shape="128x257", dtype="f32", device="apple8",
        incumbent_route="msl") == "msl"
    assert production_route_for(
        op="softmax", shape="128x257", dtype="f16", device="apple7",
        incumbent_route="msl") == "msl"
    assert production_route_for(
        op="softmax", shape="128x257", dtype="f32", device="apple7",
        timing_domain="device", incumbent_route="msl") == "msl"


def test_production_promotions_match_retained_apple7_ledger():
    root = Path(__file__).resolve().parents[2]
    ledger = json.loads((root / "benchmarks/baselines/apple7_gemm_route_ledger.json")
                        .read_text(encoding="utf-8"))
    promoted_e2e = {
        (row["op"], row["shape"], row["dtype"]): row["selected_route"]
        for row in ledger["decisions"]
        if row["timing_domain"] == "end_to_end"
        and row["status"] == "promote_candidate"
    }
    assert promoted_e2e == {
        ("softmax", "128x257", "f32"): "mpsgraph",
        ("softmax", "256x256", "f32"): "mpsgraph",
    }
    for (op, shape, dtype), route in promoted_e2e.items():
        assert production_route_for(
            op=op, shape=shape, dtype=dtype, device="apple7",
            incumbent_route="msl") == route


def test_attention_promotions_match_retained_apple7_ledger():
    root = Path(__file__).resolve().parents[2]
    ledger = json.loads(
        (root / "benchmarks/baselines/apple7_attention_route_ledger.json")
        .read_text(encoding="utf-8"))
    promoted_e2e = {
        (row["op"], row["shape"], row["dtype"]): row["selected_route"]
        for row in ledger["decisions"]
        if row["timing_domain"] == "end_to_end"
        and row["status"] == "promote_candidate"
    }
    assert len(promoted_e2e) == 8
    assert set(promoted_e2e.values()) == {"mpsgraph_bsmm"}
    for (op, shape, dtype), route in promoted_e2e.items():
        assert production_route_for(
            op=op, shape=shape, dtype=dtype, device="apple7",
            incumbent_route="online_msl_variant") == route
    promoted_device = {
        (row["op"], row["shape"], row["dtype"]): row["selected_route"]
        for row in ledger["decisions"]
        if row["timing_domain"] == "device"
        and row["status"] == "promote_candidate"
    }
    for (op, shape, dtype), route in promoted_device.items():
        assert production_route_for(
            op=op, shape=shape, dtype=dtype, device="apple7",
            timing_domain="device",
            incumbent_route="online_msl_variant") == route
    assert ledger["attention_forward_proof"]["complete"] is True
