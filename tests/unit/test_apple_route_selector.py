"""Unit coverage for evidence-gated Apple route selection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tessera.compiler.apple_route_selector import (
    AppleRouteMeasurement,
    AppleRouteContext,
    ROUTE_REPORT_SCHEMA_VERSION,
    STRICT_ROUTE_LEDGER_SCHEMA,
    STRICT_PACKAGE_SUBGRAPH_SCOPE,
    legacy_route_ledger_inventory,
    aggregate_stable_route_reports,
    load_route_measurements,
    load_strict_route_ledger,
    package_route_selected,
    production_route_decision,
    production_route_for,
    seal_strict_route_ledger,
    select_route,
)


_CONTEXT = AppleRouteContext(
    device="apple7",
    physical_device="Apple M1 Max",
    os_version="26.5.2",
    sdk_version="26.4",
    compiler_fingerprint="sha256:compiler",
    runtime_fingerprint="sha256:runtime",
)


def _strict_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": STRICT_ROUTE_LEDGER_SCHEMA,
        "selection_scope": "runtime_route",
        "measured_at": "2026-07-18T12:00:00Z",
        "expires_at": "2026-08-18T12:00:00Z",
        "context": _CONTEXT.as_mapping(),
        "source_report_count": 2,
        "source_report_digests": ["sha256:" + "a" * 64,
                                  "sha256:" + "b" * 64],
        "decisions": [{
            "device": "apple7", "op": "softmax", "shape": "128x257",
            "dtype": "f32", "timing_domain": "end_to_end",
            "incumbent_route": "msl", "selected_route": "mpsgraph",
            "status": "promote_candidate",
            "selected_evidence": {
                "provenance": "native_gpu", "correctness": True,
                "device": "apple7", "timing_domain": "end_to_end",
            },
        }],
    }
    payload.update(overrides)
    return payload


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


def test_strict_sealing_requires_producer_context_and_preserves_native_evidence():
    reports = [_report(_stable_row("mps", 1000, 800),
                       _stable_row("simdgroup_matrix", 850, 700)) for _ in range(2)]
    stable = aggregate_stable_route_reports(reports)
    with __import__("pytest").raises(ValueError, match="producer-captured context"):
        seal_strict_route_ledger(stable, reports)
    reports = [{**report, "context": _CONTEXT.as_mapping()} for report in reports]
    sealed = seal_strict_route_ledger(stable, reports)
    assert sealed["schema"] == STRICT_ROUTE_LEDGER_SCHEMA
    assert sealed["selection_scope"] == "runtime_route"
    assert all(row["selected_evidence"]["provenance"] == "native_gpu"
               for row in sealed["decisions"])
    assert len(sealed["source_report_digests"]) == 2


def test_strict_sealing_preserves_unselectable_rows_outside_admitted_decisions():
    reports = [{
        **_report(_stable_row("mps", 1000, None),
                   _stable_row("simdgroup_matrix", 850, None)),
        "context": _CONTEXT.as_mapping(),
    } for _ in range(2)]
    sealed = seal_strict_route_ledger(aggregate_stable_route_reports(reports), reports)
    assert len(sealed["decisions"]) == 1
    assert sealed["ineligible_decisions"] == [{
        "op": "matmul", "shape": "64x64x64", "dtype": "f32",
        "device": "apple7", "timing_domain": "device",
        "incumbent_route": "mps", "status": "ineligible",
        "reason": "incumbent paired evidence is incomplete",
    }]


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


def test_production_promotions_are_exact_device_shape_and_domain(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(_strict_payload()), encoding="utf-8")
    assert production_route_for(
        op="softmax", shape="128x257", dtype="f32", device="apple7",
        incumbent_route="msl", ledger_path=ledger, context=_CONTEXT) == "mpsgraph"
    assert production_route_for(
        op="softmax", shape="128x257", dtype="f32", device="apple8",
        incumbent_route="msl", ledger_path=ledger, context=_CONTEXT) == "msl"
    assert production_route_for(
        op="softmax", shape="128x257", dtype="f16", device="apple7",
        incumbent_route="msl", ledger_path=ledger, context=_CONTEXT) == "msl"
    assert production_route_for(
        op="softmax", shape="128x257", dtype="f32", device="apple7",
        timing_domain="device", incumbent_route="msl", ledger_path=ledger,
        context=_CONTEXT) == "msl"
    decision = production_route_decision(
        op="softmax", shape="128x257", dtype="f32", device="apple7",
        incumbent_route="msl", ledger_path=ledger, context=_CONTEXT)
    assert decision.selected_from_ledger is True
    assert decision.citation == f"{ledger}#decision[0]"
    assert decision.rejected_evidence == ()


def test_legacy_ledgers_are_not_admitted_as_production_evidence():
    root = Path(__file__).resolve().parents[2]
    legacy = root / "benchmarks/baselines/apple7_attention_route_ledger.json"
    admitted = load_strict_route_ledger(legacy, context=_CONTEXT)
    assert admitted.routes == {}
    assert admitted.rejected == ("schema_mismatch",)


def test_package_subgraph_ledger_cannot_select_a_runtime_route(tmp_path):
    ledger = tmp_path / "package-ledger.json"
    ledger.write_text(json.dumps(_strict_payload(
        selection_scope=STRICT_PACKAGE_SUBGRAPH_SCOPE,
    )), encoding="utf-8")
    admitted = load_strict_route_ledger(ledger, context=_CONTEXT)
    assert admitted.routes == {}
    assert admitted.rejected == ("wrong_selection_scope",)


def test_legacy_route_ledger_inventory_requires_remeasurement():
    root = Path(__file__).resolve().parents[2] / "benchmarks" / "baselines"
    records = legacy_route_ledger_inventory(root)
    assert {record.path.name for record in records} >= {
        "apple7_attention_route_ledger.json",
        "apple7_attention_backward_route_ledger.json",
        "apple7_epilogue_route_ledger.json",
        "apple7_gemm_route_ledger.json",
    }
    by_name = {record.path.name: record for record in records}
    for name in ("apple7_gemm_route_ledger.json",
                 "apple7_attention_route_ledger.json",
                 "apple7_attention_backward_route_ledger.json",
                 "apple7_epilogue_route_ledger.json"):
        assert by_name[name].migration_state == "remeasured_strict_v2"
        assert by_name[name].strict_ledger_path is not None
    assert all(record.migration_state == "remeasured_strict_v2"
               for record in by_name.values())


def test_strict_loader_rejects_stale_context_reference_and_wrong_domain(tmp_path):
    now = datetime(2026, 7, 20, tzinfo=timezone.utc)
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(_strict_payload()), encoding="utf-8")
    assert len(load_strict_route_ledger(path, context=_CONTEXT, now=now).routes) == 1

    stale = _strict_payload(expires_at="2026-07-19T00:00:00Z")
    path.write_text(json.dumps(stale), encoding="utf-8")
    assert "stale_evidence" in load_strict_route_ledger(
        path, context=_CONTEXT, now=now).rejected

    wrong = _CONTEXT.as_mapping() | {"physical_device": "Apple M9"}
    path.write_text(json.dumps(_strict_payload(context=wrong)), encoding="utf-8")
    assert "context_mismatch:physical_device" in load_strict_route_ledger(
        path, context=_CONTEXT, now=now).rejected

    payload = _strict_payload()
    decision = payload["decisions"][0]  # type: ignore[index]
    decision["selected_evidence"]["provenance"] = "reference_cpu"  # type: ignore[index]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert any("reference_provenance" in reason for reason in
               load_strict_route_ledger(path, context=_CONTEXT, now=now).rejected)

    payload = _strict_payload()
    decision = payload["decisions"][0]  # type: ignore[index]
    decision["selected_evidence"]["timing_domain"] = "device"  # type: ignore[index]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert any("wrong_evidence_domain" in reason for reason in
               load_strict_route_ledger(path, context=_CONTEXT, now=now).rejected)


def test_strict_loader_rejects_missing_independent_source_digests(tmp_path):
    path = tmp_path / "ledger.json"
    payload = _strict_payload(source_report_digests=["sha256:" + "a" * 64])
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_strict_route_ledger(
        path, context=_CONTEXT, now=datetime(2026, 7, 20, tzinfo=timezone.utc),
    ).rejected == ("missing_or_invalid_source_reports",)
