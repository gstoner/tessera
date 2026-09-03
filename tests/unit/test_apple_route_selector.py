"""Unit coverage for evidence-gated Apple route selection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tests._support.apple import (
    STRICT_PROMOTION_RULES as _PROMOTION_RULES,
    strict_promotion_evidence as promotion_evidence,
)
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
        "promotion_rules": dict(_PROMOTION_RULES),
        "decisions": [{
            "device": "apple7", "op": "softmax", "shape": "128x257",
            "dtype": "f32", "timing_domain": "end_to_end",
            "incumbent_route": "msl", "selected_route": "mpsgraph",
            "status": "promote_candidate",
            "route_evidence": {"mpsgraph": promotion_evidence()},
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


def test_stable_aggregation_promotes_only_a_repeated_per_domain_winner():
    # Three reports, not two: a promotion now needs enough independent runs to
    # measure its own dispersion (`min_promotion_runs`). The subject under test
    # is still that each timing domain is decided on its own evidence.
    reports = [
        _report(_stable_row("mps", 1000, 800),
                _stable_row("simdgroup_matrix", 850, 700)),
        _report(_stable_row("mps", 1050, 820),
                _stable_row("simdgroup_matrix", 880, 710)),
        _report(_stable_row("mps", 1020, 810),
                _stable_row("simdgroup_matrix", 865, 705)),
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


def test_strict_sealing_preserves_isolated_package_subgraph_scope():
    reports = [{
        **_report(_stable_row("live", 1000, None),
                  _stable_row("package", 850, None)),
        "selection_scope": STRICT_PACKAGE_SUBGRAPH_SCOPE,
        "context": _CONTEXT.as_mapping(),
    } for _ in range(2)]
    stable = aggregate_stable_route_reports(
        reports, incumbent_routes={"matmul": "live"},
    )
    sealed = seal_strict_route_ledger(
        stable, reports, selection_scope=STRICT_PACKAGE_SUBGRAPH_SCOPE,
    )
    assert sealed["selection_scope"] == STRICT_PACKAGE_SUBGRAPH_SCOPE
    with __import__("pytest").raises(ValueError, match="scope must match"):
        seal_strict_route_ledger(stable, reports)


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
        _report(_stable_row("mps", 1005, 805),
                _stable_row("simdgroup_matrix", 905, 702)),
    ]
    ledger = aggregate_stable_route_reports(reports)
    decisions = {row["timing_domain"]: row for row in ledger["decisions"]}
    assert decisions["end_to_end"]["selected_route"] == "mps"
    # A candidate that is 10% faster in one run and 9% slower in the next is
    # not merely "not promoted" -- it is unmeasured, and the ledger now says
    # so rather than recording the same bare `retain_incumbent` it would use
    # for a route that is simply slower.
    assert decisions["end_to_end"]["status"] == \
        "retain_incumbent_unstable_candidate"
    assert decisions["end_to_end"]["route_evidence"]["simdgroup_matrix"][
        "stability_verdict"] == "unstable_evidence"
    assert decisions["end_to_end"]["route_evidence"]["simdgroup_matrix"][
        "promotable"] is False
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
        _report(_stable_row("mps", 1200, 960),
                _stable_row("simdgroup_matrix", 1020, 816)),
    ]
    ledger = aggregate_stable_route_reports(reports)
    assert all(row["selected_route"] == "simdgroup_matrix"
               for row in ledger["decisions"])
    assert all(not row["route_evidence"]["mps"]["absolute_time_stable"]
               for row in ledger["decisions"])


def test_production_promotions_are_exact_device_shape_and_domain(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(_strict_payload()), encoding="utf-8")
    # Evidence expiry is part of the contract under test. Pin this test inside
    # the fixture's validity interval instead of coupling it to wall time.
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert production_route_for(
        op="softmax", shape="128x257", dtype="f32", device="apple7",
        incumbent_route="msl", ledger_path=ledger, context=_CONTEXT,
        now=now) == "mpsgraph"
    assert production_route_for(
        op="softmax", shape="128x257", dtype="f32", device="apple8",
        incumbent_route="msl", ledger_path=ledger, context=_CONTEXT,
        now=now) == "msl"
    assert production_route_for(
        op="softmax", shape="128x257", dtype="f16", device="apple7",
        incumbent_route="msl", ledger_path=ledger, context=_CONTEXT,
        now=now) == "msl"
    assert production_route_for(
        op="softmax", shape="128x257", dtype="f32", device="apple7",
        timing_domain="device", incumbent_route="msl", ledger_path=ledger,
        context=_CONTEXT, now=now) == "msl"
    decision = production_route_decision(
        op="softmax", shape="128x257", dtype="f32", device="apple7",
        incumbent_route="msl", ledger_path=ledger, context=_CONTEXT, now=now)
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


def test_strict_loader_refuses_a_promotion_its_own_rules_reject(tmp_path):
    """A `promote_candidate` status is a claim, not a credential.

    The loader validated provenance exhaustively -- schema, scope, exact
    context, freshness, source digests, native dispatch, correctness, timing
    domain, device, duplicates -- and the promotion criteria not at all, so the
    status string certified itself. Every case below is a ledger that passes
    every provenance check and still must not be served.
    """
    path = tmp_path / "ledger.json"
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)

    def _rejected(evidence_overrides=None, **payload_overrides):
        payload = _strict_payload(**payload_overrides)
        if evidence_overrides is not None:
            decision = payload["decisions"][0]  # type: ignore[index]
            decision["route_evidence"] = {  # type: ignore[index]
                "mpsgraph": promotion_evidence(**evidence_overrides)}
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_strict_route_ledger(path, context=_CONTEXT, now=now)

    # Sanity: the fixture is admissible before each mutation, or the negatives
    # below would pass for the wrong reason.
    assert len(_rejected({}).routes) == 1

    for overrides, expected in (
        ({"paired_win_fractions": [0.0, 0.0]}, "paired_win_fraction_below_minimum"),
        ({"paired_median_speedups": [0.01, 0.02]}, "speedup_below_minimum"),
        ({"cross_run_speedup_spread": 0.9}, "speedup_spread_above_maximum"),
        ({"placement_and_numerical_proof": False}, "requires_native_dispatch"),
        ({"resource_evidence_retained": False}, "requires_resource_evidence"),
        ({"paired_measurement": False}, "requires_interleaved_paired_trials"),
    ):
        ledger = _rejected(overrides)
        assert ledger.routes == {}, overrides
        assert any(expected in reason for reason in ledger.rejected), (
            f"{overrides} -> {ledger.rejected}")

    # Evidence absent entirely: unverifiable is not the same as fine.
    payload = _strict_payload()
    payload["decisions"][0].pop("route_evidence")  # type: ignore[index]
    path.write_text(json.dumps(payload), encoding="utf-8")
    ledger = load_strict_route_ledger(path, context=_CONTEXT, now=now)
    assert ledger.routes == {}
    assert any("missing_route_evidence" in r for r in ledger.rejected)

    # Per-run metrics shorter than the declared source-report count. The rules
    # are spelled `*_each_run`, and non-emptiness does not check "each": a
    # promotion truncated to one median and one win fraction used to return no
    # violations -- and truncation makes the row look BETTER, because a spread
    # over one element is 0.0 and clears any cap. Dropping evidence must never
    # improve a verdict (review finding on PR #673).
    payload = _strict_payload()
    decision = payload["decisions"][0]  # type: ignore[index]
    decision["route_evidence"] = {"mpsgraph": promotion_evidence(  # type: ignore[index]
        paired_median_speedups=[0.31], paired_win_fractions=[1.0],
        cross_run_speedup_spread=0.0)}
    path.write_text(json.dumps(payload), encoding="utf-8")
    ledger = load_strict_route_ledger(path, context=_CONTEXT, now=now)
    assert ledger.routes == {}
    assert any("per_run_metrics_short" in r for r in ledger.rejected), ledger.rejected

    # The two per-run lists disagreeing with each other.
    ledger = _rejected({"paired_win_fractions": [1.0]})
    assert ledger.routes == {}
    assert any("per_run_metric_length_mismatch" in r for r in ledger.rejected)

    # More per-run metrics than there were independent reports: the extra runs
    # came from somewhere the ledger does not account for.
    ledger = _rejected({"paired_median_speedups": [0.31, 0.29, 0.28],
                        "paired_win_fractions": [1.0, 1.0, 1.0]})
    assert ledger.routes == {}
    assert any("per_run_metrics_exceed_reports" in r for r in ledger.rejected)

    # Rules block absent: nothing to hold the promotion to, so fail CLOSED.
    ledger = _rejected({}, promotion_rules={})
    assert ledger.routes == {}
    assert any("promotion_rules_incomplete" in r for r in ledger.rejected)

    # A `retain_incumbent` row is not a promotion and is not held to the
    # promotion thresholds -- it kept the route that was already in production.
    payload = _strict_payload()
    decision = payload["decisions"][0]  # type: ignore[index]
    decision["status"] = "retain_incumbent"
    decision["selected_route"] = "msl"
    decision["route_evidence"] = {"msl": {"present_in_all_runs": True}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert len(load_strict_route_ledger(
        path, context=_CONTEXT, now=now).routes) == 1


def _promoted_row(**evidence):
    """A minimal promotion row carrying whatever evidence a case needs."""
    base = {
        "paired_median_speedups": [0.2, 0.21, 0.19],
        "paired_win_fractions": [1.0, 1.0, 1.0],
        "pooled_paired_win_fraction": 1.0,
        "speedup_lower_confidence_bound": 0.18,
        "cross_run_speedup_spread": 0.02,
        "placement_and_numerical_proof": True,
        "repeated_measurement": True,
        "paired_measurement": True,
        "resource_evidence_retained": True,
    }
    base.update(evidence)
    return {
        "status": "promote_candidate", "selected_route": "cand",
        "route_evidence": {"cand": base},
    }


_CURRENT_RULES = {
    "minimum_speedup_fraction_each_run": 0.05,
    "minimum_paired_win_fraction_each_run": 0.5,
    "paired_win_fraction_each_run_is_strict": True,
    "minimum_pooled_paired_win_fraction": 0.75,
    "minimum_speedup_lower_confidence_bound": 0.05,
    "minimum_promotion_runs": 3,
    "cross_run_speedup_spread_is_diagnostic_only": True,
    "maximum_cross_run_speedup_spread": 0.05,
}


def test_a_truncated_confidence_bound_rule_set_is_unverifiable_not_fine():
    """Each new threshold guards its own check, so absence must not pass.

    Every check in the confidence-bound rule set is written as `if threshold is
    not None`, which means a ledger that declares the bound and drops
    `minimum_promotion_runs` skips the run-count check while still reading as
    complete -- and a two-report promotion the aggregator would never make gets
    admitted. A missing threshold is not a pass (Decisions #21a, #30).
    """
    from tessera.compiler.apple_route_selector import promotion_rule_violations

    row = _promoted_row()
    assert promotion_rule_violations(row, _CURRENT_RULES) == []
    for dropped in ("minimum_promotion_runs",
                    "minimum_pooled_paired_win_fraction"):
        rules = {k: v for k, v in _CURRENT_RULES.items() if k != dropped}
        assert promotion_rule_violations(row, rules) == [
            f"promotion_rules_incomplete:{dropped}"], dropped

    # And the check the dropped threshold was guarding still bites when present.
    short = _promoted_row(paired_median_speedups=[0.2, 0.21],
                          paired_win_fractions=[1.0, 1.0])
    assert any(v.startswith("promotion_runs_below_minimum")
               for v in promotion_rule_violations(short, _CURRENT_RULES))


def test_the_per_run_win_floor_is_the_strict_majority_the_aggregator_applied():
    """The verifier must reject exactly what the producer rejects.

    Aggregation admits a run only on `fraction > 0.5`, but a re-derivation
    comparing `v < min_win` accepts a run sitting exactly on 0.5 -- so a
    foreign or hand-edited ledger could carry a promotion this aggregator would
    never have made. The comparison travels in the rules rather than being
    assumed, and absent the flag stays non-strict for the pre-pooling ledgers
    that meant `>= 0.75`.
    """
    from tessera.compiler.apple_route_selector import promotion_rule_violations

    tied = _promoted_row(paired_win_fractions=[1.0, 0.5, 1.0],
                         pooled_paired_win_fraction=0.83)
    assert promotion_rule_violations(tied, _CURRENT_RULES) == [
        "paired_win_fraction_below_minimum"]
    # Without the flag the same row is admitted -- which is why old ledgers,
    # whose 0.75 floor was never strict, keep verifying unchanged.
    legacy = {k: v for k, v in _CURRENT_RULES.items()
              if k != "paired_win_fraction_each_run_is_strict"}
    assert promotion_rule_violations(tied, legacy) == []


def test_sealed_rules_declare_the_comparison_the_aggregator_used():
    """The producer and the rules it seals must not drift apart."""
    reports = [
        _report(_stable_row("mps", 1000, 800),
                _stable_row("simdgroup_matrix", 850, 700)) for _ in range(3)]
    rules = aggregate_stable_route_reports(reports)["promotion_rules"]
    assert rules["paired_win_fraction_each_run_is_strict"] is True
    assert rules["minimum_paired_win_fraction_each_run"] == 0.5
    assert rules["minimum_promotion_runs"] == 3
    assert rules["minimum_pooled_paired_win_fraction"] == 0.75
