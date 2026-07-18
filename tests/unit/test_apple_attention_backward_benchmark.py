"""Portable contracts for the Apple attention-backward producer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks/apple_gpu/benchmark_attention_backward.py"
SELECTOR = (ROOT / "benchmarks/apple_gpu" /
            "select_stable_attention_backward_routes.py")


def _module():
    spec = importlib.util.spec_from_file_location("apple_attention_bwd_bench", BENCHMARK)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _selector_module():
    spec = importlib.util.spec_from_file_location(
        "apple_attention_bwd_selector", SELECTOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shape_parser_requires_positive_b_sq_sk_d():
    benchmark = _module()
    assert benchmark._shape("2x17x19x128") == (2, 17, 19, 128)
    for invalid in ("1x2x3", "1x0x3x4", "1xbadx3x4"):
        with pytest.raises(ValueError):
            benchmark._shape(invalid)


def test_benchmark_default_routes_cover_every_native_candidate():
    benchmark = _module()
    assert benchmark.ROUTES == (
        "serial_recompute", "atomic", "split_reduced")


def test_closure_matrix_covers_semantics_dtypes_and_long_context():
    benchmark = _module()
    cases = benchmark._closure_cases()
    assert {case["dtype"] for case in cases} == {"f32", "f16", "bf16"}
    assert any(case["q_heads"] > case["kv_heads"] > 1 for case in cases)
    assert any(case["kv_heads"] == 1 for case in cases)
    assert any(case["bias"] and case["window"] and case["softcap"]
               for case in cases)
    assert max(case["sk"] for case in cases) >= 1025
    assert len({benchmark._case_key(case) for case in cases}) == len(cases)


def test_backward_ledger_requires_two_complete_dtype_route_runs(monkeypatch):
    selector = _selector_module()
    monkeypatch.setattr(
        selector, "aggregate_stable_route_reports",
        lambda reports, **kwargs: {"decisions": [], "report_count": len(reports)})
    rows = []
    for dtype in ("f32", "f16", "bf16"):
        for route in ("serial_recompute", "atomic", "split_reduced"):
            rows.append({
                "dtype": dtype, "route": route,
                "native_dispatched": True, "numerically_validated": True,
                "semantics": {"q_heads": 4, "kv_heads": 2},
                "telemetry": {"device_time_coverage": 1.0},
            })
    report = {"device": "apple7", "runs": rows}
    ledger = selector.build_backward_ledger([report, report])
    assert ledger["attention_backward_proof"]["complete"] is True
    incomplete = selector.build_backward_ledger([
        report, {"device": "apple7", "runs": rows[:-1]}])
    assert incomplete["attention_backward_proof"]["complete"] is False


def test_committed_backward_ledger_matches_production_promotions():
    import json

    from tessera.compiler.apple_route_selector import production_route_for

    path = ROOT / "benchmarks/baselines/apple7_attention_backward_route_ledger.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    assert ledger["attention_backward_proof"]["complete"] is True
    assert len(ledger["decisions"]) == 12
    promotions = [row for row in ledger["decisions"]
                  if row["status"] == "promote_candidate"]
    assert len(promotions) == 6
    for row in ledger["decisions"]:
        selected = production_route_for(
            op=row["op"], shape=row["shape"], dtype=row["dtype"],
            device=row["device"], timing_domain=row["timing_domain"],
            incumbent_route=row["incumbent_route"])
        assert selected == row["selected_route"]


@pytest.mark.parametrize("causal", [False, True])
def test_shared_oracle_has_correct_shapes_and_finite_gradients(causal):
    benchmark = _module()
    rng = np.random.default_rng(19)
    q = rng.normal(size=(2, 3, 5)).astype(np.float32)
    k = rng.normal(size=(2, 4, 5)).astype(np.float32)
    v = rng.normal(size=(2, 4, 5)).astype(np.float32)
    do = rng.normal(size=(2, 3, 5)).astype(np.float32)
    gradients = benchmark._reference(q, k, v, do, scale=5 ** -.5,
                                     causal=causal)
    assert [gradient.shape for gradient in gradients] == [q.shape, k.shape, v.shape]
    assert all(np.isfinite(gradient).all() for gradient in gradients)
