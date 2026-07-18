from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_PATH = (Path(__file__).resolve().parents[2] / "benchmarks" / "apple_gpu" /
         "benchmark_attention_routes.py")
_SPEC = importlib.util.spec_from_file_location("benchmark_attention_routes", _PATH)
assert _SPEC and _SPEC.loader
_BENCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCH)

_SELECTOR_PATH = (Path(__file__).resolve().parents[2] / "benchmarks" /
                  "apple_gpu" / "select_stable_attention_routes.py")
_SELECTOR_SPEC = importlib.util.spec_from_file_location(
    "select_stable_attention_routes", _SELECTOR_PATH)
assert _SELECTOR_SPEC and _SELECTOR_SPEC.loader
_SELECTOR = importlib.util.module_from_spec(_SELECTOR_SPEC)
_SELECTOR_SPEC.loader.exec_module(_SELECTOR)


def test_attention_reference_composes_gqa_bias_window_and_softcap():
    rng = np.random.default_rng(25)
    q = rng.normal(size=(1, 4, 3, 8)).astype(np.float32)
    k = rng.normal(size=(1, 2, 7, 8)).astype(np.float32)
    v = rng.normal(size=(1, 2, 7, 8)).astype(np.float32)
    bias = rng.normal(size=(1, 4, 3, 7)).astype(np.float32)
    out = _BENCH._reference(
        q, k, v, q_heads=4, kv_heads=2, scale=8 ** -0.5,
        causal=True, bias=bias, window=3, softcap=2.0)
    assert out.shape == q.shape
    assert np.isfinite(out).all()


def test_attention_candidate_capabilities_never_invent_unavailable_routes(monkeypatch):
    from tessera import runtime as rt

    monkeypatch.setattr(rt.DeviceTensor, "is_metal", staticmethod(lambda: False))
    report = _BENCH.characterize(reps=2, trials=2)
    assert report["runs"] == []
    assert "unavailable" in report["skipped_apple_gpu"].lower()


def test_attention_ledger_requires_two_complete_forward_proof_runs(monkeypatch):
    monkeypatch.setattr(
        _SELECTOR, "aggregate_stable_route_reports",
        lambda reports, **kwargs: {"decisions": [], "report_count": len(reports)})
    variant = {"native_dispatched": True, "numerically_validated": True}
    resident = {
        **variant, "device_time_coverage": 1.0, "device_time_median_ns": 17}
    report = {
        "device": "apple7", "variant_coverage": [variant],
        "resident_comparison": [resident]}
    ledger = _SELECTOR.build_attention_ledger([report, report])
    assert ledger["attention_forward_proof"]["complete"] is True
    assert ledger["attention_forward_proof"]["native_storage_dtypes"] == [
        "f32", "f16", "bf16"]
    incomplete = _SELECTOR.build_attention_ledger([
        report, {**report, "resident_comparison": [
            {**resident, "device_time_coverage": 0.0}]}])
    assert incomplete["attention_forward_proof"]["complete"] is False
