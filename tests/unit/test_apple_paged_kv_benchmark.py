from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = (Path(__file__).resolve().parents[2] / "benchmarks" / "apple_gpu" /
            "benchmark_resident_paged_kv.py")
    spec = importlib.util.spec_from_file_location("benchmark_resident_paged_kv", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_paged_kv_report_keeps_timing_domains_and_routes_separate():
    report = _module().run_benchmark(
        ["7x8x4x2"], block_size=2, warmup=0, reps=1, runs=2)
    assert report["schema"] == "tessera.apple.resident_paged_kv.v1"
    assert isinstance(report["device"], str)
    assert len(report["rows"]) == 4
    assert all(row["device"] == report["device"] for row in report["rows"])
    assert {row["route"] for row in report["rows"]} == {"staged", "direct"}
    assert all(row["correctness"] for row in report["rows"])
    assert all(row["non_identity_page_table"] for row in report["rows"])
    assert {d["timing_domain"] for d in report["decisions"]} == {
        "device", "end_to_end"}


def test_legacy_paged_kv_corpus_cannot_bypass_strict_production_ledger():
    from tessera.compiler.apple_route_selector import production_route_for
    for domain in ("device", "end_to_end"):
        assert production_route_for(
            op="resident_paged_kv", shape="127x64x32x1", dtype="f32",
            device="apple7", timing_domain=domain,
            incumbent_route="staged") == "staged"
    assert production_route_for(
        op="resident_paged_kv", shape="128x64x32x1", dtype="f32",
        device="apple7", incumbent_route="staged") == "staged"


def test_committed_paged_kv_corpus_preserves_device_provenance():
    path = (Path(__file__).resolve().parents[2] / "benchmarks" / "baselines" /
            "apple7_resident_paged_kv_two_run.json")
    report = json.loads(path.read_text())
    assert report["device"] == "apple7"
    assert all(row["device"] == "apple7" for row in report["rows"])
