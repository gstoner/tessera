from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = (Path(__file__).resolve().parents[2] / "benchmarks" / "apple_gpu" /
            "benchmark_resident_replay.py")
    spec = importlib.util.spec_from_file_location("benchmark_resident_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resident_replay_report_has_paired_routes_and_timing_domains():
    report = _module().run_benchmark(
        ["1x4x3"], tokens=2, warmup=0, reps=1, runs=2)
    assert report["schema"] == "tessera.apple.resident_replay.v1"
    assert isinstance(report["device"], str)
    assert len(report["rows"]) == 4
    assert {row["route"] for row in report["rows"]} == {
        "fused_block", "resident_ring"}
    assert all(row["device"] == report["device"] for row in report["rows"])
    assert all(row["correctness"] for row in report["rows"]
               if row["available"])
    assert all(row["correctness"] is None for row in report["rows"]
               if not row["available"])
    assert {d["timing_domain"] for d in report["decisions"]} == {
        "device", "end_to_end"}


def test_resident_replay_reports_unavailable_route_without_metal(monkeypatch):
    module = _module()

    class _UnavailableHandle:
        resident_inputs = False

        def close(self):
            return None

    monkeypatch.setattr(
        module.rt, "apple_gpu_resident_ssm_replay_state_handle",
        lambda *args, **kwargs: _UnavailableHandle())
    report = module.run_benchmark(
        ["1x4x3"], tokens=2, warmup=0, reps=1, runs=1)
    resident = next(row for row in report["rows"]
                    if row["route"] == "resident_ring")
    assert resident["available"] is False
    assert resident["unavailable_reason"] == "resident_inputs_unavailable"
    assert resident["correctness"] is None
    assert resident["timing_domain_end_to_end_ns"] is None
    assert resident["timing_domain_device_ns"] is None


def test_resident_replay_selector_keeps_timing_domains_separate():
    from tessera.compiler.ssm_replay import select_apple_serving_route
    assert select_apple_serving_route(
        1, 128, 64, 16, device="apple7",
        timing_domain="end_to_end") == "fused_block"
    assert select_apple_serving_route(
        1, 128, 64, 16, device="apple7",
        timing_domain="device") == "fused_block"
    assert select_apple_serving_route(
        1, 128, 64, 8, device="apple7") == "fused_block"


def test_committed_resident_replay_corpus_preserves_device_provenance():
    path = (Path(__file__).resolve().parents[2] / "benchmarks" / "baselines" /
            "apple7_resident_replay_two_run.json")
    report = json.loads(path.read_text())
    assert report["device"] == "apple7"
    assert all(row["device"] == "apple7" for row in report["rows"])
