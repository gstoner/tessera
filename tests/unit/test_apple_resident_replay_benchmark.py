from __future__ import annotations

import importlib.util
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
    assert len(report["rows"]) == 4
    assert {row["route"] for row in report["rows"]} == {
        "fused_block", "resident_ring"}
    assert all(row["correctness"] for row in report["rows"])
    assert {d["timing_domain"] for d in report["decisions"]} == {
        "device", "end_to_end"}


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
