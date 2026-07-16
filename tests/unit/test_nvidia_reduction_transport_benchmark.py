"""Host-free contract for the TEST-5 reduction/transport recorder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[2]
PATH = ROOT / "benchmarks/nvidia/record_reduction_transport_baseline.py"
SPEC = importlib.util.spec_from_file_location("nvidia_reduction_transport", PATH)
assert SPEC and SPEC.loader
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_recorder_declares_distinct_timing_domains_and_all_missing_routes(monkeypatch):
    monkeypatch.setattr("tessera.runtime._nvidia_device_name", lambda: "sm_120")
    monkeypatch.setattr(bench, "benchmark_cases", lambda: [
        ("reduction_sum", "257x1025", "f32", "generated_row_reduce", lambda: None, lambda: .1),
        ("moe_dispatch", "257x193", "f32", "generated_gather", lambda: None, lambda: .2),
    ])
    monkeypatch.setattr(bench, "_wall", lambda fn: .3)
    rows = bench.record(reps=2, warmup=0, margin=2)
    assert {r["timing_domain"] for r in rows} == {"end_to_end", "device_event"}
    assert all(r["max_latency_ms"] == r["median_ms"] * 2 for r in rows)
    assert all(r["selected_route"] in r["mode"] for r in rows)
