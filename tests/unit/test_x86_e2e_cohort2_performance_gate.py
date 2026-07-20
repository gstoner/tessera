from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/x86/benchmark_x86_e2e_cohort2.py"
BASELINE = ROOT / "benchmarks/baselines/x86_avx512_e2e_cohort2_comparison.json"
SPEC = importlib.util.spec_from_file_location("x86_e2e_cohort2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_recorded_cohort2_route_disposition() -> None:
    data = json.loads(BASELINE.read_text())
    assert data["schema"] == benchmark.SCHEMA
    assert data["work_item"] == "X86-E2E-2"
    assert data["all_correct"] is True
    assert data["selector_changed"] is False
    assert {row["family"] for row in data["rows"]} == {
        "argreduce", "scan", "rmsnorm", "layernorm", "rope", "alibi",
    }
    assert all(row["correct"] for row in data["rows"])
