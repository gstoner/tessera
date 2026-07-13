from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "benchmarks/baselines/rocm_gfx1151_sparse_redesign.json"
SCRIPT = ROOT / "benchmarks/rocm/benchmark_sparse_redesign.py"


def _load():
    spec = importlib.util.spec_from_file_location("rocm_sparse_redesign", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


benchmark = _load()


def test_sparse_redesign_baseline_is_complete():
    doc = json.loads(BASELINE.read_text())
    assert doc["schema"] == "tessera.benchmark.comparative-ratchet.v1"
    assert doc["target"] == "rocm:gfx1151"
    rows = doc["rows"]
    assert {row["op"] for row in rows} == {"sparse_attention", "sparse_topk"}
    assert len([row for row in rows if row["op"] == "sparse_topk"]) == 2
    for row in rows:
        assert row["candidate_ms"] > 0
        assert row["max_candidate_ms"] > row["candidate_ms"]
        assert row["speedup"] >= row["min_speedup"] >= 1.0


def test_live_sparse_redesign_within_comparative_ratchet():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None or not rt._rocm_wmma_runtime_available():
        pytest.skip("needs tessera-opt and a live AMD GPU")

    expected = {(row["op"], row["shape"]): row
                for row in json.loads(BASELINE.read_text())["rows"]}
    measured = benchmark.measure_cases(rt, reps=5)
    assert {(row["op"], row["shape"]) for row in measured} == set(expected)
    failures = []
    for row in measured:
        gate = expected[(row["op"], row["shape"])]
        if row["candidate_ms"] > gate["max_candidate_ms"]:
            failures.append(f"{row['op']} {row['shape']} latency regressed")
        if row["speedup"] < gate["min_speedup"]:
            failures.append(f"{row['op']} {row['shape']} lost redesign win")
    assert not failures, "; ".join(failures)
