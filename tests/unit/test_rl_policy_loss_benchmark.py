from __future__ import annotations

import importlib.util
import math
from pathlib import Path

_BENCH = (Path(__file__).resolve().parents[2] / "benchmarks" / "rl"
          / "benchmark_policy_losses.py")


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("rl_policy_loss_bench", _BENCH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rl_policy_loss_benchmark_rows_are_reference_only():
    bench = _load_benchmark()
    report = bench.build_report([(2, 3, 5)], reps=1, seed=123)
    rows = report["rows"]
    assert {r["name"] for r in rows} == {
        "ppo_policy_loss", "grpo_policy_loss", "cispo_policy_loss",
    }
    for row in rows:
        assert row["variant_kind"] == "python_reference"
        assert row["target"] == "reference_cpu"
        assert row["executor"] == "python_reference"
        assert row["compiler_path"] is None
        assert row["correctness"] is None
        assert row["skip_reason"] is None
        assert math.isfinite(row["loss"])
        assert row["timing_ms"] is not None and row["timing_ms"] >= 0.0


def test_rl_policy_loss_benchmark_is_deterministic_for_same_seed():
    bench = _load_benchmark()
    a = bench.build_report([(2, 3, 5)], reps=1, seed=77)["rows"]
    b = bench.build_report([(2, 3, 5)], reps=1, seed=77)["rows"]
    assert [(r["name"], r["loss"]) for r in a] == [
        (r["name"], r["loss"]) for r in b
    ]
