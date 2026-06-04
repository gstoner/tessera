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


def test_rl_policy_loss_benchmark_rows_are_honest_by_execution_tier():
    bench = _load_benchmark()
    report = bench.build_report([(2, 3, 5)], reps=1, seed=123)
    rows = report["rows"]
    assert {r["name"] for r in rows} == {
        "ppo_policy_loss", "grpo_policy_loss", "cispo_policy_loss",
    }
    py_rows = [r for r in rows if r["variant_kind"] == "python_reference"]
    assert {r["name"] for r in py_rows} == {
        "ppo_policy_loss", "grpo_policy_loss", "cispo_policy_loss",
    }
    for row in py_rows:
        assert row["variant_kind"] == "python_reference"
        assert row["target"] == "reference_cpu"
        assert row["executor"] == "python_reference"
        assert row["compiler_path"] is None
        assert row["correctness"] is None
        assert row["skip_reason"] is None
        assert math.isfinite(row["loss"])
        assert row["timing_ms"] is not None and row["timing_ms"] >= 0.0

    decomp = [r for r in rows
              if r["variant_kind"] == "compiler_decomposed_reference"]
    assert len(decomp) == 1
    assert decomp[0]["name"] == "ppo_policy_loss"
    assert decomp[0]["executor"] == "compiler_decomposed_reference"
    assert decomp[0]["compiler_path"] == "tessera-rl-loss-decompose"
    assert decomp[0]["runtime_status"] == "reference"
    assert math.isfinite(decomp[0]["loss"])

    gated = [r for r in rows
             if r["variant_kind"] == "compiler_visible_non_executable"]
    assert {r["name"] for r in gated} == {"grpo_policy_loss", "cispo_policy_loss"}
    for row in gated:
        assert row["executor"] is None
        assert row["runtime_status"] == "compiler_visible_non_executable"
        assert row["correctness"] is None
        assert row["timing_ms"] is None
        assert row["skip_reason"]

    gpu = [r for r in rows if r["variant_kind"] == "apple_gpu_value_target_ir"]
    assert len(gpu) == 1
    assert gpu[0]["name"] == "ppo_policy_loss"
    assert gpu[0]["target"] == "apple_gpu"
    assert gpu[0]["compiler_path"] == "apple_value_target_ir"
    if gpu[0]["executor"] is None:
        assert gpu[0]["loss"] is None
        assert gpu[0]["correctness"] is None
        assert gpu[0]["timing_ms"] is None
        assert gpu[0]["skip_reason"]
    else:
        assert gpu[0]["executor"] == "apple_gpu_value_target_ir"
        assert gpu[0]["runtime_status"] == "success"
        assert gpu[0]["correctness"] is not None and gpu[0]["correctness"] < 1e-4


def test_rl_policy_loss_benchmark_is_deterministic_for_same_seed():
    bench = _load_benchmark()
    a = bench.build_report([(2, 3, 5)], reps=1, seed=77)["rows"]
    b = bench.build_report([(2, 3, 5)], reps=1, seed=77)["rows"]
    assert [(r["name"], r["variant_kind"], r["loss"]) for r in a] == [
        (r["name"], r["variant_kind"], r["loss"]) for r in b
    ]
