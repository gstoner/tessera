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
        "ppo_policy_loss_masked",
        "ppo_policy_loss_ref_kl",
        "ppo_policy_loss_entropy",
        "ppo_policy_loss_full",
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
    assert {r["name"] for r in decomp} == {
        "ppo_policy_loss",
        "ppo_policy_loss_masked",
        "ppo_policy_loss_ref_kl",
        "ppo_policy_loss_entropy",
        "ppo_policy_loss_full",
        "grpo_policy_loss",
        "cispo_policy_loss",
    }
    for row in decomp:
        assert row["executor"] == "compiler_decomposed_reference"
        assert row["compiler_path"] == "tessera-rl-loss-decompose"
        assert row["runtime_status"] == "reference"
        assert math.isfinite(row["loss"])
        assert row["correctness"] is not None
        assert row["correctness"] < 1e-4

    gpu = [r for r in rows if r["variant_kind"] == "apple_gpu_value_target_ir"]
    assert {r["name"] for r in gpu} == {
        "ppo_policy_loss",
        "ppo_policy_loss_masked",
        "ppo_policy_loss_ref_kl",
        "ppo_policy_loss_entropy",
        "ppo_policy_loss_full",
    }
    assert all(r["name"] not in {"grpo_policy_loss", "cispo_policy_loss"}
               for r in gpu)
    for row in gpu:
        assert row["target"] == "apple_gpu"
        assert row["compiler_path"] == "apple_value_target_ir"
        if row["executor"] is None:
            assert row["loss"] is None
            assert row["correctness"] is None
            assert row["timing_ms"] is None
            assert row["skip_reason"]
        else:
            assert row["executor"] == "apple_gpu_value_target_ir"
            assert row["runtime_status"] == "success"
            assert row["correctness"] is not None and row["correctness"] < 1e-4


def test_rl_policy_loss_benchmark_is_deterministic_for_same_seed():
    bench = _load_benchmark()
    a = bench.build_report([(2, 3, 5)], reps=1, seed=77)["rows"]
    b = bench.build_report([(2, 3, 5)], reps=1, seed=77)["rows"]
    assert [(r["name"], r["variant_kind"], r["loss"]) for r in a] == [
        (r["name"], r["variant_kind"], r["loss"]) for r in b
    ]
