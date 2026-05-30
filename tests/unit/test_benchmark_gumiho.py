"""Smoke test for benchmarks/apple_gpu/benchmark_gumiho.py.

Runs the driver with tiny args and validates the standard JSON schema + that the
expected op/mode rows are present. Keeps the perf driver honest as the example
and runtime evolve (numbers are machine-dependent, so only structure is checked).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_BENCH = (Path(__file__).resolve().parents[2] / "benchmarks" / "apple_gpu"
          / "benchmark_gumiho.py")
_EX = Path(__file__).resolve().parents[2] / "examples" / "advanced" / "gumiho"


@pytest.fixture(scope="module")
def bench_main():
    if not _BENCH.exists():
        pytest.skip("benchmark_gumiho.py not present")
    if str(_EX) not in sys.path:
        sys.path.insert(0, str(_EX))
    spec = importlib.util.spec_from_file_location("benchmark_gumiho", _BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


def test_benchmark_emits_schema(tmp_path, bench_main):
    out = tmp_path / "gumiho.json"
    rc = bench_main([
        "--reps", "1", "--prompts", "2", "--max-new-tokens", "8",
        "--train-steps", "120", "--target", "numpy", "--output", str(out),
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    runs = payload["runs"]
    assert runs, "no benchmark rows emitted"

    required = {"backend", "op", "shape", "dtype", "mode", "latency_ms",
                "tflops", "memory_bw_gb_s", "device", "tessera_version"}
    for row in runs:
        assert required <= set(row), f"row missing schema keys: {row}"
        assert row["backend"] == "apple_gpu"

    ops = {(r["op"], r["mode"]) for r in runs}
    assert ("gumiho_decode", "vanilla") in ops
    assert ("gumiho_decode", "speculative_trained") in ops
    assert ("gumiho_serial_draft", "resident") in ops
    assert ("gumiho_serial_draft", "per_op") in ops


def test_trained_beats_vanilla_tokens_per_pass(bench_main, tmp_path):
    out = tmp_path / "g.json"
    bench_main(["--reps", "1", "--prompts", "2", "--max-new-tokens", "8",
                "--train-steps", "200", "--target", "numpy", "--output", str(out)])
    runs = json.loads(out.read_text())["runs"]
    by_mode = {r["mode"]: r for r in runs if r["op"] == "gumiho_decode"}
    assert by_mode["vanilla"]["tokens_per_step"] == 1.0
    assert by_mode["speculative_trained"]["tokens_per_step"] > 1.5
