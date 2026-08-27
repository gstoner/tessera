"""Live SM120 dense-Krylov multi-CTA performance ratchet."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from tests._support.nvidia import nvidia_mma_runtime_available


ROOT = Path(__file__).parents[3]
BASELINE = ROOT / "benchmarks/baselines/nvidia_sm120_solver_krylov_performance.json"
RECORDER = ROOT / "benchmarks/nvidia/record_solver_krylov_baseline.py"
_spec = importlib.util.spec_from_file_location("nvidia_solver_krylov_live", RECORDER)
assert _spec and _spec.loader
recorder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recorder)
_gate_spec = importlib.util.spec_from_file_location("perf_gate", ROOT / "benchmarks/perf_gate.py")
assert _gate_spec and _gate_spec.loader
perf_gate = importlib.util.module_from_spec(_gate_spec)
_gate_spec.loader.exec_module(perf_gate)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.performance
@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="live NVIDIA CUDA required")
def test_dense_krylov_within_repeated_median_ratchet_and_scales_ctas() -> None:
    if not BASELINE.is_file():
        pytest.skip("record NVIDIA solver Krylov baseline on sm_120 first")
    with tempfile.TemporaryDirectory(prefix="tessera-solver-krylov-") as tmp:
        output = Path(tmp) / "rows.json"
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT / "python"), str(ROOT), env.get("PYTHONPATH", "")]
        )
        subprocess.run(
            [sys.executable, str(RECORDER), "--reps", "5", "--warmup", "2",
             "--device-reps", "3", "--output", str(output)],
            cwd=ROOT, env=env, check=True, capture_output=True, text=True,
        )
        measured = json.loads(output.read_text())["rows"]
    rows = [{**row, "latency_ms": row["median_ms"]} for row in measured]
    failures = perf_gate.evaluate_ratchet(rows, json.loads(BASELINE.read_text()))
    assert not failures, "\n".join(failures)
    for algorithm in ("cg", "gmres"):
        device_rows = [
            row for row in measured
            if row["op"] == f"dense_{algorithm}" and row["timing_domain"] == "device_event"
        ]
        ordered = sorted(device_rows, key=lambda row: int(row["shape"].split("x")[0]))
        assert [row["reduction_ctas"] for row in ordered] == sorted(
            {row["reduction_ctas"] for row in ordered}
        )
        assert all(row["correctness_gate"] == "known_solution_plus_fp32_true_residual"
                   for row in ordered)
