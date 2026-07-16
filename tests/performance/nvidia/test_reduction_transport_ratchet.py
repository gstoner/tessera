"""Live TEST-5 ratchet for CUDA reductions and MoE transport."""
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
BASELINE = ROOT / "benchmarks/baselines/nvidia_sm120_reduction_transport.json"
RECORDER = ROOT / "benchmarks/nvidia/record_reduction_transport_baseline.py"
_spec = importlib.util.spec_from_file_location("nvidia_reduction_transport_live", RECORDER)
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
def test_reduction_transport_routes_within_repeated_median_ratchet():
    if not BASELINE.is_file():
        pytest.skip("record NVIDIA reduction/transport baseline on sm_120 first")
    # The measured lane contains unrelated JIT/candidate tests.  Record in a
    # fresh interpreter so allocations and compiled-kernel caches from those
    # tests cannot alter this route's end-to-end baseline.
    with tempfile.TemporaryDirectory(prefix="tessera-test5-") as tmp:
        output = Path(tmp) / "rows.json"
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT / "python"), str(ROOT), env.get("PYTHONPATH", "")])
        subprocess.run([sys.executable, str(RECORDER), "--reps", "10",
                        "--warmup", "3", "--device-reps", "100",
                        "--output", str(output)], cwd=ROOT, env=env,
                       check=True, capture_output=True, text=True)
        measured = json.loads(output.read_text())["rows"]
    rows = [{**row, "latency_ms": row["median_ms"]} for row in measured]
    assert not (failures := perf_gate.evaluate_ratchet(
        rows, json.loads(BASELINE.read_text()))), "\n".join(failures)
