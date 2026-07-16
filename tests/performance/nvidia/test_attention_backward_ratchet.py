"""Live repeated-median ratchet for NVIDIA Flash-Attention backward."""
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
BASELINE = ROOT / "benchmarks/baselines/nvidia_sm120_attention_backward.json"
RECORDER = ROOT / "benchmarks/nvidia/record_attention_backward_baseline.py"
SPEC = importlib.util.spec_from_file_location("perf_gate_bwd", ROOT / "benchmarks/perf_gate.py")
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(gate)


@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.performance
@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="live NVIDIA CUDA required")
def test_attention_backward_within_dual_domain_ratchet():
    with tempfile.TemporaryDirectory(prefix="tessera-attn-bwd-") as tmp:
        measured = Path(tmp) / "measured.json"
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT / "python"), str(ROOT), env.get("PYTHONPATH", "")])
        subprocess.run([sys.executable, str(RECORDER), "--reps", "10",
                        "--device-reps", "20", "--output", str(measured)],
                       cwd=ROOT, env=env, check=True, capture_output=True, text=True)
        rows = json.loads(measured.read_text())["rows"]
    rows = [{**row, "latency_ms": row["median_ms"]} for row in rows]
    failures = gate.evaluate_ratchet(rows, json.loads(BASELINE.read_text()))
    assert not failures, "\n".join(failures)
