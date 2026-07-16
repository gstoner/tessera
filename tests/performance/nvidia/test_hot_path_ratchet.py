"""Live NVIDIA sm_120 hot-path performance ratchet."""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
import pytest
from tests._support.nvidia import nvidia_mma_runtime_available

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "benchmarks" / "baselines" / "nvidia_sm120_hot_paths.json"
RECORDER = ROOT / "benchmarks" / "nvidia" / "record_hot_path_baseline.py"

def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod

perf_gate = _load(ROOT / "benchmarks" / "perf_gate.py", "perf_gate")
recorder = _load(RECORDER, "nvidia_record_hot_path_baseline")

@pytest.mark.slow
@pytest.mark.hardware_nvidia
@pytest.mark.performance
@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="live NVIDIA GPU (sm_120) required")
def test_live_hot_paths_within_ratchet():
    if not BASELINE.is_file(): pytest.skip("nvidia baseline not recorded yet — run the recorder first")
    from tessera import runtime as rt
    rows = [{"op": op, "shape": shape, "dtype": dtype, "mode": mode, "latency_ms": recorder._median_ms(thunk, reps=10)} for op, shape, dtype, mode, thunk in recorder.hot_path_cases(rt)]
    assert not (failures := perf_gate.evaluate_ratchet(rows, json.loads(BASELINE.read_text()))), "\n".join(failures)
