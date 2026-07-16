from __future__ import annotations

import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[2] / "benchmarks/nvidia/record_attention_backward_baseline.py"
SPEC = importlib.util.spec_from_file_location("nvidia_attn_bwd_benchmark", PATH)
assert SPEC and SPEC.loader
bench = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(bench)


def test_backward_shapes_include_regular_and_ragged_production_rows():
    assert (1, 8, 128, 64) in bench.SHAPES
    assert (1, 8, 257, 64) in bench.SHAPES


def test_backward_recorder_declares_both_timing_domains(monkeypatch):
    monkeypatch.setattr("tessera.runtime._nvidia_device_name", lambda: None)
    assert bench.record(reps=1, warmup=0, device_reps=1) == []
