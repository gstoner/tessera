from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PATH = Path(__file__).parents[2] / "benchmarks/nvidia/record_attention_backward_baseline.py"
SPEC = importlib.util.spec_from_file_location("nvidia_attn_bwd_benchmark", PATH)
assert SPEC and SPEC.loader
bench = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(bench)

MATRIX_PATH = (Path(__file__).parents[2] /
               "benchmarks/nvidia/record_attention_backward_schedule_matrix.py")
MATRIX_SPEC = importlib.util.spec_from_file_location(
    "nvidia_attn_bwd_schedule_matrix", MATRIX_PATH)
assert MATRIX_SPEC and MATRIX_SPEC.loader
matrix = importlib.util.module_from_spec(MATRIX_SPEC)
MATRIX_SPEC.loader.exec_module(matrix)


def test_backward_shapes_include_regular_and_ragged_production_rows():
    assert (1, 8, 128, 64) in bench.SHAPES
    assert (1, 8, 257, 64) in bench.SHAPES


def test_backward_recorder_declares_both_timing_domains(monkeypatch):
    monkeypatch.setattr("tessera.runtime._nvidia_device_name", lambda: None)
    assert bench.record(reps=1, warmup=0, device_reps=1) == []


def test_backward_schedule_matrix_covers_d64_d128_and_ragged_gqa():
    cases = {name: (shape, options) for name, shape, options in matrix._cases()}
    assert cases["mha_d64"][0][-1] == 64
    assert cases["causal_mha_d128"][0][-1] == 128
    assert cases["ragged_gqa"][0][1] > cases["ragged_gqa"][0][2]
    assert cases["ragged_gqa"][0][3] != cases["ragged_gqa"][0][4]
    assert matrix.ROUTES == ("atomic", "split_reduced")
    assert matrix.NOISE == 0.03


def test_committed_backward_schedule_matrix_is_stable_and_resource_linked():
    path = (Path(__file__).parents[2] / "benchmarks/baselines/"
            "nvidia_sm120_attention_backward_schedules.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["production_route"] == "atomic"
    assert data["selector_changed"] is False
    assert len(data["rows"]) == 6
    assert all(row["stable"] for row in data["rows"])
    assert all(row["resource_evidence_complete"] for row in data["rows"])
    for row in data["rows"]:
        assert row["sampling"]["run_cohorts"] == (
            "balanced_abba_disjoint_samples")
        assert all(len(run["device_batch_medians_ms"]) == 5
                   for run in row["runs"])
        assert all(len(run["end_to_end_batch_medians_ms"]) == 10
                   for run in row["runs"])
        if row["candidate"] == "split_reduced":
            assert row["observed_bitwise_repeatable"] is True
            assert row["workspace_bytes"] > 0
