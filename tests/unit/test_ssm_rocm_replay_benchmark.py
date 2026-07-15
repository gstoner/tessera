"""ROCm ReplaySSM benchmark schema and analytical traffic guards."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_PATH = (Path(__file__).resolve().parents[2]
         / "benchmarks/rocm/benchmark_ssm_replay.py")
_SPEC = importlib.util.spec_from_file_location("benchmark_ssm_replay_rocm", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)
_BASELINE = (Path(__file__).resolve().parents[2]
             / "benchmarks/baselines/rocm_gfx1151_ssm_replay_matrix.json")


def test_rocm_replay_traffic_model_reduces_summary_writes():
    summary = bench.summary_state_bytes_per_token(128, 128)
    replay = bench.replay_state_bytes_per_token(128, 128, 65)
    assert summary / replay > 20


def test_rocm_replay_benchmark_shape_parser_guards_rank():
    assert bench.parse_shape("2x64x32") == (2, 64, 32)
    try:
        bench.parse_shape("2x64")
    except ValueError as exc:
        assert "BxDxN" in str(exc)
    else:
        raise AssertionError("invalid benchmark shape was accepted")


def test_rocm_replay_compiler_matrix_deduplicates_summary_and_filters_schedule(
    monkeypatch,
):
    from tessera import runtime as rt
    monkeypatch.setattr(rt, "_rocm_chip", lambda: "gfx1151")
    monkeypatch.setattr(
        bench, "_summary_row",
        lambda shape, tokens, reps: {"mode": "summary", "tokens": tokens})
    monkeypatch.setattr(
        bench, "_output_only_row",
        lambda shape, tokens, reps, capacity=None:
            {"mode": "output_only", "tokens": tokens, "capacity": capacity})
    monkeypatch.setattr(
        bench, "_replay_row",
        lambda shape, tokens, chunk, slots, reps, capacity=None:
            {"mode": "async_ring", "tokens": tokens, "capacity": capacity,
             "chunk": chunk, "async_slots": slots})
    rows = bench.run_matrix(
        ["1x8x8"], [(16, 8), (16, 16), (64, 16)], reps=1,
        schedules=((4, 2), (16, 4)))
    assert sum(row["mode"] == "summary" for row in rows) == 2
    assert not any(row["mode"] == "async_ring" and row["capacity"] == 8
                   and row["chunk"] == 16 for row in rows)
    assert all(row["compiler_path"] == "rocm_replayssm_hip" for row in rows)


def test_committed_rocm_replay_matrix_is_exact_and_oracle_gated():
    payload = json.loads(_BASELINE.read_text())
    matrix = payload["matrix"]
    assert payload["evidence_arch"] == "gfx1151"
    assert payload["compiler_path"] == "rocm_replayssm_hip"
    assert matrix["measured_rows"] == 75
    assert matrix["losing_rows"] == 0
    assert matrix["max_abs_error"] < 1e-6
    aggregate = payload["aggregate_speedup_vs_summary"]
    assert aggregate["output_only_wall"]["min"] > 1.0
    assert aggregate["async_chunk16_wall"]["min"] > 4.0
