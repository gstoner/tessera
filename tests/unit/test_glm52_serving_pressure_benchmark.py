"""Smoke tests for the GLM-5.2 combined serving-pressure benchmark."""

from __future__ import annotations

from benchmarks.rl.benchmark_glm52_serving_pressure import (
    plan_full_glm52_serving_pressure,
    run_scaled_glm52_serving_pressure,
)


def test_full_plan_is_shape_only_1m_context():
    out = plan_full_glm52_serving_pressure()
    assert out["model"] == "glm5_2"
    assert out["context_length"] == 1_048_576
    assert out["index_topk"] == 2048
    assert out["mtp_steps"] == 4
    assert out["indexer_calls_saved_per_token"] > 0
    assert out["kv_bytes_per_request"] > 0
    assert out["materialized_verify_bytes_per_request"] > out["kv_bytes_per_request"]


def test_scaled_serving_pressure_runs_and_reports_combined_metrics():
    out = run_scaled_glm52_serving_pressure(tokens=16, seed=0)
    assert out["model"] == "glm5_2_scaled"
    assert out["tokens"] == 16
    assert out["mtp_steps"] == 4
    assert len(out["accepted_length_by_step"]) == 4
    assert out["indexer_calls_saved_per_token"] > 0
    assert out["kv_bytes_per_request"] > 0
    assert out["materialized_verify_bytes_per_request"] > out["kv_bytes_per_request"]
    assert out["tokens_per_second"] > 0
