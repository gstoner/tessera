from __future__ import annotations

import json

from tessera.compiler.model_analyzer import (
    MODEL_ANALYZER_RESULT_SCHEMA_VERSION,
    run_model_analyzer_manifest,
    write_model_analyzer_result,
)
from tessera.compiler.profiling_plan import (
    DEVICE_ACTIVITY,
    INTRA_KERNEL,
    MODEL_ANALYZER,
    RUNTIME_API,
    ModelAnalyzerSweep,
    model_analyzer_manifest,
    plan_profile,
)


def _manifest():
    plan = plan_profile(
        "cpu",
        features=[RUNTIME_API, DEVICE_ACTIVITY, INTRA_KERNEL, MODEL_ANALYZER],
        model_name="tiny",
        kernels=("matmul",),
        analyzer_sweep=ModelAnalyzerSweep(
            mode="quick",
            batch_sizes=(1, 4),
            instance_counts=(1, 2),
            dynamic_batching=(False, True),
        ),
    )
    return model_analyzer_manifest(plan).to_dict()


def test_model_analyzer_runner_expands_search_space_and_selects_best_latency():
    result = run_model_analyzer_manifest(_manifest())

    assert result["schema"] == MODEL_ANALYZER_RESULT_SCHEMA_VERSION
    assert result["target"] == "cpu"
    assert result["trial_count"] == 8
    assert result["runner"]["status"] == "available"
    assert result["best"]["batch_size"] == 1
    assert result["best"]["instance_count"] == 2
    assert result["best"]["dynamic_batching"] is True
    assert result["best"]["status"] == "estimated"
    assert result["trials"][0]["measured"] is False


def test_model_analyzer_runner_accepts_real_measurement_hook(tmp_path):
    def measure(trial, manifest):
        return {
            "latency_ms": 10.0 / trial.instance_count,
            "throughput_qps": trial.batch_size * 100.0,
            "memory_bytes": trial.batch_size * 1024,
            "metadata": {"target": manifest["target"]},
        }

    result = run_model_analyzer_manifest(_manifest(), measure=measure)
    out = tmp_path / "result.json"
    write_model_analyzer_result(result, out)
    payload = json.loads(out.read_text())

    assert payload["trials"][0]["measured"] is True
    assert payload["trials"][0]["metadata"]["target"] == "cpu"
    assert payload["best"]["instance_count"] == 2
