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


def test_model_analyzer_runner_attaches_profiler_status_and_merged_traces(tmp_path):
    trace_path = tmp_path / "merged.json"
    trace_path.write_text("{}\n")
    status = {
        "schema": "tessera.profiler_provider_status.v1",
        "provider": "apple",
        "target": "apple_gpu",
        "status": "compiled_shell",
        "diagnostics": {"native_proof_required": "fresh-process"},
    }

    def measure(trial, _manifest):
        return {
            "latency_ms": 2.0,
            "throughput_qps": 100.0,
            "memory_bytes": 1024,
            "metadata": {
                "route": "mtl4",
                "fallback_reason": "counter buffers unavailable",
            },
        }

    result = run_model_analyzer_manifest(
        _manifest(),
        measure=measure,
        provider_statuses=(status,),
        merged_trace_paths=(trace_path,),
    )

    assert result["provider_statuses"][0]["provider"] == "apple"
    assert result["provider_status_summary"]["providers"]["apple"] == "compiled_shell"
    assert result["provider_status_summary"]["native_claims_allowed"] is False
    assert result["merged_traces"][0]["path"] == str(trace_path)
    assert "fallback_bound" in result["bottleneck_labels"]
    assert "apple_provider_unproven" in result["bottleneck_labels"]


def test_model_analyzer_reports_unmet_provider_requirements():
    manifest = {
        **_manifest(),
        "provider_requirements": {
            "providers": ["nvidia"],
            "features": ["runtime_api", "device_activity", "counters"],
        },
    }
    status = {
        "schema": "tessera.profiler_provider_status.v1",
        "provider": "nvidia",
        "target": "nvidia",
        "status": "planned",
        "diagnostics": {"native_proof_required": "CUPTI callback/activity proof"},
    }

    result = run_model_analyzer_manifest(manifest, provider_statuses=(status,))

    assert result["provider_requirements"]["met"] is False
    assert result["provider_requirements"]["unmet"][0]["provider"] == "nvidia"
    assert result["provider_requirements"]["unmet"][0]["status"] == "planned"
    assert "provider_requirements_unmet" in result["bottleneck_labels"]
