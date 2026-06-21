from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tessera.compiler.accelerator_profiler_context import AcceleratorProfilerContext
from tessera.compiler.apple_profiler_context import AppleProfilerContext
from tessera.compiler.model_analyzer import run_model_analyzer_manifest
from tessera.compiler.profiler_context import (
    PROFILER_CONTEXT_SCHEMA_VERSION,
    build_profiler_context_artifact,
    load_profiler_context_artifact,
    validate_profiler_context_artifact,
    write_profiler_context_artifact,
)
from tessera.compiler.profiling_plan import (
    ModelAnalyzerSweep,
    model_analyzer_manifest,
    plan_profile,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "profiler"


def test_profiler_context_artifact_normalizes_mixed_system_samples(tmp_path: Path) -> None:
    apple = AppleProfilerContext(
        gpu_usage=0.70,
        total_bandwidth_gbs=360,
        achievable_bandwidth_gbs=400,
    ).to_dict()
    nvidia = AcceleratorProfilerContext(
        vendor="nvidia",
        gpu_utilization=0.95,
        memory_utilization=0.20,
        memory_used_fraction=0.10,
    ).to_dict()

    artifact = build_profiler_context_artifact(
        target="mixed",
        samples=(apple, nvidia),
        source_status="planned",
    )

    assert artifact["schema"] == PROFILER_CONTEXT_SCHEMA_VERSION
    assert artifact["provider"] == "mixed-system-context"
    assert artifact["sample_count"] == 2
    assert artifact["bottleneck_summary"]["bottlenecks"] == {
        "bandwidth_bound": 1,
        "compute_bound": 1,
    }

    path = write_profiler_context_artifact(artifact, tmp_path / "context.json")
    assert load_profiler_context_artifact(path) == artifact


def test_profiler_context_artifact_validation_rejects_bad_counts() -> None:
    payload = {
        "schema": PROFILER_CONTEXT_SCHEMA_VERSION,
        "target": "nvidia",
        "provider": "nvidia-system-context",
        "source_status": "planned",
        "sample_count": 2,
        "bottleneck_summary": {},
        "samples": [],
    }

    with pytest.raises(ValueError, match="sample_count"):
        validate_profiler_context_artifact(payload)


def test_model_analyzer_result_accepts_profiler_context_summary() -> None:
    plan = plan_profile(
        "nvidia",
        features=("model_analyzer", "runtime_api", "device_activity"),
        model_name="tiny",
        analyzer_sweep=ModelAnalyzerSweep(batch_sizes=(1,), instance_counts=(1,), dynamic_batching=(False,)),
    )
    manifest = model_analyzer_manifest(plan).to_dict()
    context = build_profiler_context_artifact(
        target="nvidia",
        samples=(
            AcceleratorProfilerContext(
                vendor="nvidia",
                gpu_utilization=0.70,
                memory_utilization=0.90,
                memory_used_fraction=0.10,
            ).to_dict(),
        ),
    )

    result = run_model_analyzer_manifest(manifest, context_artifact=context)

    assert result["context_summary"]["provider"] == "nvidia-system-context"
    assert result["context_summary"]["dominant_bottleneck"] == "bandwidth_bound"
    assert result["trial_count"] == 1


def test_tprof_report_renders_context_section(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    context_path = tmp_path / "context.json"
    status_path = tmp_path / "provider_status.json"
    out_path = tmp_path / "report.html"
    trace_path.write_text(json.dumps({"traceEvents": []}))
    context = build_profiler_context_artifact(
        target="rocm",
        samples=(
            AcceleratorProfilerContext(
                vendor="rocm",
                gpu_utilization=0.95,
                memory_utilization=0.20,
                memory_used_fraction=0.10,
                power_watts=380,
                power_limit_watts=400,
            ).to_dict(),
        ),
    )
    write_profiler_context_artifact(context, context_path)
    status_path.write_text(json.dumps({
        "schema": "tessera.profiler_provider_status.v1",
        "provider": "rocm",
        "target": "rocm",
        "status": "planned",
        "diagnostics": {"rocprofiler_sdk": "compile-gated"},
    }))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/profiler/scripts/tprof_report.py"),
            "--in",
            str(trace_path),
            "--out",
            str(out_path),
            "--context-json",
            str(context_path),
            "--provider-status-json",
            str(status_path),
        ],
        check=True,
    )

    html = out_path.read_text()
    assert "System Context" in html
    assert "rocm-system-context" in html
    assert "power_capped" in html
    assert "Provider Status" in html
    assert "rocprofiler_sdk" in html


def test_context_golden_fixture_validates() -> None:
    payload = load_profiler_context_artifact(FIXTURES / "context_nvidia_mock.json")

    assert payload["schema"] == "tessera.profiler_context.v1"
    assert payload["provider"] == "nvidia-system-context"
    assert payload["bottleneck_summary"]["dominant_bottleneck"] == "bandwidth_bound"
