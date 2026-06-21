from __future__ import annotations

import json

import pytest

from tessera.compiler.profiling_plan import (
    COUNTERS,
    DEVICE_ACTIVITY,
    INTRA_KERNEL,
    MODEL_ANALYZER,
    MODEL_ANALYZER_SCHEMA_VERSION,
    PLANNED,
    RUNTIME_API,
    TRACE_SCHEMA_VERSION,
    ModelAnalyzerSweep,
    model_analyzer_manifest,
    normalize_profiler_target,
    plan_intra_kernel_probes,
    plan_profile,
    provider_capabilities,
)


def test_nvidia_plan_maps_advanced_features_to_vendor_providers() -> None:
    plan = plan_profile(
        "sm90",
        features=[RUNTIME_API, DEVICE_ACTIVITY, COUNTERS, INTRA_KERNEL, MODEL_ANALYZER],
        model_name="llama-block",
        kernels=("matmul", "flash_attn"),
        analyzer_sweep=ModelAnalyzerSweep(mode="quick", batch_sizes=(1, 4), instance_counts=(1, 2)),
    )

    payload = plan.to_dict()
    providers = {cap["feature"]: cap["provider"] for cap in payload["capabilities"]}

    assert payload["schema"] == TRACE_SCHEMA_VERSION
    assert payload["target"] == "nvidia"
    assert payload["model_name"] == "llama-block"
    assert payload["kernels"] == ["matmul", "flash_attn"]
    assert providers[RUNTIME_API] == "cupti-callback-api"
    assert providers[DEVICE_ACTIVITY] == "cupti-activity-api"
    assert providers[INTRA_KERNEL] == "cupti-pc-sampling+compiler-instrumentation"
    assert payload["summary"]["planned"] == 5


def test_apple_gpu_plan_is_explicitly_planned_not_native_proof() -> None:
    plan = plan_profile("apple_gpu", features=[RUNTIME_API, DEVICE_ACTIVITY, COUNTERS, INTRA_KERNEL])

    by_feature = {cap.feature: cap for cap in plan.capabilities}

    assert by_feature[DEVICE_ACTIVITY].provider == "metal-system-trace"
    assert by_feature[COUNTERS].provider == "metal-counter-sample-buffer"
    assert by_feature[INTRA_KERNEL].provider == "compiler-instrumentation"
    assert all(cap.status == PLANNED for cap in plan.capabilities)
    assert "outside this sandbox" in by_feature[RUNTIME_API].notes


def test_rocm_plan_uses_rocprofiler_sdk_surfaces() -> None:
    caps = {cap.feature: cap for cap in provider_capabilities("rocm_gfx942")}

    assert caps[RUNTIME_API].provider == "rocprofiler-sdk-tracing"
    assert caps[DEVICE_ACTIVITY].provider == "rocprofiler-sdk-dispatch-tracing"
    assert caps[COUNTERS].provider == "rocprofiler-sdk-counters"
    assert caps[INTRA_KERNEL].provider == "rocprofiler-sdk-pc-sampling+thread-trace"


def test_cpu_model_analyzer_sweep_serializes() -> None:
    plan = plan_profile(
        "cpu",
        features=[MODEL_ANALYZER],
        analyzer_sweep={
            "mode": "brute",
            "batch_sizes": [1, 8],
            "instance_counts": [1],
            "dynamic_batching": [False, True],
            "latency_budget_ms": 25.0,
        },
    )

    payload = json.loads(plan.to_json())

    assert payload["target"] == "cpu"
    assert payload["capabilities"][0]["status"] == "available"
    assert payload["analyzer_sweep"]["mode"] == "brute"
    assert payload["analyzer_sweep"]["latency_budget_ms"] == 25.0


def test_intra_kernel_probe_contract_serializes_when_requested() -> None:
    plan = plan_profile("cpu", features=[INTRA_KERNEL], kernels=("matmul", "softmax"))
    payload = plan.to_dict()

    assert payload["intra_kernel_probes"]
    assert {probe["kernel"] for probe in payload["intra_kernel_probes"]} == {"matmul", "softmax"}
    assert {probe["phase"] for probe in payload["intra_kernel_probes"]} == {
        "prologue",
        "mainloop",
        "epilogue",
    }
    assert payload["intra_kernel_probes"][0]["payload_fields"] == [
        "kernel",
        "phase",
        "tile",
        "program_id",
    ]
    assert payload["intra_kernel_probes"][0]["probe_name"] == "matmul.prologue"
    assert payload["intra_kernel_probes"][0]["source_op"] == "matmul"
    assert payload["intra_kernel_probes"][0]["schedule"] == "target_ir"


def test_model_analyzer_manifest_exposes_runner_contract() -> None:
    plan = plan_profile(
        "sm90",
        features=[RUNTIME_API, DEVICE_ACTIVITY, COUNTERS, INTRA_KERNEL, MODEL_ANALYZER],
        model_name="decoder",
        kernels=("matmul",),
        analyzer_sweep=ModelAnalyzerSweep(
            mode="quick",
            batch_sizes=(1, 8),
            instance_counts=(1, 2),
            latency_budget_ms=20.0,
        ),
    )

    manifest = model_analyzer_manifest(plan, objective="throughput_qps").to_dict()

    assert manifest["schema"] == MODEL_ANALYZER_SCHEMA_VERSION
    assert manifest["target"] == "nvidia"
    assert manifest["model_name"] == "decoder"
    assert manifest["search"]["batch_sizes"] == [1, 8]
    assert manifest["objective"]["primary"] == "throughput_qps"
    assert manifest["objective"]["latency_budget_ms"] == 20.0
    assert manifest["runner"]["provider"] == "tessera-model-analyzer+nvidia-triton-model-analyzer"
    assert manifest["runner"]["status"] == "planned"
    assert manifest["telemetry"]["provider_status_required"] is True
    assert manifest["telemetry"]["merged_trace_required"] is True
    assert manifest["telemetry"]["required_features"] == [RUNTIME_API, DEVICE_ACTIVITY]
    assert manifest["intra_kernel_probes"][0]["kernel"] == "matmul"


def test_plan_intra_kernel_probes_rejects_bad_metric() -> None:
    with pytest.raises(ValueError, match="unsupported intra-kernel probe metric"):
        plan_intra_kernel_probes(("k",), metrics=("mystery",))


def test_unknown_features_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown profiling feature"):
        plan_profile("nvidia", features=["surprise"])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("cuda", "nvidia"),
        ("nvidia_sm90", "nvidia"),
        ("rocm_gfx942", "rocm"),
        ("gfx942", "rocm"),
        ("metal", "apple_gpu"),
        ("apple_cpu", "cpu"),
    ],
)
def test_profiler_target_aliases(raw: str, expected: str) -> None:
    assert normalize_profiler_target(raw) == expected
