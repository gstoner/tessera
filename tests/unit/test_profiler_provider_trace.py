from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tessera.compiler.profiler_provider_trace import (
    PROVIDER_TRACE_SCHEMA_VERSION,
    build_provider_trace_artifact,
    normalize_cupti_activity_record,
    normalize_cupti_callback_record,
    normalize_metal_command_buffer_record,
    normalize_metal_counter_record,
    normalize_rocprofiler_activity_record,
    normalize_rocprofiler_api_record,
    normalize_rocprofiler_counter_record,
    normalize_rocprofiler_thread_trace_record,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "profiler"


def test_rocprofiler_api_dispatch_counter_thread_trace_order_contract() -> None:
    api = normalize_rocprofiler_api_record({
        "domain": "HIP_API",
        "name": "hipLaunchKernel",
        "start_us": 10,
        "end_us": 14,
        "correlation_id": 7,
    })
    dispatch = normalize_rocprofiler_activity_record({
        "activity": "dispatch",
        "kernel_name": "matmul",
        "start_us": 15,
        "end_us": 40,
        "correlation_id": 7,
        "dispatch_id": 11,
    })
    counter = normalize_rocprofiler_counter_record({
        "metric": "SQ_WAVES",
        "value": 123,
        "timestamp_us": 41,
        "dispatch_id": 11,
    })
    thread = normalize_rocprofiler_thread_trace_record({
        "kernel_name": "matmul",
        "start_us": 15,
        "end_us": 40,
        "dispatch_id": 11,
        "shader_engine_mask": "0x1",
        "target_cu": 0,
    })
    artifact = build_provider_trace_artifact(
        provider="rocprofiler",
        records=(api, dispatch, counter, thread),
    )

    assert artifact["schema"] == PROVIDER_TRACE_SCHEMA_VERSION
    assert artifact["summary"]["kinds"] == {
        "counter": 1,
        "device_activity": 1,
        "runtime_api": 1,
        "thread_trace": 1,
    }
    assert artifact["summary"]["correlation_count"] == 2
    assert artifact["traceEvents"][0]["cat"] == "runtime_api"
    assert artifact["traceEvents"][1]["cat"] == "device_activity"
    assert artifact["traceEvents"][2]["ph"] == "C"
    assert artifact["traceEvents"][3]["cat"] == "intra_kernel"


def test_metal_command_buffer_and_counter_correlation() -> None:
    command = normalize_metal_command_buffer_record({
        "label": "tessera_apple.gpu.profiler_probe:mainloop",
        "start_us": 100,
        "end_us": 140,
        "command_buffer_id": "cb0",
        "kernel": "matmul",
    })
    counter = normalize_metal_counter_record({
        "counter": "gpu_cycles",
        "value": 4096,
        "timestamp_us": 125,
        "command_buffer_id": "cb0",
        "probe": "mainloop",
    })
    artifact = build_provider_trace_artifact(provider="metal", records=(command, counter))

    assert artifact["summary"]["kinds"] == {"command_buffer": 1, "counter": 1}
    assert artifact["traceEvents"][0]["cat"] == "device_activity"
    assert artifact["traceEvents"][0]["args"]["correlation_id"] == "cb0"
    assert artifact["traceEvents"][1]["args"]["probe"] == "mainloop"


def test_cupti_callback_and_activity_share_correlation_id() -> None:
    callback = normalize_cupti_callback_record({
        "domain": "runtime",
        "name": "cudaLaunchKernel",
        "start_ns": 1_000,
        "end_ns": 3_000,
        "correlationId": 99,
    })
    activity = normalize_cupti_activity_record({
        "activity": "kernel",
        "kernel_name": "matmul",
        "start_ns": 4_000,
        "end_ns": 12_000,
        "correlationId": 99,
        "streamId": 2,
    })
    artifact = build_provider_trace_artifact(provider="cupti", records=(callback, activity))

    assert artifact["summary"]["correlation_count"] == 1
    assert artifact["traceEvents"][0]["args"]["correlation_id"] == 99
    assert artifact["traceEvents"][1]["args"]["correlation_id"] == 99
    assert artifact["traceEvents"][1]["cat"] == "device_activity"


def test_tprof_provider_trace_cli_writes_artifact_and_trace_json(tmp_path: Path) -> None:
    raw = [
        {
            "record_type": "api",
            "domain": "HSA_API",
            "name": "hsa_queue_create",
            "start_us": 1,
            "end_us": 2,
            "correlation_id": "hsa-1",
        },
        {
            "record_type": "dispatch",
            "kernel_name": "matmul",
            "start_us": 3,
            "end_us": 9,
            "correlation_id": "hsa-1",
        },
    ]
    raw_path = tmp_path / "rocprofiler_raw.json"
    artifact_path = tmp_path / "provider_trace.json"
    trace_path = tmp_path / "trace.json"
    raw_path.write_text(json.dumps(raw))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/profiler/scripts/tprof_provider_trace.py"),
            "--provider",
            "rocprofiler",
            "--input",
            str(raw_path),
            "--out",
            str(artifact_path),
            "--trace-out",
            str(trace_path),
        ],
        check=True,
    )

    artifact = json.loads(artifact_path.read_text())
    trace = json.loads(trace_path.read_text())
    assert artifact["provider"] == "rocprofiler"
    assert artifact["record_count"] == 2
    assert trace["traceEvents"][0]["cat"] == "runtime_api"
    assert trace["traceEvents"][1]["cat"] == "device_activity"

    report_path = tmp_path / "report.html"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/profiler/scripts/tprof_report.py"),
            "--in",
            str(trace_path),
            "--out",
            str(report_path),
        ],
        check=True,
    )
    assert "matmul" in report_path.read_text()


def test_tprof_provider_trace_cli_accepts_repeated_inputs(tmp_path: Path) -> None:
    raw_a = tmp_path / "a.json"
    raw_b = tmp_path / "b.json"
    out = tmp_path / "provider.json"
    raw_a.write_text(json.dumps([{
        "record_type": "api",
        "name": "hipMemcpy",
        "start_us": 1,
        "end_us": 2,
        "correlation_id": 1,
    }]))
    raw_b.write_text(json.dumps([{
        "record_type": "dispatch",
        "kernel_name": "matmul",
        "start_us": 3,
        "end_us": 5,
        "correlation_id": 1,
        "launch_id": "launch-1",
        "probe_name": "matmul.mainloop",
    }]))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/profiler/scripts/tprof_provider_trace.py"),
            "--provider",
            "rocprofiler",
            "--input",
            str(raw_a),
            "--input",
            str(raw_b),
            "--out",
            str(out),
        ],
        check=True,
    )

    payload = json.loads(out.read_text())
    assert payload["source_status"] == "file_batch"
    assert payload["record_count"] == 2
    assert payload["summary"]["launch_count"] == 1
    assert payload["summary"]["probe_count"] == 1


def test_provider_trace_golden_fixture_normalizes_all_rocprofiler_record_kinds(tmp_path: Path) -> None:
    out = tmp_path / "provider.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/profiler/scripts/tprof_provider_trace.py"),
            "--provider",
            "rocprofiler",
            "--input",
            str(FIXTURES / "provider_trace_rocprofiler_raw.json"),
            "--out",
            str(out),
        ],
        check=True,
    )

    payload = json.loads(out.read_text())
    assert payload["summary"]["kinds"] == {
        "counter": 1,
        "device_activity": 1,
        "runtime_api": 1,
        "thread_trace": 1,
    }
    assert payload["summary"]["correlation_count"] == 2


def test_provider_trace_cli_reports_bad_input_cleanly() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/profiler/scripts/tprof_provider_trace.py"),
            "--provider",
            "cupti",
            "--input",
            str(FIXTURES / "missing.json"),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "error:" in proc.stderr
    assert "missing.json" in proc.stderr
