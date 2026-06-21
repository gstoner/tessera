from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tessera.compiler.accelerator_profiler_context import AcceleratorProfilerContext
from tessera.compiler.profiler_context import build_profiler_context_artifact, write_profiler_context_artifact
from tessera.compiler.profiler_provider_trace import build_provider_trace_artifact, write_provider_trace_artifact
from tessera.compiler.profiler_provider_status import provider_status_artifact
from tessera.compiler.profiler_trace_merge import (
    MERGED_PROFILER_TRACE_SCHEMA_VERSION,
    merge_profiler_traces,
    validate_merged_profiler_trace,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "profiler"


def test_merge_profiler_traces_combines_runtime_provider_and_context() -> None:
    runtime = {
        "traceEvents": [
            {"name": "tsrLaunchKernel", "cat": "runtime_api", "ph": "X", "ts": 5, "dur": 2, "args": {}}
        ]
    }
    provider = build_provider_trace_artifact(
        provider="rocprofiler",
        records=[
            {
                "record_type": "dispatch",
                "kernel_name": "matmul",
                "start_us": 7,
                "end_us": 13,
                "correlation_id": 42,
                "launch_id": "launch-1",
                "probe_name": "matmul.mainloop",
                "dropped_records": 2,
            }
        ],
    )
    context = build_profiler_context_artifact(
        target="rocm",
        samples=[
            AcceleratorProfilerContext(
                vendor="rocm",
                gpu_utilization=0.70,
                memory_utilization=0.90,
                memory_used_fraction=0.20,
            ).to_dict()
        ],
    )

    merged = merge_profiler_traces(
        runtime_trace=runtime,
        provider_traces=(provider,),
        context_artifact=context,
        provider_statuses=(
            provider_status_artifact(provider="rocm", target="rocm", status="planned"),
        ),
    )

    assert merged["schema"] == MERGED_PROFILER_TRACE_SCHEMA_VERSION
    assert merged["summary"]["categories"] == {
        "device_activity": 1,
        "host_context": 1,
        "provider_status": 1,
        "runtime_api": 1,
    }
    assert merged["summary"]["correlation_count"] == 1
    assert merged["summary"]["launch_count"] == 1
    assert merged["summary"]["probe_count"] == 1
    assert merged["summary"]["dropped_records"] == 2
    assert merged["context_summary"]["provider"] == "rocm-system-context"
    assert merged["provider_statuses"][0]["provider"] == "rocm"
    validate_merged_profiler_trace(merged)


def test_tprof_merge_trace_cli_writes_report_ready_trace(tmp_path: Path) -> None:
    runtime_path = tmp_path / "runtime.trace.json"
    provider_path = tmp_path / "provider.json"
    context_path = tmp_path / "context.json"
    status_path = tmp_path / "status.json"
    merged_path = tmp_path / "merged.json"
    runtime_path.write_text(json.dumps({
        "traceEvents": [
            {"name": "tsrMemcpy", "cat": "device_activity", "ph": "X", "ts": 1, "dur": 3, "args": {}}
        ]
    }))
    write_provider_trace_artifact(
        build_provider_trace_artifact(
            provider="cupti",
            records=[
                {
                    "record_type": "activity",
                    "activity": "kernel",
                    "kernel_name": "matmul",
                    "start_us": 4,
                    "end_us": 12,
                    "correlationId": 9,
                }
            ],
        ),
        provider_path,
    )
    write_profiler_context_artifact(
        build_profiler_context_artifact(
            target="nvidia",
            samples=[
                AcceleratorProfilerContext(
                    vendor="nvidia",
                    gpu_utilization=0.95,
                    memory_utilization=0.20,
                    memory_used_fraction=0.20,
                ).to_dict()
            ],
        ),
        context_path,
    )
    status_path.write_text(json.dumps(provider_status_artifact(
        provider="nvidia",
        target="nvidia",
        status="planned",
    )))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/profiler/scripts/tprof_merge_trace.py"),
            "--runtime-trace",
            str(runtime_path),
            "--provider-trace",
            str(provider_path),
            "--context-json",
            str(context_path),
            "--provider-status",
            str(status_path),
            "--out",
            str(merged_path),
        ],
        check=True,
    )

    merged = json.loads(merged_path.read_text())
    assert merged["schema"] == MERGED_PROFILER_TRACE_SCHEMA_VERSION
    assert merged["summary"]["event_count"] == 4
    names = [event["name"] for event in merged["traceEvents"]]
    assert "profiler_context.summary" in names
    assert "provider_status.nvidia" in names
    assert merged["traceEvents"][-1]["name"] == "matmul"
    assert merged["provider_statuses"][0]["provider"] == "nvidia"
    validate_merged_profiler_trace(merged)


def test_tprof_merge_trace_cli_accepts_committed_fixtures(tmp_path: Path) -> None:
    provider_path = tmp_path / "provider.json"
    merged_path = tmp_path / "merged.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/profiler/scripts/tprof_provider_trace.py"),
            "--provider",
            "rocprofiler",
            "--input",
            str(FIXTURES / "provider_trace_rocprofiler_raw.json"),
            "--out",
            str(provider_path),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/profiler/scripts/tprof_merge_trace.py"),
            "--runtime-trace",
            str(FIXTURES / "runtime_trace_min.json"),
            "--provider-trace",
            str(provider_path),
            "--context-json",
            str(FIXTURES / "context_nvidia_mock.json"),
            "--out",
            str(merged_path),
        ],
        check=True,
    )

    merged = json.loads(merged_path.read_text())
    assert merged["summary"]["categories"]["runtime_api"] == 2
    assert merged["summary"]["categories"]["device_activity"] == 1
    assert merged["summary"]["categories"]["counters"] == 1
    assert merged["summary"]["categories"]["intra_kernel"] == 1


def test_tprof_merge_trace_cli_reports_malformed_context_cleanly(tmp_path: Path) -> None:
    bad_context = tmp_path / "bad_context.json"
    out = tmp_path / "merged.json"
    bad_context.write_text(json.dumps({"schema": "wrong"}))

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/profiler/scripts/tprof_merge_trace.py"),
            "--context-json",
            str(bad_context),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "error:" in proc.stderr
    assert "unsupported profiler context" in proc.stderr


def test_merge_profiler_traces_rejects_non_numeric_timestamp() -> None:
    runtime = {
        "traceEvents": [
            {"name": "bad", "cat": "runtime_api", "ph": "X", "ts": "not-a-number", "args": {}}
        ]
    }

    try:
        merge_profiler_traces(runtime_trace=runtime)
    except ValueError as exc:
        assert "non-numeric ts" in str(exc)
    else:
        raise AssertionError("expected non-numeric timestamp to be rejected")
