from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

from tessera.compiler.profiler_rocm_native import (
    ROCmCaptureRequest,
    build_rocprofv3_command,
    collect_rocprofv3,
    collect_rtg_tracer,
    normalize_rocprofv3_json,
    profiler_activity_interval_ns,
    validate_rocm_native_capture,
)


def _official_like_payload() -> dict:
    return {
        "rocprofiler-sdk-tool": [{
            "callback_records": {
                "hip_api": [{
                    "name": "hipLaunchKernel",
                    "start_timestamp": 1_000,
                    "end_timestamp": 2_000,
                    "correlation_id": 7,
                }],
            },
            "buffer_records": {
                "kernel_dispatch": [{
                    "kernel_name": "tessera_gemm",
                    "start_timestamp": 3_000,
                    "end_timestamp": 13_000,
                    "correlation_id": 7,
                    "dispatch_id": 11,
                    "queue_id": 2,
                }],
                "counter_collection": [{
                    "Counter_Name": "SQ_WAVES",
                    "Counter_Value": 64,
                    "Start_Timestamp": 3_000,
                    "Dispatch_Id": 11,
                }],
                "pc_sampling_host_trap": [{
                    "sample_timestamp": 8_000,
                    "dispatch_id": 11,
                    "exec_mask": 0xFFFF,
                    "instruction": "v_fma_f32",
                }],
            },
        }],
    }


def test_rocprofv3_json_normalizes_activity_counters_and_pc_samples() -> None:
    rows = normalize_rocprofv3_json(_official_like_payload())
    kinds = {row.get("record_type", row.get("kind")) for row in rows}
    assert {"api", "dispatch", "counter", "intra_kernel"} <= kinds
    dispatch = next(row for row in rows if row.get("record_type") == "dispatch")
    assert dispatch["kernel_name"] == "tessera_gemm"
    assert dispatch["end_ns"] - dispatch["begin_ns"] == 10_000


def test_rocprofv3_command_owns_trace_counter_and_pc_options(tmp_path: Path) -> None:
    request = ROCmCaptureRequest(
        application=("./kernel", "--shape", "256"),
        output_directory=tmp_path,
        counters=("SQ_WAVES", "TCC_MISS"),
        pc_sampling=True,
        kernel_include_regex="tessera.*",
    )
    command = build_rocprofv3_command(request)
    assert command[:2] == ["rocprofv3", "--runtime-trace"]
    assert "--kernel-trace" in command and "--pmc" in command
    assert "--pc-sampling-beta-enabled" in command
    assert command[-3:] == ["./kernel", "--shape", "256"]


def test_wsl_or_missing_kfd_capture_fails_closed_without_fabricating_records(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        "tessera.compiler.profiler_rocm_native.probe_rocm_native_capabilities",
        lambda **_: {
            "rocprofiler_native_interface": False,
            "wsl": True,
            "dev_kfd": False,
            "dev_dxg": True,
            "rocprofv3": "rocprofv3",
        },
    )
    artifact = collect_rocprofv3(ROCmCaptureRequest(
        application=("./kernel",), output_directory=tmp_path,
    ))
    assert artifact["status"] == "blocked"
    assert artifact["reason"] == "ROCPROFILER_DEVICE_INTERFACE_UNAVAILABLE"
    assert artifact["process"]["returncode"] is None
    assert artifact["provider_trace"]["record_count"] == 0
    validate_rocm_native_capture(artifact)


def test_rocprofiler_timeout_normalizes_byte_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "tessera.compiler.profiler_rocm_native.probe_rocm_native_capabilities",
        lambda **_: {"rocprofiler_native_interface": True},
    )

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1, output=b"partial out", stderr=b"partial err")

    monkeypatch.setattr("tessera.compiler.profiler_rocm_native.subprocess.run", timeout)
    artifact = collect_rocprofv3(ROCmCaptureRequest(
        application=("./kernel",), output_directory=tmp_path,
    ))
    assert artifact["status"] == "blocked"
    assert artifact["process"]["timed_out"] is True
    assert artifact["process"]["stdout"] == "partial out"
    assert artifact["process"]["stderr"] == "partial err"


def test_activity_interval_uses_only_native_dispatch_records() -> None:
    rows = normalize_rocprofv3_json(_official_like_payload())
    from tessera.compiler.profiler_provider_trace import build_provider_trace_artifact, records_from_raw

    trace = build_provider_trace_artifact(
        provider="rocprofiler", records=records_from_raw("rocprofiler", rows),
        source_status="native",
    )
    capture = {
        "schema": "tessera.profiler_rocm_native_capture.v1",
        "provider": "rocprofiler",
        "status": "collected",
        "reason": None,
        "fresh_process": True,
        "process": {"clean_exit": True},
        "provider_trace": trace,
        "eligible_for_promotion": False,
    }
    assert profiler_activity_interval_ns(capture) == 10_000


def test_rtg_missing_library_is_a_structured_fresh_process_block(tmp_path: Path) -> None:
    artifact = collect_rtg_tracer(
        application=("/bin/true",), output_directory=tmp_path,
        rtg_library=tmp_path / "missing.so",
    )
    assert artifact["status"] == "blocked"
    assert artifact["reason"] == "RTG_LIBRARY_UNAVAILABLE"
    assert artifact["fresh_process"] is True
    assert artifact["environment"]["RTG_HSA_HOST_DISPATCH"] == "1"
    assert artifact["environment"]["RTG_HIP_API_FILTER"] == "all"
    assert artifact["eligible_for_promotion"] is False
    validate_rocm_native_capture(artifact)


def test_rtg_fresh_process_requires_ordered_raw_dispatch_record(
    tmp_path: Path, monkeypatch,
) -> None:
    library = tmp_path / "rtg_tracer.so"
    library.write_bytes(b"test")

    def run(command, **kwargs):
        environment = kwargs["env"]
        trace = Path(environment["RTG_FILE_PREFIX"].replace("%p", "123"))
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_text(
            "HSA: pid:123 tid:7 dispatch queue:0xabc agent:1 signal:9 "
            "name:'tessera_gemm' start:1000 stop:9000 id:4\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("tessera.compiler.profiler_rocm_native.subprocess.run", run)
    artifact = collect_rtg_tracer(
        application=("./application",), output_directory=tmp_path / "trace",
        rtg_library=library,
    )
    assert artifact["status"] == "collected"
    assert artifact["proof"]["dispatch_activity_seen"] is True
    assert artifact["dispatch_records"][0]["duration_ns"] == 8000
    assert artifact["process"]["teardown_complete"] is True
    assert artifact["eligible_for_promotion"] is False


def test_capture_survives_a_filesystem_timestamp_behind_the_wall_clock(
    tmp_path: Path, monkeypatch,
) -> None:
    """Trace files must be selected by a state diff, not by comparing their
    mtime against a `time.time_ns()` reading taken before the run.

    Those are two different clocks. On Linux an inode timestamp comes from the
    COARSE clock (one scheduler tick of granularity) while `time.time_ns()`
    reads the fine clock, so a file written microseconds after the reference
    can carry an mtime that rounds BELOW it. The file was then filtered out, no
    dispatch records were parsed, and a capture that actually succeeded
    reported `blocked` — intermittently, and more often the faster the traced
    application. This backdates the trace by 5 ms to make that deterministic.
    """
    platform.platform()  # warm uname()'s cache before subprocess.run is patched
    library = tmp_path / "rtg_tracer.so"
    library.write_bytes(b"test")

    def run(command, **kwargs):
        environment = kwargs["env"]
        trace = Path(environment["RTG_FILE_PREFIX"].replace("%p", "123"))
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_text(
            "HSA: pid:123 tid:7 dispatch queue:0xabc agent:1 signal:9 "
            "name:'tessera_gemm' start:1000 stop:9000 id:4\n",
            encoding="utf-8",
        )
        backdated = time.time_ns() - 5_000_000
        os.utime(trace, ns=(backdated, backdated))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("tessera.compiler.profiler_rocm_native.subprocess.run", run)
    artifact = collect_rtg_tracer(
        application=("./application",), output_directory=tmp_path / "trace",
        rtg_library=library,
    )
    assert artifact["status"] == "collected", artifact.get("reason")
    assert artifact["proof"]["dispatch_activity_seen"] is True


def _fail_the_first_walk(monkeypatch) -> None:
    """Make the next `rglob` list part of the tree and then fail.

    The realistic shape of the defect: a file rotated or cleaned away by
    another process between listing and `stat`, or a permission error partway
    through. Patched at `rglob` rather than `stat` because whether `is_file()`
    routes through `Path.stat` changed between CPython 3.12 and 3.14.
    """
    real_rglob = Path.rglob
    remaining = {"failures": 1}

    def flaky_rglob(self, pattern, *args, **kwargs):
        entries = list(real_rglob(self, pattern, *args, **kwargs))
        if remaining["failures"]:
            remaining["failures"] -= 1
            yield from entries[:1]
            raise PermissionError(13, "Permission denied", str(self))
        yield from entries

    monkeypatch.setattr(Path, "rglob", flaky_rglob)


def test_snapshot_of_an_incomplete_walk_is_not_an_empty_directory(
    tmp_path: Path, monkeypatch,
) -> None:
    from tessera.compiler.profiler_rocm_native import _snapshot_files

    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "b.json").write_text("{}", encoding="utf-8")
    assert _snapshot_files(tmp_path) is not None
    assert _snapshot_files(tmp_path / "empty").keys() == set()

    _fail_the_first_walk(monkeypatch)
    assert _snapshot_files(tmp_path) is None


def test_unreadable_baseline_blocks_instead_of_claiming_stale_traces(
    tmp_path: Path, monkeypatch,
) -> None:
    """A baseline that could not be established must not read as "the output
    directory was empty" — against an empty baseline every trace left by an
    earlier run looks freshly written, and the capture would report `collected`
    on someone else's evidence.
    """
    monkeypatch.setattr(
        "tessera.compiler.profiler_rocm_native.probe_rocm_native_capabilities",
        lambda **_: {"rocprofiler_native_interface": True},
    )
    stale = tmp_path / "results.json"
    stale.write_text(json.dumps(_official_like_payload()), encoding="utf-8")
    (tmp_path / "other.json").write_text("{}", encoding="utf-8")
    launched: list[tuple] = []

    def run(command, **kwargs):
        launched.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("tessera.compiler.profiler_rocm_native.subprocess.run", run)
    _fail_the_first_walk(monkeypatch)
    artifact = collect_rocprofv3(ROCmCaptureRequest(
        application=("./kernel",), output_directory=tmp_path,
    ))
    assert artifact["status"] == "blocked"
    assert artifact["reason"] == "ROCPROFILER_OUTPUT_BASELINE_UNREADABLE"
    assert launched == []
    assert artifact["process"]["returncode"] is None
    assert artifact["proof"]["dispatch_activity_seen"] is False
    assert artifact["provider_trace"]["record_count"] == 0
    validate_rocm_native_capture(artifact)


def test_rtg_unreadable_baseline_blocks_with_a_named_reason(
    tmp_path: Path, monkeypatch,
) -> None:
    library = tmp_path / "rtg_tracer.so"
    library.write_bytes(b"test")
    output = tmp_path / "trace"
    output.mkdir()
    (output / "stale.txt").write_text(
        "HSA: pid:123 tid:7 dispatch queue:0xabc agent:1 signal:9 "
        "name:'tessera_gemm' start:1000 stop:9000 id:4\n",
        encoding="utf-8",
    )

    def run(command, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("the traced application must not be launched")

    monkeypatch.setattr("tessera.compiler.profiler_rocm_native.subprocess.run", run)
    _fail_the_first_walk(monkeypatch)
    artifact = collect_rtg_tracer(
        application=("./application",), output_directory=output,
        rtg_library=library,
    )
    assert artifact["status"] == "blocked"
    assert artifact["reason"] == "RTG_OUTPUT_BASELINE_UNREADABLE"
    assert artifact["dispatch_records"] == []
    validate_rocm_native_capture(artifact)
