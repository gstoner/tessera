"""Fresh-process ROCm native profiler capture and evidence normalization.

The collector deliberately uses the supported ``rocprofv3`` tool boundary.
Installed libraries are capability hints only: native proof is derived from
records written by the child process.  WSL hosts without ``/dev/kfd`` fail
closed before invoking ROCprofiler, which otherwise aborts the process.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .profiler_provider_trace import build_provider_trace_artifact, records_from_raw


ROCM_NATIVE_CAPTURE_SCHEMA_VERSION = "tessera.profiler_rocm_native_capture.v1"
_RTG_DISPATCH = re.compile(
    r"HSA: pid:(?P<pid>\d+) tid:(?P<tid>\d+) dispatch queue:(?P<queue>.*?) "
    r"agent:(?P<agent>\d+) signal:(?P<signal>\d+) name:'(?P<name>.*?)' "
    r"start:(?P<start>\d+) stop:(?P<stop>\d+) id:(?P<dispatch>\d+)"
)


class ROCmNativeCaptureError(ValueError):
    """Raised when a native capture request or artifact is inconsistent."""


@dataclass(frozen=True)
class ROCmCaptureRequest:
    application: tuple[str, ...]
    output_directory: Path
    rocprofv3: str = "rocprofv3"
    counters: tuple[str, ...] = ()
    pc_sampling: bool = False
    pc_sampling_method: str = "host_trap"
    pc_sampling_unit: str = "time"
    pc_sampling_interval: int = 1
    kernel_include_regex: str | None = None
    timeout_seconds: int = 120


def is_wsl() -> bool:
    release = platform.release().lower()
    return "microsoft" in release or "wsl" in release


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subprocess_output_text(value: bytes | str | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def probe_rocm_native_capabilities(
    *, rocprofv3: str | None = None, rtg_library: str | Path | None = None,
) -> dict[str, Any]:
    requested = rocprofv3 or "rocprofv3"
    requested_path = Path(requested)
    executable = (
        str(requested_path.resolve())
        if requested_path.is_file()
        else shutil.which(requested)
    )
    kfd = Path("/dev/kfd")
    dxg = Path("/dev/dxg")
    rtg = Path(rtg_library).resolve() if rtg_library else None
    return {
        "platform": platform.platform(),
        "wsl": is_wsl(),
        "dev_kfd": kfd.exists(),
        "dev_kfd_readable": os.access(kfd, os.R_OK | os.W_OK),
        "dev_dxg": dxg.exists(),
        "rocprofv3": executable,
        "rocprofiler_native_interface": bool(
            executable and kfd.exists() and os.access(kfd, os.R_OK | os.W_OK)
        ),
        "rtg_library": str(rtg) if rtg else None,
        "rtg_library_available": bool(rtg and rtg.is_file()),
    }


def build_rocprofv3_command(request: ROCmCaptureRequest) -> list[str]:
    if not request.application:
        raise ROCmNativeCaptureError("ROCm native capture requires an application")
    if request.pc_sampling_method not in {"host_trap", "stochastic"}:
        raise ROCmNativeCaptureError("unsupported ROCprofiler PC-sampling method")
    if request.pc_sampling_unit not in {"time", "cycles", "instructions"}:
        raise ROCmNativeCaptureError("unsupported ROCprofiler PC-sampling unit")
    if request.pc_sampling_interval <= 0:
        raise ROCmNativeCaptureError("PC-sampling interval must be positive")
    command = [
        request.rocprofv3,
        "--runtime-trace",
        "--kernel-trace",
        "--memory-copy-trace",
        "--stats",
        "--output-format",
        "json",
        "--output-directory",
        str(request.output_directory),
        "--output-file",
        "tessera",
    ]
    if request.kernel_include_regex:
        command.extend(["--kernel-include-regex", request.kernel_include_regex])
    if request.counters:
        command.extend(["--pmc", *request.counters])
    if request.pc_sampling:
        command.extend([
            "--pc-sampling-beta-enabled",
            "--pc-sampling-method",
            request.pc_sampling_method,
            "--pc-sampling-unit",
            request.pc_sampling_unit,
            "--pc-sampling-interval",
            str(request.pc_sampling_interval),
        ])
    command.extend(["--", *request.application])
    return command


def _first(record: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    lowered = {str(key).lower(): value for key, value in record.items()}
    for key in keys:
        if key in record:
            return record[key]
        if key.lower() in lowered:
            return lowered[key.lower()]
    return default


def _walk_record_groups(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_record_groups(child, (*path, str(key)))
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, Mapping):
                yield path, child
            yield from _walk_record_groups(child, path)


def normalize_rocprofv3_json(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Translate ROCprofiler-SDK JSON records into Tessera raw records.

    ROCprofiler uses capitalized CSV-style fields for counter output and nested
    lower-case structures for tracing/PC sampling.  Matching is therefore
    case-insensitive and path-aware, without guessing missing timestamps.
    """

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path, record in _walk_record_groups(payload):
        path_text = "/".join(path).lower()
        keys = {str(key).lower() for key in record}
        if not keys:
            continue
        raw: dict[str, Any] | None = None
        start_ns = _first(record, "start_timestamp", "start_ns", "begin_ns")
        end_ns = _first(record, "end_timestamp", "end_ns")
        correlation = _first(record, "correlation_id", "correlationId", "corr_id")
        dispatch = _first(record, "dispatch_id", "dispatchId")
        kernel = _first(record, "kernel_name", "kernel", "name")

        if "counter_collection" in path_text or {"counter_name", "counter_value"} <= keys:
            raw = {
                "record_type": "counter",
                "metric": _first(record, "counter_name", "metric", "name"),
                "value": _first(record, "counter_value", "value", default=0.0),
                "dispatch_id": dispatch,
                "correlation_id": correlation,
                "timestamp_us": float(start_ns or 0) / 1000.0,
                "kernel_name": kernel,
                "agent_id": _first(record, "agent_id"),
            }
        elif "pc_sample" in path_text or "pc_sampling" in path_text or "exec_mask" in keys:
            timestamp = _first(record, "sample_timestamp", "timestamp", default=0)
            raw = {
                "provider": "rocprofiler",
                "kind": "intra_kernel",
                "name": str(kernel or "rocprofiler.pc_sample"),
                "ts_us": float(timestamp or 0) / 1000.0,
                "duration_us": 0.0,
                "correlation_id": correlation or dispatch,
                "thread_id": _first(record, "agent_id", default=0),
                "value": None,
                "args": {
                    "record_source": "rocprofiler_pc_sampling",
                    "dispatch_id": dispatch,
                    "exec_mask": _first(record, "exec_mask"),
                    "instruction": _first(record, "instruction"),
                    "instruction_comment": _first(record, "instruction_comment"),
                    "pc": _first(record, "pc"),
                },
            }
        elif "kernel_dispatch" in path_text or "dispatch" in path_text or "dispatch_info" in keys:
            if start_ns is not None and end_ns is not None:
                raw = {
                    "record_type": "dispatch",
                    "activity": "dispatch",
                    "kernel_name": kernel or "rocprofiler.dispatch",
                    "begin_ns": start_ns,
                    "end_ns": end_ns,
                    "correlation_id": correlation,
                    "dispatch_id": dispatch,
                    "queue_id": _first(record, "queue_id"),
                    "agent_id": _first(record, "agent_id"),
                }
        elif "hip_api" in path_text or "hsa_api" in path_text or "callback" in path_text:
            if start_ns is not None and end_ns is not None:
                domain = "HSA_API" if "hsa" in path_text else "HIP_API"
                raw = {
                    "record_type": "api",
                    "domain": domain,
                    "name": _first(record, "name", "operation", default="rocprofiler.api"),
                    "begin_ns": start_ns,
                    "end_ns": end_ns,
                    "correlation_id": correlation,
                    "thread_id": _first(record, "thread_id", default=0),
                }
        if raw is None:
            continue
        canonical = json.dumps(raw, sort_keys=True, default=str)
        if canonical not in seen:
            seen.add(canonical)
            rows.append(raw)
    return rows


def _load_native_json(
    directory: Path, *, modified_after_ns: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*.json")):
        if modified_after_ns is not None and path.stat().st_mtime_ns < modified_after_ns:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            documents.append({"path": str(path), "sha256": sha256_file(path)})
            rows.extend(normalize_rocprofv3_json(payload))
    return documents, rows


def _proof_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    types = [str(row.get("record_type", row.get("kind", ""))).lower() for row in rows]
    domains = [str(row.get("domain", "")).upper() for row in rows]
    return {
        "hip_callback_seen": any(kind == "api" and domain == "HIP_API" for kind, domain in zip(types, domains)),
        "hsa_callback_seen": any(kind == "api" and domain == "HSA_API" for kind, domain in zip(types, domains)),
        "dispatch_activity_seen": any("dispatch" in kind for kind in types),
        "counter_records_seen": any("counter" in kind for kind in types),
        "pc_samples_seen": any(kind == "intra_kernel" for kind in types),
    }


def _parse_rtg_dispatches(files: Sequence[Path]) -> list[dict[str, Any]]:
    dispatches: list[dict[str, Any]] = []
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            match = _RTG_DISPATCH.search(line)
            if match is None:
                continue
            fields = match.groupdict()
            start_ns = int(fields["start"])
            end_ns = int(fields["stop"])
            if end_ns <= start_ns:
                continue
            dispatches.append({
                "kernel_name": fields["name"],
                "pid": int(fields["pid"]),
                "thread_id": int(fields["tid"]),
                "queue": fields["queue"].strip(),
                "agent_id": int(fields["agent"]),
                "signal": int(fields["signal"]),
                "dispatch_id": int(fields["dispatch"]),
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": end_ns - start_ns,
                "source_file": str(path),
            })
    return dispatches


def profiler_activity_interval_ns(capture: Mapping[str, Any]) -> int | None:
    """Return the native dispatch envelope without substituting another clock."""

    if capture.get("provider") != "rocprofiler" or capture.get("status") != "collected":
        return None
    trace = capture.get("provider_trace")
    if not isinstance(trace, Mapping):
        return None
    starts: list[float] = []
    ends: list[float] = []
    for record in trace.get("records", []):
        if not isinstance(record, Mapping) or record.get("kind") != "device_activity":
            continue
        start = record.get("ts_us")
        duration = record.get("duration_us")
        if not isinstance(start, (int, float)) or not isinstance(duration, (int, float)):
            continue
        if duration <= 0:
            continue
        starts.append(float(start))
        ends.append(float(start) + float(duration))
    if not starts:
        return None
    interval = int(round((max(ends) - min(starts)) * 1000.0))
    return interval if interval > 0 else None


def collect_rocprofv3(request: ROCmCaptureRequest) -> dict[str, Any]:
    capabilities = probe_rocm_native_capabilities(rocprofv3=request.rocprofv3)
    command = build_rocprofv3_command(request)
    request.output_directory.mkdir(parents=True, exist_ok=True)
    started_ns = time.time_ns()
    returncode: int | None = None
    stdout = ""
    stderr = ""
    timed_out = False
    blocked_reason: str | None = None
    if not capabilities["rocprofiler_native_interface"]:
        blocked_reason = "ROCPROFILER_DEVICE_INTERFACE_UNAVAILABLE"
    else:
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
                start_new_session=True,
            )
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = _subprocess_output_text(exc.stdout)
            stderr = _subprocess_output_text(exc.stderr)
        except OSError as exc:
            stderr = str(exc)
            blocked_reason = "ROCPROFILER_EXEC_FAILED"
    ended_ns = time.time_ns()
    documents, rows = _load_native_json(
        request.output_directory, modified_after_ns=started_ns,
    )
    proof = _proof_from_rows(rows)
    collected = returncode == 0 and proof["dispatch_activity_seen"]
    if blocked_reason is None and not collected:
        blocked_reason = "ROCPROFILER_NATIVE_RECORDS_MISSING"
    provider_trace = build_provider_trace_artifact(
        provider="rocprofiler",
        records=records_from_raw("rocprofiler", rows),
        source_status="native" if collected else "native_failed",
        source=str(request.output_directory),
    )
    artifact = {
        "schema": ROCM_NATIVE_CAPTURE_SCHEMA_VERSION,
        "provider": "rocprofiler",
        "status": "collected" if collected else "blocked",
        "reason": None if collected else blocked_reason,
        "fresh_process": True,
        "command": command,
        "application": list(request.application),
        "capabilities": capabilities,
        "requested": {
            "runtime_trace": True,
            "dispatch_activity": True,
            "counters": list(request.counters),
            "pc_sampling": request.pc_sampling,
            "pc_sampling_method": request.pc_sampling_method if request.pc_sampling else None,
            "pc_sampling_unit": request.pc_sampling_unit if request.pc_sampling else None,
            "pc_sampling_interval": request.pc_sampling_interval if request.pc_sampling else None,
        },
        "process": {
            "returncode": returncode,
            "signal": -returncode if isinstance(returncode, int) and returncode < 0 else None,
            "timed_out": timed_out,
            "started_ns": started_ns,
            "ended_ns": ended_ns,
            "duration_ns": ended_ns - started_ns,
            "stdout": stdout,
            "stderr": stderr,
            "clean_exit": returncode == 0,
        },
        "output_documents": documents,
        "proof": proof,
        "provider_trace": provider_trace,
        "eligible_for_regression": collected,
        "eligible_for_promotion": False,
    }
    validate_rocm_native_capture(artifact)
    return artifact


def collect_rtg_tracer(
    *, application: Sequence[str], output_directory: Path, rtg_library: Path,
    timeout_seconds: int = 120, converter: Path | None = None,
) -> dict[str, Any]:
    if not application:
        raise ROCmNativeCaptureError("rtg_tracer capture requires an application")
    output_directory.mkdir(parents=True, exist_ok=True)
    prefix = output_directory / "rtg_trace_%p"
    capabilities = probe_rocm_native_capabilities(rtg_library=rtg_library)
    environment = dict(os.environ)
    environment.update({
        "HSA_TOOLS_LIB": str(rtg_library.resolve()),
        "RTG_FILE_PREFIX": str(prefix),
        "RTG_HIP_API_FILTER": "all",
        "RTG_HSA_HOST_DISPATCH": "1",
        "RTG_PROFILE": "1",
        "RTG_PROFILE_COPY": "0",
    })
    started_ns = time.time_ns()
    returncode: int | None = None
    stdout = ""
    stderr = ""
    timed_out = False
    reason: str | None = None
    if not capabilities["rtg_library_available"]:
        reason = "RTG_LIBRARY_UNAVAILABLE"
    else:
        try:
            completed = subprocess.run(
                list(application), check=False, capture_output=True, text=True,
                timeout=timeout_seconds, start_new_session=True, env=environment,
            )
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = _subprocess_output_text(exc.stdout)
            stderr = _subprocess_output_text(exc.stderr)
        except OSError as exc:
            stderr = str(exc)
            reason = "RTG_EXEC_FAILED"
    ended_ns = time.time_ns()
    files = [
        path for path in sorted(output_directory.rglob("*"))
        if path.is_file() and path.stat().st_mtime_ns >= started_ns
    ]
    trace_json: Path | None = None
    converter_returncode: int | None = None
    converter_error: str | None = None
    if converter and converter.is_file() and files:
        trace_json = output_directory / "rtg_trace.json"
        try:
            conversion = subprocess.run(
                [str(converter), "-k", "-o", str(trace_json), *map(str, files)],
                check=False, capture_output=True, text=True, timeout=timeout_seconds,
            )
            converter_returncode = conversion.returncode
            converter_error = conversion.stderr.strip() or None
        except (OSError, subprocess.TimeoutExpired) as exc:
            converter_error = str(exc)
    trace_events: list[Mapping[str, Any]] = []
    if trace_json and trace_json.is_file():
        try:
            payload = json.loads(trace_json.read_text(encoding="utf-8"))
            trace_events = list(payload.get("traceEvents", [])) if isinstance(payload, Mapping) else []
        except json.JSONDecodeError:
            trace_events = []
    gpu_events = [event for event in trace_events if str(event.get("cat", "")).lower() in {"gpu", "hsa", "kernel"}]
    dispatches = _parse_rtg_dispatches(files)
    collected = returncode == 0 and bool(dispatches)
    if reason is None and not collected:
        reason = "RTG_NATIVE_RECORDS_MISSING"
    artifact = {
        "schema": ROCM_NATIVE_CAPTURE_SCHEMA_VERSION,
        "provider": "rtg_hsa_dispatch",
        "status": "collected" if collected else "blocked",
        "reason": None if collected else reason,
        "fresh_process": True,
        "command": list(application),
        "application": list(application),
        "capabilities": capabilities,
        "environment": {
            key: environment[key]
            for key in (
                "HSA_TOOLS_LIB",
                "RTG_FILE_PREFIX",
                "RTG_HIP_API_FILTER",
                "RTG_HSA_HOST_DISPATCH",
                "RTG_PROFILE",
                "RTG_PROFILE_COPY",
            )
        },
        "requested": {"hsa_host_dispatch": True, "hip_api_filter": "all", "profile_copy": False},
        "process": {
            "returncode": returncode,
            "signal": -returncode if isinstance(returncode, int) and returncode < 0 else None,
            "timed_out": timed_out,
            "started_ns": started_ns,
            "ended_ns": ended_ns,
            "duration_ns": ended_ns - started_ns,
            "stdout": stdout,
            "stderr": stderr,
            "clean_exit": returncode == 0,
            "teardown_complete": returncode == 0 and not timed_out,
        },
        "output_documents": [
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in files
        ],
        "proof": {
            "dispatch_activity_seen": bool(dispatches),
            "raw_dispatch_records_seen": bool(dispatches),
            "trace_files_seen": bool(files),
            "converter_returncode": converter_returncode,
            "converter_error": converter_error,
            "chrome_gpu_event_count": len(gpu_events),
        },
        "dispatch_records": dispatches,
        "provider_trace": None,
        "eligible_for_regression": collected,
        "eligible_for_promotion": False,
    }
    validate_rocm_native_capture(artifact)
    return artifact


def validate_rocm_native_capture(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != ROCM_NATIVE_CAPTURE_SCHEMA_VERSION:
        raise ROCmNativeCaptureError("unsupported ROCm native capture schema")
    if payload.get("provider") not in {"rocprofiler", "rtg_hsa_dispatch"}:
        raise ROCmNativeCaptureError("unsupported ROCm native provider")
    if payload.get("status") not in {"collected", "blocked"}:
        raise ROCmNativeCaptureError("unsupported ROCm native capture status")
    if payload.get("fresh_process") is not True:
        raise ROCmNativeCaptureError("native capture must run in a fresh process")
    process = payload.get("process")
    if not isinstance(process, Mapping):
        raise ROCmNativeCaptureError("native capture requires process evidence")
    if payload.get("status") == "collected" and process.get("clean_exit") is not True:
        raise ROCmNativeCaptureError("collected native capture requires clean process exit")
    proof = payload.get("proof")
    if not isinstance(proof, Mapping):
        raise ROCmNativeCaptureError("native capture requires proof fields")
    if payload.get("status") == "collected" and proof.get("dispatch_activity_seen") is not True:
        raise ROCmNativeCaptureError("collected native capture requires dispatch activity")
    if payload.get("provider") == "rtg_hsa_dispatch" and payload.get("status") == "collected":
        records = payload.get("dispatch_records")
        if not isinstance(records, list) or not records:
            raise ROCmNativeCaptureError("collected rtg_tracer capture requires dispatch records")
    if payload.get("status") != "collected" and not payload.get("reason"):
        raise ROCmNativeCaptureError("blocked native capture requires a reason")
    if payload.get("provider") == "rtg_hsa_dispatch" and payload.get("eligible_for_promotion"):
        raise ROCmNativeCaptureError("rtg_tracer evidence is never promotion-eligible")
    if payload.get("eligible_for_promotion"):
        raise ROCmNativeCaptureError("native capture alone cannot promote a schedule")


__all__ = [
    "ROCM_NATIVE_CAPTURE_SCHEMA_VERSION",
    "ROCmCaptureRequest",
    "ROCmNativeCaptureError",
    "build_rocprofv3_command",
    "collect_rocprofv3",
    "collect_rtg_tracer",
    "is_wsl",
    "normalize_rocprofv3_json",
    "profiler_activity_interval_ns",
    "probe_rocm_native_capabilities",
    "validate_rocm_native_capture",
]
