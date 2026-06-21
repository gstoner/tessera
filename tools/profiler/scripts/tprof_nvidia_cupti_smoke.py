#!/usr/bin/env python3
"""CUPTI/NVML native proof helper.

The helper is intentionally conservative: library visibility is recorded as
diagnostic metadata, but native availability requires proof that a CUPTI
subscriber, callback record, activity buffer, and device activity record were
observed on an NVIDIA GPU.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import platform
import sys
from pathlib import Path
from typing import Any


def _repo_python_path() -> Path:
    return Path(__file__).resolve().parents[3] / "python"


def _load_library(candidates: tuple[str, ...]) -> tuple[ctypes.CDLL | None, str | None, str | None]:
    last_error: str | None = None
    for name in candidates:
        path = ctypes.util.find_library(name) or name
        try:
            return ctypes.CDLL(path), path, None
        except OSError as exc:
            last_error = str(exc)
    return None, None, last_error


def _nvml_probe() -> dict[str, Any]:
    lib, path, error = _load_library(("nvidia-ml", "libnvidia-ml.so.1", "nvml"))
    proof: dict[str, Any] = {
        "probe_api": "NVML",
        "nvml_library": path,
        "nvidia_gpu_visible": False,
    }
    if lib is None:
        proof["error_type"] = "OSError"
        proof["error"] = error or "NVML library was not found"
        return proof
    try:
        init = getattr(lib, "nvmlInit_v2", None) or getattr(lib, "nvmlInit", None)
        shutdown = getattr(lib, "nvmlShutdown", None)
        count_fn = getattr(lib, "nvmlDeviceGetCount_v2", None) or getattr(lib, "nvmlDeviceGetCount", None)
        if init:
            init.restype = ctypes.c_int
            rc = init()
            proof["nvml_init_result"] = int(rc)
            if rc != 0:
                proof["error_type"] = "NVMLInitError"
                proof["error"] = f"nvmlInit returned {rc}"
                return proof
        if count_fn:
            count = ctypes.c_uint(0)
            count_fn.argtypes = [ctypes.POINTER(ctypes.c_uint)]
            count_fn.restype = ctypes.c_int
            rc = count_fn(ctypes.byref(count))
            proof["nvml_device_count_result"] = int(rc)
            proof["device_count"] = int(count.value)
            proof["nvidia_gpu_visible"] = rc == 0 and count.value > 0
        if shutdown:
            shutdown.restype = ctypes.c_int
            shutdown()
    except Exception as exc:  # pragma: no cover - depends on host NVIDIA stack
        proof["error_type"] = type(exc).__name__
        proof["error"] = str(exc)
    return proof


def _collect_proof() -> dict[str, Any]:
    cupti, cupti_path, cupti_error = _load_library(("cupti", "libcupti.so", "libcupti.so.12"))
    nvml = _nvml_probe()
    proof: dict[str, Any] = {
        "fresh_process": True,
        "platform": platform.platform(),
        "proof_api": "CUPTI subscriber/callback/activity proof",
        "cupti_library": cupti_path,
        "cupti_visible": cupti is not None,
        "nvidia_gpu_visible": bool(nvml.get("nvidia_gpu_visible")),
        "nvml": nvml,
        "subscriber_created": False,
        "callback_seen": False,
        "activity_buffer_seen": False,
        "activity_seen": False,
        "metric_discovery_seen": False,
    }
    if cupti_error:
        proof["cupti_error"] = cupti_error
    return proof


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(_repo_python_path()))
    from tessera.compiler.profiler_provider_status import collect_provider_status

    parser = argparse.ArgumentParser(
        prog="tprof-nvidia-cupti-smoke",
        description="Emit NVIDIA provider status from a best-effort CUPTI/NVML proof probe.",
    )
    parser.add_argument("--out", help="Write provider status JSON to this path.")
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="Return success when NVIDIA hardware/CUPTI proof is unavailable.",
    )
    args = parser.parse_args(argv)

    proof = _collect_proof()
    status = collect_provider_status("nvidia", native_proof=proof)
    status["diagnostics"]["smoke_script"] = "tprof-nvidia-cupti-smoke"
    status["diagnostics"]["collection_blocked"] = status["status"] != "native_available"

    text = json.dumps(status, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    else:
        sys.stdout.write(text)

    if status["status"] == "native_available":
        return 0
    return 0 if args.allow_unavailable else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
