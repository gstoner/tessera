#!/usr/bin/env python3
"""ROCprofiler-SDK native proof helper.

This script is safe on non-ROCm hosts: it emits a provider-status artifact with
diagnostics instead of failing the whole profiler path. A hardware proof job can
wire real ROCprofiler-SDK callbacks later and set the proof booleans true only
after observing HIP/HSA callbacks and dispatch/activity records.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any


def _repo_python_path() -> Path:
    return Path(__file__).resolve().parents[3] / "python"


def _find_library(candidates: tuple[str, ...]) -> tuple[str | None, str | None]:
    last_error: str | None = None
    for name in candidates:
        path = ctypes.util.find_library(name) or name
        try:
            ctypes.CDLL(path)
            return path, None
        except OSError as exc:
            last_error = str(exc)
    return None, last_error


def _amdsmi_probe() -> dict[str, Any]:
    proof: dict[str, Any] = {
        "probe_api": "amdsmi",
        "amdsmi_python_available": importlib.util.find_spec("amdsmi") is not None,
        "amd_gpu_visible": False,
    }
    if not proof["amdsmi_python_available"]:
        proof["error_type"] = "ModuleNotFoundError"
        proof["error"] = "amdsmi Python package is not importable"
        return proof
    try:
        import amdsmi  # type: ignore

        init = getattr(amdsmi, "amdsmi_init", None)
        shutdown = getattr(amdsmi, "amdsmi_shut_down", None)
        sockets_fn = getattr(amdsmi, "amdsmi_get_socket_handles", None)
        devices_fn = getattr(amdsmi, "amdsmi_get_processor_handles", None)
        if init:
            init()
        try:
            sockets = sockets_fn() if sockets_fn else []
            devices = []
            if devices_fn:
                for socket in sockets or [None]:
                    try:
                        devices.extend(devices_fn(socket) if socket is not None else devices_fn())
                    except TypeError:
                        devices.extend(devices_fn())
            proof["amd_gpu_visible"] = bool(devices)
            proof["device_count"] = len(devices)
        finally:
            if shutdown:
                shutdown()
    except Exception as exc:  # pragma: no cover - depends on host ROCm stack
        proof["error_type"] = type(exc).__name__
        proof["error"] = str(exc)
    return proof


def _collect_proof() -> dict[str, Any]:
    rocprofiler_path, rocprofiler_error = _find_library((
        "rocprofiler-sdk",
        "rocprofiler64",
        "librocprofiler-sdk.so",
    ))
    amdsmi = _amdsmi_probe()
    proof: dict[str, Any] = {
        "fresh_process": True,
        "platform": platform.platform(),
        "proof_api": "ROCprofiler-SDK context/tool/callback/activity proof",
        "rocprofiler_sdk_library": rocprofiler_path,
        "rocprofiler_sdk_visible": rocprofiler_path is not None,
        "amd_gpu_visible": bool(amdsmi.get("amd_gpu_visible")),
        "amdsmi": amdsmi,
        "context_created": False,
        "tool_registered": False,
        "hip_callback_seen": False,
        "hsa_callback_seen": False,
        "dispatch_activity_seen": False,
        "counter_discovery_seen": False,
    }
    if rocprofiler_error:
        proof["rocprofiler_sdk_error"] = rocprofiler_error
    return proof


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(_repo_python_path()))
    from tessera.compiler.profiler_provider_status import collect_provider_status

    parser = argparse.ArgumentParser(
        prog="tprof-rocm-native-smoke",
        description="Emit ROCm provider status from a best-effort native proof probe.",
    )
    parser.add_argument("--out", help="Write provider status JSON to this path.")
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="Return success when ROCm hardware/SDK proof is unavailable.",
    )
    args = parser.parse_args(argv)

    proof = _collect_proof()
    status = collect_provider_status("rocm", native_proof=proof)
    status["diagnostics"]["smoke_script"] = "tprof-rocm-native-smoke"
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
