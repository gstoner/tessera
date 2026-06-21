#!/usr/bin/env python3
"""Fresh-process Apple Metal profiler proof helper."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


def _repo_python_path() -> Path:
    return Path(__file__).resolve().parents[3] / "python"


def _load_metal() -> tuple[ctypes.CDLL | None, str | None, str | None]:
    framework = ctypes.util.find_library("Metal")
    candidates = [
        framework,
        "/System/Library/Frameworks/Metal.framework/Metal",
    ]
    last_error: str | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate), candidate, None
        except OSError as exc:
            last_error = str(exc)
    return None, framework, last_error or "Metal framework was not found"


def _load_objc() -> tuple[ctypes.CDLL | None, str | None]:
    path = ctypes.util.find_library("objc")
    if not path:
        return None, "objc runtime library was not found"
    try:
        return ctypes.CDLL(path), None
    except OSError as exc:
        return None, str(exc)


def _objc_sel(objc: ctypes.CDLL, name: bytes) -> int:
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    return int(objc.sel_registerName(name) or 0)


def _objc_msg_obj(objc: ctypes.CDLL, obj: int, sel: int, *extra: Any) -> int:
    objc.objc_msgSend.restype = ctypes.c_void_p
    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *[ctypes.c_ulong for _ in extra]]
    return int(objc.objc_msgSend(obj, sel, *extra) or 0)


def _objc_msg_ulong(objc: ctypes.CDLL, obj: int, sel: int) -> int:
    objc.objc_msgSend.restype = ctypes.c_ulong
    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return int(objc.objc_msgSend(obj, sel))


def _objc_msg_cstr(objc: ctypes.CDLL, obj: int, sel: int) -> str | None:
    objc.objc_msgSend.restype = ctypes.c_char_p
    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    raw = objc.objc_msgSend(obj, sel)
    return raw.decode("utf-8") if raw else None


def _prove_counter_discovery(device: int | None) -> dict[str, Any]:
    proof: dict[str, Any] = {
        "proof_api": "MTLDevice.counterSets",
        "counter_discovery_available": False,
        "counter_set_count": 0,
    }
    if not device:
        proof["error_type"] = "MetalUnavailable"
        proof["error"] = "Metal device is not visible"
        return proof
    objc, error = _load_objc()
    if objc is None:
        proof["error_type"] = "OSError"
        proof["error"] = error
        return proof
    try:
        counter_sets = _objc_msg_obj(objc, device, _objc_sel(objc, b"counterSets"))
        if not counter_sets:
            proof["error_type"] = "CounterSetsUnavailable"
            proof["error"] = "counterSets returned nil"
            return proof
        count = _objc_msg_ulong(objc, counter_sets, _objc_sel(objc, b"count"))
        proof["counter_set_count"] = count
        proof["counter_discovery_available"] = True
        if count:
            first = _objc_msg_obj(objc, counter_sets, _objc_sel(objc, b"objectAtIndex:"), 0)
            name_obj = _objc_msg_obj(objc, first, _objc_sel(objc, b"name")) if first else 0
            proof["first_counter_set"] = (
                _objc_msg_cstr(objc, name_obj, _objc_sel(objc, b"UTF8String"))
                if name_obj else None
            )
    except Exception as exc:  # pragma: no cover - platform dependent
        proof["error_type"] = type(exc).__name__
        proof["error"] = str(exc)
    return proof


def _prove_command_buffer_timestamp(adapter_library: str | None) -> dict[str, Any]:
    proof: dict[str, Any] = {
        "proof_api": "tprof_metal_capture_command_buffer_timestamp",
        "adapter_library": adapter_library,
        "timestamp_available": False,
    }
    if not adapter_library:
        proof["error_type"] = "AdapterLibraryMissing"
        proof["error"] = "pass --adapter-library or set TPROF_METAL_ADAPTER_LIB"
        return proof
    try:
        lib = ctypes.CDLL(adapter_library)
        capture = lib.tprof_metal_capture_command_buffer_timestamp
        capture.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        capture.restype = ctypes.c_bool
        start = ctypes.c_double(0.0)
        end = ctypes.c_double(0.0)
        error = ctypes.c_char_p()
        ok = capture(
            b"tessera.profiler.apple.smoke",
            ctypes.c_uint64(1),
            ctypes.byref(start),
            ctypes.byref(end),
            ctypes.byref(error),
        )
        proof["timestamp_available"] = bool(ok and end.value >= start.value)
        proof["start_us"] = start.value
        proof["end_us"] = end.value
        if error.value:
            proof["error"] = error.value.decode("utf-8")
        if not proof["timestamp_available"] and "error" not in proof:
            proof["error"] = "timestamp capture returned false"
    except Exception as exc:  # pragma: no cover - platform/build dependent
        proof["error_type"] = type(exc).__name__
        proof["error"] = str(exc)
    return proof


def _prove_metal_visibility(
    *,
    prove_counters: bool = False,
    prove_command_buffer: bool = False,
    adapter_library: str | None = None,
) -> dict[str, Any]:
    proof: dict[str, Any] = {
        "fresh_process": True,
        "platform": platform.platform(),
        "is_darwin": platform.system() == "Darwin",
        "proof_api": "MTLCreateSystemDefaultDevice",
        "metal_visible": False,
    }
    if platform.system() != "Darwin":
        proof["error_type"] = "UnsupportedPlatform"
        proof["error"] = "Apple Metal proof requires macOS"
        return proof

    metal, path, load_error = _load_metal()
    proof["metal_framework_path"] = path
    if metal is None:
        proof["error_type"] = "OSError"
        proof["error"] = load_error
        return proof

    try:
        create_device = metal.MTLCreateSystemDefaultDevice
        create_device.restype = ctypes.c_void_p
        create_device.argtypes = []
        device = create_device()
    except Exception as exc:  # pragma: no cover - platform dependent
        proof["error_type"] = type(exc).__name__
        proof["error"] = str(exc)
        return proof

    proof["metal_device_ptr"] = hex(device) if device else None
    proof["metal_visible"] = bool(device)
    if not device:
        proof["error_type"] = "MetalUnavailable"
        proof["error"] = "MTLCreateSystemDefaultDevice returned nil"
    if prove_counters:
        proof["counter_discovery"] = _prove_counter_discovery(device)
    if prove_command_buffer:
        proof["command_buffer_timestamp"] = _prove_command_buffer_timestamp(adapter_library)
    return proof


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(_repo_python_path()))
    from tessera.compiler.profiler_provider_status import collect_provider_status

    parser = argparse.ArgumentParser(
        prog="tprof-apple-metal-smoke",
        description="Run a fresh-process Apple Metal visibility proof and print provider status JSON.",
    )
    parser.add_argument("--out", help="Write provider status JSON to this path.")
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="Return success even when macOS/Metal is unavailable. Useful for CI.",
    )
    parser.add_argument(
        "--prove-counters",
        action="store_true",
        help="Also probe MTLDevice.counterSets discovery without collecting counters.",
    )
    parser.add_argument(
        "--prove-command-buffer",
        action="store_true",
        help="Also call the compiled Metal command-buffer timestamp probe from a tprof runtime library.",
    )
    parser.add_argument(
        "--adapter-library",
        default=None,
        help="Shared library exporting tprof_metal_capture_command_buffer_timestamp. Defaults to TPROF_METAL_ADAPTER_LIB.",
    )
    args = parser.parse_args(argv)

    proof = _prove_metal_visibility(
        prove_counters=args.prove_counters,
        prove_command_buffer=args.prove_command_buffer,
        adapter_library=args.adapter_library or os.environ.get("TPROF_METAL_ADAPTER_LIB"),
    )
    status = collect_provider_status("apple", native_proof=proof)
    status["diagnostics"]["smoke_script"] = "tprof-apple-metal-smoke"
    status["diagnostics"]["collection_blocked"] = not bool(proof.get("metal_visible"))

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
