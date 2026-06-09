"""
runtime.py — Python ctypes wrapper around the Tessera runtime C ABI (Phase 6)

Provides ``TesseraRuntime``, a high-level Python interface over the C ABI
defined in ``tessera_runtime.h``.  When the compiled shared library is not
found the module transparently falls back to a pure-Python mock so tests and
development workflows always succeed.

Usage::

    from tessera.runtime import TesseraRuntime, DeviceKind

    rt = TesseraRuntime()
    rt.init()
    dev = rt.get_device(0)             # CPU device (always present)
    props = rt.get_device_props(dev)
    buf = rt.malloc(dev, 1024)
    rt.memset(buf, 0, 1024)
    rt.free(buf)
    rt.shutdown()
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, List, Mapping, Optional

from .telemetry import TELEMETRY_SCHEMA_VERSION, make_event, telemetry_report
from .compiler.capabilities import get_target_capability, normalize_target, runtime_status as compiler_runtime_status
from .compiler.execution_matrix import executor_for_metadata as _exec_row_for_metadata


# ---------------------------------------------------------------------------
# Status codes (mirror TsrStatus)
# ---------------------------------------------------------------------------

class TsrStatus(IntEnum):
    SUCCESS          = 0
    INVALID_ARGUMENT = 1
    NOT_FOUND        = 2
    ALREADY_EXISTS   = 3
    OUT_OF_MEMORY    = 4
    UNIMPLEMENTED    = 5
    INTERNAL         = 6
    DEVICE_ERROR     = 7


class DeviceKind(IntEnum):
    CPU  = 0
    CUDA = 1
    HIP  = 2


class MemcpyKind(IntEnum):
    HOST_TO_DEVICE   = 0
    DEVICE_TO_HOST   = 1
    DEVICE_TO_DEVICE = 2
    HOST_TO_HOST     = 3


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TesseraRuntimeError(RuntimeError):
    """Raised when a C ABI call returns a non-SUCCESS status."""

    def __init__(self, fn_name: str, status: TsrStatus, detail: str = ""):
        msg = f"{fn_name} failed: {status.name}"
        if detail:
            msg += f" — {detail}"
        super().__init__(msg)
        self.status = status
        self.fn_name = fn_name


# ---------------------------------------------------------------------------
# Device properties
# ---------------------------------------------------------------------------

@dataclass
class DeviceProps:
    kind: DeviceKind
    name: str
    logical_tile_threads_max: int
    concurrent_tiles_hint: int

    def __repr__(self) -> str:
        return (
            f"DeviceProps(kind={self.kind.name}, name={self.name!r}, "
            f"tile_threads={self.logical_tile_threads_max}, "
            f"concurrent={self.concurrent_tiles_hint})"
        )


@dataclass(frozen=True)
class RuntimeProfile:
    cpu_wall_ms: float | None = None
    kernel_elapsed_ms: float | None = None
    memory_bytes: int | None = None
    launch_overhead_ms: float | None = None


@dataclass(frozen=True)
class BackendCapability:
    name: str
    available: bool
    executable: bool
    reason: str = ""
    dtypes: tuple[str, ...] = ()
    features: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeArtifact:
    graph_ir: str = ""
    schedule_ir: str = ""
    tile_ir: str = ""
    target_ir: str = ""
    metadata: dict[str, Any] | None = None
    abi_signature: str = ""

    @property
    def artifact_hash(self) -> str:
        payload = self.to_json(include_hash=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_ir": self.graph_ir,
            "schedule_ir": self.schedule_ir,
            "tile_ir": self.tile_ir,
            "target_ir": self.target_ir,
            "metadata": self.metadata or {},
            "abi_signature": self.abi_signature,
            "artifact_hash": self.artifact_hash,
        }

    def to_json(self, *, include_hash: bool = True) -> str:
        data = {
            "graph_ir": self.graph_ir,
            "schedule_ir": self.schedule_ir,
            "tile_ir": self.tile_ir,
            "target_ir": self.target_ir,
            "metadata": self.metadata or {},
            "abi_signature": self.abi_signature,
        }
        if include_hash:
            data["artifact_hash"] = hashlib.sha256(
                json.dumps(data, sort_keys=True).encode("utf-8")
            ).hexdigest()
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str | bytes) -> "RuntimeArtifact":
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        data = json.loads(payload)
        return cls(
            graph_ir=data.get("graph_ir", ""),
            schedule_ir=data.get("schedule_ir", ""),
            tile_ir=data.get("tile_ir", ""),
            target_ir=data.get("target_ir", ""),
            metadata=data.get("metadata") or {},
            abi_signature=data.get("abi_signature", ""),
        )


_last_profile = RuntimeProfile()
_apple_cpu_runtime: ctypes.CDLL | None = None
_apple_gpu_runtime: ctypes.CDLL | None = None


# bfloat16 dtype is exposed by the optional ml_dtypes package. We import lazily
# and tolerate it missing so the rest of the runtime works for users who do
# not need the bf16 fast path. ml_dtypes is the de-facto standard now (JAX,
# TensorFlow Probability, PyTorch's experimental numpy bridge all use it).
def _bfloat16_dtype() -> Any:
    try:
        import ml_dtypes
        return ml_dtypes.bfloat16
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Mock implementation (no shared library required)
# ---------------------------------------------------------------------------

class _MockHandle:
    """Opaque handle returned by mock operations."""

    _counter = 0

    def __init__(self, kind: str, data: Any = None):
        _MockHandle._counter += 1
        self._id = _MockHandle._counter
        self.kind = kind
        self.data = data  # e.g. bytearray for Buffer
        self._destroyed = False

    def __repr__(self) -> str:
        return f"<MockHandle id={self._id} kind={self.kind}>"


class _MockBackend:
    """
    Pure-Python mock of the Tessera runtime C ABI.
    All operations succeed silently; memory is simulated with bytearray.
    """

    def __init__(self):
        self._initialized = False
        self._devices: List[_MockHandle] = []
        self._last_error = ""

    def init(self) -> TsrStatus:
        if not self._initialized:
            self._devices = [
                _MockHandle("device",
                            DeviceProps(DeviceKind.CPU,
                                        "Tessera CPU Backend (mock)",
                                        2048, 8))
            ]
            self._initialized = True
        return TsrStatus.SUCCESS

    def shutdown(self) -> TsrStatus:
        self._devices.clear()
        self._initialized = False
        return TsrStatus.SUCCESS

    def get_device_count(self) -> int:
        return len(self._devices)

    def get_device(self, index: int) -> _MockHandle:
        if index < 0 or index >= len(self._devices):
            raise TesseraRuntimeError("tsrGetDevice", TsrStatus.NOT_FOUND,
                                       f"no device at index {index}")
        return self._devices[index]

    def get_device_props(self, dev: _MockHandle) -> DeviceProps:
        return dev.data

    def malloc(self, dev: _MockHandle, size: int) -> _MockHandle:
        if size < 0:
            raise TesseraRuntimeError("tsrMalloc", TsrStatus.INVALID_ARGUMENT,
                                       "size must be >= 0")
        return _MockHandle("buffer", bytearray(size))

    def free(self, buf: _MockHandle) -> TsrStatus:
        if buf.kind != "buffer":
            return TsrStatus.INVALID_ARGUMENT
        buf._destroyed = True
        buf.data = None
        return TsrStatus.SUCCESS

    def memset(self, buf: _MockHandle, value: int, size: int) -> TsrStatus:
        if buf.data is None:
            return TsrStatus.INVALID_ARGUMENT
        n = min(size, len(buf.data))
        for i in range(n):
            buf.data[i] = value & 0xFF
        return TsrStatus.SUCCESS

    def memcpy(self, dst: _MockHandle, src: _MockHandle,
               size: int, kind: MemcpyKind) -> TsrStatus:
        if dst.data is None or src.data is None:
            return TsrStatus.INVALID_ARGUMENT
        n = min(size, len(dst.data), len(src.data))
        dst.data[:n] = src.data[:n]
        return TsrStatus.SUCCESS

    def map(self, buf: _MockHandle) -> bytes:
        if buf.data is None:
            return b""
        return bytes(buf.data)

    def unmap(self, buf: _MockHandle) -> TsrStatus:
        return TsrStatus.SUCCESS

    def create_stream(self, dev: _MockHandle) -> _MockHandle:
        return _MockHandle("stream", dev)

    def destroy_stream(self, stream: _MockHandle) -> TsrStatus:
        stream._destroyed = True
        return TsrStatus.SUCCESS

    def stream_sync(self, stream: _MockHandle) -> TsrStatus:
        return TsrStatus.SUCCESS

    def create_event(self, dev: _MockHandle) -> _MockHandle:
        return _MockHandle("event", {"timestamp_ns": 0, "signaled": False})

    def record_event(self, event: _MockHandle, stream: _MockHandle) -> TsrStatus:
        import time
        event.data["timestamp_ns"] = int(time.perf_counter_ns())
        event.data["signaled"] = True
        return TsrStatus.SUCCESS

    def wait_event(self, event: _MockHandle, stream: _MockHandle) -> TsrStatus:
        return TsrStatus.SUCCESS

    def event_sync(self, event: _MockHandle) -> TsrStatus:
        return TsrStatus.SUCCESS

    def destroy_event(self, event: _MockHandle) -> TsrStatus:
        event._destroyed = True
        return TsrStatus.SUCCESS

    def event_get_timestamp(self, event: _MockHandle) -> int:
        return event.data.get("timestamp_ns", 0)

    def get_version(self) -> tuple:
        return (0, 1, 0)

    def get_last_error(self) -> str:
        return self._last_error


# ---------------------------------------------------------------------------
# ctypes-based backend (real shared library)
# ---------------------------------------------------------------------------

def _find_library() -> Optional[str]:
    """Search for libtessera_runtime in common locations."""
    candidates = [
        "libtessera_runtime.dylib",
        "libtessera_runtime.so",
        "tessera_runtime.dll",
    ]
    search_dirs = [
        Path(__file__).parent.parent.parent.parent / "build" / "lib",
        Path(__file__).parent.parent.parent.parent / "build",
        Path("/usr/local/lib"),
        Path("/usr/lib"),
    ]
    # Environment override
    if env := os.environ.get("TESSERA_RUNTIME_LIB"):
        return env

    for d in search_dirs:
        for name in candidates:
            p = d / name
            if p.exists():
                return str(p)

    found = ctypes.util.find_library("tessera_runtime")
    return found


def _elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TesseraRuntime:
    """
    High-level Python wrapper over the Tessera runtime C ABI.

    Automatically falls back to a pure-Python mock when the compiled
    ``libtessera_runtime`` shared library is not found on the system.

    Parameters
    ----------
    lib_path : str, optional
        Explicit path to the shared library.  If not provided, the runtime
        searches standard locations and the ``TESSERA_RUNTIME_LIB`` env var.
    mock : bool
        Force mock mode regardless of library availability.
    """

    def __init__(
        self,
        lib_path: Optional[str] = None,
        *,
        mock: bool = False,
    ) -> None:
        self._mock_mode: bool = mock
        self._lib: Any = None
        self._backend: Any = None
        self._initialized: bool = False
        self._telemetry_events: list[dict[str, Any]] = []

        if not mock:
            path = lib_path or _find_library()
            if path:
                try:
                    self._lib = ctypes.CDLL(path)
                    self._setup_ctypes()
                except OSError:
                    self._lib = None

        if self._lib is None:
            self._mock_mode = True
            self._backend = _MockBackend()

    # ------------------------------------------------------------------
    # ctypes function declarations
    # ------------------------------------------------------------------

    def _setup_ctypes(self) -> None:
        """Declare argtypes / restypes for every ABI entry point."""
        lib = self._lib
        vp = ctypes.c_void_p
        i32 = ctypes.c_int
        u32 = ctypes.c_uint32
        u64 = ctypes.c_uint64
        sz = ctypes.c_size_t

        lib.tsrInit.restype = i32
        lib.tsrShutdown.restype = i32

        lib.tsrGetDeviceCount.argtypes = [ctypes.POINTER(i32)]
        lib.tsrGetDeviceCount.restype = i32

        lib.tsrGetDevice.argtypes = [i32, ctypes.POINTER(vp)]
        lib.tsrGetDevice.restype = i32

        lib.tsrGetDeviceProps.argtypes = [vp, ctypes.c_void_p]
        lib.tsrGetDeviceProps.restype = i32

        lib.tsrMalloc.argtypes = [vp, sz, ctypes.POINTER(vp)]
        lib.tsrMalloc.restype = i32

        lib.tsrFree.argtypes = [vp]
        lib.tsrFree.restype = i32

        lib.tsrMemset.argtypes = [vp, i32, sz]
        lib.tsrMemset.restype = i32

        lib.tsrMemcpy.argtypes = [vp, vp, sz, i32]
        lib.tsrMemcpy.restype = i32

        lib.tsrMap.argtypes = [vp, ctypes.POINTER(vp), ctypes.POINTER(sz)]
        lib.tsrMap.restype = i32

        lib.tsrUnmap.argtypes = [vp]
        lib.tsrUnmap.restype = i32

        lib.tsrCreateStream.argtypes = [vp, ctypes.POINTER(vp)]
        lib.tsrCreateStream.restype = i32

        lib.tsrDestroyStream.argtypes = [vp]
        lib.tsrDestroyStream.restype = i32

        lib.tsrStreamSynchronize.argtypes = [vp]
        lib.tsrStreamSynchronize.restype = i32

        lib.tsrCreateEvent.argtypes = [vp, ctypes.POINTER(vp)]
        lib.tsrCreateEvent.restype = i32

        lib.tsrRecordEvent.argtypes = [vp, vp]
        lib.tsrRecordEvent.restype = i32

        lib.tsrWaitEvent.argtypes = [vp, vp]
        lib.tsrWaitEvent.restype = i32

        lib.tsrEventSynchronize.argtypes = [vp]
        lib.tsrEventSynchronize.restype = i32

        lib.tsrDestroyEvent.argtypes = [vp]
        lib.tsrDestroyEvent.restype = i32

        lib.tsrEventGetTimestamp.argtypes = [vp, ctypes.POINTER(u64)]
        lib.tsrEventGetTimestamp.restype = i32

        lib.tsrGetLastError.restype = ctypes.c_char_p
        lib.tsrClearLastError.restype = None

        lib.tsrGetVersion.argtypes = [ctypes.POINTER(i32),
                                       ctypes.POINTER(i32),
                                       ctypes.POINTER(i32)]
        lib.tsrGetVersion.restype = None

        if hasattr(lib, "tsrNativeGemmF32"):
            lib.tsrNativeGemmF32.argtypes = [
                vp,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                i32,
                i32,
                i32,
            ]
            lib.tsrNativeGemmF32.restype = i32

        if hasattr(lib, "tsrGetWorkerThreadCount"):
            lib.tsrGetWorkerThreadCount.argtypes = [vp, ctypes.POINTER(u32)]
            lib.tsrGetWorkerThreadCount.restype = i32

    def _check(self, fn_name: str, status: int, extra: str = "") -> None:
        """Raise TesseraRuntimeError if status != SUCCESS."""
        s = TsrStatus(status)
        if s != TsrStatus.SUCCESS:
            detail = extra
            if self._lib:
                err = self._lib.tsrGetLastError()
                if err:
                    detail = err.decode() if isinstance(err, bytes) else err
            raise TesseraRuntimeError(fn_name, s, detail)

    def _record_event(
        self,
        name: str,
        *,
        op: str | None = None,
        latency_ms: float | None = None,
        memory_bytes: int | None = None,
        status: str = "ok",
        kernel_id: str | None = None,
        device: str | int | None = None,
        stream: str | int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = make_event(
            name,
            source="runtime",
            op=op or name,
            arch="cpu" if self._mock_mode else "native",
            kernel_id=kernel_id,
            device=device,
            stream=stream,
            latency_ms=latency_ms,
            memory_bytes=memory_bytes,
            status=status,
            metadata={
                "mock": self._mock_mode,
                **dict(metadata or {}),
            },
        )
        self._telemetry_events.append(event)
        return event

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Initialise the runtime and all available backends."""
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            self._backend.init()
        else:
            self._check("tsrInit", self._lib.tsrInit())
        self._initialized = True
        self._record_event("runtime.init", latency_ms=_elapsed_ms(start_ns))

    def shutdown(self) -> None:
        """Release all runtime resources."""
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            self._backend.shutdown()
        else:
            self._check("tsrShutdown", self._lib.tsrShutdown())
        self._initialized = False
        self._record_event("runtime.shutdown", latency_ms=_elapsed_ms(start_ns))

    # ------------------------------------------------------------------
    # Device enumeration
    # ------------------------------------------------------------------

    def get_device_count(self) -> int:
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            count = self._backend.get_device_count()
            self._record_event(
                "runtime.device_count",
                latency_ms=_elapsed_ms(start_ns),
                metadata={"count": count},
            )
            return count
        count = ctypes.c_int(0)
        self._check("tsrGetDeviceCount",
                    self._lib.tsrGetDeviceCount(ctypes.byref(count)))
        self._record_event(
            "runtime.device_count",
            latency_ms=_elapsed_ms(start_ns),
            metadata={"count": count.value},
        )
        return count.value

    def get_device(self, index: int) -> Any:
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            dev = self._backend.get_device(index)
            self._record_event(
                "runtime.get_device",
                latency_ms=_elapsed_ms(start_ns),
                device=index,
            )
            return dev
        handle = ctypes.c_void_p(0)
        self._check("tsrGetDevice",
                    self._lib.tsrGetDevice(index, ctypes.byref(handle)))
        self._record_event(
            "runtime.get_device",
            latency_ms=_elapsed_ms(start_ns),
            device=index,
        )
        return handle

    def get_device_props(self, dev: Any) -> DeviceProps:
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            props = self._backend.get_device_props(dev)
            self._record_event(
                "runtime.device_props",
                latency_ms=_elapsed_ms(start_ns),
                device=getattr(dev, "_id", None),
                metadata={"kind": props.kind.name, "name": props.name},
            )
            return props
        # C struct layout for tsrDeviceProps
        class _Props(ctypes.Structure):
            _fields_ = [
                ("kind",                     ctypes.c_int),
                ("name",                     ctypes.c_char * 128),
                ("logical_tile_threads_max", ctypes.c_uint32),
                ("concurrent_tiles_hint",    ctypes.c_uint32),
            ]
        p = _Props()
        self._check("tsrGetDeviceProps",
                    self._lib.tsrGetDeviceProps(dev, ctypes.byref(p)))
        props = DeviceProps(
            kind=DeviceKind(p.kind),
            name=p.name.decode(),
            logical_tile_threads_max=p.logical_tile_threads_max,
            concurrent_tiles_hint=p.concurrent_tiles_hint,
        )
        self._record_event(
            "runtime.device_props",
            latency_ms=_elapsed_ms(start_ns),
            metadata={"kind": props.kind.name, "name": props.name},
        )
        return props

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def malloc(self, dev: Any, size: int) -> Any:
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            buf = self._backend.malloc(dev, size)
            self._record_event(
                "runtime.malloc",
                latency_ms=_elapsed_ms(start_ns),
                memory_bytes=size,
                device=getattr(dev, "_id", None),
            )
            return buf
        handle = ctypes.c_void_p(0)
        self._check("tsrMalloc",
                    self._lib.tsrMalloc(dev, size, ctypes.byref(handle)))
        self._record_event("runtime.malloc", latency_ms=_elapsed_ms(start_ns), memory_bytes=size)
        return handle

    def free(self, buf: Any) -> None:
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            self._backend.free(buf)
        else:
            self._check("tsrFree", self._lib.tsrFree(buf))
        self._record_event("runtime.free", latency_ms=_elapsed_ms(start_ns))

    def memset(self, buf: Any, value: int, size: int) -> None:
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            self._backend.memset(buf, value, size)
        else:
            self._check("tsrMemset", self._lib.tsrMemset(buf, value, size))
        self._record_event(
            "runtime.memset",
            latency_ms=_elapsed_ms(start_ns),
            memory_bytes=size,
            metadata={"value": value},
        )

    def memcpy(self, dst: Any, src: Any, size: int,
               kind: MemcpyKind = MemcpyKind.DEVICE_TO_DEVICE) -> None:
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            self._backend.memcpy(dst, src, size, kind)
        else:
            self._check("tsrMemcpy",
                        self._lib.tsrMemcpy(dst, src, size, int(kind)))
        self._record_event(
            "runtime.memcpy",
            latency_ms=_elapsed_ms(start_ns),
            memory_bytes=size,
            metadata={"kind": kind.name},
        )

    def map(self, buf: Any) -> bytes:
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            data = self._backend.map(buf)
            self._record_event(
                "runtime.map",
                latency_ms=_elapsed_ms(start_ns),
                memory_bytes=len(data),
            )
            return data
        host_ptr = ctypes.c_void_p(0)
        size = ctypes.c_size_t(0)
        self._check("tsrMap",
                    self._lib.tsrMap(buf, ctypes.byref(host_ptr),
                                     ctypes.byref(size)))
        self._record_event("runtime.map", latency_ms=_elapsed_ms(start_ns), memory_bytes=size.value)
        if not host_ptr.value or not size.value:
            return b""
        return (ctypes.c_char * size.value).from_address(host_ptr.value).raw

    def unmap(self, buf: Any) -> None:
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            self._backend.unmap(buf)
        else:
            self._check("tsrUnmap", self._lib.tsrUnmap(buf))
        self._record_event("runtime.unmap", latency_ms=_elapsed_ms(start_ns))

    # ------------------------------------------------------------------
    # Streams & events
    # ------------------------------------------------------------------

    def create_stream(self, dev: Any) -> Any:
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            stream = self._backend.create_stream(dev)
            self._record_event(
                "runtime.create_stream",
                latency_ms=_elapsed_ms(start_ns),
                device=getattr(dev, "_id", None),
                stream=getattr(stream, "_id", None),
            )
            return stream
        h = ctypes.c_void_p(0)
        self._check("tsrCreateStream",
                    self._lib.tsrCreateStream(dev, ctypes.byref(h)))
        self._record_event("runtime.create_stream", latency_ms=_elapsed_ms(start_ns), stream=h.value)
        return h

    def destroy_stream(self, stream: Any) -> None:
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            self._backend.destroy_stream(stream)
        else:
            self._check("tsrDestroyStream",
                        self._lib.tsrDestroyStream(stream))
        self._record_event(
            "runtime.destroy_stream",
            latency_ms=_elapsed_ms(start_ns),
            stream=getattr(stream, "_id", None),
        )

    def stream_sync(self, stream: Any) -> None:
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            self._backend.stream_sync(stream)
        else:
            self._check("tsrStreamSynchronize",
                        self._lib.tsrStreamSynchronize(stream))
        self._record_event(
            "runtime.stream_sync",
            latency_ms=_elapsed_ms(start_ns),
            stream=getattr(stream, "_id", None),
        )

    def create_event(self, dev: Any) -> Any:
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            event = self._backend.create_event(dev)
            self._record_event(
                "runtime.create_event",
                latency_ms=_elapsed_ms(start_ns),
                device=getattr(dev, "_id", None),
                metadata={"event": getattr(event, "_id", None)},
            )
            return event
        h = ctypes.c_void_p(0)
        self._check("tsrCreateEvent",
                    self._lib.tsrCreateEvent(dev, ctypes.byref(h)))
        self._record_event("runtime.create_event", latency_ms=_elapsed_ms(start_ns), metadata={"event": h.value})
        return h

    def record_event(self, event: Any, stream: Any) -> None:
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            self._backend.record_event(event, stream)
        else:
            self._check("tsrRecordEvent",
                        self._lib.tsrRecordEvent(event, stream))
        self._record_event(
            "runtime.record_event",
            latency_ms=_elapsed_ms(start_ns),
            stream=getattr(stream, "_id", None),
            metadata={
                "event": getattr(event, "_id", None),
                "timestamp_ns": self.event_get_timestamp(event),
            },
        )

    def wait_event(self, event: Any, stream: Any) -> None:
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            self._backend.wait_event(event, stream)
        else:
            self._check("tsrWaitEvent",
                        self._lib.tsrWaitEvent(event, stream))
        self._record_event(
            "runtime.wait_event",
            latency_ms=_elapsed_ms(start_ns),
            stream=getattr(stream, "_id", None),
            metadata={"event": getattr(event, "_id", None)},
        )

    def event_sync(self, event: Any) -> None:
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            self._backend.event_sync(event)
        else:
            self._check("tsrEventSynchronize",
                        self._lib.tsrEventSynchronize(event))
        self._record_event(
            "runtime.event_sync",
            latency_ms=_elapsed_ms(start_ns),
            metadata={"event": getattr(event, "_id", None)},
        )

    def destroy_event(self, event: Any) -> None:
        start_ns = time.perf_counter_ns()
        if self._mock_mode:
            self._backend.destroy_event(event)
        else:
            self._check("tsrDestroyEvent",
                        self._lib.tsrDestroyEvent(event))
        self._record_event(
            "runtime.destroy_event",
            latency_ms=_elapsed_ms(start_ns),
            metadata={"event": getattr(event, "_id", None)},
        )

    def event_get_timestamp(self, event: Any) -> int:
        """Return event timestamp in nanoseconds."""
        if self._mock_mode:
            return self._backend.event_get_timestamp(event)
        ts = ctypes.c_uint64(0)
        self._check("tsrEventGetTimestamp",
                    self._lib.tsrEventGetTimestamp(event, ctypes.byref(ts)))
        return ts.value

    # ------------------------------------------------------------------
    # Version / diagnostics
    # ------------------------------------------------------------------

    def get_version(self) -> tuple:
        if self._mock_mode:
            return self._backend.get_version()
        major = ctypes.c_int(0)
        minor = ctypes.c_int(0)
        patch = ctypes.c_int(0)
        self._lib.tsrGetVersion(ctypes.byref(major),
                                ctypes.byref(minor),
                                ctypes.byref(patch))
        return (major.value, minor.value, patch.value)

    def native_gemm_f32(self, dev: Any, a: Any, b: Any, c: Any, m: int, n: int, k: int) -> None:
        if self._mock_mode:
            raise TesseraRuntimeError("tsrNativeGemmF32", TsrStatus.UNIMPLEMENTED, "native GEMM requires the C runtime")
        if not hasattr(self._lib, "tsrNativeGemmF32"):
            raise TesseraRuntimeError("tsrNativeGemmF32", TsrStatus.UNIMPLEMENTED, "runtime ABI lacks tsrNativeGemmF32")
        self._check(
            "tsrNativeGemmF32",
            self._lib.tsrNativeGemmF32(
                dev,
                a,
                b,
                c,
                ctypes.c_int(m),
                ctypes.c_int(n),
                ctypes.c_int(k),
            ),
        )

    def get_worker_thread_count(self, dev: Any) -> int:
        if self._mock_mode:
            return 0
        if not hasattr(self._lib, "tsrGetWorkerThreadCount"):
            raise TesseraRuntimeError("tsrGetWorkerThreadCount", TsrStatus.UNIMPLEMENTED, "runtime ABI lacks tsrGetWorkerThreadCount")
        count = ctypes.c_uint32(0)
        self._check("tsrGetWorkerThreadCount", self._lib.tsrGetWorkerThreadCount(dev, ctypes.byref(count)))
        return int(count.value)

    def get_last_error(self) -> str:
        if self._mock_mode:
            return self._backend.get_last_error()
        err = self._lib.tsrGetLastError()
        return err.decode() if isinstance(err, bytes) else (err or "")

    def telemetry_events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._telemetry_events]

    def telemetry_report(self) -> dict[str, Any]:
        events = self.telemetry_events()
        return {
            "schema": TELEMETRY_SCHEMA_VERSION,
            "source": "runtime",
            "summary": telemetry_report(events),
            "events": events,
        }

    def clear_telemetry(self) -> None:
        self._telemetry_events.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_mock(self) -> bool:
        """True when operating in pure-Python mock mode."""
        return self._mock_mode

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def __repr__(self) -> str:
        mode = "mock" if self._mock_mode else "native"
        return f"TesseraRuntime(mode={mode!r}, initialized={self._initialized})"

    def __enter__(self) -> "TesseraRuntime":
        self.init()
        return self

    def __exit__(self, *_) -> None:
        if self._initialized:
            self.shutdown()


def _runtime_snapshot() -> tuple[TesseraRuntime | None, list[DeviceProps]]:
    rt = TesseraRuntime()
    try:
        rt.init()
        devices = [rt.get_device_props(rt.get_device(i)) for i in range(rt.get_device_count())]
        return rt, devices
    except Exception:
        return None, []
    finally:
        if rt.is_initialized:
            rt.shutdown()


def available_backends() -> list[str]:
    """Return runtime backends discoverable through the current C ABI/mock."""
    _, devices = _runtime_snapshot()
    names = []
    for props in devices:
        if props.kind == DeviceKind.CPU:
            names.append("cpu")
        elif props.kind == DeviceKind.CUDA:
            names.append("cuda")
        elif props.kind == DeviceKind.HIP:
            names.append("rocm")
    return names or ["cpu"]


def backend_capabilities(target: str = "cpu") -> BackendCapability:
    """Describe current backend support without guessing executable kernels."""
    target = normalize_target(target)
    compiler_cap = get_target_capability(target)
    backends = available_backends()
    runtime_backend = compiler_cap.runtime_backend
    backend_name = "cuda" if runtime_backend == "cuda" else "rocm" if runtime_backend == "hip" else target
    if backend_name not in backends and target not in {"apple_cpu", "apple_gpu"}:
        return BackendCapability(
            name=target,
            available=False,
            executable=False,
            reason=f"{target} backend is not available",
        )
    if target == "cpu":
        return BackendCapability(
            name="cpu",
            available=True,
            executable=True,
            reason="CPU host-tile runtime is available; generated artifact launch remains limited",
            dtypes=("fp32", "f32"),
            features=("host_tile_launch", "memory", "events"),
        )
    status = compiler_runtime_status(target)
    return BackendCapability(
        name=target,
        available=True,
        executable=status == "ready",
        reason=compiler_cap.reason or f"{target} runtime status is {status}",
        dtypes=compiler_cap.supported_dtypes,
        features=compiler_cap.features,
    )


def query_backend(target: str = "cpu") -> dict[str, Any]:
    cap = backend_capabilities(target)
    result: dict[str, Any] = {
        "name": cap.name,
        "available": cap.available,
        "executable": cap.executable,
        "reason": cap.reason,
        "dtypes": list(cap.dtypes),
        "features": list(cap.features),
    }
    # Phase 3: for apple_gpu, attach the *observed* runtime capability snapshot
    # (the canonical source — a real Metal-4 probe). It is always explicit:
    # runtime_available=false + empty capabilities when the runtime can't load,
    # so this never makes a silent "Metal 4 full" claim.
    if target == "apple_gpu":
        from ._apple_gpu_dispatch import apple_gpu_capabilities_snapshot

        result["observed_capabilities"] = apple_gpu_capabilities_snapshot()
    return result


def compile(module_ir: str | RuntimeArtifact, target: str | None = None, options: dict[str, Any] | None = None) -> RuntimeArtifact:
    """Create a lightweight artifact container for the current compiler output.

    This is intentionally a containerization API, not a claim that Target IR can
    execute through the runtime yet.
    """
    if isinstance(module_ir, RuntimeArtifact):
        metadata = dict(module_ir.metadata or {})
        metadata.update({
            "target": target or metadata.get("target", "cpu"),
            "options": options or metadata.get("options", {}) or {},
        })
        return RuntimeArtifact(
            graph_ir=module_ir.graph_ir,
            schedule_ir=module_ir.schedule_ir,
            tile_ir=module_ir.tile_ir,
            target_ir=module_ir.target_ir,
            metadata=metadata,
            abi_signature=module_ir.abi_signature,
        )
    meta: dict[str, Any] = {
        "target": target or "cpu",
        "options": options or {},
        "runtime_status": "artifact_only",
    }
    # Phase 3: pure containerization of an apple_gpu module is an ARTIFACT, not a
    # runtime claim — stamp the metal_artifact descriptor (never metal_runtime /
    # mtl4_runtime) and its required_capabilities. No observed_capabilities: no
    # runtime probe ran here.
    if (target or "") == "apple_gpu":
        from .compiler.apple_target_descriptor import (
            METAL_ARTIFACT,
            apple_target_descriptor,
        )

        desc = apple_target_descriptor(METAL_ARTIFACT)
        meta["target_descriptor"] = desc
        meta["required_capabilities"] = desc["required_capabilities"]
    return RuntimeArtifact(
        graph_ir=str(module_ir),
        metadata=meta,
        abi_signature=f"tessera.runtime.v1.{target or 'cpu'}",
    )


def load_artifact(path_or_bytes: str | bytes | os.PathLike[str] | RuntimeArtifact) -> RuntimeArtifact:
    if isinstance(path_or_bytes, RuntimeArtifact):
        return path_or_bytes
    if isinstance(path_or_bytes, bytes):
        return RuntimeArtifact.from_json(path_or_bytes)
    if isinstance(path_or_bytes, str):
        payload = path_or_bytes.lstrip()
        if payload.startswith(("{", "[")):
            return RuntimeArtifact.from_json(path_or_bytes)
        candidate = Path(path_or_bytes)
        if candidate.exists():
            return RuntimeArtifact.from_json(candidate.read_text(encoding="utf-8"))
        return RuntimeArtifact.from_json(path_or_bytes)
    if isinstance(path_or_bytes, os.PathLike) and Path(path_or_bytes).exists():
        return RuntimeArtifact.from_json(Path(path_or_bytes).read_text(encoding="utf-8"))
    raise TypeError(f"unsupported artifact input: {type(path_or_bytes)!r}")


# G6 — executor functions, keyed by ExecutionRow.executor_id from the matrix.
# launch() consults the matrix to pick the executor; adding a new backend
# executor is now (1) add the function here, (2) add it to KNOWN_EXECUTORS in
# execution_matrix.py, (3) add the matrix row. No new branch in launch().
#
# Executors return either:
#   - the raw output (matrix row's execution_kind is reported as-is), or
#   - a (output, override_execution_kind) tuple — for executors with an internal
#     fallback chain (e.g. native_cpu -> reference_cpu) that need to report a
#     different execution_kind on success than the row's default.
def _execute_cpu_native_or_jit(artifact, args):
    """G6.1 — CPU executor that preserves the native_cpu -> reference_cpu
    fallback chain from the prior inline path. Tries the AMX/native CPU
    artifact first; if it raises, falls back to the JIT numpy reference path
    and reports execution_kind='reference_cpu' on success. Returns
    (output, execution_kind) so launch() can override the matrix row's
    default execution_kind on the fallback path."""
    try:
        return _execute_native_cpu_artifact(artifact, args), "native_cpu"
    except Exception:
        # Re-raise the JIT error if THAT fails too — preserves the existing
        # "invalid_artifact" reporting (the native exc is dropped because the
        # JIT path is the canonical CPU fallback and its error is more
        # informative than the AMX one).
        return _execute_jit_cpu_artifact(artifact, args), "reference_cpu"


def _executor_table():
    # Lazily resolved: these symbols are defined later in this file.
    return {
        "apple_cpu_accelerate": _execute_apple_cpu_accelerate_artifact,
        "apple_gpu_mps":        _execute_apple_gpu_mps_artifact,
        "apple_value_target_ir": _execute_apple_value_target_ir_artifact,
        "apple_gpu_value_target_ir": _execute_apple_value_target_ir_gpu_artifact,
        "native_cpu":           _execute_cpu_native_or_jit,
        "jit_cpu_numpy":        _execute_jit_cpu_artifact,
    }


def _first_failing_gate_for_metadata(metadata: dict, target: str):
    """B.2 — resolve the audit-named "first failing gate" for an unsupported
    (target, op) combination. Returns a ``GateResult`` (with ``.gate`` and
    ``.detail``) or ``None`` if every gate passes / the gate module can't be
    consulted.

    **C.2 — canonical short-circuit.** If the artifact carries the canonical
    answer stamped by :meth:`CompileResult.to_runtime_artifact`
    (``metadata["canonical_first_failing_gate"]``), trust it: the upstream
    canonical_compile already evaluated every gate, so re-running them here
    would duplicate truth and risk drift. Otherwise (legacy artifacts +
    direct ``runtime.compile`` containerization), fall back to evaluating
    the gates at launch time using ``metadata["ops"]`` as the op signal.
    """
    # C.2 — trust the canonical answer if it's already on the artifact.
    if "canonical_first_failing_gate" in metadata:
        gate_name = metadata.get("canonical_first_failing_gate")
        if gate_name is None:
            return None
        try:
            from .compiler.pipeline_gates import GateResult, STATUS_FAIL
        except Exception:
            return None
        return GateResult(
            gate=str(gate_name),
            status=STATUS_FAIL,
            detail=str(metadata.get("canonical_first_failing_gate_detail", "")),
        )
    try:  # noqa: SIM105 — import here so runtime startup stays cheap
        from .compiler.pipeline_gates import first_failing_gate as _ffg
    except Exception:
        return None
    op_name: str | None = None
    ops = metadata.get("ops")
    if isinstance(ops, list) and ops:
        head = ops[0]
        if isinstance(head, dict):
            raw = head.get("op_name") or head.get("name")
            if isinstance(raw, str) and raw:
                op_name = raw.removeprefix("tessera.")
    try:
        return _ffg(target, op_name)
    except Exception:
        return None


def launch(kernel: RuntimeArtifact, args: Any, stream: Any = None) -> dict[str, Any]:
    """Launch executable CPU artifacts or return a structured non-success result."""
    global _last_profile
    start_ns = time.perf_counter_ns()
    artifact = load_artifact(kernel)
    metadata = artifact.metadata or {}
    target = str(metadata.get("target", "cpu"))
    cap = backend_capabilities(target)
    # G6 + G6.1 — matrix-driven dispatch for **every** executable row (Apple
    # CPU/GPU + the CPU native_cpu/jit_cpu_numpy rows). The matrix lookup
    # returns an ExecutionRow naming the executor_id; _executor_table() resolves
    # it to a Python function. Adding a new backend executor is now (function +
    # KNOWN_EXECUTORS + matrix row) — no new branch in launch().
    row = _exec_row_for_metadata(metadata)
    if (row is not None and row.executable and row.executor_id is not None
            and metadata.get("executable") is True
            and row.executor_id in _executor_table()):
        executor = _executor_table()[row.executor_id]
        arch = row.target
        kid = str(metadata.get("kernel_id", row.executor_id))
        try:
            output = executor(artifact, args)
        except Exception as exc:
            _last_profile = RuntimeProfile(launch_overhead_ms=0.0)
            telemetry = make_event(
                "runtime.launch", source="runtime", op="artifact_launch",
                arch=arch, kernel_id=kid, graph_hash=artifact.artifact_hash,
                status="invalid_artifact",
                metadata={"compiler_path": row.compiler_path,
                          "execution_kind": row.execution_kind,
                          "reason": str(exc)},
            )
            return {
                "ok": False, "runtime_status": "invalid_artifact",
                "compiler_path": row.compiler_path,
                "execution_kind": row.execution_kind,
                "artifact_hash": artifact.artifact_hash,
                "reason": str(exc), "telemetry": telemetry,
            }
        # G6.1 — executors with an internal fallback chain (e.g. CPU
        # native_cpu -> reference_cpu) may return (output, override_kind) to
        # report a different execution_kind on the fallback path. Plain output
        # uses the matrix row's default execution_kind.
        exec_kind = row.execution_kind
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], str):
            output, exec_kind = output
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        _last_profile = RuntimeProfile(cpu_wall_ms=elapsed_ms, launch_overhead_ms=elapsed_ms)
        meta_ok = {"compiler_path": row.compiler_path,
                   "execution_kind": exec_kind,
                   "execution_mode": row.execution_mode}
        telemetry = make_event(
            "runtime.launch", source="runtime", op="artifact_launch",
            arch=arch, kernel_id=kid, graph_hash=artifact.artifact_hash,
            latency_ms=elapsed_ms, status="ok", metadata=meta_ok,
        )
        return {
            "ok": True, "runtime_status": "success",
            "compiler_path": row.compiler_path,
            "execution_kind": exec_kind,
            "artifact_hash": artifact.artifact_hash, "output": output,
            "telemetry": telemetry,
            "profile": {"cpu_wall_ms": elapsed_ms, "launch_overhead_ms": elapsed_ms},
        }
    if target != "cpu":
        # G4 — consult the single-source execution matrix instead of hard-coding
        # the "non-CPU = unimplemented" rule. Today the matrix has no executor
        # for non-Apple non-CPU targets (NVIDIA/ROCm), so the
        # behavior is the same; but adding a new backend executor only requires
        # one matrix row + one runtime fn (no second hard-coded branch here).
        # The Apple-CPU and Apple-GPU branches above are themselves entries in
        # the matrix — they handle their compiler_path before reaching here.
        _last_profile = RuntimeProfile(launch_overhead_ms=0.0)
        unim_status = "unimplemented" if cap.available else "missing_backend"
        # B.2 — surface the audit-named first failing gate, not just a generic
        # "unwired" message. Caller gets a structured `first_failing_gate` key
        # for machine-readable use; `reason` leads with the named gate.
        gate = _first_failing_gate_for_metadata(metadata, target)
        if gate is not None:
            unim_reason = (
                f"unsupported: first failing gate `{gate.gate}` — "
                f"{gate.detail}. (see docs/audit/op_target_conformance.md)"
            )
            gate_name: str | None = gate.gate
            gate_detail = gate.detail
        else:
            unim_reason = (f"{target} generated artifact execution is not wired "
                           f"to the runtime ABI yet (see "
                           f"docs/audit/generated/runtime_execution_matrix.md)")
            gate_name = None
            gate_detail = ""
        telemetry = make_event(
            "runtime.launch",
            source="runtime",
            op="artifact_launch",
            arch=target,
            kernel_id=str(metadata.get("kernel_id", "artifact")),
            graph_hash=artifact.artifact_hash,
            status=unim_status,
            metadata={
                "compiler_path": str(metadata.get("compiler_path", "artifact_only")),
                "execution_kind": str(metadata.get("execution_kind", "artifact_only")),
                "reason": unim_reason,
                "first_failing_gate": gate_name,
                "first_failing_gate_detail": gate_detail,
            },
        )
        return {
            "ok": False,
            "runtime_status": unim_status,
            "compiler_path": str(metadata.get("compiler_path", "artifact_only")),
            "execution_kind": str(metadata.get("execution_kind", "artifact_only")),
            "artifact_hash": artifact.artifact_hash,
            "reason": unim_reason,
            "first_failing_gate": gate_name,
            "first_failing_gate_detail": gate_detail,
            "telemetry": telemetry,
        }
    # G6.1 — the inline CPU branches (native_cpu / jit_cpu_numpy) used to live
    # here; they are now routed through the matrix dispatcher above via the
    # `native_cpu` and `jit_cpu_numpy` executor_ids in _executor_table(). The
    # native_cpu -> reference_cpu fallback chain is preserved inside
    # _execute_cpu_native_or_jit, which returns (output, override_kind) so the
    # dispatcher reports execution_kind='reference_cpu' on the fallback path.

    _last_profile = RuntimeProfile(launch_overhead_ms=0.0)
    # B.2 — augment CPU fall-through with the audit-named first failing gate.
    # Most artifacts that reach here are CPU artifacts whose ops aren't in the
    # native_cpu / jit_cpu_numpy capability set; the gate names which axis.
    gate = _first_failing_gate_for_metadata(metadata, "cpu")
    upstream_reason = str(metadata.get(
        "reason", "Generated artifact is not executable by the runtime"))
    if gate is not None:
        reason = (f"unsupported: first failing gate `{gate.gate}` — "
                  f"{gate.detail}. ({upstream_reason})")
        gate_name = gate.gate
        gate_detail = gate.detail
    else:
        reason = upstream_reason
        gate_name = None
        gate_detail = ""
    telemetry = make_event(
        "runtime.launch",
        source="runtime",
        op="artifact_launch",
        arch="cpu",
        kernel_id=str(metadata.get("kernel_id", "artifact")),
        graph_hash=artifact.artifact_hash,
        status="unsupported" if cap.available else "missing_backend",
        metadata={
            "compiler_path": str(metadata.get("compiler_path", "artifact_only")),
            "execution_kind": str(metadata.get("execution_kind", "artifact_only")),
            "reason": reason,
            "first_failing_gate": gate_name,
            "first_failing_gate_detail": gate_detail,
        },
    )
    return {
        "ok": False,
        "runtime_status": "unsupported" if cap.available else "missing_backend",
        "compiler_path": str(metadata.get("compiler_path", "artifact_only")),
        "execution_kind": str(metadata.get("execution_kind", "artifact_only")),
        "artifact_hash": artifact.artifact_hash,
        "reason": reason,
        "first_failing_gate": gate_name,
        "first_failing_gate_detail": gate_detail,
        "telemetry": telemetry,
    }


def get_last_profile() -> RuntimeProfile:
    return _last_profile


def runtime_smoke_telemetry(*, mock: bool = True, bytes_size: int = 64) -> dict[str, Any]:
    """Exercise the CPU runtime spine and return telemetry-compatible JSON."""

    rt = TesseraRuntime(mock=mock)
    rt.init()
    count = rt.get_device_count()
    dev = rt.get_device(0)
    props = rt.get_device_props(dev)
    buf = rt.malloc(dev, bytes_size)
    rt.memset(buf, 0, bytes_size)
    mapped = rt.map(buf)
    rt.unmap(buf)
    stream = rt.create_stream(dev)
    event = rt.create_event(dev)
    rt.record_event(event, stream)
    timestamp_ns = rt.event_get_timestamp(event)
    rt.wait_event(event, stream)
    rt.event_sync(event)
    rt.destroy_event(event)
    rt.stream_sync(stream)
    rt.destroy_stream(stream)
    rt.free(buf)
    rt.shutdown()
    events = rt.telemetry_events()

    return {
        "schema": TELEMETRY_SCHEMA_VERSION,
        "runtime_status": "success",
        "device_count": count,
        "device": {
            "kind": props.kind.name,
            "name": props.name,
            "logical_tile_threads_max": props.logical_tile_threads_max,
            "concurrent_tiles_hint": props.concurrent_tiles_hint,
        },
        "mapped_bytes": len(mapped),
        "event_timestamp_ns": timestamp_ns,
        "telemetry_summary": telemetry_report(events),
        "telemetry_events": events,
    }


def _execute_native_cpu_metadata(metadata: Mapping[str, Any], args: Any) -> Any:
    """Run an eligible CPU f32 rank-2 GEMM through libtessera_runtime."""

    import numpy as np

    arg_names = list(metadata.get("arg_names") or [])
    ops = list(metadata.get("ops") or [])
    output_name = metadata.get("output_name")
    if len(ops) != 1 or str(ops[0].get("op_name")) not in {"tessera.matmul", "tessera.gemm"}:
        raise ValueError("native_cpu currently supports a single matmul/gemm op")
    values = _bind_launch_args(args, arg_names)
    operand_names = [str(name) for name in ops[0].get("operands", [])]
    if len(operand_names) != 2:
        raise ValueError("native_cpu GEMM requires two operands")
    a = np.ascontiguousarray(_as_numpy(values[operand_names[0]]), dtype=np.float32)
    b = np.ascontiguousarray(_as_numpy(values[operand_names[1]]), dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("native_cpu GEMM requires rank-2 operands")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"native_cpu GEMM K mismatch: {a.shape[1]} != {b.shape[0]}")
    out = np.empty((a.shape[0], b.shape[1]), dtype=np.float32)

    rt = TesseraRuntime(mock=False)
    rt.init()
    try:
        dev = rt.get_device(0)
        rt.native_gemm_f32(
            dev,
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(a.shape[0]),
            int(b.shape[1]),
            int(a.shape[1]),
        )
    finally:
        rt.shutdown()
    if output_name:
        values[str(output_name)] = out
    return out


def _execute_native_cpu_artifact(artifact: RuntimeArtifact, args: Any) -> Any:
    return _execute_native_cpu_metadata(artifact.metadata or {}, args)


def _execute_jit_cpu_artifact(artifact: RuntimeArtifact, args: Any) -> Any:
    import numpy as np

    metadata = artifact.metadata or {}
    arg_names = list(metadata.get("arg_names") or [])
    output_name = metadata.get("output_name")
    ops = list(metadata.get("ops") or [])
    if not arg_names or not output_name or not ops:
        raise ValueError("executable CPU artifact is missing arg_names, output_name, or ops")

    values = _bind_launch_args(args, arg_names)
    for op in ops:
        op_name = str(op.get("op_name", ""))
        result = op.get("result")
        operand_names = [str(name) for name in op.get("operands", [])]
        kwargs = dict(op.get("kwargs") or {})
        missing = [name for name in operand_names if name not in values]
        if missing:
            raise ValueError(f"artifact requires operand(s): {', '.join(missing)}")
        operands = [_as_numpy(values[name]) for name in operand_names]
        if not result:
            raise ValueError(f"artifact op {op_name!r} has no result")
        values[str(result)] = _execute_runtime_cpu_op(op_name, operands, kwargs, np)

    if output_name not in values:
        raise ValueError(f"artifact did not produce output {output_name!r}")
    return values[output_name]


# Ops the apple_cpu *artifact* (metadata) path dispatches through Accelerate's
# cblas_sgemm family. _apple_cpu_dispatch_matmul selects rank-2 vs rank-3
# (batched) by operand shape, so tessera.batched_gemm shares the same entry —
# Sprint 6 added it so the default artifact path routes batched matmul through
# the real (batched) GEMM instead of the numpy reference fall-through.
_APPLE_CPU_ACCELERATE_OPS = frozenset(
    {"tessera.matmul", "tessera.gemm", "tessera.batched_gemm"})


def _execute_apple_cpu_accelerate_artifact(artifact: RuntimeArtifact, args: Any) -> Any:
    """Public entry: dispatch an apple_cpu artifact. Delegates to the metadata
    dispatcher so the JIT hot-path can skip the artifact wrapper + hash + JSON
    serialization entirely (see `JitFn._apple_cpu_fast_call`)."""

    return _execute_apple_cpu_accelerate_metadata(artifact.metadata or {}, args)


def _execute_apple_cpu_accelerate_metadata(metadata: Mapping[str, Any], args: Any) -> Any:
    """Run an apple_cpu plan from a metadata dict directly: matmul/gemm via
    Accelerate (cblas_sgemm), every other supported op via the numpy reference
    path used by the default `cpu` target. Multi-op programs are first-class —
    ops execute in the order captured by `metadata["ops"]` and intermediate
    values stay in the same `values` dict the cpu reference path uses.

    Phase 8.2 launch-overhead reduction: `JitFn` calls this directly so per-call
    artifact construction (~0.5 ms for small GEMMs) is bypassed.
    """

    import numpy as np

    arg_names = list(metadata.get("arg_names") or [])
    output_name = metadata.get("output_name")
    ops = list(metadata.get("ops") or [])
    if not ops:
        raise ValueError("apple_cpu_accelerate artifact has no ops")

    values = _bind_launch_args(args, arg_names)
    for op in ops:
        op_name = str(op.get("op_name", ""))
        result = op.get("result")
        operand_names = [str(name) for name in op.get("operands", [])]
        kwargs = dict(op.get("kwargs") or {})
        missing = [name for name in operand_names if name not in values]
        if missing:
            raise ValueError(f"artifact requires operand(s): {', '.join(missing)}")
        if not result:
            raise ValueError(f"artifact op {op_name!r} has no result")

        if op_name in _APPLE_CPU_ACCELERATE_OPS:
            values[str(result)] = _apple_cpu_dispatch_matmul(
                op_name, [_as_numpy(values[name]) for name in operand_names], np
            )
        else:
            # Fall through to the numpy CPU reference path. This keeps the
            # apple_cpu target strictly more capable than the default `cpu`
            # target: same op coverage, plus Accelerate dispatch for matmul.
            operands = [_as_numpy(values[name]) for name in operand_names]
            values[str(result)] = _execute_runtime_cpu_op(op_name, operands, kwargs, np)

    if output_name not in values:
        raise ValueError(f"artifact did not produce output {output_name!r}")
    return values[output_name]


def _apple_cpu_dispatch_matmul(op_name: str, operands: list[Any], np: Any) -> Any:
    """Dispatch a single matmul/gemm op through Accelerate's cblas_sgemm.

    Falls back to numpy.matmul if the input shape/dtype combination is outside
    the Phase 8.2 fast-path (f32, rank-2). The fallback keeps multi-op programs
    that mix shapes/dtypes runnable end-to-end while only the eligible matmuls
    pay the Accelerate dispatch cost.
    """

    if len(operands) != 2:
        raise ValueError(f"{op_name!r} requires exactly two operands")
    a = np.asarray(operands[0])
    b = np.asarray(operands[1])

    rank2_fast_path = (
        a.dtype == np.float32
        and b.dtype == np.float32
        and a.ndim == 2
        and b.ndim == 2
    )

    # Phase 8.2 Item #3: rank-3 batched matmul via Accelerate's cblas_sgemm
    # looped over the batch dimension. Activates when both inputs are
    # 3-D float32 with matching leading batch dimensions and the batched
    # ctypes symbol is available.
    rank3_batched_path = (
        a.dtype == np.float32
        and b.dtype == np.float32
        and a.ndim == 3
        and b.ndim == 3
        and a.shape[0] == b.shape[0]
    )

    # Phase 8.2 Item #4: rank-2 fp16 matmul via BNNS (with cblas fallback
    # internal to the runtime symbol). Apple Silicon runs fp16 natively;
    # falls through to numpy for shapes/dtypes outside the supported envelope.
    fp16_fast_path = (
        a.dtype == np.float16
        and b.dtype == np.float16
        and a.ndim == 2
        and b.ndim == 2
    )

    # Phase 8.2 follow-up: rank-2 bf16 matmul via BNNSDataTypeBFloat16. The
    # ml_dtypes.bfloat16 numpy dtype is a 2-byte type byte-compatible with
    # the C ABI's uint16_t bf16 layout, so we can pass through .view(np.uint16)
    # to ctypes. Activates only when ml_dtypes is installed (soft dep) and
    # both inputs use that dtype.
    bf16_dtype = _bfloat16_dtype()
    bf16_fast_path = (
        bf16_dtype is not None
        and a.dtype == bf16_dtype
        and b.dtype == bf16_dtype
        and a.ndim == 2
        and b.ndim == 2
    )

    if rank2_fast_path:
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"matmul shape mismatch: {a.shape} x {b.shape}")
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a, dtype=np.float32)
        if not b.flags.c_contiguous:
            b = np.ascontiguousarray(b, dtype=np.float32)

        out = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
        gemm = _apple_cpu_gemm_f32()
        gemm(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(a.shape[0]),
            ctypes.c_int32(b.shape[1]),
            ctypes.c_int32(a.shape[1]),
        )
        return out

    if fp16_fast_path:
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"matmul shape mismatch: {a.shape} x {b.shape}")
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a, dtype=np.float16)
        if not b.flags.c_contiguous:
            b = np.ascontiguousarray(b, dtype=np.float16)
        gemm_f16 = _apple_cpu_gemm_f16()
        if gemm_f16 is not None:
            out = np.zeros((a.shape[0], b.shape[1]), dtype=np.float16)
            gemm_f16(
                a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int32(a.shape[0]),
                ctypes.c_int32(b.shape[1]),
                ctypes.c_int32(a.shape[1]),
            )
            return out
        # Older runtime build without the fp16 symbol — convert to f32, do
        # the matmul, convert back. Same numerical contract as the C++ shim's
        # internal fallback path.
        out_f32 = np.matmul(a.astype(np.float32), b.astype(np.float32))
        return out_f32.astype(np.float16)

    if bf16_fast_path:
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"matmul shape mismatch: {a.shape} x {b.shape}")
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a, dtype=bf16_dtype)
        if not b.flags.c_contiguous:
            b = np.ascontiguousarray(b, dtype=bf16_dtype)
        gemm_bf16 = _apple_cpu_gemm_bf16()
        if gemm_bf16 is not None:
            out = np.zeros((a.shape[0], b.shape[1]), dtype=bf16_dtype)
            gemm_bf16(
                a.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                b.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int32(a.shape[0]),
                ctypes.c_int32(b.shape[1]),
                ctypes.c_int32(a.shape[1]),
            )
            return out
        # Older runtime build without the bf16 symbol — same conversion-based
        # fallback as fp16 above. ml_dtypes' bfloat16 is convertible to/from
        # float32 directly via numpy astype.
        out_f32 = np.matmul(a.astype(np.float32), b.astype(np.float32))
        return out_f32.astype(bf16_dtype)

    if rank3_batched_path:
        if a.shape[2] != b.shape[1]:
            raise ValueError(f"batched matmul shape mismatch: {a.shape} x {b.shape}")
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a, dtype=np.float32)
        if not b.flags.c_contiguous:
            b = np.ascontiguousarray(b, dtype=np.float32)

        batch, M, K = a.shape
        _, _, N = b.shape
        out = np.zeros((batch, M, N), dtype=np.float32)
        batched = _apple_cpu_gemm_f32_batched()
        if batched is not None:
            batched(
                a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int32(batch),
                ctypes.c_int32(M),
                ctypes.c_int32(N),
                ctypes.c_int32(K),
                ctypes.c_int32(M * K),
                ctypes.c_int32(K * N),
                ctypes.c_int32(M * N),
            )
        else:
            # Older runtime build without the batched symbol — loop in Python
            # calling the per-batch gemm. Same numerical result, slightly more
            # ctypes-call overhead.
            gemm = _apple_cpu_gemm_f32()
            for i in range(batch):
                gemm(
                    a[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    b[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    out[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_int32(M),
                    ctypes.c_int32(N),
                    ctypes.c_int32(K),
                )
        return out

    # Anything else: fall back to numpy. Phase 8.4 will route bf16/fp16 here
    # through BNNS.
    return np.matmul(a, b)


def _apple_cpu_gemm_f32() -> Any:
    runtime = _load_apple_cpu_runtime()
    gemm = runtime.tessera_apple_cpu_gemm_f32
    gemm.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    gemm.restype = None
    return gemm


# ── Apple Value Target IR — CPU value-call execution (sprint 2 → 3) ─────────
# The narrow, executable runtime path for the value lane: dispatch the C ABI
# `symbol` named in a tessera_apple.cpu.call value op (read from
# metadata["apple_value_calls"]), not a parallel op-name matcher.  Sprint 3
# completes the Apple CPU linalg family (cholesky / tri_solve / cholesky_solve /
# lu / qr / svd), including multi-result ops; GPU/package calls stay gated.

_F32P = ctypes.POINTER(ctypes.c_float)
_I32P = ctypes.POINTER(ctypes.c_int32)


# Resolved+configured ctypes entries, keyed by symbol. Each Apple-CPU value
# symbol has a fixed (argtypes, restype), so the bound function object is reused
# across dispatches instead of re-resolving the dylib export and re-assigning
# argtypes/restype on every call.
_APPLE_CPU_FN_CACHE: dict[str, Any] = {}


def _apple_cpu_linalg_fn(symbol: str, argtypes: list, restype: Any = ctypes.c_int32) -> Any:
    """Resolve an Accelerate C ABI entry by symbol name (cached by symbol).

    `restype` defaults to int32 (the LAPACK `info` convention shared by the
    linalg family); the GEMM symbol returns void, so callers pass restype=None.
    """
    cached = _APPLE_CPU_FN_CACHE.get(symbol)
    if cached is not None:
        return cached
    runtime = _load_apple_cpu_runtime()
    fn = getattr(runtime, symbol, None)
    if fn is None:
        raise ValueError(
            f"apple_cpu runtime lacks {symbol} — rebuild TesseraAppleRuntime "
            f"(the prebuilt dylib predates the linalg lane)")
    fn.argtypes = argtypes
    fn.restype = restype
    _APPLE_CPU_FN_CACHE[symbol] = fn
    return fn


def _apple_cpu_cholesky_f32() -> Any:  # back-compat name (kept for tests)
    return _apple_cpu_linalg_fn("tessera_apple_cpu_cholesky_f32",
                                [_F32P, _F32P, ctypes.c_int32])


def _as_f32_2d(x, np, *, square: bool = False, name: str = "input"):
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got shape {tuple(a.shape)}")
    if square and a.shape[0] != a.shape[1]:
        raise ValueError(f"{name} must be square, got shape {tuple(a.shape)}")
    return a


def _b(call: Mapping[str, Any], key: str, default: bool) -> int:
    """Read a bool linalg attr off the value call (1/0 for the C ABI)."""
    v = call.get(key, default)
    return 1 if bool(v) else 0


def _dispatch_cholesky(inputs, call, np):
    a = _as_f32_2d(inputs[0], np, square=True, name="cholesky A")
    n = int(a.shape[0])
    out = np.zeros((n, n), dtype=np.float32)
    info = _apple_cpu_cholesky_f32()(a.ctypes.data_as(_F32P),
                                     out.ctypes.data_as(_F32P), n)
    if info != 0:
        raise ValueError(f"cholesky failed (info={info}; not positive definite?)")
    return out


def _dispatch_tri_solve(inputs, call, np):
    a = _as_f32_2d(inputs[0], np, square=True, name="tri_solve A")
    b = _as_f32_2d(inputs[1], np, name="tri_solve B")
    n, k = int(a.shape[0]), int(b.shape[1])
    x = np.zeros((n, k), dtype=np.float32)
    fn = _apple_cpu_linalg_fn(
        "tessera_apple_cpu_tri_solve_f32",
        [_F32P, _F32P, _F32P, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_int32, ctypes.c_int32])
    info = fn(a.ctypes.data_as(_F32P), b.ctypes.data_as(_F32P),
              x.ctypes.data_as(_F32P), n, k,
              _b(call, "lower", True), _b(call, "trans", False),
              _b(call, "unit_diag", False))
    if info != 0:
        raise ValueError(f"tri_solve failed (info={info})")
    return x


def _dispatch_cholesky_solve(inputs, call, np):
    a = _as_f32_2d(inputs[0], np, square=True, name="cholesky_solve A")
    b = _as_f32_2d(inputs[1], np, name="cholesky_solve B")
    n, k = int(a.shape[0]), int(b.shape[1])
    x = np.zeros((n, k), dtype=np.float32)
    fn = _apple_cpu_linalg_fn(
        "tessera_apple_cpu_cholesky_solve_f32",
        [_F32P, _F32P, _F32P, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32])
    info = fn(a.ctypes.data_as(_F32P), b.ctypes.data_as(_F32P),
              x.ctypes.data_as(_F32P), n, k, _b(call, "lower", True))
    if info != 0:
        raise ValueError(f"cholesky_solve failed (info={info})")
    return x


def _dispatch_lu(inputs, call, np):
    a = _as_f32_2d(inputs[0], np, square=True, name="lu A")
    n = int(a.shape[0])
    lu = np.zeros((n, n), dtype=np.float32)
    piv = np.zeros((n,), dtype=np.int32)
    fn = _apple_cpu_linalg_fn(
        "tessera_apple_cpu_lu_f32", [_F32P, _F32P, _I32P, ctypes.c_int32])
    info = fn(a.ctypes.data_as(_F32P), lu.ctypes.data_as(_F32P),
              piv.ctypes.data_as(_I32P), n)
    if info < 0:
        raise ValueError(f"lu failed (info={info})")
    return (lu, piv)  # SSA order: (packed LU, pivots)


def _dispatch_qr(inputs, call, np):
    a = _as_f32_2d(inputs[0], np, name="qr A")
    m, n = int(a.shape[0]), int(a.shape[1])
    if m < n:
        raise ValueError(f"qr value-call requires M>=N (reduced), got {m}x{n}")
    q = np.zeros((m, n), dtype=np.float32)
    r = np.zeros((n, n), dtype=np.float32)
    fn = _apple_cpu_linalg_fn(
        "tessera_apple_cpu_qr_f32",
        [_F32P, _F32P, _F32P, ctypes.c_int32, ctypes.c_int32])
    info = fn(a.ctypes.data_as(_F32P), q.ctypes.data_as(_F32P),
              r.ctypes.data_as(_F32P), m, n)
    if info != 0:
        raise ValueError(f"qr failed (info={info})")
    return (q, r)  # SSA order: (Q, R)


def _dispatch_svd(inputs, call, np):
    if call.get("full_matrices"):
        raise ValueError(
            "svd value-call: full_matrices=true is a named follow-on; the CPU "
            "value symbol tessera_apple_cpu_svd_f32 produces the reduced SVD")
    a = _as_f32_2d(inputs[0], np, name="svd A")
    m, n = int(a.shape[0]), int(a.shape[1])
    kc = min(m, n)
    u = np.zeros((m, kc), dtype=np.float32)
    s = np.zeros((kc,), dtype=np.float32)
    v = np.zeros((kc, n), dtype=np.float32)
    fn = _apple_cpu_linalg_fn(
        "tessera_apple_cpu_svd_f32",
        [_F32P, _F32P, _F32P, _F32P, ctypes.c_int32, ctypes.c_int32])
    info = fn(a.ctypes.data_as(_F32P), u.ctypes.data_as(_F32P),
              s.ctypes.data_as(_F32P), v.ctypes.data_as(_F32P), m, n)
    if info != 0:
        raise ValueError(f"svd failed (info={info})")
    return (u, s, v)  # SSA order: (U, S, V)


def _dispatch_matmul(inputs, call, np):
    """Sprint 5: dense rank-2 (M,K)@(K,N)->(M,N) via Accelerate cblas_sgemm
    (`tessera_apple_cpu_gemm_f32`, which returns void).

    dtype: the compiler gates dtype upstream — only the static rank-2 *f32*
    envelope is ever routed here (the Graph IR target-capability verifier
    rejects f16/bf16 matmul on CPU). At this boundary `_as_f32_2d` **coerces**
    inputs to contiguous fp32; it does not reject a non-f32 input. Shape is
    validated: rank-2 (via `_as_f32_2d`) and K-consistency here."""
    a = _as_f32_2d(inputs[0], np, name="matmul A")
    b = _as_f32_2d(inputs[1], np, name="matmul B")
    m, ka = int(a.shape[0]), int(a.shape[1])
    kb, n = int(b.shape[0]), int(b.shape[1])
    if ka != kb:
        raise ValueError(
            f"matmul contraction mismatch: (M,K)={a.shape} @ (K,N)={b.shape}")
    # GEMM writes C = A@B with beta=0, so the output is fully overwritten — no
    # need to pre-zero. np.empty avoids the wasted memset.
    c = np.empty((m, n), dtype=np.float32)
    fn = _apple_cpu_linalg_fn(
        "tessera_apple_cpu_gemm_f32",
        [_F32P, _F32P, _F32P, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32],
        restype=None)
    fn(a.ctypes.data_as(_F32P), b.ctypes.data_as(_F32P),
       c.ctypes.data_as(_F32P), m, n, ka)
    return c


def _as_2d(x, np, dtype, *, name: str):
    a = np.ascontiguousarray(np.asarray(x, dtype=dtype))
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got shape {tuple(a.shape)}")
    return a


def _dispatch_matmul_f16(inputs, call, np):
    """Sprint 7: rank-2 f16 matmul via the Apple BNNS GEMM C ABI
    (`tessera_apple_cpu_gemm_f16`, uint16/half ABI, void). Inputs are coerced to
    contiguous f16; the f16 result is returned as an f16 array (tests compare
    through an fp32 view). Fails with a named error if the symbol is absent —
    never a silent fp32 substitution."""
    _u16 = ctypes.POINTER(ctypes.c_uint16)
    a = _as_2d(inputs[0], np, np.float16, name="matmul(f16) A")
    b = _as_2d(inputs[1], np, np.float16, name="matmul(f16) B")
    m, ka = int(a.shape[0]), int(a.shape[1])
    kb, n = int(b.shape[0]), int(b.shape[1])
    if ka != kb:
        raise ValueError(f"matmul contraction mismatch: (M,K)={a.shape} @ (K,N)={b.shape}")
    fn = _apple_cpu_gemm_f16()
    if fn is None:
        raise ValueError(
            "apple_cpu runtime lacks tessera_apple_cpu_gemm_f16 — rebuild "
            "TesseraAppleRuntime (the prebuilt dylib predates the f16 GEMM symbol)")
    c = np.empty((m, n), dtype=np.float16)
    fn(a.ctypes.data_as(_u16), b.ctypes.data_as(_u16), c.ctypes.data_as(_u16),
       m, n, ka)
    return c


def _dispatch_matmul_bf16(inputs, call, np):
    """Sprint 7: rank-2 bf16 matmul via the Apple BNNS GEMM C ABI
    (`tessera_apple_cpu_gemm_bf16`). The Python boundary uses
    ml_dtypes.bfloat16 (a 2-byte numpy dtype, byte-compatible with the C ABI's
    uint16 bf16 layout). If ml_dtypes is unavailable, fail with a named
    unsupported-dependency error — never silently fall back to fp32."""
    _u16 = ctypes.POINTER(ctypes.c_uint16)
    bf16 = _bfloat16_dtype()
    if bf16 is None:
        raise ValueError(
            "bf16 value matmul requires the optional `ml_dtypes` dependency "
            "(pip install ml_dtypes) — it is unavailable, and falling back to "
            "fp32 would silently change the dtype contract")
    a = _as_2d(inputs[0], np, bf16, name="matmul(bf16) A")
    b = _as_2d(inputs[1], np, bf16, name="matmul(bf16) B")
    m, ka = int(a.shape[0]), int(a.shape[1])
    kb, n = int(b.shape[0]), int(b.shape[1])
    if ka != kb:
        raise ValueError(f"matmul contraction mismatch: (M,K)={a.shape} @ (K,N)={b.shape}")
    fn = _apple_cpu_gemm_bf16()
    if fn is None:
        raise ValueError(
            "apple_cpu runtime lacks tessera_apple_cpu_gemm_bf16 — rebuild "
            "TesseraAppleRuntime (the prebuilt dylib predates the bf16 GEMM symbol)")
    c = np.empty((m, n), dtype=bf16)
    fn(a.view(np.uint16).ctypes.data_as(_u16),
       b.view(np.uint16).ctypes.data_as(_u16),
       c.view(np.uint16).ctypes.data_as(_u16), m, n, ka)
    return c


def _as_f32_3d(x, np, *, name: str = "input"):
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if a.ndim != 3:
        raise ValueError(f"{name} must be a 3-D batched matrix, got shape {tuple(a.shape)}")
    return a


def _dispatch_batched_matmul(inputs, call, np):
    """Sprint 6: dense rank-3 batched matmul C[b]=A[b]@B[b] via the Accelerate
    batched GEMM C ABI (`tessera_apple_cpu_gemm_f32_batched`, void, beta=0).

    Strict envelope (the compiler only routes static rank-3 f32 here, and this
    re-checks at runtime): exactly 2 operands, both rank-3, matching batch and
    K, no broadcasting. dtype is **coerced** to contiguous fp32 at the boundary
    (`_as_f32_3d`) — the dtype gate is upstream. Strides are the tight per-batch
    element counts (M*K / K*N / M*N)."""
    a = _as_f32_3d(inputs[0], np, name="batched_gemm A")
    b = _as_f32_3d(inputs[1], np, name="batched_gemm B")
    batch, m, ka = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
    bb, kb, n = int(b.shape[0]), int(b.shape[1]), int(b.shape[2])
    if batch != bb:
        raise ValueError(
            f"batched_gemm batch mismatch: A batch={batch}, B batch={bb} "
            f"(no broadcasting)")
    if ka != kb:
        raise ValueError(
            f"batched_gemm contraction mismatch: A K={ka}, B K={kb}")
    c = np.empty((batch, m, n), dtype=np.float32)  # ABI writes every slice (beta=0)
    fn = _apple_cpu_linalg_fn(
        "tessera_apple_cpu_gemm_f32_batched",
        [_F32P, _F32P, _F32P, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32],
        restype=None)
    fn(a.ctypes.data_as(_F32P), b.ctypes.data_as(_F32P), c.ctypes.data_as(_F32P),
       batch, m, n, ka,
       m * ka, ka * n, m * n)  # strideA, strideB, strideC (elements per batch)
    return c


# symbol -> (handler, expected operand count). The allowlist is its key set.
_APPLE_VALUE_CPU_DISPATCH: dict[str, tuple] = {
    "tessera_apple_cpu_cholesky_f32":       (_dispatch_cholesky, 1),
    "tessera_apple_cpu_tri_solve_f32":      (_dispatch_tri_solve, 2),
    "tessera_apple_cpu_cholesky_solve_f32": (_dispatch_cholesky_solve, 2),
    "tessera_apple_cpu_lu_f32":             (_dispatch_lu, 1),
    "tessera_apple_cpu_qr_f32":             (_dispatch_qr, 1),
    "tessera_apple_cpu_svd_f32":            (_dispatch_svd, 1),
    # Sprint 5: first non-linalg executable value op — fp32 rank-2 matmul/gemm.
    # Both op_kinds ("matmul"/"gemm") emit this one symbol in the IR.
    "tessera_apple_cpu_gemm_f32":           (_dispatch_matmul, 2),
    # Sprint 6: fp32 rank-3 batched matmul (op_kind "batched_gemm").
    "tessera_apple_cpu_gemm_f32_batched":   (_dispatch_batched_matmul, 2),
    # Sprint 7: rank-2 f16 / bf16 matmul via BNNS (op_kind "matmul").
    "tessera_apple_cpu_gemm_f16":           (_dispatch_matmul_f16, 2),
    "tessera_apple_cpu_gemm_bf16":          (_dispatch_matmul_bf16, 2),
}
_APPLE_VALUE_CPU_SYMBOLS: frozenset[str] = frozenset(_APPLE_VALUE_CPU_DISPATCH)


def _execute_apple_value_target_ir_artifact(artifact: RuntimeArtifact, args: Any) -> Any:
    """Execute an Apple value-target-IR artifact by dispatching the C ABI symbol
    named in metadata["apple_value_calls"].

    Scope: exactly one tessera_apple.cpu.call with status == "executable" and a
    symbol on the CPU allowlist (the full Apple CPU linalg family). Single-result
    ops return one ndarray; multi-result ops return a tuple in SSA result order
    (lu→(LU,pivots), qr→(Q,R), svd→(U,S,V)). Multi-op programs, GPU kernel_call /
    package_call, and off-allowlist symbols raise (so launch() reports
    invalid_artifact rather than silently falling back to a parallel matcher)."""
    import numpy as np

    metadata = artifact.metadata or {}
    calls = list(metadata.get("apple_value_calls") or [])
    if not calls:
        raise ValueError("apple_value_target_ir artifact carries no apple_value_calls")
    if len(calls) != 1:
        raise ValueError(
            f"apple_value_target_ir: only single value-call programs are "
            f"executable so far (got {len(calls)}); multi-op is a named follow-on")
    call = calls[0]
    op = str(call.get("op", ""))
    if op != "tessera_apple.cpu.call":
        raise ValueError(
            f"apple_value_target_ir: only tessera_apple.cpu.call is executable; "
            f"'{op}' (GPU kernel_call / package_call are gated follow-ons)")
    if call.get("status") != "executable":
        raise ValueError(
            f"apple_value_target_ir: value call status is "
            f"{call.get('status')!r}, not 'executable'")
    symbol = str(call.get("symbol", ""))
    entry = _APPLE_VALUE_CPU_DISPATCH.get(symbol)
    if entry is None:
        raise ValueError(
            f"apple_value_target_ir: symbol {symbol!r} is not in the executable "
            f"CPU allowlist {sorted(_APPLE_VALUE_CPU_SYMBOLS)} "
            f"(op_kind={call.get('op_kind')!r})")
    handler, want_operands = entry

    # Bind inputs from arg names if present, else positionally.
    arg_names = list(metadata.get("arg_names") or [])
    if arg_names:
        bound = _bind_launch_args(args, arg_names)
        inputs = [bound[name] for name in arg_names if name in bound]
    elif isinstance(args, (list, tuple)):
        inputs = list(args)
    else:
        inputs = [args]
    # Exact operand count — the value executor contract is one value call with a
    # precise arity. Accepting extra operands would let a malformed/aliased call
    # silently ignore inputs; reject rather than dispatch a partial read.
    if len(inputs) != want_operands:
        raise ValueError(
            f"apple_value_target_ir: {call.get('op_kind')} value-call needs "
            f"exactly {want_operands} input(s), got {len(inputs)}")

    return handler(inputs, call, np)


def _apple_cpu_gemm_f32_batched() -> Any:
    """Phase 8.2 Item #3: batched cblas_sgemm wrapper for rank-3 matmul.

    Falls back to single-GEMM dispatch via ctypes lookup. If the prebuilt
    library predates Phase 8.2 Item #3 the symbol may be missing; in that
    case the caller should use the per-batch loop in Python.
    """

    runtime = _load_apple_cpu_runtime()
    sym = getattr(runtime, "tessera_apple_cpu_gemm_f32_batched", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,  # batch
        ctypes.c_int32,  # M
        ctypes.c_int32,  # N
        ctypes.c_int32,  # K
        ctypes.c_int32,  # strideA
        ctypes.c_int32,  # strideB
        ctypes.c_int32,  # strideC
    ]
    sym.restype = None
    return sym


def _apple_cpu_gemm_f16() -> Any:
    """Phase 8.2 Item #4: fp16 matmul via BNNS (Accelerate) with cblas_sgemm
    fallback. Inputs/outputs use IEEE-754 half encoding (numpy.float16
    layout). Symbol may be absent on older runtime builds."""

    runtime = _load_apple_cpu_runtime()
    sym = getattr(runtime, "tessera_apple_cpu_gemm_f16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_cpu_gemm_bf16() -> Any:
    """Phase 8.2 follow-up: bf16 matmul via BNNS (BNNSDataTypeBFloat16) with
    a bit-shift fp32 conversion fallback inside the runtime symbol. The
    Python boundary uses ml_dtypes.bfloat16 (a 2-byte numpy dtype) which is
    byte-compatible with the C ABI's uint16_t bf16 layout. Symbol may be
    absent on older runtime builds; in that case bf16 inputs fall through to
    the fp32 numpy path."""

    runtime = _load_apple_cpu_runtime()
    sym = getattr(runtime, "tessera_apple_cpu_gemm_bf16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _load_apple_cpu_runtime() -> ctypes.CDLL:
    global _apple_cpu_runtime
    if _apple_cpu_runtime is not None:
        return _apple_cpu_runtime
    candidates = []
    env = os.environ.get("TESSERA_APPLE_CPU_RUNTIME_LIB")
    if env:
        candidates.append(Path(env))
    root = Path(__file__).resolve().parents[2]
    candidates.extend([
        root / "build/src/compiler/codegen/Tessera_Apple_Backend/libTesseraAppleRuntime.dylib",
        root / "build/src/compiler/codegen/Tessera_Apple_Backend/libTesseraAppleRuntime.so",
    ])
    for candidate in candidates:
        if candidate.exists():
            _apple_cpu_runtime = ctypes.CDLL(str(candidate))
            return _apple_cpu_runtime

    built = _build_apple_cpu_runtime_shared(root)
    _apple_cpu_runtime = ctypes.CDLL(str(built))
    return _apple_cpu_runtime


_APPLE_GPU_MPS_OPS = frozenset(
    {"tessera.matmul", "tessera.gemm", "tessera.batched_gemm"}
)
_APPLE_GPU_MSL_OPS = frozenset({
    "tessera.rope",
    "tessera.flash_attn",
    "tessera.softmax",
    "tessera.softmax_safe",
    "tessera.gelu",
})

# 2026-05-29 — MetalPerformanceShadersGraph-backed Tier-1 / long-tail lane.
# op_name -> unary opcode (must match apple_gpu_runtime.mm mpsg_unary_node).
_APPLE_GPU_UNARY_OPCODES = {
    "tessera.relu": 0,
    "tessera.sigmoid": 1,
    "tessera.sigmoid_safe": 1,
    "tessera.tanh": 2,
    "tessera.softplus": 3,
    "tessera.silu": 4,
    "tessera.exp": 6,
    "tessera.log": 7,
    "tessera.sqrt": 8,
    "tessera.rsqrt": 9,
    "tessera.neg": 10,
    "tessera.negative": 10,
    "tessera.abs": 11,
    "tessera.absolute": 11,
    # Batch 1 (2026-06-08) — float-output elementwise math (MPSGraph nodes).
    "tessera.sin": 12, "tessera.cos": 13, "tessera.tan": 14,
    "tessera.asin": 15, "tessera.acos": 16, "tessera.atan": 17,
    "tessera.sinh": 18, "tessera.cosh": 19, "tessera.erf": 20, "tessera.erfc": 21,
    "tessera.expm1": 22, "tessera.log1p": 23, "tessera.reciprocal": 24,
    "tessera.sign": 25, "tessera.floor": 26, "tessera.ceil": 27,
    "tessera.round": 28, "tessera.trunc": 29,
    # Batch 2 (2026-06-08) — unary predicates / logical / bitwise → f32 mask.
    "tessera.isfinite": 30, "tessera.isinf": 31, "tessera.isnan": 32,
    "tessera.logical_not": 33, "tessera.bitwise_not": 34,
}
# Batch 1 (2026-06-08) — binary float math + comparison → f32 mask. op_name ->
# opcode in apple_gpu_runtime.mm mpsg_binary_node. add/sub/mul/div/maximum/minimum
# reuse the existing C nodes (0-5); 7-10 math; 11-16 comparison.
_APPLE_GPU_BINARY_OPCODES = {
    "tessera.add": 0, "tessera.sub": 1, "tessera.mul": 2, "tessera.div": 3,
    "tessera.maximum": 4, "tessera.minimum": 5,
    "tessera.pow": 7, "tessera.atan2": 8, "tessera.mod": 9, "tessera.floor_div": 10,
    "tessera.eq": 11, "tessera.ne": 12, "tessera.lt": 13, "tessera.le": 14,
    "tessera.gt": 15, "tessera.ge": 16,
    # Batch 2 (2026-06-08) — logical (→ f32 mask) + bitwise (int32) binary.
    "tessera.logical_and": 17, "tessera.logical_or": 18, "tessera.logical_xor": 19,
    "tessera.bitwise_and": 20, "tessera.bitwise_or": 21, "tessera.bitwise_xor": 22,
}
# op_name -> rowop kind (0 layer_norm, 1 rmsnorm, 3 log_softmax). Softmax stays
# on its dedicated MSL path for single-op; the MPSGraph softmax symbol is used
# by the f16/bf16 fused-chain completion below.
_APPLE_GPU_ROWOP_KINDS = {
    "tessera.layer_norm": 0,
    "tessera.rmsnorm": 1,
    "tessera.rmsnorm_safe": 1,
    "tessera.log_softmax": 3,
}
# Batch 2 (2026-06-08) — composed on the GPU binary lane (no dedicated kernel):
# clamp/clip = max(min(x,hi),lo); where = c*a + (1-c)*b.
_APPLE_GPU_COMPOSE_OPS = frozenset({"tessera.clamp", "tessera.clip", "tessera.where"})
# Batch 3 (2026-06-08) — regression / CE losses composed from the GPU opcode
# lanes (per-element recipe + reduce). One dispatcher, no dedicated kernels.
_APPLE_GPU_LOSS_COMPOSE_OPS = frozenset({
    "tessera.loss.mse", "tessera.loss.mae", "tessera.loss.huber",
    "tessera.loss.smooth_l1", "tessera.loss.log_cosh", "tessera.loss.vlb",
    "tessera.loss.ddpm_noise_pred", "tessera.loss.binary_cross_entropy",
    "tessera.loss.cross_entropy",
    "tessera.loss.kl_divergence", "tessera.loss.js_divergence",
})
# Group/instance/weight norm composed from the rowop (layer_norm) + reduce lanes.
_APPLE_GPU_NORM_COMPOSE_OPS = frozenset({
    "tessera.group_norm", "tessera.instance_norm", "tessera.weight_norm",
})
# Standard attention family (Sub-sprint A) — thin wrappers over the proven GQA
# flash-attention kernel (multi_head/gqa/mqa/mla_decode/gated_attention).
_APPLE_GPU_ATTN_WRAPPER_OPS = frozenset({
    "tessera.multi_head_attention", "tessera.gqa_attention",
    "tessera.mqa_attention", "tessera.mla_decode", "tessera.gated_attention",
})
# Linear / recurrent attention family (Sub-sprint B) via the quadratic-parallel
# form (φ(Q)φ(K)ᵀ ⊙ causal[⊙decay]) @ V — two GPU bmms + a mask multiply.
_APPLE_GPU_LINEAR_ATTN_OPS = frozenset({
    "tessera.linear_attn", "tessera.linear_attn_state",
    "tessera.lightning_attention", "tessera.power_attn", "tessera.retention",
})
# NSA masked-softmax attention (Sub-sprint C) — compressed-block (plain) +
# sliding-window (structured causal/window mask) via bmm→+mask→softmax→bmm.
_APPLE_GPU_MASKED_ATTN_OPS = frozenset({
    "tessera.attn_compressed_blocks", "tessera.attn_sliding_window",
})
# Delta-rule attention family (Sub-sprint D) — gated_deltanet / kimi_delta /
# modified_delta as the quadratic form with a per-token column-weight mask.
_APPLE_GPU_DELTA_ATTN_OPS = frozenset({
    "tessera.gated_deltanet", "tessera.kimi_delta_attention",
    "tessera.modified_delta_attention",
})
_APPLE_GPU_MPSGRAPH_OPS = (
    frozenset(_APPLE_GPU_UNARY_OPCODES)
    | frozenset(_APPLE_GPU_BINARY_OPCODES)
    | frozenset(_APPLE_GPU_ROWOP_KINDS)
    | _APPLE_GPU_COMPOSE_OPS
    | frozenset({"tessera.silu_mul"})
)
# 2026-05-29 — Tier-2 projections routed through the matmul / bmm lane.
_APPLE_GPU_PROJECTION_OPS = frozenset(
    {"tessera.linear_general", "tessera.qkv_projection"}
)
# 2026-05-29 — Tier-3 reductions / scans via the MPSGraph reduce lane.
# op_name -> (kind, op_code); kinds: "reduce" (scalar per row), "arg" (int
# index), "scan" (cumulative, same shape).
_APPLE_GPU_REDUCE_OPS = {
    "tessera.reduce": ("reduce", 0),   # sum
    "tessera.mean": ("reduce", 1),
    "tessera.amax": ("reduce", 2),
    "tessera.amin": ("reduce", 3),
    "tessera.prod": ("reduce", 4),
    "tessera.var": ("reduce", 5),
    "tessera.std": ("reduce", 6),
    "tessera.argmax": ("arg", 0),
    "tessera.argmin": ("arg", 1),
    "tessera.cumsum": ("scan", 0),
    "tessera.cumprod": ("scan", 1),
    # Batch 2 (2026-06-08) — reduce/scan opcode completions.
    "tessera.logsumexp": ("reduce", 7),
    "tessera.cummax": ("scan", 2),
    "tessera.cummin": ("scan", 3),
    "tessera.max": ("reduce", 2),   # reduce-max (alias of amax over an axis)
    "tessera.min": ("reduce", 3),   # reduce-min (alias of amin)
}
_APPLE_GPU_REDUCTION_OPS = frozenset(_APPLE_GPU_REDUCE_OPS)
# 2026-05-30 — Tier-3 convolutions: conv2d via the MPSGraph convolution2D node
# (NHWC/HWIO); conv3d via im2col + a GPU MPSGraph batched matmul (NDHWC/DHWIO).
_APPLE_GPU_CONV_OPS = frozenset({"tessera.conv2d", "tessera.conv3d"})
# GPU linear-algebra lane (MPSMatrix) — only the registered Graph IR ops:
# tessera.cholesky (1 operand) + tessera.tri_solve (2 operands, `lower` kwarg).
_APPLE_GPU_LINALG_OPS = frozenset({"tessera.cholesky", "tessera.tri_solve"})
# Mamba-2 selective state-space scan — chunked-parallel SSD with its batched
# contractions on the Metal bmm lane (scalar-state A; (D,N) A falls back).
_APPLE_GPU_SSM_OPS = frozenset({"tessera.selective_ssm"})
# Ragged grouped matmul (MoE expert-FFN compute core) — per-group MPS matmul.
_APPLE_GPU_MOE_OPS = frozenset({"tessera.grouped_gemm"})
# LDT candidate-axis ops with dedicated Metal kernels (popcount intrinsic,
# innermost-axis nonzero count).
_APPLE_GPU_LDT_OPS = frozenset({"tessera.popcount", "tessera.count_nonzero", "tessera.loss.z_loss", "tessera.loss.asymmetric_bce", "tessera.loss.load_balance_loss", "tessera.masked_categorical"})
# Geometric-algebra (Clifford Cl(3,0)) flat-coefficient lane — the canonical
# tessera.ops projection of the tessera.ga.* Multivector surface. The dispatcher
# calls the GA lane, which internally routes Cl(3,0) f32 to the cl30 MSL kernels.
_APPLE_GPU_CLIFFORD_OPS = frozenset({
    "tessera.clifford_geometric_product", "tessera.clifford_wedge",
    "tessera.clifford_left_contraction", "tessera.clifford_inner",
    "tessera.clifford_rotor_sandwich",
    "tessera.clifford_reverse", "tessera.clifford_grade_involution",
    "tessera.clifford_conjugate", "tessera.clifford_grade_projection",
    "tessera.clifford_hodge_star",
    "tessera.clifford_ext_deriv", "tessera.clifford_vec_deriv",
    "tessera.clifford_codiff",
    "tessera.clifford_exp", "tessera.clifford_log",
    "tessera.clifford_norm", "tessera.clifford_norm_squared",
})
# Energy-based-model flat-array lane — canonical tessera.ops projection of the
# tensor-clean tessera.ebm.* subset; the dispatcher calls the EBM lane, which
# internally routes f32 inputs to the dedicated EBM MSL kernels.
_APPLE_GPU_EBM_OPS = frozenset({
    "tessera.ebm_energy_quadratic", "tessera.ebm_self_verify",
    "tessera.ebm_refinement", "tessera.ebm_inner_step",
})
# EBM training losses (CD / PCD / score-matching / ISM / DSM) — MPSGraph
# reductions over energy/score tensors. reduction="mean" runs on GPU.
_APPLE_GPU_EBM_LOSS_OPS = frozenset({
    "tessera.loss.contrastive_divergence", "tessera.loss.persistent_cd",
    "tessera.loss.score_matching", "tessera.loss.implicit_score_matching",
    "tessera.loss.denoising_score_matching",
})
_APPLE_GPU_RUNTIME_OPS = (
    _APPLE_GPU_MPS_OPS | _APPLE_GPU_MSL_OPS | _APPLE_GPU_MPSGRAPH_OPS
    | _APPLE_GPU_PROJECTION_OPS | _APPLE_GPU_REDUCTION_OPS | _APPLE_GPU_CONV_OPS
    | _APPLE_GPU_LINALG_OPS | _APPLE_GPU_SSM_OPS | _APPLE_GPU_MOE_OPS
    | _APPLE_GPU_LDT_OPS | _APPLE_GPU_CLIFFORD_OPS | _APPLE_GPU_EBM_OPS
    | _APPLE_GPU_EBM_LOSS_OPS | _APPLE_GPU_LOSS_COMPOSE_OPS
    | _APPLE_GPU_NORM_COMPOSE_OPS | _APPLE_GPU_ATTN_WRAPPER_OPS
    | _APPLE_GPU_LINEAR_ATTN_OPS | _APPLE_GPU_MASKED_ATTN_OPS
    | _APPLE_GPU_DELTA_ATTN_OPS
)


def _execute_apple_gpu_mps_artifact(artifact: RuntimeArtifact, args: Any) -> Any:
    """Public entry: dispatch an apple_gpu MPS artifact. Delegates to the
    metadata dispatcher so the JIT hot-path can skip the artifact wrapper +
    hash + JSON serialization (see `JitFn._apple_gpu_fast_call`)."""

    return _execute_apple_gpu_mps_metadata(artifact.metadata or {}, args)


def _execute_apple_gpu_mps_metadata(metadata: Mapping[str, Any], args: Any) -> Any:
    """Phase 8.3 + 8.4: run an apple_gpu plan from a metadata dict directly.

    Single-op programs dispatch through the per-op envelope (matmul/gemm via
    MPS, rope/flash_attn/softmax/gelu via custom MSL). Phase 8.4.3: a
    recognized 2-op fusion chain (matmul -> softmax) collapses to a single
    fused MSL kernel call, skipping the host-side intermediate.

    `_is_apple_gpu_mps_executable` in driver.py rejects anything outside
    these patterns at compile time, so the per-op fallthrough below is just
    a defensive guard against gating drift.
    """

    import numpy as np

    arg_names = list(metadata.get("arg_names") or [])
    output_name = metadata.get("output_name")
    ops = list(metadata.get("ops") or [])
    if not ops:
        raise ValueError("apple_gpu_mps artifact has no ops")

    values = _bind_launch_args(args, arg_names)

    # Phase 8.4.8 (Stage 3 SwiGLU Performance Plan) — 4-op fusion dispatch.
    # The longest known fusion. Check before any shorter chain so the most-
    # specific match wins. Chain shape: gate matmul → up matmul → silu_mul
    # → down matmul, with both gate/up consuming the same %x.
    if _apple_gpu_metadata_is_swiglu_chain(ops):
        m_gate, m_up, _silu, m_down = ops[0], ops[1], ops[2], ops[3]
        x_name = m_gate.get("operands", [None])[0]
        if x_name is None:
            raise ValueError("swiglu fusion: gate matmul missing operand")
        wg_name = m_gate.get("operands", [None, None])[1]
        wu_name = m_up.get("operands", [None, None])[1]
        wd_name = m_down.get("operands", [None, None])[1]
        if wg_name is None or wu_name is None or wd_name is None:
            raise ValueError("swiglu fusion: missing weight operand name")
        result_name = m_down.get("result")
        if not result_name:
            raise ValueError("swiglu fusion: missing tail matmul result name")
        x_arr = _as_numpy(values[str(x_name).lstrip("%")])
        wg_arr = _as_numpy(values[str(wg_name).lstrip("%")])
        wu_arr = _as_numpy(values[str(wu_name).lstrip("%")])
        wd_arr = _as_numpy(values[str(wd_name).lstrip("%")])
        values[str(result_name)] = _apple_gpu_dispatch_swiglu(
            x_arr, wg_arr, wu_arr, wd_arr, np
        )
        if output_name not in values:
            raise ValueError(f"artifact did not produce output {output_name!r}")
        return values[output_name]

    # Phase 8.4.3 + 8.4.5 — fused chain dispatch. Longest match wins, so
    # check the 3-op pattern (matmul -> softmax -> matmul) before the 2-op
    # pattern (matmul -> softmax).
    if _apple_gpu_metadata_is_matmul_softmax_matmul_chain(ops):
        first, _sm, third = ops[0], ops[1], ops[2]
        a_b = [_as_numpy(values[name]) for name in first.get("operands", [])]
        # The third op's second operand is C (the V tensor).
        third_operands = [str(n) for n in third.get("operands", [])]
        if len(third_operands) < 2:
            raise ValueError("matmul_softmax_matmul fusion: tail matmul missing C operand")
        c_name = third_operands[1]
        c = _as_numpy(values[c_name])
        result_name = third.get("result")
        if not result_name:
            raise ValueError("matmul_softmax_matmul fusion: missing tail matmul result name")
        values[str(result_name)] = _apple_gpu_dispatch_matmul_softmax_matmul(
            a_b + [c], np
        )
        if output_name not in values:
            raise ValueError(f"artifact did not produce output {output_name!r}")
        return values[output_name]

    if _apple_gpu_metadata_is_matmul_softmax_chain(ops):
        first, second = ops[0], ops[1]
        operands = [_as_numpy(values[name]) for name in first.get("operands", [])]
        result_name = second.get("result")
        if not result_name:
            raise ValueError("matmul_softmax fusion: missing softmax result name")
        values[str(result_name)] = _apple_gpu_dispatch_matmul_softmax(
            operands, np
        )
        if output_name not in values:
            raise ValueError(f"artifact did not produce output {output_name!r}")
        return values[output_name]

    # Phase 8.4.7 — matmul -> gelu and matmul -> rmsnorm 2-op fusions.
    if _apple_gpu_metadata_is_matmul_postlude_chain(ops, "tessera.gelu"):
        first, second = ops[0], ops[1]
        operands = [_as_numpy(values[name]) for name in first.get("operands", [])]
        result_name = second.get("result")
        if not result_name:
            raise ValueError("matmul_gelu fusion: missing gelu result name")
        values[str(result_name)] = _apple_gpu_dispatch_matmul_gelu(operands, np)
        if output_name not in values:
            raise ValueError(f"artifact did not produce output {output_name!r}")
        return values[output_name]

    if _apple_gpu_metadata_is_matmul_postlude_chain(
        ops, "tessera.rmsnorm", "tessera.rmsnorm_safe"
    ):
        first, second = ops[0], ops[1]
        operands = [_as_numpy(values[name]) for name in first.get("operands", [])]
        result_name = second.get("result")
        if not result_name:
            raise ValueError("matmul_rmsnorm fusion: missing rmsnorm result name")
        kwargs = dict(second.get("kwargs") or {})
        # Default eps matches the python runtime: 1e-5 for rmsnorm,
        # 1e-6 for rmsnorm_safe.
        eps_default = 1e-6 if str(second.get("op_name")) == "tessera.rmsnorm_safe" else 1e-5
        eps = float(kwargs.get("eps", eps_default))
        values[str(result_name)] = _apple_gpu_dispatch_matmul_rmsnorm(
            operands, eps, np
        )
        if output_name not in values:
            raise ValueError(f"artifact did not produce output {output_name!r}")
        return values[output_name]

    # P6 — linear + bias (+ activation) fused via the MPP matmul2d epilogue.
    # matmul -> add(bias) [-> gelu|relu|silu]. bias is the add's 2nd operand.
    bias_act = _apple_gpu_metadata_matmul_bias_act(ops)
    if bias_act is not None:
        m, addop = ops[0], ops[1]
        ab = [_as_numpy(values[name]) for name in m.get("operands", [])]
        add_operands = [str(n) for n in addop.get("operands", [])]
        if len(add_operands) < 2:
            raise ValueError("matmul_bias fusion: add op missing bias operand")
        bias = _as_numpy(values[add_operands[1].lstrip("%")])
        result_name = ops[-1].get("result")
        if not result_name:
            raise ValueError("matmul_bias fusion: missing result name")
        values[str(result_name)] = _apple_gpu_dispatch_matmul_bias_act(ab, bias, bias_act, np)
        if output_name not in values:
            raise ValueError(f"artifact did not produce output {output_name!r}")
        return values[output_name]

    for op in ops:
        op_name = str(op.get("op_name", ""))
        result = op.get("result")
        operand_names = [str(name) for name in op.get("operands", [])]
        kwargs = dict(op.get("kwargs") or {})
        missing = [name for name in operand_names if name not in values]
        if missing:
            raise ValueError(f"artifact requires operand(s): {', '.join(missing)}")
        if not result:
            raise ValueError(f"artifact op {op_name!r} has no result")

        if op_name in _APPLE_GPU_MPS_OPS:
            values[str(result)] = _apple_gpu_dispatch_matmul(
                op_name, [_as_numpy(values[name]) for name in operand_names], np
            )
        elif op_name == "tessera.rope":
            values[str(result)] = _apple_gpu_dispatch_rope(
                op_name, [_as_numpy(values[name]) for name in operand_names], np
            )
        elif op_name == "tessera.flash_attn":
            values[str(result)] = _apple_gpu_dispatch_flash_attn(
                op_name,
                [_as_numpy(values[name]) for name in operand_names],
                kwargs,
                np,
            )
        elif op_name in {"tessera.softmax", "tessera.softmax_safe"}:
            values[str(result)] = _apple_gpu_dispatch_softmax(
                op_name,
                [_as_numpy(values[name]) for name in operand_names],
                kwargs,
                np,
            )
        elif op_name == "tessera.gelu":
            values[str(result)] = _apple_gpu_dispatch_gelu(
                op_name,
                [_as_numpy(values[name]) for name in operand_names],
                np,
            )
        elif op_name in _APPLE_GPU_UNARY_OPCODES:
            values[str(result)] = _apple_gpu_dispatch_unary(
                op_name,
                [_as_numpy(values[name]) for name in operand_names],
                np,
            )
        elif op_name in _APPLE_GPU_BINARY_OPCODES:
            values[str(result)] = _apple_gpu_dispatch_mpsgraph_binary(
                op_name,
                [_as_numpy(values[name]) for name in operand_names],
                kwargs,
                np,
            )
        elif op_name in ("tessera.clamp", "tessera.clip"):
            values[str(result)] = _apple_gpu_dispatch_clamp(
                op_name,
                [_as_numpy(values[name]) for name in operand_names],
                kwargs,
                np,
            )
        elif op_name == "tessera.where":
            values[str(result)] = _apple_gpu_dispatch_where(
                [_as_numpy(values[name]) for name in operand_names],
                np,
            )
        elif op_name in _APPLE_GPU_LOSS_COMPOSE_OPS:
            values[str(result)] = _apple_gpu_dispatch_loss(
                op_name,
                [_as_numpy(values[name]) for name in operand_names],
                kwargs,
                np,
            )
        elif op_name in _APPLE_GPU_NORM_COMPOSE_OPS:
            values[str(result)] = _apple_gpu_dispatch_norm(
                op_name,
                [_as_numpy(values[name]) for name in operand_names],
                kwargs,
                np,
            )
        elif op_name in _APPLE_GPU_ATTN_WRAPPER_OPS:
            values[str(result)] = _apple_gpu_dispatch_attn_wrapper(
                op_name,
                [_as_numpy(values[name]) for name in operand_names],
                kwargs,
                np,
            )
        elif op_name in _APPLE_GPU_LINEAR_ATTN_OPS:
            values[str(result)] = _apple_gpu_dispatch_linear_attn(
                op_name,
                [_as_numpy(values[name]) for name in operand_names],
                kwargs,
                np,
            )
        elif op_name in _APPLE_GPU_MASKED_ATTN_OPS:
            values[str(result)] = _apple_gpu_dispatch_masked_attn(
                op_name,
                [_as_numpy(values[name]) for name in operand_names],
                kwargs,
                np,
            )
        elif op_name in _APPLE_GPU_DELTA_ATTN_OPS:
            values[str(result)] = _apple_gpu_dispatch_delta_attn(
                op_name,
                [_as_numpy(values[name]) for name in operand_names],
                kwargs,
                np,
            )
        elif op_name in _APPLE_GPU_ROWOP_KINDS:
            values[str(result)] = _apple_gpu_dispatch_rowop(
                op_name,
                [_as_numpy(values[name]) for name in operand_names],
                kwargs,
                np,
            )
        elif op_name == "tessera.silu_mul":
            values[str(result)] = _apple_gpu_dispatch_silu_mul(
                [_as_numpy(values[name]) for name in operand_names],
                np,
            )
        elif op_name == "tessera.linear_general":
            values[str(result)] = _apple_gpu_dispatch_linear_general(
                [_as_numpy(values[name]) for name in operand_names],
                kwargs,
                np,
            )
        elif op_name == "tessera.qkv_projection":
            values[str(result)] = _apple_gpu_dispatch_qkv_projection(
                [_as_numpy(values[name]) for name in operand_names],
                np,
            )
        elif op_name in _APPLE_GPU_REDUCE_OPS:
            values[str(result)] = _apple_gpu_dispatch_reduce(
                op_name,
                [_as_numpy(values[name]) for name in operand_names],
                kwargs,
                np,
            )
        elif op_name == "tessera.conv2d":
            values[str(result)] = _apple_gpu_dispatch_conv2d(
                [_as_numpy(values[name]) for name in operand_names],
                kwargs,
                np,
            )
        elif op_name == "tessera.conv3d":
            values[str(result)] = _apple_gpu_dispatch_conv3d(
                [_as_numpy(values[name]) for name in operand_names],
                kwargs,
                np,
            )
        elif op_name in _APPLE_GPU_LINALG_OPS:
            values[str(result)] = _apple_gpu_dispatch_linalg(
                op_name,
                [_as_numpy(values[name]) for name in operand_names],
                kwargs,
                np,
            )
        elif op_name in _APPLE_GPU_SSM_OPS:
            values[str(result)] = _apple_gpu_dispatch_selective_ssm(
                [_as_numpy(values[name]) for name in operand_names],
                kwargs,
                np,
            )
        elif op_name in _APPLE_GPU_MOE_OPS:
            values[str(result)] = _apple_gpu_dispatch_grouped_gemm(
                [_as_numpy(values[name]) for name in operand_names],
                kwargs,
                np,
            )
        elif op_name == "tessera.popcount":
            values[str(result)] = _apple_gpu_dispatch_popcount(
                [_as_numpy(values[name]) for name in operand_names], kwargs, np)
        elif op_name == "tessera.count_nonzero":
            values[str(result)] = _apple_gpu_dispatch_count_nonzero(
                [_as_numpy(values[name]) for name in operand_names], kwargs, np)
        elif op_name == "tessera.loss.z_loss":
            values[str(result)] = _apple_gpu_dispatch_z_loss(
                [_as_numpy(values[name]) for name in operand_names], kwargs, np)
        elif op_name == "tessera.loss.asymmetric_bce":
            values[str(result)] = _apple_gpu_dispatch_asymmetric_bce(
                [_as_numpy(values[name]) for name in operand_names], kwargs, np)
        elif op_name == "tessera.loss.load_balance_loss":
            values[str(result)] = _apple_gpu_dispatch_load_balance_loss(
                [_as_numpy(values[name]) for name in operand_names], kwargs, np)
        elif op_name == "tessera.masked_categorical":
            values[str(result)] = _apple_gpu_dispatch_masked_categorical(
                [_as_numpy(values[name]) for name in operand_names], kwargs, np)
        elif op_name in _APPLE_GPU_CLIFFORD_OPS:
            values[str(result)] = _apple_gpu_dispatch_clifford(
                op_name,
                [_as_numpy(values[name]) for name in operand_names], kwargs, np)
        elif op_name in _APPLE_GPU_EBM_OPS:
            values[str(result)] = _apple_gpu_dispatch_ebm(
                op_name,
                [_as_numpy(values[name]) for name in operand_names], kwargs, np)
        elif op_name in _APPLE_GPU_EBM_LOSS_OPS:
            values[str(result)] = _apple_gpu_dispatch_ebm_loss(
                op_name,
                [_as_numpy(values[name]) for name in operand_names], kwargs, np)
        else:
            # Phase 8.4.x will broaden further; today single-op gating in
            # driver.py is the authoritative envelope. A non-MPS, non-MSL op
            # here means the gating + dispatcher are out of sync.
            raise ValueError(
                f"apple_gpu runtime path does not support op {op_name!r} "
                f"(envelope: {sorted(_APPLE_GPU_RUNTIME_OPS)})"
            )

    if output_name not in values:
        raise ValueError(f"artifact did not produce output {output_name!r}")
    return values[output_name]


def _apple_gpu_metadata_is_matmul_softmax_matmul_chain(ops: list[dict]) -> bool:
    """Phase 8.4.5 — check whether the metadata ops form a 3-op
    matmul -> softmax -> matmul fusion chain (full attention block).
    Mirrors driver.py's compile-time check so the runtime can detect
    the pattern from artifact metadata alone."""

    if len(ops) != 3:
        return False
    first, second, third = ops[0], ops[1], ops[2]
    if str(first.get("op_name", "")) not in {"tessera.matmul", "tessera.gemm"}:
        return False
    if str(second.get("op_name", "")) not in {"tessera.softmax", "tessera.softmax_safe"}:
        return False
    if str(third.get("op_name", "")) not in {"tessera.matmul", "tessera.gemm"}:
        return False

    def _strip(name: str) -> str:
        return name[1:] if name.startswith("%") else name

    sm_operands = [_strip(str(n)) for n in second.get("operands") or []]
    if len(sm_operands) != 1 or sm_operands[0] != str(first.get("result", "")):
        return False
    third_operands = [_strip(str(n)) for n in third.get("operands") or []]
    if len(third_operands) < 1 or third_operands[0] != str(second.get("result", "")):
        return False
    return True


def _apple_gpu_metadata_is_swiglu_chain(ops: list[dict]) -> bool:
    """Phase 8.4.8 (Stage 3 SwiGLU Performance Plan) — check whether the
    metadata ops form a 4-op SwiGLU MLP-block fusion chain:

        matmul(x, Wg) -> matmul(x, Wu) -> silu_mul(gate, up) -> matmul(_, Wd)

    Mirrors driver.py's compile-time `_apple_gpu_chain_kind == "swiglu"`
    check. Both gate and up matmuls must consume the same `%x` SSA value;
    otherwise the chain isn't a SwiGLU block.
    """

    if len(ops) != 4:
        return False
    m_gate, m_up, sm_op, m_down = ops[0], ops[1], ops[2], ops[3]
    if str(m_gate.get("op_name", "")) not in {"tessera.matmul", "tessera.gemm"}:
        return False
    if str(m_up.get("op_name", "")) not in {"tessera.matmul", "tessera.gemm"}:
        return False
    if str(sm_op.get("op_name", "")) != "tessera.silu_mul":
        return False
    if str(m_down.get("op_name", "")) not in {"tessera.matmul", "tessera.gemm"}:
        return False

    def _strip(name: str) -> str:
        return name[1:] if name.startswith("%") else name

    gate_operands = [_strip(str(n)) for n in m_gate.get("operands") or []]
    up_operands = [_strip(str(n)) for n in m_up.get("operands") or []]
    if len(gate_operands) < 2 or len(up_operands) < 2:
        return False
    if gate_operands[0] != up_operands[0]:
        return False  # gate and up must share %x.

    sm_operands = [_strip(str(n)) for n in sm_op.get("operands") or []]
    if len(sm_operands) != 2:
        return False
    if sm_operands[0] != str(m_gate.get("result", "")):
        return False
    if sm_operands[1] != str(m_up.get("result", "")):
        return False

    down_operands = [_strip(str(n)) for n in m_down.get("operands") or []]
    if len(down_operands) < 1:
        return False
    if down_operands[0] != str(sm_op.get("result", "")):
        return False
    return True


def _apple_gpu_metadata_is_matmul_postlude_chain(
    ops: list[dict], *postlude_op_names: str
) -> bool:
    """Phase 8.4.7 — generic matmul -> postlude 2-op chain detector.
    Used by the matmul_gelu and matmul_rmsnorm dispatchers; the second
    op's name set is configurable so the same shape works for both.
    """

    if len(ops) != 2:
        return False
    first, second = ops[0], ops[1]
    if str(first.get("op_name", "")) not in {"tessera.matmul", "tessera.gemm"}:
        return False
    if str(second.get("op_name", "")) not in set(postlude_op_names):
        return False
    operands = [str(n) for n in second.get("operands") or []]
    if len(operands) < 1:
        return False
    op0 = operands[0]
    if op0.startswith("%"):
        op0 = op0[1:]
    return op0 == str(first.get("result", ""))


def _apple_gpu_metadata_matmul_bias_act(ops: list[dict]):
    """P6 — recognize a linear+bias(+activation) chain from metadata:
    matmul/gemm -> add(bias) [-> gelu|relu|silu]. Returns the activation name
    ("none"/"relu"/"gelu"/"silu") if matched, else None. Mirrors driver.py's
    compile-time `_apple_gpu_chain_kind` matmul_bias* cases."""

    def _op0(op) -> str:
        operands = [str(n) for n in op.get("operands") or []]
        return operands[0].lstrip("%") if operands else ""

    if len(ops) not in (2, 3):
        return None
    m, addop = ops[0], ops[1]
    if str(m.get("op_name", "")) not in {"tessera.matmul", "tessera.gemm"}:
        return None
    if str(addop.get("op_name", "")) != "tessera.add":
        return None
    if _op0(addop) != str(m.get("result", "")):
        return None
    if len(ops) == 2:
        return "none"
    act = ops[2]
    act_map = {"tessera.gelu": "gelu", "tessera.relu": "relu", "tessera.silu": "silu"}
    act_name = act_map.get(str(act.get("op_name", "")))
    if act_name is None or _op0(act) != str(addop.get("result", "")):
        return None
    return act_name


def _apple_gpu_dispatch_matmul_bias_act(operands: list[Any], bias: Any,
                                        act: str, np: Any) -> Any:
    """P6 — fused ``act(A @ B + bias)`` via the MPP matmul2d epilogue when A/B are
    f16/bf16 and bias is a genuine per-output-column [N] vector; otherwise a
    correct numpy fallback (any bias shape / dtype / no Metal 4). The output is
    cast back to the input dtype so the fused result drops in for the per-op
    chain (with strictly better fp32-accumulated numerics)."""
    a = np.asarray(operands[0])
    b = np.asarray(operands[1])
    bias = np.asarray(bias)
    in_dtype = a.dtype
    bf16 = _bfloat16_dtype()
    if in_dtype == np.float16:
        dt = "f16"
    elif bf16 is not None and in_dtype == bf16:
        dt = "bf16"
    else:
        dt = None
    is_col_bias = bias.ndim == 1 and a.ndim == 2 and b.ndim == 2 and bias.shape[0] == b.shape[1]
    if dt is not None and is_col_bias and a.shape[1] == b.shape[0]:
        C, ran = apple_gpu_mtl4_matmul2d_epilogue(a, b, np, bias=bias, act=act, dtype=dt)
        if ran:
            return C.astype(in_dtype)
    # Fallback for f32 / non-column-bias / no Metal 4: keep the matmul (the O(MNK)
    # part) on the GPU via the per-dtype MPS path, then bias + act on the host
    # (broadcast add handles 1-D or 2-D bias). Correct for any dtype/shape, and a
    # win over the all-numpy eager path multi-op chains otherwise hit.
    y = np.asarray(_apple_gpu_dispatch_matmul("tessera.matmul", [a, b], np))
    y = y.astype(np.float32) + bias.astype(np.float32)
    y = _mtl4_epilogue_act_numpy(y, _MTL4_EPILOGUE_ACT[act], np)
    return y.astype(in_dtype)


def _apple_gpu_metadata_is_matmul_softmax_chain(ops: list[dict]) -> bool:
    """Phase 8.4.3 — check whether the metadata ops form a 2-op
    matmul -> softmax fusion chain. Mirrors driver.py's compile-time check
    so the runtime can detect the pattern from artifact metadata alone."""

    if len(ops) != 2:
        return False
    first, second = ops[0], ops[1]
    if str(first.get("op_name", "")) not in {"tessera.matmul", "tessera.gemm"}:
        return False
    if str(second.get("op_name", "")) not in {"tessera.softmax", "tessera.softmax_safe"}:
        return False
    sm_operands = [str(n) for n in second.get("operands") or []]
    if len(sm_operands) != 1:
        return False
    op0 = sm_operands[0]
    if op0.startswith("%"):
        op0 = op0[1:]
    return op0 == str(first.get("result", ""))


def _apple_gpu_dispatch_matmul(op_name: str, operands: list[Any], np: Any) -> Any:
    """Phase 8.3 + 8.4.4: dispatch a single rank-2 matmul through the
    apple_gpu runtime shim. Picks the runtime symbol by element type:
      - f32: native MPSDataTypeFloat32 (Phase 8.3)
      - f16: native MPSDataTypeFloat16 (Phase 8.4.4)
      - bf16: fp32-conversion path inside the shim (Phase 8.4.4)
    Other dtypes fall back to numpy.matmul.
    """

    if len(operands) != 2:
        raise ValueError(f"{op_name!r} requires exactly two operands")
    a = np.asarray(operands[0])
    b = np.asarray(operands[1])

    # Batched / rank-3+ matmul → the MPSGraph bmm lane (Tier-2 keystone). Covers
    # batched_gemm and rank-3 matmul, incl. a shared (broadcast) B operand.
    if a.ndim >= 3 or b.ndim >= 3:
        res = _apple_gpu_dispatch_bmm(a, b, np)
        return res if res is not None else np.matmul(a, b)

    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        return np.matmul(a, b)
    if a.dtype != b.dtype:
        return np.matmul(a, b)

    if a.dtype == np.float32:
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a, dtype=np.float32)
        if not b.flags.c_contiguous:
            b = np.ascontiguousarray(b, dtype=np.float32)
        # M4: capability-gated routing onto the Metal 4 lane (opt-in); falls
        # through to the default MPS path when disabled / out of envelope.
        routed = _mtl4_route_matmul_f32(a, b, np)
        if routed is not None:
            return routed
        out = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
        gemm = _apple_gpu_mps_matmul_f32()
        gemm(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(a.shape[0]),
            ctypes.c_int32(b.shape[1]),
            ctypes.c_int32(a.shape[1]),
        )
        return out

    if a.dtype == np.float16:
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a, dtype=np.float16)
        if not b.flags.c_contiguous:
            b = np.ascontiguousarray(b, dtype=np.float16)
        # P7 (2026-06-01): route the fp16 GEMV decode step (M==1) to the
        # native MPP matmul2d tensor-op — measured 3.2-3.4x faster than
        # MPS, which has a slow fp16 M==1 path. Strictly size-gated: at
        # M>=2 / square sizes MPS wins, so only M==1 flips by default.
        # See docs/apple_gpu_metal4_adoption.md (P7) + benchmarks/apple_gpu/
        # benchmark_mtl4_matmul_routing.py for the measurement.
        routed = _mtl4_route_matmul2d_f16(a, b, np)
        if routed is not None:
            return routed
        out = np.zeros((a.shape[0], b.shape[1]), dtype=np.float16)
        gemm_f16 = _apple_gpu_mps_matmul_f16()
        if gemm_f16 is not None:
            gemm_f16(
                a.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                b.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int32(a.shape[0]),
                ctypes.c_int32(b.shape[1]),
                ctypes.c_int32(a.shape[1]),
            )
            return out
        # Older runtime build without the f16 symbol — convert to f32 and back.
        return (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)

    bf16_dtype = _bfloat16_dtype()
    if bf16_dtype is not None and a.dtype == bf16_dtype:
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a, dtype=bf16_dtype)
        if not b.flags.c_contiguous:
            b = np.ascontiguousarray(b, dtype=bf16_dtype)
        # P5: default to the native MPP matmul2d tensor-op (~10x the fp32-conversion
        # MPS fallback). Falls through to the legacy path off Metal 4 / when disabled.
        routed = _mtl4_route_matmul2d_bf16(a, b, np)
        if routed is not None:
            return routed
        out = np.zeros((a.shape[0], b.shape[1]), dtype=bf16_dtype)
        gemm_bf16 = _apple_gpu_mps_matmul_bf16()
        if gemm_bf16 is not None:
            gemm_bf16(
                a.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                b.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int32(a.shape[0]),
                ctypes.c_int32(b.shape[1]),
                ctypes.c_int32(a.shape[1]),
            )
            return out
        return (a.astype(np.float32) @ b.astype(np.float32)).astype(bf16_dtype)

    return np.matmul(a, b)


def _apple_gpu_bmm_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_bmm_f32", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_bmm_f16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_bmm_f16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_bmm_bf16() -> Any:
    """Sprint 8: bf16 batched matmul GPU symbol. Honest bf16 ABI (uint16
    pointers); the runtime shim upcasts to f32 internally. May be absent on an
    older runtime build."""
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_bmm_bf16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _running_on_darwin() -> bool:
    # Wrapped in a function so mypy does not statically narrow `sys.platform`
    # on a Linux checker and flag the Darwin-only body below as unreachable.
    return sys.platform == "darwin"


def _apple_gpu_native_sparse_attn_f32() -> Any:
    """Sprint 11: bind the real Apple GPU Native Sparse Attention symbol.

    The non-Darwin stub exports the same symbol but zero-fills, so value
    execution requires Darwin + an active shared Apple GPU runtime, not symbol
    presence alone.
    """
    if not _running_on_darwin():
        return None
    try:
        from ._apple_gpu_dispatch import apple_gpu_skip_reason
        if apple_gpu_skip_reason() is not None:
            return None
    except Exception:
        return None
    runtime = _load_apple_gpu_runtime()
    cache_size = getattr(runtime, "tessera_apple_gpu_runtime_msl_cache_size", None)
    if cache_size is None:
        return None
    cache_size.argtypes = []
    cache_size.restype = ctypes.c_int32
    if int(cache_size()) < 0:
        return None
    sym = getattr(runtime, "tessera_apple_gpu_native_sparse_attn_f32", None)
    if sym is None:
        return None
    fp = ctypes.POINTER(ctypes.c_float)
    sym.argtypes = [fp, fp, fp, fp, fp] + [ctypes.c_int32] * 8
    sym.restype = None
    return sym


def _apple_gpu_ppo_policy_loss_f32() -> Any:
    """Stage 13: bind the Apple GPU PPO policy-loss value symbol.

    The non-Darwin stub intentionally returns 0 from the C ABI. The resolver
    returns the symbol when present; dispatch checks the return code so stubs or
    unavailable MPSGraph paths never become successful GPU execution.
    """
    if not _running_on_darwin():
        return None
    try:
        from ._apple_gpu_dispatch import apple_gpu_skip_reason
        if apple_gpu_skip_reason() is not None:
            return None
    except Exception:
        return None
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_ppo_policy_loss_f32", None)
    if sym is None:
        return None
    fp = ctypes.POINTER(ctypes.c_float)
    sym.argtypes = [fp, fp, fp, fp, ctypes.c_int32, ctypes.c_float]
    sym.restype = ctypes.c_int32
    return sym


def _apple_gpu_ppo_policy_loss_ex_f32() -> Any:
    """Stage 14: bind the Apple GPU PPO symbol with optional side tensors.

    This is a superset ABI for masked PPO, reference-KL PPO, and entropy PPO.
    The strict Stage 13 symbol stays available for the 3-operand compiler
    envelope; this symbol is used by benchmark/runtime artifacts that explicitly
    carry the optional tensor flags.
    """
    if not _running_on_darwin():
        return None
    try:
        from ._apple_gpu_dispatch import apple_gpu_skip_reason
        if apple_gpu_skip_reason() is not None:
            return None
    except Exception:
        return None
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_ppo_policy_loss_ex_f32", None)
    if sym is None:
        return None
    fp = ctypes.POINTER(ctypes.c_float)
    sym.argtypes = [
        fp, fp, fp, fp, fp, fp, fp,
        ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    sym.restype = ctypes.c_int32
    return sym


def _apple_gpu_ebm_energy_quadratic_value_f32() -> Any:
    """Bind the Apple GPU EBM quadratic-energy value symbol.

    This status-returning symbol is separate from the legacy void EBM ABI:
    it returns 1 only when the Metal dispatch path ran, and 0 for
    stub/unavailable paths. That prevents CPU-reference fallback from being
    labeled Apple GPU value execution.
    """
    if not _running_on_darwin():
        return None
    try:
        from ._apple_gpu_dispatch import apple_gpu_skip_reason
        if apple_gpu_skip_reason() is not None:
            return None
    except Exception:
        return None
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_ebm_energy_quadratic_value_f32",
                  None)
    if sym is None:
        return None
    fp = ctypes.POINTER(ctypes.c_float)
    sym.argtypes = [fp, fp, fp, ctypes.c_int32, ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def _apple_gpu_ebm_langevin_step_value_f32() -> Any:
    """Bind the Apple GPU EBM Langevin-step value symbol."""
    if not _running_on_darwin():
        return None
    try:
        from ._apple_gpu_dispatch import apple_gpu_skip_reason
        if apple_gpu_skip_reason() is not None:
            return None
    except Exception:
        return None
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_ebm_langevin_step_value_f32",
                  None)
    if sym is None:
        return None
    fp = ctypes.POINTER(ctypes.c_float)
    sym.argtypes = [
        fp, fp, fp, ctypes.c_float, ctypes.c_float, fp, ctypes.c_int32,
    ]
    sym.restype = ctypes.c_int32
    return sym


def _apple_gpu_ebm_refinement_value_f32() -> Any:
    """Bind the Apple GPU deterministic EBM refinement value symbol."""
    if not _running_on_darwin():
        return None
    try:
        from ._apple_gpu_dispatch import apple_gpu_skip_reason
        if apple_gpu_skip_reason() is not None:
            return None
    except Exception:
        return None
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_ebm_refinement_value_f32",
                  None)
    if sym is None:
        return None
    fp = ctypes.POINTER(ctypes.c_float)
    sym.argtypes = [fp, fp, ctypes.c_float, ctypes.c_int32, fp, ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def _apple_gpu_ebm_partition_exact_value_f32() -> Any:
    """Bind the Apple GPU scalar EBM partition value symbol."""
    if not _running_on_darwin():
        return None
    try:
        from ._apple_gpu_dispatch import apple_gpu_skip_reason
        if apple_gpu_skip_reason() is not None:
            return None
    except Exception:
        return None
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_ebm_partition_exact_value_f32",
                  None)
    if sym is None:
        return None
    fp = ctypes.POINTER(ctypes.c_float)
    sym.argtypes = [fp, ctypes.c_int32, ctypes.c_float, fp]
    sym.restype = ctypes.c_int32
    return sym


def _apple_gpu_clifford_geo_product_cl30_value_f32() -> Any:
    """Bind the Apple GPU cl30 Clifford geometric-product value symbol."""
    if not _running_on_darwin():
        return None
    try:
        from ._apple_gpu_dispatch import apple_gpu_skip_reason
        if apple_gpu_skip_reason() is not None:
            return None
    except Exception:
        return None
    runtime = _load_apple_gpu_runtime()
    sym = getattr(
        runtime, "tessera_apple_gpu_clifford_geo_product_cl30_value_f32", None)
    if sym is None:
        return None
    fp = ctypes.POINTER(ctypes.c_float)
    sym.argtypes = [fp, fp, fp, ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


_APPLE_GPU_PPO_POLICY_LOSS_AVAILABLE: bool | None = None
_APPLE_GPU_PPO_POLICY_LOSS_EX_AVAILABLE: bool | None = None
_APPLE_GPU_EBM_ENERGY_QUADRATIC_AVAILABLE: bool | None = None
_APPLE_GPU_EBM_LANGEVIN_STEP_AVAILABLE: bool | None = None
_APPLE_GPU_EBM_REFINEMENT_AVAILABLE: bool | None = None
_APPLE_GPU_EBM_PARTITION_EXACT_AVAILABLE: bool | None = None
_APPLE_GPU_CLIFFORD_GEO_PRODUCT_AVAILABLE: bool | None = None


def _ppo_policy_loss_np(
    np, logp_new, logp_old, advantages, *, clip_epsilon: float = 0.2,
    mask=None, ref_logp=None, kl_coef: float = 0.0, entropy=None,
    entropy_coef: float = 0.0,
) -> float:
    ratio = np.exp(logp_new - logp_old)
    clipped = np.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    loss = -np.minimum(ratio * advantages, clipped * advantages)
    if ref_logp is not None and kl_coef != 0.0:
        delta = ref_logp - logp_new
        loss = loss + kl_coef * (np.exp(delta) - delta - 1.0)
    if entropy is not None and entropy_coef != 0.0:
        loss = loss - entropy_coef * entropy
    if mask is not None:
        masked = loss * mask
        denom = max(float(np.sum(mask)), 1.0)
        return float(np.sum(masked) / denom)
    return float(np.mean(loss))


def _apple_gpu_ppo_policy_loss_available() -> bool:
    """True iff the PPO value C ABI runs a tiny MPSGraph numerical probe."""
    global _APPLE_GPU_PPO_POLICY_LOSS_AVAILABLE
    if _APPLE_GPU_PPO_POLICY_LOSS_AVAILABLE is not None:
        return _APPLE_GPU_PPO_POLICY_LOSS_AVAILABLE
    sym = _apple_gpu_ppo_policy_loss_f32()
    if sym is None:
        _APPLE_GPU_PPO_POLICY_LOSS_AVAILABLE = False
        return False
    try:
        import numpy as _np
        a = _np.asarray([0.1], dtype=_np.float32)
        b = _np.asarray([0.0], dtype=_np.float32)
        adv = _np.asarray([1.0], dtype=_np.float32)
        out = _np.empty((), dtype=_np.float32)
        fp = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = int(sym(fp(a), fp(b), fp(adv), fp(out),
                     ctypes.c_int32(1), ctypes.c_float(0.2)))
        expected = -min(float(_np.exp(0.1)), 1.2)
        _APPLE_GPU_PPO_POLICY_LOSS_AVAILABLE = (
            rc == 1 and bool(_np.isfinite(out)) and
            abs(float(out) - expected) < 1.0e-5)
    except Exception:
        _APPLE_GPU_PPO_POLICY_LOSS_AVAILABLE = False
    return _APPLE_GPU_PPO_POLICY_LOSS_AVAILABLE


def _apple_gpu_ppo_policy_loss_ex_available() -> bool:
    """True iff the extended PPO ABI passes strict/mask/KL/entropy probes."""
    global _APPLE_GPU_PPO_POLICY_LOSS_EX_AVAILABLE
    if _APPLE_GPU_PPO_POLICY_LOSS_EX_AVAILABLE is not None:
        return _APPLE_GPU_PPO_POLICY_LOSS_EX_AVAILABLE
    sym = _apple_gpu_ppo_policy_loss_ex_f32()
    if sym is None:
        _APPLE_GPU_PPO_POLICY_LOSS_EX_AVAILABLE = False
        return False
    try:
        import numpy as _np
        logp_old = _np.asarray([-0.2, -0.1, -0.4], dtype=_np.float32)
        logp_new = _np.asarray([-0.15, -0.18, -0.35], dtype=_np.float32)
        adv = _np.asarray([1.2, -0.7, 0.3], dtype=_np.float32)
        mask = _np.asarray([1.0, 0.0, 1.0], dtype=_np.float32)
        ref = _np.asarray([-0.21, -0.11, -0.42], dtype=_np.float32)
        ent = _np.asarray([0.4, 0.3, 0.2], dtype=_np.float32)
        out = _np.empty((), dtype=_np.float32)
        fp = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        null = ctypes.POINTER(ctypes.c_float)()
        probes: list[
            tuple[Any, Any, Any, float, float, int, int, int, Mapping[str, Any]]
        ] = [
            (null, null, null, 0.0, 0.0, 0, 0, 0, {}),
            (fp(mask), null, null, 0.0, 0.0, 1, 0, 0, {"mask": mask}),
            (null, fp(ref), null, 0.03, 0.0, 0, 1, 0,
             {"ref_logp": ref, "kl_coef": 0.03}),
            (fp(mask), fp(ref), fp(ent), 0.02, 0.01, 1, 1, 1,
             {"mask": mask, "ref_logp": ref, "kl_coef": 0.02,
              "entropy": ent, "entropy_coef": 0.01}),
        ]
        ok = True
        for mask_p, ref_p, ent_p, kl, ent_coef, has_mask, has_ref, has_ent, kw in probes:
            rc = int(sym(fp(logp_new), fp(logp_old), fp(adv), mask_p, ref_p,
                         ent_p, fp(out), ctypes.c_int32(int(logp_new.size)),
                         ctypes.c_float(0.2), ctypes.c_float(kl),
                         ctypes.c_float(ent_coef), ctypes.c_int32(has_mask),
                         ctypes.c_int32(has_ref), ctypes.c_int32(has_ent)))
            expected = _ppo_policy_loss_np(
                _np, logp_new, logp_old, adv, clip_epsilon=0.2, **kw)
            ok = ok and rc == 1 and bool(_np.isfinite(out)) and (
                abs(float(out) - expected) < 2.0e-5)
        _APPLE_GPU_PPO_POLICY_LOSS_EX_AVAILABLE = ok
    except Exception:
        _APPLE_GPU_PPO_POLICY_LOSS_EX_AVAILABLE = False
    return _APPLE_GPU_PPO_POLICY_LOSS_EX_AVAILABLE


_APPLE_VALUE_COMPILE_PIPELINE_OK: bool | None = None


def _apple_value_compile_pipeline_available() -> bool:
    """True iff the apple_gpu VALUE COMPILE pipeline yields an executable value
    artifact (metadata ``compiler_path == 'apple_value_target_ir'``) for a
    representative dotted op.

    This complements the per-op *runtime* probes below. Those bind a runtime
    kernel symbol and run a tiny numerical check — but they say nothing about
    whether the *compile* path still produces a value artifact. If the compile
    path silently degrades while a runtime kernel is present (a broken
    Graph-IR → tessera-opt seam, a missing/regressed apple value pipeline, or no
    tessera-opt at all), a ``*_value_available()`` probe would otherwise report
    True and the value tests — which assert on ``metadata['compiler_path']`` —
    would KeyError instead of skipping. Gating those probes on this check makes
    them SKIP.

    Scope: the seam/pipeline regression class is uniform across dotted value ops,
    so one representative op (``tessera.ebm.langevin_step``) suffices to detect
    it. A genuinely op-specific compile gap still relies on that op's runtime
    probe. Cached (the compile is run at most once per process).
    """
    global _APPLE_VALUE_COMPILE_PIPELINE_OK
    if _APPLE_VALUE_COMPILE_PIPELINE_OK is not None:
        return _APPLE_VALUE_COMPILE_PIPELINE_OK
    ok = False
    try:
        from .compiler.canonical_compile import canonical_compile
        from .compiler.graph_ir import (
            GraphIRFunction, GraphIRModule, IRArg, IROp, IRType,
        )

        t = "tensor<2x3xf32>"
        ty = IRType(t, ("2", "3"), "fp32")
        op = IROp(
            result="o", op_name="tessera.ebm.langevin_step",
            operands=["%a0", "%a1", "%a2"], operand_types=[t, t, t],
            result_type=t, kwargs={"eta": 0.125, "noise_scale": 0.25},
        )
        module = GraphIRModule(functions=[GraphIRFunction(
            name="f", args=[IRArg(f"a{i}", ty) for i in range(3)],
            result_types=[ty], body=[op], return_values=["%o"])])
        art = canonical_compile(
            module, target="apple_gpu",
            options={"apple_target_ir_mode": "value"}).to_runtime_artifact()
        ok = (art.metadata or {}).get("compiler_path") == "apple_value_target_ir"
    except Exception:  # noqa: BLE001 — any failure means "not available"
        ok = False
    _APPLE_VALUE_COMPILE_PIPELINE_OK = ok
    return ok


def _require_value_compile_pipeline(probe):
    """Decorator: a value-availability probe is only True if the runtime kernel
    AND the value COMPILE pipeline are both available (so tests skip, not
    KeyError, when the compile path has silently degraded)."""

    @functools.wraps(probe)
    def gated() -> bool:
        return _apple_value_compile_pipeline_available() and probe()

    return gated


@_require_value_compile_pipeline
def _apple_gpu_ebm_energy_quadratic_value_available() -> bool:
    """True iff the EBM energy value ABI runs a tiny Metal numerical probe."""
    global _APPLE_GPU_EBM_ENERGY_QUADRATIC_AVAILABLE
    if _APPLE_GPU_EBM_ENERGY_QUADRATIC_AVAILABLE is not None:
        return _APPLE_GPU_EBM_ENERGY_QUADRATIC_AVAILABLE
    sym = _apple_gpu_ebm_energy_quadratic_value_f32()
    if sym is None:
        _APPLE_GPU_EBM_ENERGY_QUADRATIC_AVAILABLE = False
        return False
    try:
        import numpy as _np
        x = _np.asarray([[1.0, 2.0, -1.0], [0.5, 0.25, -0.75]],
                        dtype=_np.float32)
        y = _np.asarray([[0.0, 1.0, 1.0], [0.25, -0.25, -0.25]],
                        dtype=_np.float32)
        out = _np.empty((2,), dtype=_np.float32)
        fp = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = int(sym(fp(x), fp(y), fp(out), ctypes.c_int32(2),
                     ctypes.c_int32(3)))
        expected = 0.5 * _np.sum((x - y) * (x - y), axis=1)
        _APPLE_GPU_EBM_ENERGY_QUADRATIC_AVAILABLE = (
            rc == 1 and bool(_np.all(_np.isfinite(out))) and
            bool(_np.allclose(out, expected, rtol=1.0e-5, atol=1.0e-6)))
    except Exception:
        _APPLE_GPU_EBM_ENERGY_QUADRATIC_AVAILABLE = False
    return _APPLE_GPU_EBM_ENERGY_QUADRATIC_AVAILABLE


@_require_value_compile_pipeline
def _apple_gpu_ebm_langevin_step_value_available() -> bool:
    """True iff the EBM Langevin value ABI runs a tiny Metal numerical probe."""
    global _APPLE_GPU_EBM_LANGEVIN_STEP_AVAILABLE
    if _APPLE_GPU_EBM_LANGEVIN_STEP_AVAILABLE is not None:
        return _APPLE_GPU_EBM_LANGEVIN_STEP_AVAILABLE
    sym = _apple_gpu_ebm_langevin_step_value_f32()
    if sym is None:
        _APPLE_GPU_EBM_LANGEVIN_STEP_AVAILABLE = False
        return False
    try:
        import numpy as _np
        y = _np.asarray([[0.5, -1.0, 2.0], [1.25, 0.0, -0.5]],
                       dtype=_np.float32)
        grad = _np.asarray([[0.25, -0.5, 1.0], [0.5, -0.25, 0.75]],
                          dtype=_np.float32)
        noise = _np.asarray([[0.1, 0.2, -0.3], [0.4, -0.5, 0.6]],
                           dtype=_np.float32)
        eta = 0.125
        noise_scale = 0.25
        out = _np.empty_like(y)
        fp = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = int(sym(fp(y), fp(grad), fp(noise), ctypes.c_float(eta),
                     ctypes.c_float(noise_scale), fp(out),
                     ctypes.c_int32(int(y.size))))
        expected = y - eta * grad + noise_scale * noise
        _APPLE_GPU_EBM_LANGEVIN_STEP_AVAILABLE = (
            rc == 1 and bool(_np.all(_np.isfinite(out))) and
            bool(_np.allclose(out, expected, rtol=1.0e-5, atol=1.0e-6)))
    except Exception:
        _APPLE_GPU_EBM_LANGEVIN_STEP_AVAILABLE = False
    return _APPLE_GPU_EBM_LANGEVIN_STEP_AVAILABLE


def _clifford_geo_product_cl30_np(np, a, b):
    """Reference cl(3,0) geometric product for blade-last 8-coeff tensors."""
    c = np.empty_like(a, dtype=np.float32)
    c[..., 0] = (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] +
                 a[..., 2] * b[..., 2] + a[..., 3] * b[..., 3] -
                 a[..., 4] * b[..., 4] - a[..., 5] * b[..., 5] -
                 a[..., 6] * b[..., 6] - a[..., 7] * b[..., 7])
    c[..., 1] = (a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0] -
                 a[..., 2] * b[..., 4] + a[..., 3] * b[..., 6] +
                 a[..., 4] * b[..., 2] - a[..., 5] * b[..., 7] -
                 a[..., 6] * b[..., 3] - a[..., 7] * b[..., 5])
    c[..., 2] = (a[..., 0] * b[..., 2] + a[..., 1] * b[..., 4] +
                 a[..., 2] * b[..., 0] - a[..., 3] * b[..., 5] -
                 a[..., 4] * b[..., 1] + a[..., 5] * b[..., 3] -
                 a[..., 6] * b[..., 7] - a[..., 7] * b[..., 6])
    c[..., 3] = (a[..., 0] * b[..., 3] - a[..., 1] * b[..., 6] +
                 a[..., 2] * b[..., 5] + a[..., 3] * b[..., 0] -
                 a[..., 4] * b[..., 7] - a[..., 5] * b[..., 2] +
                 a[..., 6] * b[..., 1] - a[..., 7] * b[..., 4])
    c[..., 4] = (a[..., 0] * b[..., 4] + a[..., 1] * b[..., 2] -
                 a[..., 2] * b[..., 1] + a[..., 3] * b[..., 7] +
                 a[..., 4] * b[..., 0] - a[..., 5] * b[..., 6] +
                 a[..., 6] * b[..., 5] + a[..., 7] * b[..., 3])
    c[..., 5] = (a[..., 0] * b[..., 5] + a[..., 1] * b[..., 7] +
                 a[..., 2] * b[..., 3] - a[..., 3] * b[..., 2] +
                 a[..., 4] * b[..., 6] + a[..., 5] * b[..., 0] -
                 a[..., 6] * b[..., 4] + a[..., 7] * b[..., 1])
    c[..., 6] = (a[..., 0] * b[..., 6] - a[..., 1] * b[..., 3] +
                 a[..., 2] * b[..., 7] + a[..., 3] * b[..., 1] -
                 a[..., 4] * b[..., 5] + a[..., 5] * b[..., 4] +
                 a[..., 6] * b[..., 0] + a[..., 7] * b[..., 2])
    c[..., 7] = (a[..., 0] * b[..., 7] + a[..., 1] * b[..., 5] +
                 a[..., 2] * b[..., 6] + a[..., 3] * b[..., 4] +
                 a[..., 4] * b[..., 3] + a[..., 5] * b[..., 1] +
                 a[..., 6] * b[..., 2] + a[..., 7] * b[..., 0])
    return c


@_require_value_compile_pipeline
def _apple_gpu_ebm_refinement_value_available() -> bool:
    """True iff deterministic EBM refinement runs a tiny Metal probe."""
    global _APPLE_GPU_EBM_REFINEMENT_AVAILABLE
    if _APPLE_GPU_EBM_REFINEMENT_AVAILABLE is not None:
        return _APPLE_GPU_EBM_REFINEMENT_AVAILABLE
    sym = _apple_gpu_ebm_refinement_value_f32()
    if sym is None:
        _APPLE_GPU_EBM_REFINEMENT_AVAILABLE = False
        return False
    try:
        import numpy as _np
        y0 = _np.asarray([[0.5, -1.0, 2.0], [1.25, 0.0, -0.5]],
                        dtype=_np.float32)
        grad = _np.asarray([[0.25, -0.5, 1.0], [0.5, -0.25, 0.75]],
                          dtype=_np.float32)
        eta = 0.125
        steps = 4
        out = _np.empty_like(y0)
        fp = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = int(sym(fp(y0), fp(grad), ctypes.c_float(eta),
                     ctypes.c_int32(steps), fp(out),
                     ctypes.c_int32(int(y0.size))))
        expected = y0 - steps * eta * grad
        _APPLE_GPU_EBM_REFINEMENT_AVAILABLE = (
            rc == 1 and bool(_np.all(_np.isfinite(out))) and
            bool(_np.allclose(out, expected, rtol=1.0e-5, atol=1.0e-6)))
    except Exception:
        _APPLE_GPU_EBM_REFINEMENT_AVAILABLE = False
    return _APPLE_GPU_EBM_REFINEMENT_AVAILABLE


@_require_value_compile_pipeline
def _apple_gpu_ebm_partition_exact_value_available() -> bool:
    """True iff scalar EBM partition exact runs a tiny Metal probe."""
    global _APPLE_GPU_EBM_PARTITION_EXACT_AVAILABLE
    if _APPLE_GPU_EBM_PARTITION_EXACT_AVAILABLE is not None:
        return _APPLE_GPU_EBM_PARTITION_EXACT_AVAILABLE
    sym = _apple_gpu_ebm_partition_exact_value_f32()
    if sym is None:
        _APPLE_GPU_EBM_PARTITION_EXACT_AVAILABLE = False
        return False
    try:
        import numpy as _np
        energies = _np.asarray([0.2, -0.3, 1.0], dtype=_np.float32)
        temperature = 0.75
        out = _np.empty((), dtype=_np.float32)
        fp = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = int(sym(fp(energies), ctypes.c_int32(int(energies.size)),
                     ctypes.c_float(temperature), fp(out)))
        scaled = -energies / temperature
        expected = float(_np.exp(_np.max(scaled)) *
                         _np.sum(_np.exp(scaled - _np.max(scaled))))
        _APPLE_GPU_EBM_PARTITION_EXACT_AVAILABLE = (
            rc == 1 and bool(_np.isfinite(out)) and
            abs(float(out) - expected) < 1.0e-5)
    except Exception:
        _APPLE_GPU_EBM_PARTITION_EXACT_AVAILABLE = False
    return _APPLE_GPU_EBM_PARTITION_EXACT_AVAILABLE


@_require_value_compile_pipeline
def _apple_gpu_clifford_geo_product_cl30_value_available() -> bool:
    """True iff cl30 Clifford geometric product runs a tiny Metal probe."""
    global _APPLE_GPU_CLIFFORD_GEO_PRODUCT_AVAILABLE
    if _APPLE_GPU_CLIFFORD_GEO_PRODUCT_AVAILABLE is not None:
        return _APPLE_GPU_CLIFFORD_GEO_PRODUCT_AVAILABLE
    sym = _apple_gpu_clifford_geo_product_cl30_value_f32()
    if sym is None:
        _APPLE_GPU_CLIFFORD_GEO_PRODUCT_AVAILABLE = False
        return False
    try:
        import numpy as _np
        a = _np.asarray([[1.0, 0.2, -0.4, 0.5, 0.1, -0.3, 0.7, -0.2],
                         [0.3, -0.6, 0.8, -0.1, 0.4, 0.2, -0.5, 0.9]],
                        dtype=_np.float32)
        b = _np.asarray([[0.5, -0.1, 0.6, 0.2, -0.7, 0.3, 0.4, -0.8],
                         [-0.2, 0.9, -0.3, 0.7, 0.1, -0.4, 0.6, 0.5]],
                        dtype=_np.float32)
        out = _np.empty_like(a)
        fp = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = int(sym(fp(a), fp(b), fp(out), ctypes.c_int32(2)))
        expected = _clifford_geo_product_cl30_np(_np, a, b)
        _APPLE_GPU_CLIFFORD_GEO_PRODUCT_AVAILABLE = (
            rc == 1 and bool(_np.all(_np.isfinite(out))) and
            bool(_np.allclose(out, expected, rtol=1.0e-5, atol=1.0e-6)))
    except Exception:
        _APPLE_GPU_CLIFFORD_GEO_PRODUCT_AVAILABLE = False
    return _APPLE_GPU_CLIFFORD_GEO_PRODUCT_AVAILABLE


# Apple GPU value-lane batched-matmul dispatch (Sprint 8). symbol -> (resolver,
# numpy-dtype-kind). The key set is the GPU value allowlist.
_APPLE_VALUE_GPU_DISPATCH: dict[str, tuple] = {
    "tessera_apple_gpu_bmm_f32": (_apple_gpu_bmm_f32, "f32"),
    "tessera_apple_gpu_bmm_f16": (_apple_gpu_bmm_f16, "f16"),
    "tessera_apple_gpu_bmm_bf16": (_apple_gpu_bmm_bf16, "bf16"),
    "tessera_apple_gpu_native_sparse_attn_f32": (
        _apple_gpu_native_sparse_attn_f32, "native_sparse_attn_f32"),
    "tessera_apple_gpu_ppo_policy_loss_f32": (
        _apple_gpu_ppo_policy_loss_f32, "ppo_policy_loss_f32"),
    "tessera_apple_gpu_ppo_policy_loss_ex_f32": (
        _apple_gpu_ppo_policy_loss_ex_f32, "ppo_policy_loss_ex_f32"),
    "tessera_apple_gpu_ebm_energy_quadratic_value_f32": (
        _apple_gpu_ebm_energy_quadratic_value_f32,
        "ebm_energy_quadratic_value_f32"),
    "tessera_apple_gpu_ebm_langevin_step_value_f32": (
        _apple_gpu_ebm_langevin_step_value_f32,
        "ebm_langevin_step_value_f32"),
    "tessera_apple_gpu_ebm_refinement_value_f32": (
        _apple_gpu_ebm_refinement_value_f32,
        "ebm_refinement_value_f32"),
    "tessera_apple_gpu_ebm_partition_exact_value_f32": (
        _apple_gpu_ebm_partition_exact_value_f32,
        "ebm_partition_exact_value_f32"),
    "tessera_apple_gpu_clifford_geo_product_cl30_value_f32": (
        _apple_gpu_clifford_geo_product_cl30_value_f32,
        "clifford_geo_product_cl30_value_f32"),
}
_APPLE_VALUE_GPU_SYMBOLS: frozenset[str] = frozenset(_APPLE_VALUE_GPU_DISPATCH)


def _dispatch_gpu_batched_matmul(inputs, call, np):
    """Sprint 8: strict rank-3 batched matmul on the Apple GPU value lane.

    Validates exactly two rank-3 operands with matching batch + K (no
    broadcasting), dispatches the symbol named in the value call, and preserves
    the dtype-specific output (f32/f16/bf16). f16/bf16 use uint16 views at the
    ctypes boundary; bf16 requires `ml_dtypes` (named error if missing, never a
    silent fp32 substitution)."""
    symbol = str(call.get("symbol", ""))
    entry = _APPLE_VALUE_GPU_DISPATCH.get(symbol)
    if entry is None:
        raise ValueError(
            f"apple_value_target_ir(gpu): symbol {symbol!r} is not in the GPU "
            f"value allowlist {sorted(_APPLE_VALUE_GPU_SYMBOLS)}")
    resolver, kind = entry
    if kind == "f32":
        npdt, half = np.float32, False
    elif kind == "f16":
        npdt, half = np.float16, True
    else:  # bf16
        npdt = _bfloat16_dtype()
        if npdt is None:
            raise ValueError(
                "bf16 GPU value matmul requires the optional `ml_dtypes` "
                "dependency (pip install ml_dtypes) — falling back to fp32 would "
                "silently change the dtype contract")
        half = True

    a = np.ascontiguousarray(np.asarray(inputs[0], dtype=npdt))
    b = np.ascontiguousarray(np.asarray(inputs[1], dtype=npdt))
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError(
            f"batched_gemm(gpu) requires rank-3 operands, got {a.ndim}-D / {b.ndim}-D")
    batch, m, ka = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
    bb, kb, n = int(b.shape[0]), int(b.shape[1]), int(b.shape[2])
    if batch != bb:
        raise ValueError(
            f"batched_gemm(gpu) batch mismatch: A batch={batch}, B batch={bb} "
            f"(no broadcasting on the value lane)")
    if ka != kb:
        raise ValueError(f"batched_gemm(gpu) contraction mismatch: A K={ka}, B K={kb}")
    sym = resolver()
    if sym is None:
        raise ValueError(
            f"apple_gpu runtime lacks {symbol} — rebuild TesseraAppleRuntime "
            f"(the prebuilt dylib predates this GPU bmm symbol)")
    out = np.empty((batch, m, n), dtype=npdt)
    dims = (ctypes.c_int32(batch), ctypes.c_int32(m), ctypes.c_int32(n),
            ctypes.c_int32(ka), ctypes.c_int32(0))  # b_broadcast=0 (exact batch)
    if half:
        def _u16(arr):
            return arr.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        sym(_u16(a), _u16(b), _u16(out), *dims)
    else:
        def _fp(arr):
            return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        sym(_fp(a), _fp(b), _fp(out), *dims)
    return out


def _dispatch_gpu_native_sparse_attn(inputs, call, np):
    """Strict fp32 rank-4 DeepSeek NSA dispatch for Apple GPU value IR."""
    symbol = str(call.get("symbol", ""))
    entry = _APPLE_VALUE_GPU_DISPATCH.get(symbol)
    if entry is None or entry[1] != "native_sparse_attn_f32":
        raise ValueError(
            f"apple_value_target_ir(gpu): symbol {symbol!r} is not the native "
            "sparse-attention value symbol")
    if len(inputs) != 4:
        raise ValueError(
            f"apple_value_target_ir(gpu): native_sparse_attn_fused value-call "
            f"needs exactly 4 input(s), got {len(inputs)}")
    q = np.ascontiguousarray(np.asarray(inputs[0], dtype=np.float32))
    k = np.ascontiguousarray(np.asarray(inputs[1], dtype=np.float32))
    v = np.ascontiguousarray(np.asarray(inputs[2], dtype=np.float32))
    gate = np.ascontiguousarray(np.asarray(inputs[3], dtype=np.float32))
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or gate.ndim != 4:
        raise ValueError(
            "native_sparse_attn_fused(gpu) requires rank-4 Q/K/V/gate tensors")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"native_sparse_attn_fused(gpu) Q/K/V shape mismatch: "
            f"{q.shape} / {k.shape} / {v.shape}")
    try:
        window = int(call["window_size"])
        block = int(call["block_size"])
        top_k = int(call["top_k"])
    except Exception as exc:
        raise ValueError(
            "native_sparse_attn_fused(gpu) requires window_size/block_size/"
            "top_k integer attrs") from exc
    causal = bool(call.get("causal", True))
    B, H, S, D = (int(x) for x in q.shape)
    if block <= 0 or window <= 0 or top_k <= 0 or S % block != 0:
        raise ValueError(
            "native_sparse_attn_fused(gpu) requires positive window/block/top_k "
            "and S divisible by block_size")
    num_blocks = S // block
    if top_k > num_blocks:
        raise ValueError(
            f"native_sparse_attn_fused(gpu) top_k={top_k} exceeds "
            f"S/block_size={num_blocks}")
    if gate.shape != (B, H, S, num_blocks):
        raise ValueError(
            f"native_sparse_attn_fused(gpu) gate shape {gate.shape} must be "
            f"({B}, {H}, {S}, {num_blocks})")
    sym = _apple_gpu_native_sparse_attn_f32()
    if sym is None:
        raise ValueError(
            "apple_gpu runtime lacks an active Metal native sparse-attention "
            "executor; the non-Darwin stub is not executable for value IR")
    out = np.empty_like(q, dtype=np.float32)
    fp = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    sym(fp(q), fp(k), fp(v), fp(gate), fp(out),
        ctypes.c_int32(B), ctypes.c_int32(H), ctypes.c_int32(S),
        ctypes.c_int32(D), ctypes.c_int32(window), ctypes.c_int32(block),
        ctypes.c_int32(top_k), ctypes.c_int32(1 if causal else 0))
    return out


def _dispatch_gpu_ppo_policy_loss(inputs, call, np):
    """fp32 PPO mean policy-loss dispatch for Apple GPU value IR.

    The strict Stage 13 symbol takes exactly three tensors. Stage 14's extended
    symbol may add mask, ref_logp, and entropy tensors when the value-call attrs
    explicitly request them. Both symbols are MPSGraph-probed before success is
    reported.
    """
    symbol = str(call.get("symbol", ""))
    entry = _APPLE_VALUE_GPU_DISPATCH.get(symbol)
    if entry is None or entry[1] not in {
        "ppo_policy_loss_f32", "ppo_policy_loss_ex_f32",
    }:
        raise ValueError(
            f"apple_value_target_ir(gpu): symbol {symbol!r} is not the PPO "
            "policy-loss value symbol")
    kind = entry[1]
    has_mask = bool(call.get("has_mask", False))
    has_ref_kl = bool(call.get("has_ref_kl", False))
    has_entropy = bool(call.get("has_entropy", False))
    want = 3 + int(has_mask) + int(has_ref_kl) + int(has_entropy)
    if len(inputs) != want:
        raise ValueError(
            f"apple_value_target_ir(gpu): ppo_policy_loss value-call needs "
            f"exactly {want} input(s), got {len(inputs)}")
    if str(call.get("reduction", "mean")) != "mean":
        raise ValueError("ppo_policy_loss(gpu) only supports reduction='mean'")
    clip = float(call.get("clip_epsilon", 0.2))
    if clip <= 0.0:
        raise ValueError("ppo_policy_loss(gpu) requires positive clip_epsilon")
    kl_coef = float(call.get("kl_coef", 0.0))
    entropy_coef = float(call.get("entropy_coef", 0.0))
    if kind == "ppo_policy_loss_f32" and (has_mask or has_ref_kl or has_entropy):
        raise ValueError(
            "ppo_policy_loss(gpu) side tensors require "
            "tessera_apple_gpu_ppo_policy_loss_ex_f32")
    if kind == "ppo_policy_loss_f32" and kl_coef != 0.0:
        raise ValueError("ppo_policy_loss(gpu) strict symbol does not support KL side terms")
    logp_new = np.ascontiguousarray(np.asarray(inputs[0], dtype=np.float32))
    logp_old = np.ascontiguousarray(np.asarray(inputs[1], dtype=np.float32))
    adv = np.ascontiguousarray(np.asarray(inputs[2], dtype=np.float32))
    if logp_new.shape != logp_old.shape or logp_new.shape != adv.shape:
        raise ValueError(
            f"ppo_policy_loss(gpu) shape mismatch: {logp_new.shape} / "
            f"{logp_old.shape} / {adv.shape}")
    if logp_new.size <= 0:
        raise ValueError("ppo_policy_loss(gpu) requires a non-empty tensor")
    out = np.empty((), dtype=np.float32)
    fp = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    if kind == "ppo_policy_loss_f32":
        sym = _apple_gpu_ppo_policy_loss_f32()
        if sym is None or not _apple_gpu_ppo_policy_loss_available():
            raise ValueError(
                "apple_gpu runtime lacks an active, numerically proven "
                "tessera_apple_gpu_ppo_policy_loss_f32 executor")
        rc = int(sym(fp(logp_new), fp(logp_old), fp(adv), fp(out),
                     ctypes.c_int32(int(logp_new.size)), ctypes.c_float(clip)))
    else:
        sym = _apple_gpu_ppo_policy_loss_ex_f32()
        if sym is None or not _apple_gpu_ppo_policy_loss_ex_available():
            raise ValueError(
                "apple_gpu runtime lacks an active, numerically proven "
                "tessera_apple_gpu_ppo_policy_loss_ex_f32 executor")
        idx = 3
        mask = ref_logp = entropy = None
        if has_mask:
            mask = np.ascontiguousarray(np.asarray(inputs[idx], dtype=np.float32))
            idx += 1
        if has_ref_kl:
            ref_logp = np.ascontiguousarray(np.asarray(inputs[idx], dtype=np.float32))
            idx += 1
        if has_entropy:
            entropy = np.ascontiguousarray(np.asarray(inputs[idx], dtype=np.float32))
        for label, arr in (("mask", mask), ("ref_logp", ref_logp),
                           ("entropy", entropy)):
            if arr is not None and arr.shape != logp_new.shape:
                raise ValueError(
                    f"ppo_policy_loss(gpu) {label} shape {arr.shape} must "
                    f"match logp shape {logp_new.shape}")
        null = ctypes.POINTER(ctypes.c_float)()
        rc = int(sym(
            fp(logp_new), fp(logp_old), fp(adv),
            fp(mask) if mask is not None else null,
            fp(ref_logp) if ref_logp is not None else null,
            fp(entropy) if entropy is not None else null,
            fp(out), ctypes.c_int32(int(logp_new.size)),
            ctypes.c_float(clip), ctypes.c_float(kl_coef),
            ctypes.c_float(entropy_coef), ctypes.c_int32(1 if has_mask else 0),
            ctypes.c_int32(1 if has_ref_kl else 0),
            ctypes.c_int32(1 if has_entropy else 0)))
    if rc != 1:
        raise ValueError(
            "apple_gpu PPO policy-loss value executor is not active "
            "(stub/unavailable MPSGraph path returned non-success)")
    return out


def _dispatch_gpu_ebm_energy_quadratic(inputs, call, np):
    """Strict fp32 rank-2 quadratic EBM energy on the Apple GPU value lane."""
    symbol = str(call.get("symbol", ""))
    entry = _APPLE_VALUE_GPU_DISPATCH.get(symbol)
    if entry is None or entry[1] != "ebm_energy_quadratic_value_f32":
        raise ValueError(
            f"apple_value_target_ir(gpu): symbol {symbol!r} is not the EBM "
            "energy value symbol")
    if len(inputs) != 2:
        raise ValueError(
            f"apple_value_target_ir(gpu): ebm_energy_quadratic value-call needs "
            f"exactly 2 input(s), got {len(inputs)}")
    x = np.ascontiguousarray(np.asarray(inputs[0], dtype=np.float32))
    y = np.ascontiguousarray(np.asarray(inputs[1], dtype=np.float32))
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("ebm_energy_quadratic(gpu) requires rank-2 x/y tensors")
    if x.shape != y.shape:
        raise ValueError(
            f"ebm_energy_quadratic(gpu) shape mismatch: {x.shape} / {y.shape}")
    if x.size <= 0:
        raise ValueError("ebm_energy_quadratic(gpu) requires a non-empty tensor")
    sym = _apple_gpu_ebm_energy_quadratic_value_f32()
    if sym is None or not _apple_gpu_ebm_energy_quadratic_value_available():
        raise ValueError(
            "apple_gpu runtime lacks an active, numerically proven "
            "tessera_apple_gpu_ebm_energy_quadratic_value_f32 executor")
    batch, dim = int(x.shape[0]), int(x.shape[1])
    out = np.empty((batch,), dtype=np.float32)
    fp = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    rc = int(sym(fp(x), fp(y), fp(out), ctypes.c_int32(batch),
                 ctypes.c_int32(dim)))
    if rc != 1:
        raise ValueError(
            "apple_gpu EBM energy value executor is not active "
            "(stub/unavailable Metal path returned non-success)")
    return out


def _dispatch_gpu_ebm_langevin_step(inputs, call, np):
    """Strict fp32 affine Langevin step on the Apple GPU value lane."""
    symbol = str(call.get("symbol", ""))
    entry = _APPLE_VALUE_GPU_DISPATCH.get(symbol)
    if entry is None or entry[1] != "ebm_langevin_step_value_f32":
        raise ValueError(
            f"apple_value_target_ir(gpu): symbol {symbol!r} is not the EBM "
            "Langevin value symbol")
    if len(inputs) != 3:
        raise ValueError(
            f"apple_value_target_ir(gpu): ebm_langevin_step value-call needs "
            f"exactly 3 input(s), got {len(inputs)}")
    y = np.ascontiguousarray(np.asarray(inputs[0], dtype=np.float32))
    grad = np.ascontiguousarray(np.asarray(inputs[1], dtype=np.float32))
    noise = np.ascontiguousarray(np.asarray(inputs[2], dtype=np.float32))
    if y.shape != grad.shape or y.shape != noise.shape:
        raise ValueError(
            f"ebm_langevin_step(gpu) shape mismatch: {y.shape} / "
            f"{grad.shape} / {noise.shape}")
    if y.size <= 0:
        raise ValueError("ebm_langevin_step(gpu) requires a non-empty tensor")
    eta = float(call.get("eta", 0.0))
    noise_scale = float(call.get("noise_scale", 0.0))
    if eta <= 0.0:
        raise ValueError("ebm_langevin_step(gpu) requires positive eta")
    if noise_scale < 0.0:
        raise ValueError("ebm_langevin_step(gpu) requires non-negative noise_scale")
    sym = _apple_gpu_ebm_langevin_step_value_f32()
    if sym is None or not _apple_gpu_ebm_langevin_step_value_available():
        raise ValueError(
            "apple_gpu runtime lacks an active, numerically proven "
            "tessera_apple_gpu_ebm_langevin_step_value_f32 executor")
    out = np.empty_like(y, dtype=np.float32)
    fp = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    rc = int(sym(fp(y), fp(grad), fp(noise), ctypes.c_float(eta),
                 ctypes.c_float(noise_scale), fp(out),
                 ctypes.c_int32(int(y.size))))
    if rc != 1:
        raise ValueError(
            "apple_gpu EBM Langevin value executor is not active "
            "(stub/unavailable Metal path returned non-success)")
    return out


def _dispatch_gpu_ebm_refinement(inputs, call, np):
    """Strict fp32 deterministic T-step EBM refinement on the GPU value lane."""
    symbol = str(call.get("symbol", ""))
    entry = _APPLE_VALUE_GPU_DISPATCH.get(symbol)
    if entry is None or entry[1] != "ebm_refinement_value_f32":
        raise ValueError(
            f"apple_value_target_ir(gpu): symbol {symbol!r} is not the EBM "
            "refinement value symbol")
    if len(inputs) != 2:
        raise ValueError(
            f"apple_value_target_ir(gpu): ebm_refinement value-call needs "
            f"exactly 2 input(s), got {len(inputs)}")
    y0 = np.ascontiguousarray(np.asarray(inputs[0], dtype=np.float32))
    grad = np.ascontiguousarray(np.asarray(inputs[1], dtype=np.float32))
    if y0.shape != grad.shape:
        raise ValueError(
            f"ebm_refinement(gpu) shape mismatch: {y0.shape} / {grad.shape}")
    if y0.size <= 0:
        raise ValueError("ebm_refinement(gpu) requires a non-empty tensor")
    eta = float(call.get("eta", 0.0))
    steps = int(call.get("steps", 0))
    if eta <= 0.0:
        raise ValueError("ebm_refinement(gpu) requires positive eta")
    if steps <= 0:
        raise ValueError("ebm_refinement(gpu) requires positive steps")
    if "temperature" in call or "noise_scale" in call:
        raise ValueError(
            "ebm_refinement(gpu) value executor is deterministic; "
            "temperature/noise_scale variants are gated")
    sym = _apple_gpu_ebm_refinement_value_f32()
    if sym is None or not _apple_gpu_ebm_refinement_value_available():
        raise ValueError(
            "apple_gpu runtime lacks an active, numerically proven "
            "tessera_apple_gpu_ebm_refinement_value_f32 executor")
    out = np.empty_like(y0, dtype=np.float32)
    fp = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    rc = int(sym(fp(y0), fp(grad), ctypes.c_float(eta),
                 ctypes.c_int32(steps), fp(out), ctypes.c_int32(int(y0.size))))
    if rc != 1:
        raise ValueError(
            "apple_gpu EBM refinement value executor is not active "
            "(stub/unavailable Metal path returned non-success)")
    return out


def _dispatch_gpu_ebm_partition_exact(inputs, call, np):
    """Strict fp32 scalar exact EBM partition on the GPU value lane."""
    symbol = str(call.get("symbol", ""))
    entry = _APPLE_VALUE_GPU_DISPATCH.get(symbol)
    if entry is None or entry[1] != "ebm_partition_exact_value_f32":
        raise ValueError(
            f"apple_value_target_ir(gpu): symbol {symbol!r} is not the EBM "
            "partition-exact value symbol")
    if len(inputs) != 1:
        raise ValueError(
            f"apple_value_target_ir(gpu): ebm_partition_exact value-call needs "
            f"exactly 1 input(s), got {len(inputs)}")
    energies = np.ascontiguousarray(np.asarray(inputs[0], dtype=np.float32))
    if energies.size <= 0:
        raise ValueError("ebm_partition_exact(gpu) requires non-empty energies")
    temperature = float(call.get("temperature", 1.0))
    reduction = str(call.get("reduction", "logsumexp"))
    if temperature <= 0.0:
        raise ValueError("ebm_partition_exact(gpu) requires positive temperature")
    if reduction != "logsumexp":
        raise ValueError(
            "ebm_partition_exact(gpu) value executor only supports "
            "reduction='logsumexp'")
    sym = _apple_gpu_ebm_partition_exact_value_f32()
    if sym is None or not _apple_gpu_ebm_partition_exact_value_available():
        raise ValueError(
            "apple_gpu runtime lacks an active, numerically proven "
            "tessera_apple_gpu_ebm_partition_exact_value_f32 executor")
    out = np.empty((), dtype=np.float32)
    fp = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    rc = int(sym(fp(energies), ctypes.c_int32(int(energies.size)),
                 ctypes.c_float(temperature), fp(out)))
    if rc != 1:
        raise ValueError(
            "apple_gpu EBM partition-exact value executor is not active "
            "(stub/unavailable Metal path returned non-success)")
    return out


def _dispatch_gpu_clifford_geometric_product(inputs, call, np):
    """Strict fp32 cl(3,0) Clifford geometric product on the GPU value lane."""
    symbol = str(call.get("symbol", ""))
    entry = _APPLE_VALUE_GPU_DISPATCH.get(symbol)
    if entry is None or entry[1] != "clifford_geo_product_cl30_value_f32":
        raise ValueError(
            f"apple_value_target_ir(gpu): symbol {symbol!r} is not the "
            "Clifford cl30 geometric-product value symbol")
    if len(inputs) != 2:
        raise ValueError(
            f"apple_value_target_ir(gpu): clifford_geometric_product value-call "
            f"needs exactly 2 input(s), got {len(inputs)}")
    a = np.ascontiguousarray(np.asarray(inputs[0], dtype=np.float32))
    b = np.ascontiguousarray(np.asarray(inputs[1], dtype=np.float32))
    if a.shape != b.shape:
        raise ValueError(
            f"clifford_geometric_product(gpu) shape mismatch: {a.shape} / {b.shape}")
    if a.ndim < 1 or a.shape[-1] != 8:
        raise ValueError(
            "clifford_geometric_product(gpu) requires blade-last cl30 tensors "
            "with last dimension 8")
    if a.size <= 0:
        raise ValueError(
            "clifford_geometric_product(gpu) requires a non-empty tensor")
    if int(call.get("p", 3)) != 3 or int(call.get("q", 0)) != 0:
        raise ValueError(
            "clifford_geometric_product(gpu) value executor only supports p=3,q=0")
    batch = int(a.size // 8)
    sym = _apple_gpu_clifford_geo_product_cl30_value_f32()
    if sym is None or not _apple_gpu_clifford_geo_product_cl30_value_available():
        raise ValueError(
            "apple_gpu runtime lacks an active, numerically proven "
            "tessera_apple_gpu_clifford_geo_product_cl30_value_f32 executor")
    out = np.empty_like(a, dtype=np.float32)
    fp = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    rc = int(sym(fp(a), fp(b), fp(out), ctypes.c_int32(batch)))
    if rc != 1:
        raise ValueError(
            "apple_gpu Clifford geometric-product value executor is not active "
            "(stub/unavailable Metal path returned non-success)")
    return out


def _execute_apple_value_target_ir_gpu_artifact(artifact: "RuntimeArtifact", args: Any) -> Any:
    """Execute an Apple GPU value-target-IR artifact.

    Scope: exactly one `tessera_apple.gpu.kernel_call` with
    an op kind and symbol on the GPU value allowlist. Sprint 8 shipped
    rank-3 `batched_gemm`; Sprint 11 adds static fp32 rank-4
    `native_sparse_attn_fused`; Stage 13 adds strict fp32 mean PPO policy
    loss; the EBM value lane adds strict fp32 quadratic energy, one-step
    Langevin, deterministic refinement, scalar partition exact, and one cl30
    Clifford geometric-product executor. Rejects
    `cpu.call`, `package_call`, multi-op programs, unknown symbols, extra
    operands, and non-executable statuses — so launch reports `invalid_artifact`
    rather than silently mis-dispatching."""
    import numpy as np

    metadata = artifact.metadata or {}
    calls = list(metadata.get("apple_value_calls") or [])
    if not calls:
        raise ValueError("apple_value_target_ir(gpu) artifact carries no apple_value_calls")
    if len(calls) != 1:
        raise ValueError(
            f"apple_value_target_ir(gpu): only single value-call programs are "
            f"executable (got {len(calls)}); multi-op is a named follow-on")
    call = calls[0]
    op = str(call.get("op", ""))
    if op != "tessera_apple.gpu.kernel_call":
        raise ValueError(
            f"apple_value_target_ir(gpu): only tessera_apple.gpu.kernel_call is "
            f"executable; '{op}' (cpu.call / package_call are gated)")
    if call.get("status") != "executable":
        raise ValueError(
            f"apple_value_target_ir(gpu): value call status is "
            f"{call.get('status')!r}, not 'executable'")
    op_kind = str(call.get("op_kind"))
    if op_kind not in {"batched_gemm", "native_sparse_attn_fused",
                       "ppo_policy_loss", "ebm_energy_quadratic",
                       "ebm_langevin_step", "ebm_refinement",
                       "ebm_partition_exact", "clifford_geometric_product"}:
        raise ValueError(
            f"apple_value_target_ir(gpu): only batched_gemm, "
            f"native_sparse_attn_fused, ppo_policy_loss, EBM value kernels, "
            f"and cl30 clifford_geometric_product execute on the GPU value "
            f"lane today "
            f"(got op_kind={call.get('op_kind')!r})")

    arg_names = list(metadata.get("arg_names") or [])
    if arg_names:
        if isinstance(args, (list, tuple)) and len(args) != len(arg_names):
            raise ValueError(
                f"apple_value_target_ir(gpu): value-call needs exactly "
                f"{len(arg_names)} named input(s), got {len(args)}")
        bound = _bind_launch_args(args, arg_names)
        inputs = [bound[name] for name in arg_names if name in bound]
    elif isinstance(args, (list, tuple)):
        inputs = list(args)
    else:
        inputs = [args]
    if op_kind == "batched_gemm":
        if len(inputs) != 2:
            raise ValueError(
                f"apple_value_target_ir(gpu): batched_gemm value-call needs "
                f"exactly 2 input(s), got {len(inputs)}")
        return _dispatch_gpu_batched_matmul(inputs, call, np)
    if op_kind == "ppo_policy_loss":
        return _dispatch_gpu_ppo_policy_loss(inputs, call, np)
    if op_kind == "ebm_energy_quadratic":
        return _dispatch_gpu_ebm_energy_quadratic(inputs, call, np)
    if op_kind == "ebm_langevin_step":
        return _dispatch_gpu_ebm_langevin_step(inputs, call, np)
    if op_kind == "ebm_refinement":
        return _dispatch_gpu_ebm_refinement(inputs, call, np)
    if op_kind == "ebm_partition_exact":
        return _dispatch_gpu_ebm_partition_exact(inputs, call, np)
    if op_kind == "clifford_geometric_product":
        return _dispatch_gpu_clifford_geometric_product(inputs, call, np)
    return _dispatch_gpu_native_sparse_attn(inputs, call, np)


def _apple_gpu_dispatch_grouped_gemm(operands: Any, kwargs: Any, np: Any) -> Any:
    """Ragged grouped matmul on Apple GPU: each contiguous token group is
    multiplied by its expert weight via a per-group MPS matmul on the Metal
    `bmm`/matmul lane (numpy fallback per group when a shape isn't supported).
    ``operands`` = (x, weights, group_sizes); group_sizes is integer routing
    metadata resolved as the third graph operand."""
    from . import _apple_gpu_backend as agb

    x = np.asarray(operands[0], dtype=np.float32)
    w = np.asarray(operands[1], dtype=np.float32)
    gs = np.asarray(operands[2]).astype(np.int64).reshape(-1)
    T = int(x.shape[0])

    # Fast path: ONE fused dispatch over the whole (T, N) output via the
    # grouped_gemm MSL kernel (folds routing in, no per-expert dispatch).
    if int(gs.sum()) == T:
        expert_ids = np.repeat(np.arange(w.shape[0], dtype=np.int32), gs)
        try:
            return np.ascontiguousarray(agb.gpu_grouped_gemm(x, w, expert_ids))
        except Exception:                           # noqa: BLE001 — fall to per-group
            pass

    # Fallback: per-group MPS matmul (also covers a malformed group_sizes sum).
    out = np.zeros((T, w.shape[2]), dtype=np.float32)
    off = 0
    for e in range(w.shape[0]):
        n = int(gs[e])
        if n:
            blk = np.ascontiguousarray(x[off:off + n])
            we = np.ascontiguousarray(w[e])
            try:
                r = agb.gpu_matmul(blk, we)
            except Exception:                       # noqa: BLE001 — per-group fallback
                r = None
            out[off:off + n] = r if r is not None else blk @ we
        off += n
    return out


def _apple_gpu_dispatch_clifford(op_name: Any, operands: Any, kwargs: Any, np: Any) -> Any:
    """Geometric-algebra (Clifford Cl(3,0)) flat-coefficient op on Metal.

    Routes through the GA lane shim (``_clifford_ops``), which internally
    dispatches Cl(3,0) f32 inputs to the ``cl30`` MSL kernels via
    ``tessera.ga.ops._try_apple_gpu_*`` (falling back to the numpy reference for
    other signatures/dtypes). The shim is the single source of GA truth — the
    runtime does not re-implement the products here.
    """
    from . import _clifford_ops as C
    short = str(op_name).split(".", 1)[1] if "." in str(op_name) else str(op_name)
    fn: Any = C.CLIFFORD_OPS[short]
    if short == "clifford_grade_projection":
        grade = kwargs.get("grade", kwargs.get("k"))
        return fn(np.asarray(operands[0]), grade=grade)
    if short in ("clifford_ext_deriv", "clifford_vec_deriv", "clifford_codiff"):
        return fn(np.asarray(operands[0]), spacing=kwargs.get("spacing"))
    return fn(*[np.asarray(o) for o in operands])


def _apple_gpu_dispatch_ebm(op_name: Any, operands: Any, kwargs: Any, np: Any) -> Any:
    """Energy-based-model flat-array op on Metal.

    Routes through the EBM lane shim (``_ebm_ops``), which internally dispatches
    f32 inputs to the dedicated EBM MSL kernels
    (``tessera_apple_gpu_ebm_{energy_quadratic,self_verify,refinement,inner_step}_f32``),
    falling back to the numpy reference otherwise. The shim is the single source
    of EBM truth — the runtime does not re-implement the energies here.
    """
    from . import _ebm_ops as B
    short = str(op_name).split(".", 1)[1] if "." in str(op_name) else str(op_name)
    fn: Any = B.EBM_OPS[short]
    ops = [np.asarray(o) for o in operands]
    if short == "ebm_energy_quadratic":
        return fn(ops[0], ops[1])
    if short == "ebm_self_verify":
        return fn(ops[0], ops[1], beta=kwargs.get("beta"))
    if short == "ebm_refinement":
        return fn(ops[0], ops[1], eta=kwargs["eta"], T=kwargs["T"])
    # ebm_inner_step
    return fn(ops[0], ops[1], eta=kwargs["eta"], noise_scale=kwargs.get("noise_scale", 0.0))


def _apple_gpu_dispatch_popcount(operands: Any, kwargs: Any, np: Any) -> Any:
    """LDT popcount on Metal (MSL `popcount` intrinsic), shape-preserving."""
    from . import _apple_gpu_backend as agb
    return agb.gpu_popcount(np.asarray(operands[0]))


def _apple_gpu_dispatch_count_nonzero(operands: Any, kwargs: Any, np: Any) -> Any:
    """LDT count_nonzero on Metal. The innermost-axis case runs the dedicated
    MSL kernel; other axes / keepdims fall back to numpy (still correct)."""
    from . import _apple_gpu_backend as agb
    x = np.asarray(operands[0])
    axis = kwargs.get("axis")
    keepdims = bool(kwargs.get("keepdims", False))
    last = x.ndim - 1
    if x.ndim >= 1 and not keepdims and axis in (-1, last):
        return agb.gpu_count_nonzero_lastaxis(x)
    return (x != 0).sum(axis=axis, keepdims=keepdims)


def _apple_gpu_dispatch_z_loss(operands: Any, kwargs: Any, np: Any) -> Any:
    """Router z-loss on Metal (MPSGraph). Only ``reduction="mean"`` runs on GPU;
    other reductions fall back to the numpy reference."""
    from . import _apple_gpu_backend as agb
    x = np.asarray(operands[0], dtype=np.float32)
    if kwargs.get("reduction", "mean") != "mean" or x.ndim < 1:
        import tessera.losses as _L
        return _L.z_loss(x, reduction=kwargs.get("reduction", "mean"))
    return np.float32(agb.gpu_z_loss(x))


def _apple_gpu_dispatch_asymmetric_bce(operands: Any, kwargs: Any, np: Any) -> Any:
    """Asymmetric BCE (mean) on Metal (MPSGraph); non-mean reductions fall back."""
    from . import _apple_gpu_backend as agb
    z = np.asarray(operands[0], dtype=np.float32)
    t = np.asarray(operands[1], dtype=np.float32)
    if kwargs.get("reduction", "mean") != "mean":
        import tessera.losses as _L
        return _L.asymmetric_bce(z, t, kwargs.get("pos_weight", 1.0),
                                 kwargs.get("neg_weight", 1.0),
                                 reduction=kwargs.get("reduction", "mean"))
    return np.float32(agb.gpu_asymmetric_bce(
        z, t, float(kwargs.get("pos_weight", 1.0)),
        float(kwargs.get("neg_weight", 1.0))))


def _apple_gpu_dispatch_ebm_loss(op_name: Any, operands: Any, kwargs: Any, np: Any) -> Any:
    """EBM training losses on Metal (MPSGraph reductions). Only reduction="mean"
    (the default) runs on GPU; sum/none fall back to the numpy reference.

      tessera.loss.contrastive_divergence / persistent_cd → mean(E⁺ − E⁻)
      tessera.loss.score_matching                        → ½·mean_all((s − t)²)
      tessera.loss.implicit_score_matching               → mean(½·Σ s² + div)
      tessera.loss.denoising_score_matching              → mean(½·Σ (s + (ỹ−y)/σ²)²)
    """
    from . import _apple_gpu_backend as agb
    import tessera.losses as _L
    short = str(op_name).rsplit(".", 1)[-1]
    reduction = kwargs.get("reduction", "mean")
    a = np.asarray(operands[0], dtype=np.float32)
    if short in ("contrastive_divergence", "persistent_cd"):
        b = np.asarray(operands[1], dtype=np.float32)
        if reduction != "mean":
            ref = (_L.contrastive_divergence_loss if short == "contrastive_divergence"
                   else _L.persistent_cd_loss)
            return ref(a, b, reduction=reduction)
        return np.float32(agb.gpu_ebm_energy_diff_mean(a, b))
    if short == "score_matching":
        b = np.asarray(operands[1], dtype=np.float32)
        if reduction != "mean":
            return _L.score_matching_loss(a, b, reduction=reduction)
        return np.float32(agb.gpu_ebm_half_mse(a, b))
    if short == "implicit_score_matching":
        div = np.asarray(operands[1], dtype=np.float32)
        if reduction != "mean" or a.ndim < 2:
            return _L.implicit_score_matching_loss(a, div, reduction=reduction)
        return np.float32(agb.gpu_ebm_ism(a, div))
    # denoising_score_matching
    yc = np.asarray(operands[1], dtype=np.float32)
    yn = np.asarray(operands[2], dtype=np.float32)
    sigma = float(kwargs["sigma"])
    if reduction != "mean" or a.ndim < 2:
        return _L.denoising_score_matching_loss(a, yc, yn, sigma, reduction=reduction)
    return np.float32(agb.gpu_ebm_dsm(a, yc, yn, 1.0 / (sigma * sigma)))


def _apple_gpu_dispatch_load_balance_loss(operands: Any, kwargs: Any, np: Any) -> Any:
    """Switch load-balance aux loss on Metal (MPSGraph). reduction="mean" + the
    default top-1 argmax assignment run on GPU; otherwise fall back."""
    from . import _apple_gpu_backend as agb
    p = np.asarray(operands[0], dtype=np.float32)
    if (kwargs.get("reduction", "mean") != "mean"
            or kwargs.get("assignment") is not None or p.ndim < 2):
        import tessera.losses as _L
        return _L.load_balance_loss(p, assignment=kwargs.get("assignment"),
                                    reduction=kwargs.get("reduction", "mean"))
    return np.float32(agb.gpu_load_balance_loss(p))


def _apple_gpu_dispatch_masked_categorical(operands: Any, kwargs: Any, np: Any) -> Any:
    """Greedy masked categorical on Metal (MPSGraph argMax). Only the greedy
    case (no rng key) runs on GPU; a keyed sample falls back to numpy."""
    from . import _apple_gpu_backend as agb
    logits = np.asarray(operands[0], dtype=np.float32)
    mask = np.asarray(operands[1], dtype=np.float32)
    if kwargs.get("key") is not None:
        import tessera as _ts
        return _ts.ops.masked_categorical(logits, mask, key=kwargs.get("key"),
                                          axis=kwargs.get("axis", -1))
    return agb.gpu_masked_categorical(logits, mask)


def _apple_gpu_dispatch_selective_ssm(operands: Any, kwargs: Any, np: Any) -> Any:
    """Mamba-2 ``selective_ssm`` on Apple GPU via the chunked-parallel SSD form
    (`_mamba_ssd.selective_ssm_parallel`), with its three batched contractions
    (state projection, C·Bᵀ gram, state update) dispatched to the Metal `bmm`
    lane. Scalar-state ``A`` (shape ``(D,)``) takes the parallel matmul form;
    general ``(D, N)`` ``A`` falls back to the sequential numpy reference."""
    from . import _mamba_ssd

    x, A, B, C, delta = (np.asarray(o) for o in operands[:5])
    gate = kwargs.get("gate")
    state = kwargs.get("state")
    chunk = int(kwargs.get("chunk_size", 64))

    if not _mamba_ssd.supports_parallel(A):
        import tessera as _ts
        return _ts.ops.selective_ssm(
            x, A, B, C, delta, gate=gate, state=state, chunk_size=chunk)

    def _gpu_bmm(p: Any, q: Any) -> Any:
        r = _apple_gpu_dispatch_bmm(p, q, np)
        return r if r is not None else np.matmul(p, q)

    return _mamba_ssd.selective_ssm_parallel(
        x, A, B, C, delta, gate=gate, state=state, chunk_size=chunk,
        matmul3d=_gpu_bmm)


def _apple_gpu_dispatch_bmm(a: Any, b: Any, np: Any) -> Any:
    """Batched / rank-3+ matmul via the MPSGraph bmm lane. A is [..., M, K];
    B is [..., K, N] (matching leading dims) or a shared/broadcast operand
    ([K, N], or leading dims all 1). Returns the result, or None when the shape
    or dtype isn't supported so the caller falls back to numpy. f32/f16 native;
    bf16 upcasts to f32 on-GPU then downcasts."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim < 3 or b.ndim < 2 or a.shape[-1] != b.shape[-2]:
        return None
    M = int(a.shape[-2])
    K = int(a.shape[-1])
    N = int(b.shape[-1])
    batch = 1
    for d in a.shape[:-2]:
        batch *= int(d)

    if b.ndim == 2:
        b_broadcast = True
    elif b.shape[:-2] == a.shape[:-2]:
        b_broadcast = False
    elif all(int(d) == 1 for d in b.shape[:-2]):
        b_broadcast = True
    else:
        return None  # mixed broadcast — let numpy handle it

    out_shape = tuple(a.shape[:-1]) + (N,)
    out_dtype = a.dtype
    bf16_dtype = _bfloat16_dtype()

    if a.dtype == np.float32 and b.dtype == np.float32:
        sym = _apple_gpu_bmm_f32()
        compute_dt, half = np.float32, False
    elif a.dtype == np.float16 and b.dtype == np.float16:
        sym = _apple_gpu_bmm_f16()
        compute_dt, half = np.float16, True
    elif (bf16_dtype is not None and a.dtype == bf16_dtype
          and b.dtype == bf16_dtype):
        sym = _apple_gpu_bmm_f32()  # upcast path
        compute_dt, half = np.float32, False
    else:
        return None
    if sym is None:
        return None

    a2 = np.ascontiguousarray(a.reshape(batch, M, K).astype(compute_dt))
    bbatch = 1 if b_broadcast else batch
    b2 = np.ascontiguousarray(b.reshape(bbatch, K, N).astype(compute_dt))
    out = np.zeros((batch, M, N), dtype=compute_dt)
    bc = ctypes.c_int32(1 if b_broadcast else 0)
    dims = (ctypes.c_int32(batch), ctypes.c_int32(M), ctypes.c_int32(N),
            ctypes.c_int32(K), bc)
    if half:
        up = lambda arr: arr.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        sym(up(a2), up(b2), up(out), *dims)
    else:
        fp = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        sym(fp(a2), fp(b2), fp(out), *dims)
    return out.reshape(out_shape).astype(out_dtype)


def _apple_gpu_dispatch_linear_general(operands: list[Any], kwargs: dict,
                                       np: Any) -> Any:
    """tessera.linear_general — axis-flexible linear projection. The common
    last-axis contraction with a rank-2 weight (x[..., K] @ W[K, N] (+ bias))
    routes through the GPU matmul / bmm lane; the general tensordot case falls
    back to a numpy reference."""
    if len(operands) < 2:
        raise ValueError("linear_general requires (x, W[, bias])")
    x = np.asarray(operands[0])
    W = np.asarray(operands[1])
    bias = np.asarray(operands[2]) if len(operands) > 2 else None
    axis = kwargs.get("axis", -1)

    if isinstance(axis, int):
        ax = axis if axis >= 0 else x.ndim + axis
        if ax == x.ndim - 1 and W.ndim == 2 and x.shape[-1] == W.shape[0]:
            y = np.asarray(
                _apple_gpu_dispatch_matmul("tessera.matmul", [x, W], np))
            return y if bias is None else (y + bias)
        axes = (ax,)
    else:
        axes = tuple(a if a >= 0 else x.ndim + a for a in axis)
    y = np.tensordot(x, W, axes=(axes, tuple(range(len(axes)))))
    return y if bias is None else (y + bias)


def _apple_gpu_dispatch_qkv_projection(operands: list[Any], np: Any) -> Any:
    """tessera.qkv_projection — y = x @ W_qkv, split into (Q, K, V) along the
    last axis. The matmul runs on the GPU matmul / bmm lane; the split is host
    glue."""
    if len(operands) < 2:
        raise ValueError("qkv_projection requires (x, W_qkv)")
    x = np.asarray(operands[0])
    W = np.asarray(operands[1])
    y = np.asarray(_apple_gpu_dispatch_matmul("tessera.matmul", [x, W], np))
    return tuple(np.split(y, 3, axis=-1))


def _apple_gpu_mpsgraph_reduce_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mpsgraph_reduce_f32", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32]
    sym.restype = None
    return sym


def _apple_gpu_gumbel_argmax_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_gumbel_argmax_f32", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
                    ctypes.c_int32, ctypes.c_float]
    sym.restype = None
    return sym


def _gumbel_noise_from_key(shape: tuple, key: Any, np: Any) -> Any:
    """Gumbel(0,1) noise g = -log(-log(u)) from the canonical Philox stream, so
    sampling is deterministic + reproducible (and bit-exact vs a CPU reference)
    without an on-GPU RNG. ``key`` is a ``tessera.rng.RNGKey``; if None, a
    seed-0 key is used."""
    from . import rng as _rng
    if key is None:
        key = _rng.RNGKey.from_seed(0)
    u = np.asarray(_rng.uniform(key, shape, dtype="fp32"))
    u = np.clip(u, 1e-9, 1.0 - 1e-7).astype(np.float32)
    return (-np.log(-np.log(u))).astype(np.float32)


def _apply_topk_topp_mask(logits: Any, top_k: int, top_p: float, np: Any) -> Any:
    """Mask logits to -inf outside the top-k / top-p (nucleus) set, per row.
    Operates on a [rows, vocab] f32 copy."""
    out = logits.astype(np.float32, copy=True)
    neg_inf = np.float32(-1e30)
    if top_k and top_k > 0 and top_k < out.shape[-1]:
        # keep the top_k largest per row; threshold = k-th largest
        kth = np.partition(out, -top_k, axis=-1)[:, -top_k][:, None]
        out = np.where(out < kth, neg_inf, out)
    if top_p and 0.0 < top_p < 1.0:
        order = np.argsort(-out, axis=-1)
        sorted_logits = np.take_along_axis(out, order, axis=-1)
        m = sorted_logits.max(-1, keepdims=True)
        probs = np.exp(sorted_logits - m)
        probs /= probs.sum(-1, keepdims=True)
        cum = np.cumsum(probs, axis=-1)
        # keep tokens up to and including the one that crosses top_p
        keep = cum - probs <= top_p
        keep[:, 0] = True  # always keep the most probable token
        mask_sorted = np.where(keep, sorted_logits, neg_inf)
        out = np.empty_like(out)
        np.put_along_axis(out, order, mask_sorted, axis=-1)
    return out


def _apple_gpu_gumbel_sample(logits: Any, np: Any, *, key: Any = None,
                             temperature: float = 1.0, top_k: int = 0,
                             top_p: float = 0.0, greedy: bool = False) -> Any:
    """GPU Gumbel-max categorical sampler — draws one token id per row of
    ``logits`` ``[..., vocab]``.

    ``argmax(logits/T + g)`` with Gumbel noise ``g`` (from the Philox ``key``)
    is an exact draw from ``softmax(logits/T)``; the per-row argmax over the
    vocab runs on-GPU (the throughput win for batched sampling). ``greedy=True``
    (or ``temperature==0``) returns the plain argmax. ``top_k`` / ``top_p``
    restrict the candidate set (host-side mask). Reproducible: same ``key`` +
    logits ⇒ same tokens. Returns int64 ids shaped like the leading dims of
    ``logits``; falls back to numpy when the GPU symbol is unavailable."""
    arr = np.asarray(logits, dtype=np.float32)
    lead = arr.shape[:-1]
    vocab = int(arr.shape[-1])
    rows2d = arr.reshape(-1, vocab)
    rows = int(rows2d.shape[0])

    masked = _apply_topk_topp_mask(rows2d, top_k, top_p, np)
    if greedy or temperature == 0.0:
        gumbel = np.zeros((rows, vocab), np.float32)
        inv_temp = 1.0
    else:
        gumbel = _gumbel_noise_from_key((rows, vocab), key, np)
        inv_temp = 1.0 / float(temperature)

    sym = _apple_gpu_gumbel_argmax_f32()
    masked = np.ascontiguousarray(masked, np.float32)
    gumbel = np.ascontiguousarray(gumbel, np.float32)
    if sym is not None:
        out = np.zeros(rows, np.int32)
        fp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        sym(fp(masked), fp(gumbel), out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(rows), ctypes.c_int32(vocab), ctypes.c_float(inv_temp))
        ids = out.astype(np.int64)
    else:
        ids = np.argmax(masked * inv_temp + gumbel, axis=-1).astype(np.int64)
    return ids.reshape(lead) if lead else ids.reshape(())


def _apple_gpu_mpsgraph_argreduce_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mpsgraph_argreduce_f32", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32]
    sym.restype = None
    return sym


def _apple_gpu_mpsgraph_scan_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mpsgraph_scan_f32", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32]
    sym.restype = None
    return sym


def _apple_gpu_dispatch_reduce(op_name: str, operands: list[Any], kwargs: dict,
                               np: Any) -> Any:
    """Reductions / scans via the MPSGraph lane. Arbitrary axis/keepdims are
    folded to a [rows, cols] last-axis reduction by transposing the reduced
    axes to the end. f16/bf16 upcast to f32 (fp32 reduction numerics). Falls
    back to numpy for non-float dtypes or when Metal is unavailable."""
    x = np.asarray(operands[0])
    kind, op = _APPLE_GPU_REDUCE_OPS[op_name]
    axis = kwargs.get("axis", -1 if kind == "scan" else None)
    keepdims = bool(kwargs.get("keepdims", False))
    ddof = int(kwargs.get("ddof", 0))
    out_dtype = x.dtype
    bf16 = _bfloat16_dtype()
    is_float = (x.dtype in (np.float32, np.float16)
                or (bf16 is not None and x.dtype == bf16))
    fp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def _ref() -> Any:
        if kind == "reduce":
            f = {0: np.sum, 1: np.mean, 2: np.amax, 3: np.amin, 4: np.prod}.get(op)
            if f is not None:
                return f(x, axis=axis, keepdims=keepdims)
            if op == 5:
                return np.var(x, axis=axis, keepdims=keepdims, ddof=ddof)
            if op == 7:  # logsumexp = log(Σ exp(x − max)) + max  (stable)
                m = np.max(x, axis=axis, keepdims=True)
                r = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
                return r if keepdims else np.squeeze(
                    r, axis=(tuple(range(x.ndim)) if axis is None else axis))
            return np.std(x, axis=axis, keepdims=keepdims, ddof=ddof)
        if kind == "arg":
            r = (np.argmax if op == 0 else np.argmin)(x, axis=axis)
            return np.expand_dims(r, axis) if (keepdims and axis is not None) else r
        scan_fn = {0: np.cumsum, 1: np.cumprod,
                   2: np.maximum.accumulate, 3: np.minimum.accumulate}[op]
        return scan_fn(x, axis=axis)

    if not is_float or x.ndim == 0:
        return _ref()
    xf = np.ascontiguousarray(x.astype(np.float32))
    n = x.ndim

    if kind == "scan":
        ax = axis if axis is not None else -1
        ax = ax if ax >= 0 else n + ax
        perm = [i for i in range(n) if i != ax] + [ax]
        inv = list(np.argsort(perm))
        xt = np.ascontiguousarray(np.transpose(xf, perm))
        inner = int(xt.shape[-1])
        outer = int(xt.size // inner) if inner else 0
        sym = _apple_gpu_mpsgraph_scan_f32()
        if sym is None:
            return _ref()
        out = np.empty((outer, inner), np.float32)
        sym(ctypes.c_int32(op), fp(xt.reshape(outer, inner)), fp(out),
            ctypes.c_int32(outer), ctypes.c_int32(inner))
        res = np.transpose(out.reshape(xt.shape), inv)
        return res.astype(out_dtype)

    # reduce / arg: normalize the reduced axes.
    if axis is None:
        axes = tuple(range(n))
    elif isinstance(axis, int):
        axes = (axis if axis >= 0 else n + axis,)
    else:
        axes = tuple(a if a >= 0 else n + a for a in axis)
    if kind == "arg" and len(axes) != 1 and axis is not None:
        return _ref()
    kept = [i for i in range(n) if i not in axes]
    perm = kept + list(axes)
    xt = np.ascontiguousarray(np.transpose(xf, perm))
    inner = 1
    for a in axes:
        inner *= int(x.shape[a])
    outer = int(xt.size // inner) if inner else 0
    kept_shape = tuple(int(x.shape[i]) for i in kept)

    if kind == "reduce":
        sym = _apple_gpu_mpsgraph_reduce_f32()
        if sym is None:
            return _ref()
        out = np.empty(max(outer, 1), np.float32)
        sym(ctypes.c_int32(op), fp(xt.reshape(outer, inner)), fp(out),
            ctypes.c_int32(outer), ctypes.c_int32(inner))
        if op in (5, 6) and ddof != 0 and inner > ddof:
            factor = inner / (inner - ddof)
            out = out * (factor if op == 5 else np.sqrt(factor))
        res = out.reshape(kept_shape) if kept_shape else out.reshape(())
        if keepdims:
            full = [int(s) for s in x.shape]
            for a in axes:
                full[a] = 1
            res = res.reshape(full)
        return res.astype(out_dtype)

    # arg
    sym = _apple_gpu_mpsgraph_argreduce_f32()
    if sym is None:
        return _ref()
    out = np.empty(max(outer, 1), np.int32)
    sym(ctypes.c_int32(op),
        fp(np.ascontiguousarray(xt.reshape(outer, inner))),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(outer), ctypes.c_int32(inner))
    res = out.astype(np.int64)
    res = res.reshape(kept_shape) if kept_shape else res.reshape(())
    if keepdims and axis is not None:
        full = [int(s) for s in x.shape]
        for a in axes:
            full[a] = 1
        res = res.reshape(full)
    return res


def _apple_gpu_conv2d_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_conv2d_f32", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.POINTER(ctypes.c_float)] * 4 + [ctypes.c_int32] * 14
    sym.restype = None
    return sym


def _apple_gpu_conv2d_f16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_conv2d_f16", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.POINTER(ctypes.c_uint16)] * 4 + [ctypes.c_int32] * 14
    sym.restype = None
    return sym


# P8 — conv2d on the Metal 4 matrix units via im2col + the matmul2d epilogue.
# A KxK conv equals im2col(activation) @ weights_reshaped, and conv's per-output-
# channel bias is exactly the epilogue's per-column bias — so a `conv → bias →
# activation` block collapses to one fused matmul2d dispatch on the cooperative
# tensor units (f16/bf16). This is the robust arbitrary-size conv lane; the native
# MPP `convolution2d` cooperative op is verified single-tile but its multi-tile
# grid convention is undocumented (see docs/apple_backend_integration_review.md P8).
# OPT-IN, OFF by default. The unfold now runs ON the GPU (an MSL im2col kernel —
# `apple_gpu_conv2d` prefers the on-device `tessera_apple_gpu_mtl4_conv2d_*`
# symbol, host im2col only as a fallback), and the matmul runs on the matrix
# units (f16/bf16, fused bias+act). But this still loses to MPSGraph's *fused*
# conv (~0.55-0.77x): materializing the `col` matrix is extra memory traffic that
# a fused/direct/Winograd conv avoids — so unlike P5 matmul (where MPS had no
# bf16 GEMM), conv has no easy default-win. A default win needs the native MPP
# `convolution2d` cooperative op (no col materialization; its multi-tile tiling
# is undocumented) or a direct conv. Toggle: TESSERA_APPLE_GPU_MTL4_CONV=1 /
# set_apple_gpu_mtl4_conv_routing(True).
_MTL4_CONV_ROUTE = os.environ.get(
    "TESSERA_APPLE_GPU_MTL4_CONV", "0") in ("1", "true", "True")


def set_apple_gpu_mtl4_conv_routing(enabled: bool) -> None:
    global _MTL4_CONV_ROUTE
    _MTL4_CONV_ROUTE = bool(enabled)


def apple_gpu_mtl4_conv_routing_enabled() -> bool:
    return _MTL4_CONV_ROUTE


def _im2col_nhwc(X: Any, kH: int, kW: int, sH: int, sW: int,
                 pH: int, pW: int, dH: int, dW: int, np: Any):
    """Vectorized NHWC im2col → (col[N*OH*OW, kH*kW*Cin], OH, OW). Zero-pads by
    (pH, pW); supports stride and dilation. Patch channel order is (kH, kW, Cin)
    to match a weights reshape of HWIO → [kH*kW*Cin, Cout]."""
    N, H, W, Cin = X.shape
    if pH or pW:
        X = np.pad(X, ((0, 0), (pH, pH), (pW, pW), (0, 0)))
    spanH, spanW = dH * (kH - 1) + 1, dW * (kW - 1) + 1
    win = np.lib.stride_tricks.sliding_window_view(X, (spanH, spanW), axis=(1, 2))
    # win: [N, OHf, OWf, Cin, spanH, spanW] -> stride on output, dilation on taps.
    win = win[:, ::sH, ::sW, :, ::dH, ::dW]               # [N,OH,OW,Cin,kH,kW]
    N, OH, OW, Cin, kh, kw = win.shape
    col = np.transpose(win, (0, 1, 2, 4, 5, 3)).reshape(N * OH * OW, kh * kw * Cin)
    return np.ascontiguousarray(col), OH, OW


def _apple_gpu_mtl4_conv2d_sym(dtype: str) -> Any:
    """P8 on-device conv symbol (GPU im2col + matmul2d epilogue). None if absent."""
    runtime = _load_apple_gpu_runtime()
    name = "tessera_apple_gpu_mtl4_conv2d_bf16" if dtype == "bf16" else "tessera_apple_gpu_mtl4_conv2d_f16"
    sym = getattr(runtime, name, None)
    if sym is None:
        return None
    u16 = ctypes.POINTER(ctypes.c_uint16)
    fp = ctypes.POINTER(ctypes.c_float)
    i32 = ctypes.c_int32
    sym.argtypes = [u16, u16, fp, fp, i32] + [i32] * 13
    sym.restype = i32
    return sym


def apple_gpu_conv2d(X: Any, W: Any, np: Any, *, bias: Any = None, act: str = "none",
                     stride=1, padding=0, dilation=1, dtype: str = "f16"):
    """P8 — 2-D convolution on the GPU matrix units via im2col + the fused
    ``matmul2d`` epilogue. ``X`` is NHWC, ``W`` is HWIO ``[kH,kW,Cin,Cout]``,
    ``bias`` is ``[Cout]``, ``act`` in {none,relu,gelu,silu}. groups=1 only.
    Prefers the on-device path (GPU im2col, ``col`` never leaves the GPU); falls
    back to a host im2col + the epilogue when the on-device symbol didn't run.
    Returns ``(Y[N,OH,OW,Cout] float32, ran_on_mtl4)``; numpy fallback off Metal 4.
    See docs/apple_backend_integration_review.md (P8)."""
    def _pair(v):
        return (int(v[0]), int(v[1])) if isinstance(v, (tuple, list)) else (int(v), int(v))
    sH, sW = _pair(stride); pH, pW = _pair(padding); dH, dW = _pair(dilation)
    N = int(X.shape[0]); kH, kW, Cin, Cout = (int(s) for s in W.shape)
    H, Wd = int(X.shape[1]), int(X.shape[2])
    OH = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    OW = (Wd + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    act_code = _MTL4_EPILOGUE_ACT.get(act)
    if act_code is None:
        raise ValueError(f"act must be one of {sorted(_MTL4_EPILOGUE_ACT)}; got {act!r}")
    in_dtype = _bfloat16_dtype() if dtype == "bf16" else np.float16
    if in_dtype is None:
        raise RuntimeError("bf16 conv requires the optional ml_dtypes package")
    Xc = np.ascontiguousarray(X, in_dtype)
    Wr = np.ascontiguousarray(np.asarray(W, in_dtype).reshape(kH * kW * Cin, Cout))
    bias_f = np.ascontiguousarray(bias, np.float32).reshape(-1) if bias is not None else None

    # On-device path: GPU im2col → matmul2d epilogue (col stays on the GPU).
    sym = _apple_gpu_mtl4_conv2d_sym(dtype)
    if sym is not None:
        u16 = ctypes.POINTER(ctypes.c_uint16)
        fp = ctypes.POINTER(ctypes.c_float)
        Y = np.empty((N * OH * OW, Cout), np.float32)
        rc = sym(Xc.view(np.uint16).ctypes.data_as(u16),
                 Wr.view(np.uint16).ctypes.data_as(u16),
                 bias_f.ctypes.data_as(fp) if bias_f is not None else None,
                 Y.ctypes.data_as(fp), ctypes.c_int32(act_code),
                 *[ctypes.c_int32(v) for v in
                   (N, H, Wd, Cin, Cout, kH, kW, sH, sW, pH, pW, dH, dW)])
        if rc == 1:
            return Y.reshape(N, OH, OW, Cout), True

    # Fallback: host im2col + the epilogue (also numpy-falls-back off Metal 4).
    col, OH2, OW2 = _im2col_nhwc(np.asarray(X, np.float32), kH, kW, sH, sW, pH, pW, dH, dW, np)
    Y, ran = apple_gpu_mtl4_matmul2d_epilogue(col, Wr, np, bias=bias_f, act=act, dtype=dtype)
    return Y.reshape(N, OH2, OW2, Cout), ran


# ── GPU linear-algebra lane — Cholesky / LU / triangular solve via MPSMatrix ──
# The one capability MPSGraph cannot provide (it has no matrix-decomposition
# ops), so these dense f32 factorizations/solves are the only GPU path for
# tessera.ops.{cholesky, solve, cholesky_solve, tri_solve}. Rank-2 f32 runs as a
# single GPU dispatch; batched (ndim>2) f32 loops the rank-2 kernel per matrix
# (MPS decomposition/solve are single-matrix per encode — no native batch);
# non-f32 / off-Metal inputs fall back to the numpy reference. Each returns
# (result, ran_on_gpu). See docs/apple_backend_integration_review.md (linalg).

def _apple_gpu_linalg_sym(name: str, argtypes: list) -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, name, None)
    if sym is None:
        return None
    sym.argtypes = argtypes
    sym.restype = ctypes.c_int32
    return sym


def _apple_gpu_chol_2d(A2d: Any, np: Any) -> Any:
    """GPU lower-Cholesky of one 2-D f32 matrix; None if it didn't run."""
    fp = ctypes.POINTER(ctypes.c_float)
    sym = _apple_gpu_linalg_sym("tessera_apple_gpu_cholesky_f32", [fp, fp, ctypes.c_int32])
    if sym is None:
        return None
    n = int(A2d.shape[0])
    L = np.empty((n, n), np.float32)
    rc = sym(np.ascontiguousarray(A2d, np.float32).ctypes.data_as(fp),
             L.ctypes.data_as(fp), ctypes.c_int32(n))
    return L if rc == 0 else None


def _apple_gpu_solve_2d(A2d: Any, B2d: Any, np: Any, symname: str) -> Any:
    """GPU full solve ``A·X = B`` for one 2-D matrix; None if it didn't run."""
    fp = ctypes.POINTER(ctypes.c_float)
    sym = _apple_gpu_linalg_sym(symname, [fp, fp, fp, ctypes.c_int32, ctypes.c_int32])
    if sym is None:
        return None
    n, nrhs = int(A2d.shape[0]), int(B2d.shape[1])
    X = np.empty((n, nrhs), np.float32)
    rc = sym(np.ascontiguousarray(A2d, np.float32).ctypes.data_as(fp),
             np.ascontiguousarray(B2d, np.float32).ctypes.data_as(fp),
             X.ctypes.data_as(fp), ctypes.c_int32(n), ctypes.c_int32(nrhs))
    return X if rc == 0 else None


def _apple_gpu_tri_2d(A2d: Any, B2d: Any, np: Any, lower: bool, trans: bool,
                      unit: bool) -> Any:
    """GPU triangular solve for one 2-D matrix; None if it didn't run."""
    fp = ctypes.POINTER(ctypes.c_float)
    i32 = ctypes.c_int32
    sym = _apple_gpu_linalg_sym("tessera_apple_gpu_tri_solve_f32", [fp, fp, fp] + [i32] * 5)
    if sym is None:
        return None
    n, nrhs = int(A2d.shape[0]), int(B2d.shape[1])
    X = np.empty((n, nrhs), np.float32)
    rc = sym(np.ascontiguousarray(A2d, np.float32).ctypes.data_as(fp),
             np.ascontiguousarray(B2d, np.float32).ctypes.data_as(fp),
             X.ctypes.data_as(fp), i32(n), i32(nrhs),
             i32(1 if lower else 0), i32(1 if trans else 0), i32(1 if unit else 0))
    return X if rc == 0 else None


def _apple_gpu_batched_linalg(A2: Any, B2: Any, core: Any, out_tail_cols: int,
                              np: Any) -> Any:
    """Loop a rank-2 GPU linalg ``core(A2d, B2d_or_None) -> ndarray|None`` over the
    leading batch dims of ``A2`` (``B2`` shares them). Returns the stacked result
    reshaped to ``A2.shape[:-2] + (n, out_tail_cols)``, or None if any slice
    didn't run on the GPU (caller numpy-falls-back the whole batch)."""
    n = int(A2.shape[-1])
    Af = A2.reshape(-1, n, n)
    Bf = None if B2 is None else B2.reshape(Af.shape[0], n, int(B2.shape[-1]))
    outs = []
    for k in range(Af.shape[0]):
        r = core(Af[k], None if Bf is None else Bf[k])
        if r is None:
            return None
        outs.append(r)
    return np.stack(outs).reshape(A2.shape[:-2] + (n, out_tail_cols)).astype(np.float32)


def _linalg_dtype_policy(A: Any, np: Any) -> Any:
    """Dtype policy for the GPU linalg lane. MPS decomposition/solve are f32-only,
    so f16/bf16 inputs run on the GPU *in f32* and the result is cast back to the
    input dtype (the bf16-matmul pattern); f32 is native; f64 (or anything
    needing >f32 precision) skips the GPU and computes on numpy in f64 so we never
    silently lose precision. Returns ``(out_dtype, gpu_ok, fallback_dtype)``."""
    dt = np.asarray(A).dtype
    bf16 = _bfloat16_dtype()
    # ml_dtypes.bfloat16 is a numpy *extension* dtype — np.issubdtype(.., floating)
    # is False for it, so check it explicitly to preserve bf16 on output.
    is_float = np.issubdtype(dt, np.floating) or (bf16 is not None and dt == bf16)
    out_dt = dt if is_float else np.float32
    gpu_ok = dt != np.float64  # f64 -> numpy (GPU would downcast to f32)
    fb_dt = np.float64 if dt == np.float64 else np.float32
    return out_dt, gpu_ok, fb_dt


def _apple_gpu_cholesky_batched_msl(A2: Any, np: Any) -> Any:
    """Batched (…, n, n) Cholesky via the grid MSL kernel (one threadgroup per
    matrix). ``A2`` is f32 contiguous. Returns L (same leading dims) or None
    (kernel unavailable, or any matrix not positive-definite → let numpy raise)."""
    sym = _apple_gpu_linalg_sym(
        "tessera_apple_gpu_cholesky_batched_f32",
        [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
         ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32])
    if sym is None:
        return None
    n = int(A2.shape[-1])
    lead = A2.shape[:-2]
    B = 1
    for d in lead:
        B *= int(d)
    fp = ctypes.POINTER(ctypes.c_float)
    ip = ctypes.POINTER(ctypes.c_int32)
    flat = np.ascontiguousarray(A2.reshape(B, n, n))
    L = np.empty((B, n, n), np.float32)
    st = np.zeros(B, np.int32)
    rc = sym(flat.ctypes.data_as(fp), L.ctypes.data_as(fp), st.ctypes.data_as(ip),
             ctypes.c_int32(B), ctypes.c_int32(n))
    if rc != 1 or bool(st.any()):     # not-PD anywhere -> numpy fallback raises
        return None
    return L.reshape(A2.shape)


def _apple_gpu_tri_solve_batched_msl(A2: Any, B2: Any, np: Any, *, lower: bool,
                                     trans: bool, unit: bool) -> Any:
    """Batched (…, n, n) / (…, n, nrhs) triangular solve via the grid MSL kernel.
    ``A2``/``B2`` are f32 contiguous with matching leading dims. Returns X (leading
    dims + (n, nrhs)) or None."""
    sym = _apple_gpu_linalg_sym(
        "tessera_apple_gpu_tri_solve_batched_f32",
        [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int32] * 6)
    if sym is None:
        return None
    n = int(A2.shape[-1])
    nrhs = int(B2.shape[-1])
    lead = A2.shape[:-2]
    B = 1
    for d in lead:
        B *= int(d)
    fp = ctypes.POINTER(ctypes.c_float)
    i32 = ctypes.c_int32
    Af = np.ascontiguousarray(A2.reshape(B, n, n))
    Bf = np.ascontiguousarray(B2.reshape(B, n, nrhs))
    X = np.empty((B, n, nrhs), np.float32)
    rc = sym(Af.ctypes.data_as(fp), Bf.ctypes.data_as(fp), X.ctypes.data_as(fp),
             i32(B), i32(n), i32(nrhs), i32(1 if lower else 0),
             i32(1 if trans else 0), i32(1 if unit else 0))
    return X.reshape(lead + (n, nrhs)) if rc == 1 else None


def apple_gpu_cholesky(A: Any, np: Any) -> Any:
    """Lower Cholesky factor ``L`` of an SPD matrix ``A`` (``A = L·Lᵀ``), matching
    ``numpy.linalg.cholesky``. f16/bf16/f32 on the GPU (f16/bf16 compute in f32 and
    cast back; rank-2 = one MPS dispatch, batched = a single grid MSL dispatch,
    one threadgroup per matrix); f64 + fallbacks on numpy. Returns
    ``(L, ran_on_gpu)`` with ``L`` in the input float dtype. Raises
    ``numpy.linalg.LinAlgError`` (via the fallback) if ``A`` is not PD."""
    out_dt, gpu_ok, fb_dt = _linalg_dtype_policy(A, np)
    A2 = np.ascontiguousarray(A, np.float32)
    if gpu_ok and A2.ndim >= 2 and A2.shape[-1] == A2.shape[-2]:
        if A2.ndim == 2:
            L = _apple_gpu_chol_2d(A2, np)                  # MPS single matrix
            if L is not None:
                return L.astype(out_dt), True
        else:
            out = _apple_gpu_cholesky_batched_msl(A2, np)   # grid MSL (≫ per-matrix)
            if out is not None:
                return out.astype(out_dt), True
    return np.linalg.cholesky(np.asarray(A, fb_dt)).astype(out_dt), False


def _apple_gpu_solve_impl(A: Any, B: Any, np: Any, symname: str) -> Any:
    """Shared body for the Cholesky/LU full-solve wrappers (``A·X = B``)."""
    out_dt, gpu_ok, fb_dt = _linalg_dtype_policy(A, np)
    A2 = np.ascontiguousarray(A, np.float32)
    B1 = np.ascontiguousarray(np.asarray(B, np.float32))
    vec = B1.ndim == A2.ndim - 1           # stacked-vector RHS (numpy convention)
    B2 = B1[..., None] if vec else B1
    core = lambda a, b: _apple_gpu_solve_2d(a, b, np, symname)
    if gpu_ok and (A2.ndim == 2 and A2.shape[0] == A2.shape[1]
            and B2.ndim == 2 and B2.shape[0] == A2.shape[0]):
        X = core(A2, B2)
        if X is not None:
            return (X[..., 0] if vec else X).astype(out_dt), True
    elif gpu_ok and (A2.ndim >= 3 and A2.shape[-1] == A2.shape[-2]
          and B2.ndim == A2.ndim and B2.shape[:-1] == A2.shape[:-1]):
        out = _apple_gpu_batched_linalg(A2, B2, core, int(B2.shape[-1]), np)
        if out is not None:
            return (out[..., 0] if vec else out).astype(out_dt), True
    # B2 (always […, n, nrhs]) keeps np.linalg.solve well-defined for batched
    # vector RHS across numpy 1.x/2.x; squeeze the trailing axis back off.
    Af = np.asarray(A, fb_dt)
    Bf = np.asarray(B, fb_dt)
    Bf = Bf[..., None] if vec else Bf
    X = np.linalg.solve(Af, Bf)
    return (X[..., 0] if vec else X).astype(out_dt), False


def apple_gpu_solve(A: Any, B: Any, np: Any) -> Any:
    """General linear solve ``A·X = B`` via LU with partial pivoting, matching
    ``numpy.linalg.solve``. f32 on the GPU (rank-2 + batched); else numpy. ``B``
    may be a vector ``[…, n]`` or matrix ``[…, n, nrhs]``. Returns
    ``(X float32, ran_on_gpu)``."""
    return _apple_gpu_solve_impl(A, B, np, "tessera_apple_gpu_solve_lu_f32")


def apple_gpu_cholesky_solve(A: Any, B: Any, np: Any) -> Any:
    """SPD linear solve ``A·X = B`` via Cholesky factorization (faster + more
    stable than LU for symmetric-positive-definite ``A``). Rank-2 uses the MPS
    factor+solve; **batched runs the grid MSL Cholesky + two grid MSL triangular
    solves** (``L·Y = B`` then ``Lᵀ·X = Y``) — all single grid dispatches. f64 /
    not-PD fall back to numpy. Returns ``(X, ran_on_gpu)``."""
    out_dt, gpu_ok, _ = _linalg_dtype_policy(A, np)
    A2 = np.ascontiguousarray(A, np.float32)
    B1 = np.ascontiguousarray(np.asarray(B, np.float32))
    if (gpu_ok and A2.ndim >= 3 and A2.shape[-1] == A2.shape[-2]):
        vec = B1.ndim == A2.ndim - 1
        B2 = B1[..., None] if vec else B1
        if B2.ndim == A2.ndim and B2.shape[:-1] == A2.shape[:-1]:
            L = _apple_gpu_cholesky_batched_msl(A2, np)            # A = L·Lᵀ
            if L is not None:
                Y = _apple_gpu_tri_solve_batched_msl(L, B2, np, lower=True,
                                                     trans=False, unit=False)
                X = (None if Y is None else
                     _apple_gpu_tri_solve_batched_msl(L, Y, np, lower=True,
                                                      trans=True, unit=False))
                if X is not None:
                    return (X[..., 0] if vec else X).astype(out_dt), True
    return _apple_gpu_solve_impl(A, B, np, "tessera_apple_gpu_solve_cholesky_f32")


def apple_gpu_tri_solve(A: Any, B: Any, np: Any, *, lower: bool = True,
                        trans: bool = False, unit: bool = False) -> Any:
    """Triangular solve ``op(tri(A))·X = B``. ``tri(A)`` is the lower (``lower``)
    or upper triangle of ``A``; ``op`` is transpose when ``trans``; the diagonal
    is unit when ``unit``. Only the relevant triangle is read (matching
    ``np.tril``/``np.triu`` + ``np.linalg.solve``). f32 on the GPU (rank-2 +
    batched). f16/bf16 compute in f32 and cast back; f64 stays on numpy. ``B`` may
    be ``[…, n]`` or ``[…, n, nrhs]``. Returns ``(X, ran_on_gpu)`` in the input
    float dtype."""
    out_dt, gpu_ok, fb_dt = _linalg_dtype_policy(A, np)
    A2 = np.ascontiguousarray(A, np.float32)
    B1 = np.ascontiguousarray(np.asarray(B, np.float32))
    vec = B1.ndim == A2.ndim - 1
    B2 = B1[..., None] if vec else B1
    core = lambda a, b: _apple_gpu_tri_2d(a, b, np, lower, trans, unit)
    if gpu_ok and (A2.ndim == 2 and A2.shape[0] == A2.shape[1]
            and B2.ndim == 2 and B2.shape[0] == A2.shape[0]):
        X = core(A2, B2)
        if X is not None:
            return (X[..., 0] if vec else X).astype(out_dt), True
    elif gpu_ok and (A2.ndim >= 3 and A2.shape[-1] == A2.shape[-2]
          and B2.ndim == A2.ndim and B2.shape[:-1] == A2.shape[:-1]):
        out = _apple_gpu_tri_solve_batched_msl(A2, B2, np, lower=lower, trans=trans,
                                               unit=unit)        # grid MSL kernel
        if out is not None:
            return (out[..., 0] if vec else out).astype(out_dt), True
    Af = np.asarray(A, fb_dt)
    Bf = np.asarray(B, fb_dt)
    Bf = Bf[..., None] if vec else Bf
    tri = np.tril(Af) if lower else np.triu(Af)
    if unit:
        tri = tri.copy()
        idx = np.arange(int(Af.shape[-1]))
        tri[..., idx, idx] = 1.0
    if trans:
        tri = np.swapaxes(tri, -1, -2)
    X = np.linalg.solve(tri, Bf)
    return (X[..., 0] if vec else X).astype(out_dt), False


def apple_gpu_qr(A: Any, np: Any) -> Any:
    """Reduced QR factorization ``A = Q·R`` (``Q`` m×n orthonormal columns, ``R``
    n×n upper-triangular) for a tall/square matrix (m ≥ n) — **GPU-resident via
    Cholesky-QR**, reusing the GPU Cholesky + triangular-solve kernels (MPS has no
    QR kernel): ``G = AᵀA``, ``R = chol(G)ᵀ`` (upper, positive diagonal), ``Q =
    A·R⁻¹`` (one lower-triangular solve). f16/bf16 compute in f32 and cast back.

    Cholesky-QR is fast but loses ~κ(A)² accuracy. The result is **verified**
    (``‖QᵀQ − I‖`` checked); if it fails the orthonormality tolerance — or the
    Gram isn't positive-definite, or the dtype is f64 — it **falls back to numpy's
    Householder QR**. So the returned ``Q`` is always orthonormal. Returns
    ``(Q, R, ran_on_gpu)``. Validate by reconstruction (``Q·R ≈ A``) /
    orthonormality (``QᵀQ ≈ I``), not elementwise vs numpy — QR is unique only up
    to column signs and this returns the positive-R-diagonal variant."""
    out_dt, gpu_ok, fb_dt = _linalg_dtype_policy(A, np)
    A2 = np.ascontiguousarray(A, np.float32)
    if gpu_ok and A2.ndim == 2 and A2.shape[0] >= A2.shape[1] >= 1:
        try:
            n = int(A2.shape[1])
            G = A2.T @ A2                            # n×n Gram (small)
            L, ran_c = apple_gpu_cholesky(G, np)     # G = L·Lᵀ, L lower
            if ran_c:
                R = np.ascontiguousarray(L.T)        # upper-tri, positive diag
                # Q·R = A  ⟺  L·Qᵀ = Aᵀ  (L = Rᵀ lower-triangular)
                Qt, ran_s = apple_gpu_tri_solve(L, A2.T, np, lower=True)
                if ran_s:
                    Q = np.ascontiguousarray(Qt.T)
                    # Verify orthonormality — Cholesky-QR degrades as κ(A)², and a
                    # barely-PD Gram still factors but yields a non-orthonormal Q.
                    ortho_err = float(np.abs(Q.T @ Q - np.eye(n, dtype=np.float32)).max())
                    if ortho_err < 1e-3:
                        return Q.astype(out_dt), R.astype(out_dt), True
        except np.linalg.LinAlgError:
            pass  # Gram not PD (ill-conditioned / rank-deficient) -> Householder
    Q, R = np.linalg.qr(np.asarray(A, fb_dt))
    return Q.astype(out_dt), R.astype(out_dt), False


def apple_gpu_svd(A: Any, np: Any, *, full_matrices: bool = False) -> Any:
    """Reduced singular value decomposition ``A = U·diag(S)·Vh`` (``U`` m×n, ``S``
    n descending, ``Vh`` n×n) — **GPU-resident via a custom one-sided Jacobi MSL
    kernel** (MPS has no SVD/eigensolver). Rotates column pairs to mutual
    orthogonality, then σ = column norms, U = normalized columns, V = accumulated
    rotations. f16/bf16 compute in f32 and cast back.

    The GPU path covers ``full_matrices=False`` for **any 2-D or batched (`…,m,n`)**
    shape — tall (``m≥n``) runs directly; **wide (``m<n``) runs on ``Aᵀ`` with U/V
    swapped**; **batched runs one threadgroup per matrix in a single grid dispatch**
    (whole-GPU utilization, ~30–95× a per-matrix loop). The result is **verified**
    (``‖U·Σ·Vh − A‖`` per matrix); on failure — or for ``full_matrices=True`` or
    f64 — it **falls back to numpy**. f16/bf16 compute in f32. Returns
    ``(U, S, Vh, ran_on_gpu)``. Validate by reconstruction / orthonormality, not
    elementwise (SVD is unique only up to signs of paired singular vectors)."""
    out_dt, gpu_ok, fb_dt = _linalg_dtype_policy(A, np)
    A2 = np.ascontiguousarray(A, np.float32)
    if gpu_ok and not full_matrices and A2.ndim >= 2:
        m, n = int(A2.shape[-2]), int(A2.shape[-1])
        transpose = m < n                       # run SVD(Aᵀ) for wide A, swap U/V
        Aw = np.ascontiguousarray(np.swapaxes(A2, -1, -2)) if transpose else A2
        r, c = int(Aw.shape[-2]), int(Aw.shape[-1])   # r >= c
        lead = Aw.shape[:-2]
        Bn = 1
        for d in lead:
            Bn *= int(d)
        # Brent–Luk (parallel tournament) is ~2–4× the sequential cyclic kernel
        # and caps at N ≤ 256 (perm in threadgroup memory); larger falls back to
        # the sequential batched kernel. Both share the same ABI + verify path.
        symname = ("tessera_apple_gpu_svd_bl_batched_f32" if c <= 256
                   else "tessera_apple_gpu_svd_batched_f32")
        sym = _apple_gpu_linalg_sym(
            symname, [ctypes.POINTER(ctypes.c_float)] * 4 + [ctypes.c_int32] * 3)
        if sym is not None and c >= 1:
            fp = ctypes.POINTER(ctypes.c_float)
            flat = np.ascontiguousarray(Aw.reshape(Bn, r, c))
            Uf = np.empty((Bn, r, c), np.float32)
            Sf = np.empty((Bn, c), np.float32)
            Vf = np.empty((Bn, c, c), np.float32)     # right vectors as columns
            rc = sym(flat.ctypes.data_as(fp), Uf.ctypes.data_as(fp),
                     Sf.ctypes.data_as(fp), Vf.ctypes.data_as(fp),
                     ctypes.c_int32(Bn), ctypes.c_int32(r), ctypes.c_int32(c))
            if rc == 1:
                order = np.argsort(-Sf, axis=-1)       # per-matrix descending
                Ss = np.take_along_axis(Sf, order, axis=-1)
                Us = np.take_along_axis(Uf, order[:, None, :], axis=-1)
                Vs = np.take_along_axis(Vf, order[:, None, :], axis=-1)
                # Verify reconstruction of what the kernel factored (the `flat`
                # stack), per matrix, before trusting the iterative result.
                recon = (Us * Ss[:, None, :]) @ np.swapaxes(Vs, -1, -2)
                denom = np.abs(flat).reshape(Bn, -1).max(axis=-1) + 1e-30
                err = float((np.abs(recon - flat).reshape(Bn, -1).max(axis=-1) / denom).max())
                if err < 1e-3:
                    if transpose:        # A = Vs·Σ·Usᵀ  (swap of the Aᵀ SVD)
                        U_b, S_b, Vh_b = Vs, Ss, np.swapaxes(Us, -1, -2)
                    else:
                        U_b, S_b, Vh_b = Us, Ss, np.swapaxes(Vs, -1, -2)
                    U_o = U_b.reshape(lead + U_b.shape[1:])     # squeezes batch for rank-2
                    S_o = S_b.reshape(lead + S_b.shape[1:])
                    Vh_o = Vh_b.reshape(lead + Vh_b.shape[1:])
                    return (np.ascontiguousarray(U_o).astype(out_dt),
                            np.ascontiguousarray(S_o).astype(out_dt),
                            np.ascontiguousarray(Vh_o).astype(out_dt), True)
    U, S, Vh = np.linalg.svd(np.asarray(A, fb_dt), full_matrices=full_matrices)
    return U.astype(out_dt), S.astype(out_dt), Vh.astype(out_dt), False


# ── GPU-native RNG lane (opt-in) — MPSMatrixRandomPhilox ──────────────────────
# Philox-family (matching Tessera's S4 RNG family) but the stream is NOT
# bit-identical to Tessera's CPU Philox — so this is a SEPARATE opt-in surface,
# deliberately NOT wired into the deterministic tessera.rng samplers (that would
# break CPU/GPU equality + check_determinism, Decision #18). Determinism here is
# by the `seed` argument under MPS's own generator. Use it for large on-device
# random fills where generating on CPU + uploading would dominate.

def _apple_gpu_random_sym(name: str) -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, name, None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int64,
                    ctypes.c_uint64, ctypes.c_float, ctypes.c_float]
    sym.restype = ctypes.c_int32
    return sym


def _apple_gpu_random(shape: Any, np: Any, *, seed: int, normal: bool,
                      a: float, b: float, symname: str) -> Any:
    """Shared body: fill ``shape`` with f32 random values on the GPU (uniform or
    normal). Returns ``(array, ran_on_gpu)``; numpy fallback off Metal."""
    shape = tuple(int(s) for s in (shape if isinstance(shape, (tuple, list)) else (shape,)))
    n = 1
    for s in shape:
        n *= s
    out = np.empty(n, np.float32)
    sym = _apple_gpu_random_sym(symname)
    if sym is not None and n > 0:
        rc = sym(out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 ctypes.c_int64(n), ctypes.c_uint64(int(seed) & 0xFFFFFFFFFFFFFFFF),
                 ctypes.c_float(a), ctypes.c_float(b))
        if rc == 1:
            return out.reshape(shape), True
    # numpy fallback — NOT the same stream as the GPU (nor as tessera.rng); this
    # path only guarantees the requested distribution + shape.
    rng = np.random.default_rng(int(seed))
    if normal:
        vals = rng.normal(a, b, size=n).astype(np.float32)
    else:
        vals = rng.uniform(a, b, size=n).astype(np.float32)
    return vals.reshape(shape), False


def apple_gpu_random_uniform(shape: Any, np: Any, *, seed: int = 0,
                             low: float = 0.0, high: float = 1.0) -> Any:
    """Opt-in GPU-native uniform f32 fill in ``[low, high)`` via
    MPSMatrixRandomPhilox. Returns ``(array float32, ran_on_gpu)``. The stream is
    Philox-family but **not** bit-identical to ``tessera.rng`` / numpy — use only
    where a separate GPU RNG stream is acceptable."""
    return _apple_gpu_random(shape, np, seed=seed, normal=False, a=float(low),
                             b=float(high), symname="tessera_apple_gpu_random_uniform_f32")


def apple_gpu_random_normal(shape: Any, np: Any, *, seed: int = 0,
                            mean: float = 0.0, std: float = 1.0) -> Any:
    """Opt-in GPU-native normal f32 fill (given ``mean``/``std``) via
    MPSMatrixRandomPhilox. Returns ``(array float32, ran_on_gpu)``. Separate
    stream from ``tessera.rng`` (see :func:`apple_gpu_random_uniform`)."""
    return _apple_gpu_random(shape, np, seed=seed, normal=True, a=float(mean),
                             b=float(std), symname="tessera_apple_gpu_random_normal_f32")


def _apple_gpu_dispatch_linalg(op_name: str, operands: list[Any], kwargs: dict,
                               np: Any) -> Any:
    """Route a Graph IR linalg op to the MPSMatrix GPU lane (numpy fallback
    inside the wrappers). Only the registered ops are handled: ``tessera.cholesky``
    (factor) and ``tessera.tri_solve`` (triangular solve, honoring the ``lower``
    attribute). Returns the result array (the executor stores values, not the
    ``(result, ran)`` tuple the public wrappers return)."""
    if op_name == "tessera.cholesky":
        return apple_gpu_cholesky(np.asarray(operands[0]), np)[0]
    if op_name == "tessera.tri_solve":
        lower = kwargs.get("lower", True)
        if isinstance(lower, str):
            lower = lower.strip().lower() not in ("0", "false", "no")
        return apple_gpu_tri_solve(np.asarray(operands[0]), np.asarray(operands[1]),
                                   np, lower=bool(lower))[0]
    raise ValueError(f"_apple_gpu_dispatch_linalg: unsupported op {op_name!r}")


def _apple_gpu_dispatch_conv2d(operands: list[Any], kwargs: dict, np: Any) -> Any:
    """Tier-3 2-D convolution via MPSGraph (NHWC source, HWIO weights).

    X is [N, H, W, Cin]; weight is [kH, kW, Cin/groups, Cout]; optional bias is
    [Cout]; output is [N, outH, outW, Cout]. ``stride``/``padding``/``dilation``
    accept an int or a 2-tuple; ``groups`` defaults to 1. f32/f16 run natively;
    bf16 runs via a host fp32 round-trip; any other dtype (or an unavailable
    runtime) returns None so the caller falls back to the numpy reference."""
    X = np.asarray(operands[0])
    W = np.asarray(operands[1])
    bias = None
    if len(operands) > 2 and operands[2] is not None:
        bias = np.asarray(operands[2])
    if X.ndim != 4 or W.ndim != 4:
        return None

    def _pair(v: Any) -> tuple[int, int]:
        if isinstance(v, (tuple, list)):
            return int(v[0]), int(v[1])
        return int(v), int(v)

    sH, sW = _pair(kwargs.get("stride", 1))
    pH, pW = _pair(kwargs.get("padding", 0))
    dH, dW = _pair(kwargs.get("dilation", 1))
    groups = int(kwargs.get("groups", 1))
    N, H, Wd, Cin = (int(s) for s in X.shape)
    kH, kW, cinG, Cout = (int(s) for s in W.shape)
    if groups <= 0 or Cin % groups or Cout % groups or cinG != Cin // groups:
        return None
    outH = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    outW = (Wd + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    if outH <= 0 or outW <= 0:
        return None
    out_dtype = X.dtype
    bf16 = _bfloat16_dtype()
    # P8 (opt-in): route f16/bf16 conv to the matmul2d epilogue lane (im2col +
    # matrix-unit matmul, fused bias). groups==1 only; falls through to the
    # native/round-trip path when off / ineligible / Metal 4 unavailable.
    if (_MTL4_CONV_ROUTE and groups == 1
            and (out_dtype == np.float16 or (bf16 is not None and out_dtype == bf16))):
        lane_dtype = "bf16" if out_dtype == bf16 else "f16"
        Y, ran = apple_gpu_conv2d(X, W, np, bias=bias, act="none",
                                  stride=(sH, sW), padding=(pH, pW),
                                  dilation=(dH, dW), dtype=lane_dtype)
        if ran:
            return Y.astype(out_dtype)
    iattrs = [ctypes.c_int32(v) for v in
              (N, H, Wd, Cin, Cout, kH, kW, sH, sW, pH, pW, dH, dW, groups)]

    if out_dtype == np.float16:
        sym = _apple_gpu_conv2d_f16()
        if sym is None:
            return None
        up = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        xh = np.ascontiguousarray(X).view(np.uint16)
        wh = np.ascontiguousarray(W).view(np.uint16)
        bh = (np.ascontiguousarray(bias).view(np.uint16)
              if bias is not None else None)
        out = np.zeros((N, outH, outW, Cout), dtype=np.uint16)
        sym(up(xh), up(wh), up(bh) if bh is not None else None, up(out), *iattrs)
        return out.view(np.float16)

    is_bf16 = bf16 is not None and out_dtype == bf16
    is_f32 = out_dtype == np.float32
    if not (is_f32 or is_bf16):
        return None
    sym = _apple_gpu_conv2d_f32()
    if sym is None:
        return None
    fp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    xf = np.ascontiguousarray(X.astype(np.float32))
    wf = np.ascontiguousarray(W.astype(np.float32))
    bf = np.ascontiguousarray(bias.astype(np.float32)) if bias is not None else None
    out = np.zeros((N, outH, outW, Cout), dtype=np.float32)
    sym(fp(xf), fp(wf), fp(bf) if bf is not None else None, fp(out), *iattrs)
    return out.astype(out_dtype)


def _apple_gpu_conv3d_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_conv3d_f32", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.POINTER(ctypes.c_float)] * 4 + [ctypes.c_int32] * 19
    sym.restype = None
    return sym


def _apple_gpu_conv3d_f16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_conv3d_f16", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.POINTER(ctypes.c_uint16)] * 4 + [ctypes.c_int32] * 19
    sym.restype = None
    return sym


def _apple_gpu_dispatch_conv3d(operands: list[Any], kwargs: dict, np: Any) -> Any:
    """Tier-3 3-D convolution via im2col + a GPU MPSGraph batched matmul
    (NDHWC source, DHWIO weights).

    X is [N, D, H, W, Cin]; weight is [kD, kH, kW, Cin/groups, Cout]; optional
    bias is [Cout]; output is [N, oD, oH, oW, Cout]. ``stride``/``padding``/
    ``dilation`` accept an int or a 3-tuple; ``groups`` defaults to 1. f32/f16
    run natively (fp32 GEMM accumulation); bf16 runs via a host fp32 round-trip;
    any other dtype (or an unavailable runtime) returns None so the caller falls
    back to the numpy reference."""
    X = np.asarray(operands[0])
    W = np.asarray(operands[1])
    bias = None
    if len(operands) > 2 and operands[2] is not None:
        bias = np.asarray(operands[2])
    if X.ndim != 5 or W.ndim != 5:
        return None

    def _triple(v: Any) -> tuple[int, int, int]:
        if isinstance(v, (tuple, list)):
            return int(v[0]), int(v[1]), int(v[2])
        return int(v), int(v), int(v)

    sD, sH, sW = _triple(kwargs.get("stride", 1))
    pD, pH, pW = _triple(kwargs.get("padding", 0))
    dD, dH, dW = _triple(kwargs.get("dilation", 1))
    groups = int(kwargs.get("groups", 1))
    N, iD, iH, iW, Cin = (int(s) for s in X.shape)
    kD, kH, kW, cinG, Cout = (int(s) for s in W.shape)
    if groups <= 0 or Cin % groups or Cout % groups or cinG != Cin // groups:
        return None

    def _out(i: int, k: int, s: int, p: int, d: int) -> int:
        return (i + 2 * p - d * (k - 1) - 1) // s + 1

    oD = _out(iD, kD, sD, pD, dD)
    oH = _out(iH, kH, sH, pH, dH)
    oW = _out(iW, kW, sW, pW, dW)
    if oD <= 0 or oH <= 0 or oW <= 0:
        return None
    out_dtype = X.dtype
    bf16 = _bfloat16_dtype()
    iattrs = [ctypes.c_int32(v) for v in
              (N, iD, iH, iW, Cin, Cout, kD, kH, kW, sD, sH, sW, pD, pH, pW,
               dD, dH, dW, groups)]

    if out_dtype == np.float16:
        sym = _apple_gpu_conv3d_f16()
        if sym is None:
            return None
        up = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        xh = np.ascontiguousarray(X).view(np.uint16)
        wh = np.ascontiguousarray(W).view(np.uint16)
        bh = (np.ascontiguousarray(bias).view(np.uint16)
              if bias is not None else None)
        out = np.zeros((N, oD, oH, oW, Cout), dtype=np.uint16)
        sym(up(xh), up(wh), up(bh) if bh is not None else None, up(out), *iattrs)
        return out.view(np.float16)

    is_bf16 = bf16 is not None and out_dtype == bf16
    is_f32 = out_dtype == np.float32
    if not (is_f32 or is_bf16):
        return None
    sym = _apple_gpu_conv3d_f32()
    if sym is None:
        return None
    fp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    xf = np.ascontiguousarray(X.astype(np.float32))
    wf = np.ascontiguousarray(W.astype(np.float32))
    bf = np.ascontiguousarray(bias.astype(np.float32)) if bias is not None else None
    out = np.zeros((N, oD, oH, oW, Cout), dtype=np.float32)
    sym(fp(xf), fp(wf), fp(bf) if bf is not None else None, fp(out), *iattrs)
    return out.astype(out_dtype)


def _apple_gpu_mla_decode_sym(suffix: str) -> Any:
    """Compressed-KV mla_decode symbol (f32 / f16 / bf16). f32 uses float
    pointers; f16/bf16 use uint16 pointers. ABI: X,Wdkv,Wuk,Wuv,Q,O + 6 ints."""
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, f"tessera_apple_gpu_mla_decode_{suffix}", None)
    if sym is None:
        return None
    ptr = (ctypes.POINTER(ctypes.c_float) if suffix == "f32"
           else ctypes.POINTER(ctypes.c_uint16))
    sym.argtypes = [ptr] * 6 + [ctypes.c_int32] * 6
    sym.restype = None
    return sym


def _apple_gpu_mla_decode_rope_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mla_decode_rope_f32", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.POINTER(ctypes.c_float)] * 10 + [ctypes.c_int32] * 8
    sym.restype = None
    return sym


def _apple_gpu_mla_decode_rope_half(suffix: str) -> Any:
    """f16 / bf16 decoupled-RoPE symbol: 5 uint16 tensor inputs + 4 f32 cos/sin
    + uint16 output (matching the C ABI)."""
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, f"tessera_apple_gpu_mla_decode_rope_{suffix}", None)
    if sym is None:
        return None
    u16 = ctypes.POINTER(ctypes.c_uint16)
    f32 = ctypes.POINTER(ctypes.c_float)
    sym.argtypes = [u16] * 5 + [f32] * 4 + [u16] + [ctypes.c_int32] * 8
    sym.restype = None
    return sym


def _apple_gpu_mla_decode_rope(Qn: Any, Qr: Any, Kn: Any, Kr: Any, V: Any,
                               cosQ: Any, sinQ: Any, cosK: Any, sinK: Any,
                               np: Any, rotation_style: str = "interleaved") -> Any:
    """MLA decode with decoupled RoPE (explicit per-head K).

    DeepSeek-style MLA: each head's query/key splits into a no-position part
    (``dn``) and a RoPE-carrying part (``dr``); the key RoPE part is shared
    across heads. Once RoPE is applied and ``[nope ; rope]`` is concatenated,
    the score is standard MHA with head_dim ``dn + dr``; the heavy attention
    runs on-GPU via the fused ``bsmm`` kernel.

    Dtype is taken from ``Qn`` (the five tensor inputs share it): **f32**/**f16**
    run natively (f16 ⇒ f16 bsmm I/O, fp32 accumulation), **bf16** via a host
    round-trip. cos/sin tables stay f32. Shapes: Qn ``[B,H,Sq,dn]``,
    Qr ``[B,H,Sq,dr]``, Kn ``[B,H,Skv,dn]``, Kr ``[B,Skv,dr]`` (shared),
    V ``[B,H,Skv,dv]``, cos/sin ``[Sq|Skv, dr/2]``. Returns O ``[B,H,Sq,dv]`` in
    the input dtype. ``rotation_style`` is ``"interleaved"`` or ``"half"``."""
    style = 0 if str(rotation_style).lower().startswith("inter") else 1
    Qn0 = np.asarray(Qn)
    out_dtype = Qn0.dtype
    bf16 = _bfloat16_dtype()
    cosQ = np.ascontiguousarray(cosQ, np.float32)
    sinQ = np.ascontiguousarray(sinQ, np.float32)
    cosK = np.ascontiguousarray(cosK, np.float32)
    sinK = np.ascontiguousarray(sinK, np.float32)
    fpf = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B, H, Sq, dn = (int(s) for s in Qn0.shape)
    dr = int(np.asarray(Qr).shape[-1])
    Skv = int(np.asarray(Kn).shape[-2])
    dv = int(np.asarray(V).shape[-1])
    iargs = [ctypes.c_int32(v) for v in (B, H, Sq, Skv, dn, dr, dv, style)]

    is_f16 = (out_dtype == np.float16)
    is_bf16 = (bf16 is not None and out_dtype == bf16)
    if is_f16 or is_bf16:
        sym = _apple_gpu_mla_decode_rope_half("f16" if is_f16 else "bf16")
        if sym is not None:
            up = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            ten = [np.ascontiguousarray(a).view(np.uint16) for a in
                   (Qn, Qr, Kn, Kr, V)]
            O = np.zeros((B, H, Sq, dv), np.uint16)
            sym(up(ten[0]), up(ten[1]), up(ten[2]), up(ten[3]), up(ten[4]),
                fpf(cosQ), fpf(sinQ), fpf(cosK), fpf(sinK), up(O), *iargs)
            return O.view(out_dtype)

    Qn = np.ascontiguousarray(Qn, np.float32)
    Qr = np.ascontiguousarray(Qr, np.float32)
    Kn = np.ascontiguousarray(Kn, np.float32)
    Kr = np.ascontiguousarray(Kr, np.float32)
    V = np.ascontiguousarray(V, np.float32)
    sym = _apple_gpu_mla_decode_rope_f32()
    if sym is None:
        return None
    O = np.zeros((B, H, Sq, dv), np.float32)
    sym(fpf(Qn), fpf(Qr), fpf(Kn), fpf(Kr), fpf(V), fpf(cosQ), fpf(sinQ),
        fpf(cosK), fpf(sinK), fpf(O), *iargs)
    return O.astype(out_dtype) if out_dtype != np.float32 else O


def _apple_gpu_mla_absorb_decode_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mla_absorb_decode_f32", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.POINTER(ctypes.c_float)] * 11 + [ctypes.c_int32] * 9
    sym.restype = None
    return sym


def _apple_gpu_mla_absorb_decode_half(suffix: str) -> Any:
    """f16 / bf16 absorb symbol: 6 uint16 tensor inputs + uint16 output, with
    f32 cos/sin tables in between (matching the C ABI)."""
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, f"tessera_apple_gpu_mla_absorb_decode_{suffix}", None)
    if sym is None:
        return None
    u16 = ctypes.POINTER(ctypes.c_uint16)
    f32 = ctypes.POINTER(ctypes.c_float)
    # q_nope,q_rope,c_kv,k_rope,Wuk_t,Wuv (u16) | cosQ,sinQ,cosK,sinK (f32) | O (u16)
    sym.argtypes = [u16] * 6 + [f32] * 4 + [u16] + [ctypes.c_int32] * 9
    sym.restype = None
    return sym


def _apple_gpu_mla_absorb_decode(q_nope: Any, q_rope: Any, c_kv: Any,
                                 k_rope: Any, Wuk_t: Any, Wuv: Any, cosQ: Any,
                                 sinQ: Any, cosK: Any, sinK: Any, np: Any,
                                 rotation_style: str = "interleaved") -> Any:
    """MLA decode with **weight absorption** + decoupled RoPE — the real MLA
    bandwidth win.

    Attention runs directly against the cached compressed latent ``c_kv``
    (shared across heads); the up-projection weights absorb into the query /
    output so per-head K/V are never materialized. The KV cache therefore stores
    only ``c_kv [B,Skv,Dl]`` + the shared ``k_rope [B,Skv,dr]``. Mathematically
    identical to the explicit-K decoupled-RoPE path.

    Dtype is taken from ``q_nope``: **f32** and **f16** run natively (f16 carries
    f16 I/O on-GPU at half the cache-read bandwidth, fp32 accumulation); **bf16**
    runs via a host fp32 round-trip. The cos/sin tables are always f32. The six
    tensor inputs must share the dtype. Shapes: q_nope ``[B,H,Sq,dn]``,
    q_rope ``[B,H,Sq,dr]``, c_kv ``[B,Skv,Dl]``, k_rope ``[B,Skv,dr]``,
    Wuk_t ``[H,dn,Dl]``, Wuv ``[H,Dl,dv]``, cos/sin ``[Sq|Skv, dr/2]``. Returns
    O ``[B,H,Sq,dv]`` in the input dtype. Returns None when unavailable."""
    style = 0 if str(rotation_style).lower().startswith("inter") else 1
    qn0 = np.asarray(q_nope)
    out_dtype = qn0.dtype
    bf16 = _bfloat16_dtype()
    cosQ = np.ascontiguousarray(cosQ, np.float32)
    sinQ = np.ascontiguousarray(sinQ, np.float32)
    cosK = np.ascontiguousarray(cosK, np.float32)
    sinK = np.ascontiguousarray(sinK, np.float32)
    fpf = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    is_f16 = (out_dtype == np.float16)
    is_bf16 = (bf16 is not None and out_dtype == bf16)
    if is_f16 or is_bf16:
        sym = _apple_gpu_mla_absorb_decode_half("f16" if is_f16 else "bf16")
        if sym is not None:
            ten = [np.ascontiguousarray(a).view(np.uint16) for a in
                   (q_nope, q_rope, c_kv, k_rope, Wuk_t, Wuv)]
            qn, qr, ckv, kr, wukt, wuv = ten
            B, H, Sq, dn = (int(s) for s in qn0.shape)
            dr = int(np.asarray(q_rope).shape[-1])
            Skv, Dl = int(np.asarray(c_kv).shape[-2]), int(np.asarray(c_kv).shape[-1])
            dv = int(np.asarray(Wuv).shape[-1])
            up = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            O = np.zeros((B, H, Sq, dv), np.uint16)
            sym(up(qn), up(qr), up(ckv), up(kr), up(wukt), up(wuv),
                fpf(cosQ), fpf(sinQ), fpf(cosK), fpf(sinK), up(O),
                ctypes.c_int32(B), ctypes.c_int32(H), ctypes.c_int32(Sq),
                ctypes.c_int32(Skv), ctypes.c_int32(dn), ctypes.c_int32(dr),
                ctypes.c_int32(dv), ctypes.c_int32(Dl), ctypes.c_int32(style))
            return O.view(out_dtype)
        # fall through to f32 if the half symbol is unavailable

    arrs = [np.ascontiguousarray(a, np.float32) for a in
            (q_nope, q_rope, c_kv, k_rope, Wuk_t, Wuv)]
    qn, qr, ckv, kr, wukt, wuv = arrs
    B, H, Sq, dn = (int(s) for s in qn.shape)
    dr = int(qr.shape[-1])
    Skv, Dl = int(ckv.shape[-2]), int(ckv.shape[-1])
    dv = int(wuv.shape[-1])
    sym = _apple_gpu_mla_absorb_decode_f32()
    if sym is None:
        return None
    O = np.zeros((B, H, Sq, dv), np.float32)
    sym(fpf(qn), fpf(qr), fpf(ckv), fpf(kr), fpf(wukt), fpf(wuv), fpf(cosQ),
        fpf(sinQ), fpf(cosK), fpf(sinK), fpf(O), ctypes.c_int32(B),
        ctypes.c_int32(H), ctypes.c_int32(Sq), ctypes.c_int32(Skv),
        ctypes.c_int32(dn), ctypes.c_int32(dr), ctypes.c_int32(dv),
        ctypes.c_int32(Dl), ctypes.c_int32(style))
    return O.astype(out_dtype) if out_dtype != np.float32 else O


def _apple_gpu_flash_attn_gqa_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_flash_attn_gqa_f32", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.POINTER(ctypes.c_float)] * 4 + [ctypes.c_int32] * 6 + [
        ctypes.c_float, ctypes.c_int32]
    sym.restype = None
    return sym


def _apple_gpu_flash_attn_gqa_f16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_flash_attn_gqa_f16", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.POINTER(ctypes.c_uint16)] * 4 + [ctypes.c_int32] * 6 + [
        ctypes.c_float, ctypes.c_int32]
    sym.restype = None
    return sym


def _apple_gpu_flash_attn_gqa_bf16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_flash_attn_gqa_bf16", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.POINTER(ctypes.c_uint16)] * 4 + [ctypes.c_int32] * 6 + [
        ctypes.c_float, ctypes.c_int32]
    sym.restype = None
    return sym


def _apple_gpu_dispatch_gqa(Q: Any, K: Any, V: Any, num_q_heads: int,
                            num_kv_heads: int, np: Any, scale: float | None = None,
                            causal: bool = False) -> Any:
    """Native GQA/MQA flash attention with KV-group indexing (no repeated KV).
    Q is [..., q_heads, Sq, D]; K/V are [..., kv_heads, Sk, D] (kv_heads <
    q_heads). The leading dims fold to the batch; query head h reads KV group
    h // (q_heads/kv_heads). f32/f16 run natively; bf16 runs natively via a
    fp32-conversion kernel; any other float dtype upcasts to f32. Returns the
    attention output shaped like Q, or None (caller falls back) when the shape /
    dtype is unsupported."""
    import math as _math
    Q = np.asarray(Q)
    K = np.asarray(K)
    V = np.asarray(V)
    if Q.ndim < 3 or K.ndim < 3 or V.ndim < 3:
        return None
    if num_q_heads % max(num_kv_heads, 1) != 0:
        return None
    Sq, D = int(Q.shape[-2]), int(Q.shape[-1])
    Sk = int(K.shape[-2])
    if D > 256 or K.shape[-1] != D or V.shape[-1] != D:
        return None
    out_dtype = Q.dtype
    Bq = int(np.prod(Q.shape[:-2]))   # total query heads
    Gkv = int(np.prod(K.shape[:-2]))  # total kv heads
    if Bq != (Bq // num_q_heads) * num_q_heads or Gkv != (Bq // num_q_heads) * num_kv_heads:
        return None
    sc = float(scale) if scale is not None else 1.0 / _math.sqrt(D)

    # ml_dtypes.bfloat16 is the bf16 boundary dtype when available.
    try:
        import ml_dtypes as _ml_dtypes
        _bf16 = _ml_dtypes.bfloat16
    except Exception:  # pragma: no cover - soft dep
        _bf16 = None

    is_f16 = (out_dtype == np.float16)
    is_bf16 = (_bf16 is not None and out_dtype == _bf16)

    # --- Native half-width paths (uint16 ABI; no f32 round-trip on host) ---
    if is_f16 or is_bf16:
        sym = _apple_gpu_flash_attn_gqa_f16() if is_f16 else _apple_gpu_flash_attn_gqa_bf16()
        if sym is not None:
            qh = np.ascontiguousarray(Q).reshape(Bq, Sq, D).view(np.uint16)
            kh = np.ascontiguousarray(K).reshape(Gkv, Sk, D).view(np.uint16)
            vh = np.ascontiguousarray(V).reshape(Gkv, Sk, D).view(np.uint16)
            outh = np.zeros((Bq, Sq, D), dtype=np.uint16)
            up = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            sym(up(qh), up(kh), up(vh), up(outh),
                ctypes.c_int32(Bq), ctypes.c_int32(num_q_heads),
                ctypes.c_int32(num_kv_heads), ctypes.c_int32(Sq),
                ctypes.c_int32(Sk), ctypes.c_int32(D), ctypes.c_float(sc),
                ctypes.c_int32(1 if causal else 0))
            return outh.reshape(Q.shape).view(out_dtype)
        # Native symbol unavailable (non-Apple / old runtime) — fall through to f32.

    sym = _apple_gpu_flash_attn_gqa_f32()
    if sym is None:
        return None
    qf = np.ascontiguousarray(Q.astype(np.float32)).reshape(Bq, Sq, D)
    kf = np.ascontiguousarray(K.astype(np.float32)).reshape(Gkv, Sk, D)
    vf = np.ascontiguousarray(V.astype(np.float32)).reshape(Gkv, Sk, D)
    out = np.zeros((Bq, Sq, D), dtype=np.float32)
    fp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    sym(fp(qf), fp(kf), fp(vf), fp(out),
        ctypes.c_int32(Bq), ctypes.c_int32(num_q_heads),
        ctypes.c_int32(num_kv_heads), ctypes.c_int32(Sq), ctypes.c_int32(Sk),
        ctypes.c_int32(D), ctypes.c_float(sc), ctypes.c_int32(1 if causal else 0))
    return out.reshape(Q.shape).astype(out_dtype)


def _apple_gpu_bsmm_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mpsgraph_bsmm_f32", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.POINTER(ctypes.c_float)] * 4 + [ctypes.c_int32] * 5 + [
        ctypes.c_float]
    sym.restype = None
    return sym


def _apple_gpu_bsmm_f16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mpsgraph_bsmm_f16", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.POINTER(ctypes.c_uint16)] * 4 + [ctypes.c_int32] * 5 + [
        ctypes.c_float]
    sym.restype = None
    return sym


def _apple_gpu_dispatch_batched_attention(Q: Any, K: Any, V: Any, np: Any,
                                          scale: float | None = None) -> Any:
    """Fused batched attention O = softmax((Q @ Kᵀ) * scale) @ V in a single
    dispatch (vs the bmm + softmax + bmm compose). Q/K/V are [..., T, D]; the
    leading dims fold to the batch. f32 + f16 native (fp32 compute); bf16
    upcasts. Returns None for unsupported shapes/dtypes so the caller can fall
    back to the compose path."""
    import math as _math
    Q = np.asarray(Q)
    K = np.asarray(K)
    V = np.asarray(V)
    if Q.ndim < 2 or K.ndim < 2 or V.ndim < 2:
        return None
    T, D = int(Q.shape[-2]), int(Q.shape[-1])
    Sk = int(K.shape[-2])
    if K.shape[-1] != D or V.shape[-1] != D or V.shape[-2] != Sk:
        return None
    batch = int(np.prod(Q.shape[:-2])) if Q.ndim > 2 else 1
    if (int(np.prod(K.shape[:-2])) if K.ndim > 2 else 1) != batch:
        return None
    out_dtype = Q.dtype
    sc = float(scale) if scale is not None else 1.0 / _math.sqrt(D)
    bf16 = _bfloat16_dtype()
    if Q.dtype == np.float32:
        sym, half = _apple_gpu_bsmm_f32(), False
    elif Q.dtype == np.float16:
        sym, half = _apple_gpu_bsmm_f16(), True
    elif bf16 is not None and Q.dtype == bf16:
        sym, half = _apple_gpu_bsmm_f32(), False  # upcast
    else:
        return None
    if sym is None:
        return None
    dt = np.float16 if half else np.float32
    a = np.ascontiguousarray(Q.astype(dt)).reshape(batch, T, D)
    # B = Kᵀ per batch: [batch, D, Sk]
    bmat = np.ascontiguousarray(
        K.astype(dt).reshape(batch, Sk, D).transpose(0, 2, 1))
    c = np.ascontiguousarray(V.astype(dt)).reshape(batch, Sk, D)
    out = np.zeros((batch, T, D), dtype=dt)
    dims = (ctypes.c_int32(batch), ctypes.c_int32(T), ctypes.c_int32(Sk),
            ctypes.c_int32(D), ctypes.c_int32(D), ctypes.c_float(sc))
    if half:
        up = lambda arr: arr.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        sym(up(a), up(bmat), up(c), up(out), *dims)
    else:
        fp = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        sym(fp(a), fp(bmat), fp(c), fp(out), *dims)
    return out.reshape(Q.shape).astype(out_dtype)


def _apple_gpu_dispatch_attn_wrapper(op_name: str, operands: list[Any],
                                     kwargs: dict, np: Any) -> Any:
    """Standard attention family (Sub-sprint A) routed through the proven GQA
    flash-attention kernel.  Each op is a thin wrapper over softmax((QKᵀ)·scale)
    @V — multi_head_attention reshapes [B,S,H*D]→[B,H,S,D]; gqa/mqa pick the
    KV-group counts; mla_decode optionally projects the latent K/V; gated_attention
    multiplies by a (sigmoid) gate.  Reshapes are host-side views; the attention
    FLOPs run on the GPU.  Returns None on unsupported shapes/dtypes so the
    op-loop falls back to the numpy reference."""
    short = str(op_name)
    scale = kwargs.get("scale")
    causal = bool(kwargs.get("causal", False))
    Q = np.asarray(operands[0])

    if short == "tessera.multi_head_attention":
        num_heads = int(kwargs["num_heads"])
        K = np.asarray(operands[1])
        V = np.asarray(operands[2])
        if Q.ndim != 3:
            return None
        B, Sq, hd = Q.shape
        if hd % num_heads != 0:
            return None
        D = hd // num_heads

        def split(t: Any) -> Any:
            St = int(t.shape[1])
            return np.ascontiguousarray(
                t.reshape(B, St, num_heads, D).transpose(0, 2, 1, 3))

        out = _apple_gpu_dispatch_gqa(
            split(Q), split(K), split(V), num_heads, num_heads, np,
            scale=scale, causal=causal)
        if out is None:
            return None
        out = np.asarray(out)
        return np.ascontiguousarray(out.transpose(0, 2, 1, 3).reshape(B, Sq, hd))

    if short in ("tessera.gqa_attention", "tessera.mqa_attention"):
        K = np.asarray(operands[1])
        V = np.asarray(operands[2])
        if Q.ndim != 4:
            return None
        if short == "tessera.mqa_attention":
            nq, nkv = int(Q.shape[1]), 1
        else:
            nq = int(kwargs["num_query_heads"])
            nkv = int(kwargs["num_kv_heads"])
        return _apple_gpu_dispatch_gqa(Q, K, V, nq, nkv, np, scale=scale, causal=causal)

    if short == "tessera.mla_decode":
        K = np.asarray(operands[1])
        V = np.asarray(operands[2])
        w_k = operands[3] if len(operands) > 3 else kwargs.get("W_k")
        w_v = operands[4] if len(operands) > 4 else kwargs.get("W_v")
        if w_k is not None:
            K = np.matmul(K, np.asarray(w_k, K.dtype))
        if w_v is not None:
            V = np.matmul(V, np.asarray(w_v, V.dtype))
        if Q.ndim != 4 or K.ndim != 4:
            return None
        return _apple_gpu_dispatch_gqa(
            Q, K, V, int(Q.shape[1]), int(K.shape[1]), np, scale=scale, causal=causal)

    if short == "tessera.gated_attention":
        K = np.asarray(operands[1])
        V = np.asarray(operands[2])
        gate = np.asarray(operands[3])
        if Q.ndim != 4:
            return None
        attn = _apple_gpu_dispatch_gqa(
            Q, K, V, int(Q.shape[1]), int(Q.shape[1]), np, scale=scale, causal=causal)
        if attn is None:
            return None
        attn = np.asarray(attn)
        act = kwargs.get("gate_activation", "sigmoid")
        if act == "sigmoid":
            gate = 1.0 / (1.0 + np.exp(-gate.astype(np.float64))).astype(attn.dtype)
        elif act not in ("identity", "none"):
            raise ValueError("gate_activation must be 'sigmoid', 'identity', or 'none'")
        return attn * np.broadcast_to(np.asarray(gate, attn.dtype), attn.shape)

    return None


def _apple_gpu_linear_attn_fmap(x: Any, name: str, np: Any) -> Any:
    """Linear-attention feature map on the GPU lanes where a direct opcode
    exists (relu / polynomial_2); identity is a no-op; elu (= elu(x)+1, the
    always-positive kernel feature) is the one cheap host fallback."""
    if name == "identity":
        return np.ascontiguousarray(x.astype(np.float32))
    xf = np.ascontiguousarray(x.astype(np.float32))
    if name == "relu":
        return np.asarray(_apple_gpu_dispatch_unary("tessera.relu", [xf], np), np.float32)
    if name == "polynomial_2":
        return np.asarray(
            _apple_gpu_dispatch_mpsgraph_binary("tessera.mul", [xf, xf], {}, np),
            np.float32)
    if name == "elu":  # elu(x) + 1 = where(x > 0, x + 1, exp(x))
        return np.where(xf > 0, xf + 1.0, np.exp(xf)).astype(np.float32)
    raise ValueError(f"unknown linear-attn feature_map {name!r}")


def _apple_gpu_dispatch_linear_attn(op_name: str, operands: list[Any],
                                    kwargs: dict, np: Any) -> Any:
    """Linear / recurrent attention family (Sub-sprint B) via the quadratic-
    parallel form ``O = (φ(Q)φ(K)ᵀ ⊙ causal[⊙decay]) @ V`` — two GPU batched
    matmuls plus an elementwise mask multiply.  Covers linear_attn (+_state),
    lightning_attention, power_attn, retention.  The causal/decay mask is a
    structured host-side constant (tril ⊙ cumprod-decay ratio); the QKᵀ and
    PV matmuls and the feature map run on the GPU.  Returns None on
    unsupported shapes so the op-loop falls back to the numpy reference."""
    short = str(op_name)
    Q = np.asarray(operands[0])
    K = np.asarray(operands[1])
    V = np.asarray(operands[2])
    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
        return None
    B, H, S, _Dq = Q.shape
    out_dtype = np.result_type(Q, K, V)
    causal = bool(kwargs.get("causal", True))
    decay = kwargs.get("decay")
    fm = kwargs.get("feature_map", "identity")
    want_state = (short == "tessera.linear_attn_state")
    deg = None

    if short == "tessera.lightning_attention":
        fm = "identity"
    elif short == "tessera.power_attn":
        deg = int(kwargs.get("deg", 2))
        fm = "polynomial_2" if deg == 2 else "identity"
    elif short == "tessera.retention":
        deg = int(kwargs.get("deg", 2))
        fm = "polynomial_2" if deg == 2 else "identity"
        log_g = kwargs.get("log_g")
        if log_g is not None:
            decay = np.exp(np.asarray(log_g, np.float64))

    # deg != 2 pre-powers Q/K then uses the identity map (matches the references).
    Qf = Q.astype(np.float64)
    Kf = K.astype(np.float64)
    if deg is not None and deg != 2:
        Qf = Qf ** deg
        Kf = Kf ** deg

    phiQ = _apple_gpu_linear_attn_fmap(Qf, fm, np)              # [B,H,S,Dq]
    phiK = _apple_gpu_linear_attn_fmap(Kf, fm, np)              # [B,H,S,Dq]
    Vf = np.ascontiguousarray(V.astype(np.float32))            # [B,H,S,Dv]
    phiK_T = np.ascontiguousarray(np.swapaxes(phiK, -1, -2))   # [B,H,Dq,S]

    if want_state:
        if decay is not None:
            dc = np.cumprod(np.asarray(decay, np.float64), axis=2)         # [B,H,S]
            tail = (dc[:, :, -1:] / dc).astype(np.float32)                 # Π_{s>t} decay
            Vw = np.ascontiguousarray(Vf * tail[:, :, :, None])
        else:
            Vw = Vf
        state = _apple_gpu_dispatch_bmm(phiK_T, Vw, np)        # [B,H,Dq,Dv]
        if state is None:
            return None
        return np.asarray(state).astype(out_dtype)

    A = _apple_gpu_dispatch_bmm(phiQ, phiK_T, np)             # [B,H,S,S]
    if A is None:
        return None
    A = np.ascontiguousarray(np.asarray(A, np.float32))
    if causal:
        tril = np.tril(np.ones((S, S), np.float32))
        if decay is not None:
            dc = np.cumprod(np.asarray(decay, np.float64), axis=2)         # [B,H,S]
            ratio = (dc[:, :, :, None] / dc[:, :, None, :]).astype(np.float32)
            mask = np.ascontiguousarray(tril[None, None] * ratio)          # [B,H,S,S]
        else:
            mask = np.ascontiguousarray(np.broadcast_to(tril, (B, H, S, S)))
        A = np.asarray(
            _apple_gpu_dispatch_mpsgraph_binary("tessera.mul", [A, mask], {}, np),
            np.float32)
    O = _apple_gpu_dispatch_bmm(np.ascontiguousarray(A), Vf, np)   # [B,H,S,Dv]
    if O is None:
        return None
    return np.asarray(O).astype(out_dtype)


def _apple_gpu_dispatch_delta_attn(op_name: str, operands: list[Any],
                                   kwargs: dict, np: Any) -> Any:
    """Delta-rule attention family (Sub-sprint D) — gated_deltanet,
    kimi_delta_attention, modified_delta_attention.  Each sequential delta
    recurrence is algebraically the quadratic form O = (QKᵀ ⊙ mask) @ V where
    the mask carries a per-token *column* weight c_r = β_r·decay_ratio
    [/(1+‖K_r‖‖V_r‖) for the modified rule], plus an optional output gate.
    Two GPU batched matmuls + a host-constructed mask + a GPU mask multiply;
    returns None for the (rare) return_state path so it falls back to numpy."""
    short = str(op_name)
    if kwargs.get("return_state"):
        return None
    Q = np.asarray(operands[0])
    K = np.asarray(operands[1])
    V = np.asarray(operands[2])
    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
        return None
    if not bool(kwargs.get("causal", True)):
        return None  # non-causal delta reference uses a different closed form
    B, H, S, _ = Q.shape
    out_dtype = np.result_type(Q, K, V)
    modified = (short == "tessera.modified_delta_attention")
    beta = kwargs.get("beta")
    decay = kwargs.get("decay")
    gate = kwargs.get("gate")

    Kf = K.astype(np.float64)
    Vf = V.astype(np.float64)
    phiQ = np.ascontiguousarray(Q.astype(np.float32))
    phiK_T = np.ascontiguousarray(np.swapaxes(K.astype(np.float32), -1, -2))
    A = _apple_gpu_dispatch_bmm(phiQ, phiK_T, np)             # [B,H,S,S]
    if A is None:
        return None
    A = np.ascontiguousarray(np.asarray(A, np.float32))

    # Per-token column weight c_r and the causal[⊙decay] mask (host constant).
    c = np.ones((B, H, S), np.float64)
    if beta is not None:
        c = c * np.asarray(beta, np.float64)
    if modified:
        nK = np.linalg.norm(Kf, axis=-1)
        nV = np.linalg.norm(Vf, axis=-1)
        c = c / (1.0 + nK * nV)
    tril = np.tril(np.ones((S, S), np.float64))
    if decay is not None:
        dc = np.cumprod(np.asarray(decay, np.float64), axis=2)
        ratio = dc[:, :, :, None] / dc[:, :, None, :]
        mask = tril[None, None] * ratio
    else:
        mask = np.broadcast_to(tril, (B, H, S, S))
    mask = np.ascontiguousarray((mask * c[:, :, None, :]).astype(np.float32))
    A = np.asarray(
        _apple_gpu_dispatch_mpsgraph_binary("tessera.mul", [A, mask], {}, np), np.float32)
    O = _apple_gpu_dispatch_bmm(np.ascontiguousarray(A),
                                np.ascontiguousarray(V.astype(np.float32)), np)  # [B,H,S,Dv]
    if O is None:
        return None
    O = np.asarray(O, np.float32)
    if gate is not None:
        g = np.asarray(
            _apple_gpu_dispatch_unary("tessera.sigmoid", [np.ascontiguousarray(
                np.asarray(gate, np.float32))], np), np.float32)
        O = np.asarray(
            _apple_gpu_dispatch_mpsgraph_binary(
                "tessera.mul", [np.ascontiguousarray(O),
                                np.ascontiguousarray(np.broadcast_to(g, O.shape))], {}, np),
            np.float32)
    return O.astype(out_dtype)


def _apple_gpu_dispatch_masked_attn(op_name: str, operands: list[Any],
                                    kwargs: dict, np: Any) -> Any:
    """NSA masked-softmax attention (Sub-sprint C).  attn_compressed_blocks is
    plain softmax(QK_cᵀ·scale)@V_c (the proven batched-attention kernel);
    attn_sliding_window adds a structured causal/window additive mask before the
    softmax.  Both run QKᵀ / softmax / PV on the GPU; the window mask is a
    host-side structured constant.  Returns None on unsupported shapes."""
    import math as _math
    short = str(op_name)
    Q = np.asarray(operands[0])
    K = np.asarray(operands[1])
    V = np.asarray(operands[2])
    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
        return None
    D = int(Q.shape[-1])
    scale = 1.0 / _math.sqrt(D)

    if short == "tessera.attn_compressed_blocks":
        return _apple_gpu_dispatch_batched_attention(Q, K, V, np, scale=scale)

    if short == "tessera.attn_sliding_window":
        window_size = int(kwargs["window_size"])
        causal = bool(kwargs.get("causal", True))
        if window_size <= 0:
            return None
        B, H, S, _ = Q.shape
        Sk = int(K.shape[2])
        out_dtype = np.result_type(Q, K, V)
        Qs = np.ascontiguousarray((Q.astype(np.float32)) * np.float32(scale))
        Kt = np.ascontiguousarray(np.swapaxes(K.astype(np.float32), -1, -2))
        A = _apple_gpu_dispatch_bmm(Qs, Kt, np)                      # [B,H,S,Sk]
        if A is None:
            return None
        A = np.ascontiguousarray(np.asarray(A, np.float32))
        i = np.arange(S)[:, None]
        j = np.arange(Sk)[None, :]
        if causal:
            outside = (j > i) | (j < i - window_size + 1)
        else:
            outside = (j > i + window_size // 2) | (j < i - window_size // 2)
        bias = np.where(outside, np.float32(-1e30), np.float32(0.0))  # [S,Sk]
        bias = np.ascontiguousarray(np.broadcast_to(bias, (B, H, S, Sk)))
        A = np.asarray(
            _apple_gpu_dispatch_mpsgraph_binary("tessera.add", [A, bias], {}, np),
            np.float32)
        P2 = _apple_gpu_dispatch_softmax(
            "tessera.softmax", [np.ascontiguousarray(A.reshape(B * H * S, Sk))], {}, np)
        P = np.ascontiguousarray(np.asarray(P2, np.float32).reshape(B, H, S, Sk))
        O = _apple_gpu_dispatch_bmm(P, np.ascontiguousarray(V.astype(np.float32)), np)
        if O is None:
            return None
        return np.asarray(O).astype(out_dtype)

    return None


def _apple_gpu_mps_matmul_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = runtime.tessera_apple_gpu_mps_matmul_f32
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_mps_matmul_f16() -> Any:
    """Phase 8.4.4 — fp16 matmul via MPSDataTypeFloat16. Inputs/outputs are
    bit-pattern uint16_t* (numpy float16 layout). Symbol may be absent on
    older runtime builds; the dispatcher falls through to fp32 conversion."""

    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mps_matmul_f16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_mps_matmul_bf16() -> Any:
    """Phase 8.4.4 — bf16 matmul. The runtime shim does fp32 conversion
    inside since MPS doesn't natively support bf16 matrix descriptors as of
    macOS 14. ml_dtypes.bfloat16 dtype is byte-compatible with the C ABI."""

    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mps_matmul_bf16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_dispatch_rope(op_name: str, operands: list[Any], np: Any) -> Any:
    """Phase 8.4 + 8.4.4.1: dispatch a single rank-2 rope through the apple_gpu
    runtime shim's custom MSL kernel. Picks the runtime symbol by element type:
      - f32: native MSL kernel (Phase 8.4)
      - f16: native MSL `half` kernel (Phase 8.4.4.1)
      - bf16: fp32-conversion path inside the shim (Phase 8.4.4.1)
    Inputs outside the supported envelope fall back to the numpy reference.
    """

    if len(operands) != 2:
        raise ValueError(f"{op_name!r} requires exactly two operands")
    x = np.asarray(operands[0])
    theta = np.asarray(operands[1])

    if (
        x.ndim != 2 or theta.ndim != 2
        or x.shape != theta.shape
        or x.shape[1] % 2 != 0
        or x.dtype != theta.dtype
    ):
        return _runtime_rope(np, x, theta)

    if x.dtype == np.float32:
        if not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=np.float32)
        if not theta.flags.c_contiguous:
            theta = np.ascontiguousarray(theta, dtype=np.float32)
        out = np.zeros(x.shape, dtype=np.float32)
        rope = _apple_gpu_rope_f32()
        rope(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            theta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(x.shape[0]),
            ctypes.c_int32(x.shape[1]),
        )
        return out

    if x.dtype == np.float16:
        if not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=np.float16)
        if not theta.flags.c_contiguous:
            theta = np.ascontiguousarray(theta, dtype=np.float16)
        out = np.zeros(x.shape, dtype=np.float16)
        rope_f16 = _apple_gpu_rope_f16()
        if rope_f16 is not None:
            rope_f16(
                x.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                theta.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int32(x.shape[0]),
                ctypes.c_int32(x.shape[1]),
            )
            return out
        # Older runtime build without f16 — fall back via fp32.
        return _runtime_rope(np, x.astype(np.float32), theta.astype(np.float32)).astype(np.float16)

    bf16_dtype = _bfloat16_dtype()
    if bf16_dtype is not None and x.dtype == bf16_dtype:
        if not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=bf16_dtype)
        if not theta.flags.c_contiguous:
            theta = np.ascontiguousarray(theta, dtype=bf16_dtype)
        out = np.zeros(x.shape, dtype=bf16_dtype)
        rope_bf16 = _apple_gpu_rope_bf16()
        if rope_bf16 is not None:
            rope_bf16(
                x.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                theta.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int32(x.shape[0]),
                ctypes.c_int32(x.shape[1]),
            )
            return out
        return _runtime_rope(np, x.astype(np.float32), theta.astype(np.float32)).astype(bf16_dtype)

    return _runtime_rope(np, x, theta)


def _apple_gpu_rope_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = runtime.tessera_apple_gpu_rope_f32
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_rope_f16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_rope_f16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_rope_bf16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_rope_bf16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_dispatch_flash_attn(op_name: str, operands: list[Any],
                                   kwargs: Mapping[str, Any], np: Any) -> Any:
    """Phase 8.4.1 + 8.4.4.2: dispatch a single rank-3 flash-attention
    forward through the apple_gpu runtime shim's custom MSL kernel. Picks
    symbol by element type (f32, f16, bf16). Inputs outside the supported
    envelope (rank, dtype, head_dim > 256) fall back to the numpy reference.
    """

    if len(operands) < 3:
        raise ValueError(f"{op_name!r} requires Q, K, V operands")
    q = np.asarray(operands[0])
    k = np.asarray(operands[1])
    v = np.asarray(operands[2])

    if not (
        q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        and q.shape[0] == k.shape[0] == v.shape[0]
        and k.shape[1] == v.shape[1]
        and q.shape[2] == k.shape[2] == v.shape[2]
        and q.shape[2] <= 256
        and q.dtype == k.dtype == v.dtype
    ):
        return _runtime_flash_attn(np, q, k, v, kwargs)

    B, Sq, D = q.shape
    Sk = k.shape[1]
    scale = kwargs.get("scale", None)
    scale = (1.0 / float(np.sqrt(D))) if scale is None else float(scale)
    causal = 1 if bool(kwargs.get("causal", False)) else 0

    if q.dtype == np.float32:
        if not q.flags.c_contiguous:
            q = np.ascontiguousarray(q, dtype=np.float32)
        if not k.flags.c_contiguous:
            k = np.ascontiguousarray(k, dtype=np.float32)
        if not v.flags.c_contiguous:
            v = np.ascontiguousarray(v, dtype=np.float32)
        out = np.zeros((B, Sq, D), dtype=np.float32)
        flash_attn = _apple_gpu_flash_attn_f32()
        flash_attn(
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(B), ctypes.c_int32(Sq), ctypes.c_int32(Sk),
            ctypes.c_int32(D), ctypes.c_float(scale), ctypes.c_int32(causal),
        )
        return out

    if q.dtype == np.float16:
        if not q.flags.c_contiguous:
            q = np.ascontiguousarray(q, dtype=np.float16)
        if not k.flags.c_contiguous:
            k = np.ascontiguousarray(k, dtype=np.float16)
        if not v.flags.c_contiguous:
            v = np.ascontiguousarray(v, dtype=np.float16)
        out = np.zeros((B, Sq, D), dtype=np.float16)
        flash_attn_f16 = _apple_gpu_flash_attn_f16()
        if flash_attn_f16 is not None:
            flash_attn_f16(
                q.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                k.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                v.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int32(B), ctypes.c_int32(Sq), ctypes.c_int32(Sk),
                ctypes.c_int32(D), ctypes.c_float(scale), ctypes.c_int32(causal),
            )
            return out
        return _runtime_flash_attn(
            np, q.astype(np.float32), k.astype(np.float32), v.astype(np.float32), kwargs
        ).astype(np.float16)

    bf16_dtype = _bfloat16_dtype()
    if bf16_dtype is not None and q.dtype == bf16_dtype:
        if not q.flags.c_contiguous:
            q = np.ascontiguousarray(q, dtype=bf16_dtype)
        if not k.flags.c_contiguous:
            k = np.ascontiguousarray(k, dtype=bf16_dtype)
        if not v.flags.c_contiguous:
            v = np.ascontiguousarray(v, dtype=bf16_dtype)
        out = np.zeros((B, Sq, D), dtype=bf16_dtype)
        flash_attn_bf16 = _apple_gpu_flash_attn_bf16()
        if flash_attn_bf16 is not None:
            flash_attn_bf16(
                q.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                k.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                v.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int32(B), ctypes.c_int32(Sq), ctypes.c_int32(Sk),
                ctypes.c_int32(D), ctypes.c_float(scale), ctypes.c_int32(causal),
            )
            return out
        return _runtime_flash_attn(
            np, q.astype(np.float32), k.astype(np.float32), v.astype(np.float32), kwargs
        ).astype(bf16_dtype)

    return _runtime_flash_attn(np, q, k, v, kwargs)


def _apple_gpu_flash_attn_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = runtime.tessera_apple_gpu_flash_attn_f32
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # Q
        ctypes.POINTER(ctypes.c_float),  # K
        ctypes.POINTER(ctypes.c_float),  # V
        ctypes.POINTER(ctypes.c_float),  # O
        ctypes.c_int32,                  # B
        ctypes.c_int32,                  # Sq
        ctypes.c_int32,                  # Sk
        ctypes.c_int32,                  # D
        ctypes.c_float,                  # scale
        ctypes.c_int32,                  # causal
    ]
    sym.restype = None
    return sym


def _apple_gpu_flash_attn_f16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_flash_attn_f16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_float, ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_flash_attn_bf16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_flash_attn_bf16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_float, ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_dispatch_softmax(op_name: str, operands: list[Any],
                                kwargs: Mapping[str, Any], np: Any) -> Any:
    """Phase 8.4.2 + 8.4.4.1: dispatch a single rank-2 softmax (axis=-1)
    through the apple_gpu runtime shim's custom MSL kernel. Picks symbol by
    element type (f32, f16, bf16). Inputs outside the supported envelope
    fall back to the numpy reference.
    """

    if len(operands) < 1:
        raise ValueError(f"{op_name!r} requires one operand")
    x = np.asarray(operands[0])
    axis = int(kwargs.get("axis", -1))
    if x.ndim != 2 or (axis != -1 and axis != 1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)

    if x.dtype == np.float32:
        if not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=np.float32)
        M, K = x.shape
        out = np.zeros((M, K), dtype=np.float32)
        softmax = _apple_gpu_softmax_f32()
        softmax(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(M),
            ctypes.c_int32(K),
        )
        return out

    if x.dtype == np.float16:
        if not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=np.float16)
        M, K = x.shape
        out = np.zeros((M, K), dtype=np.float16)
        softmax_f16 = _apple_gpu_softmax_f16()
        if softmax_f16 is not None:
            softmax_f16(
                x.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int32(M),
                ctypes.c_int32(K),
            )
            return out
        e = np.exp(x.astype(np.float32) - np.max(x.astype(np.float32), axis=-1, keepdims=True))
        return (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float16)

    bf16_dtype = _bfloat16_dtype()
    if bf16_dtype is not None and x.dtype == bf16_dtype:
        if not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=bf16_dtype)
        M, K = x.shape
        out = np.zeros((M, K), dtype=bf16_dtype)
        softmax_bf16 = _apple_gpu_softmax_bf16()
        if softmax_bf16 is not None:
            softmax_bf16(
                x.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int32(M),
                ctypes.c_int32(K),
            )
            return out
        e = np.exp(x.astype(np.float32) - np.max(x.astype(np.float32), axis=-1, keepdims=True))
        return (e / np.sum(e, axis=-1, keepdims=True)).astype(bf16_dtype)

    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _apple_gpu_softmax_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = runtime.tessera_apple_gpu_softmax_f32
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_softmax_f16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_softmax_f16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_softmax_bf16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_softmax_bf16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_dispatch_gelu(op_name: str, operands: list[Any], np: Any) -> Any:
    """Phase 8.4.2 + 8.4.4.1: dispatch a single rank-2 gelu through the
    apple_gpu runtime shim's custom MSL kernel. Picks symbol by element type
    (f32, f16, bf16). Tanh-approximation matching the numpy reference."""

    if len(operands) < 1:
        raise ValueError(f"{op_name!r} requires one operand")
    x = np.asarray(operands[0])
    if x.ndim != 2:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    if x.dtype == np.float32:
        if not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=np.float32)
        M, K = x.shape
        out = np.zeros((M, K), dtype=np.float32)
        gelu = _apple_gpu_gelu_f32()
        gelu(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(M * K),
        )
        return out

    if x.dtype == np.float16:
        if not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=np.float16)
        M, K = x.shape
        out = np.zeros((M, K), dtype=np.float16)
        gelu_f16 = _apple_gpu_gelu_f16()
        if gelu_f16 is not None:
            gelu_f16(
                x.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int32(M * K),
            )
            return out
        x32 = x.astype(np.float32)
        return (0.5 * x32 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x32 + 0.044715 * x32**3)))).astype(np.float16)

    bf16_dtype = _bfloat16_dtype()
    if bf16_dtype is not None and x.dtype == bf16_dtype:
        if not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=bf16_dtype)
        M, K = x.shape
        out = np.zeros((M, K), dtype=bf16_dtype)
        gelu_bf16 = _apple_gpu_gelu_bf16()
        if gelu_bf16 is not None:
            gelu_bf16(
                x.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int32(M * K),
            )
            return out
        x32 = x.astype(np.float32)
        return (0.5 * x32 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x32 + 0.044715 * x32**3)))).astype(bf16_dtype)

    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _apple_gpu_gelu_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = runtime.tessera_apple_gpu_gelu_f32
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_gelu_f16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_gelu_f16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_gelu_bf16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_gelu_bf16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


# ---------------------------------------------------------------------------
# MetalPerformanceShadersGraph lane (2026-05-29) — Tier-1 activations /
# normalizations and the long tail. One parametrized runner per shape class
# in apple_gpu_runtime.mm; these helpers pick the symbol by element type and
# fall back to numpy when Metal is unavailable. bf16 upcasts to f32, runs on
# the GPU, and downcasts (mirrors the bf16 matmul path).
# ---------------------------------------------------------------------------
def _apple_gpu_mpsgraph_unary_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mpsgraph_unary_f32", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
    ]
    sym.restype = None
    return sym


def _apple_gpu_mpsgraph_unary_f16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mpsgraph_unary_f16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int64,
    ]
    sym.restype = None
    return sym


def _apple_gpu_mpsgraph_binary_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mpsgraph_binary_f32", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
    ]
    sym.restype = None
    return sym


def _apple_gpu_cf_scan_f32() -> Any:
    """Phase-G Rung 0 control-flow scan symbol (MPSGraph forLoop). None when
    unavailable."""
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_cf_scan_f32", None)
    if sym is None:
        return None
    fp = ctypes.POINTER(ctypes.c_float)
    sym.argtypes = [fp, fp, fp, fp, fp,
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def apple_gpu_cf_scan(Wh: Any, Wx: Any, xseq: Any, init: Any, np: Any) -> Any:
    """Bounded scan ``carry_{i+1} = tanh(carry_i @ Wh + x_i @ Wx)`` lowered to a
    single MPSGraph control-flow executable; returns the per-step carries
    ``ys`` of shape ``[T, d]``. Falls back to a numpy scan off Metal."""
    Wh = np.ascontiguousarray(Wh, np.float32)
    Wx = np.ascontiguousarray(Wx, np.float32)
    xseq = np.ascontiguousarray(xseq, np.float32)
    init = np.ascontiguousarray(init, np.float32).reshape(-1)
    T, m = xseq.shape
    d = Wh.shape[0]
    ys = np.empty((T, d), np.float32)
    sym = _apple_gpu_cf_scan_f32()
    fp = ctypes.POINTER(ctypes.c_float)
    rc = 0
    if sym is not None:
        rc = sym(Wh.ctypes.data_as(fp), Wx.ctypes.data_as(fp),
                 xseq.ctypes.data_as(fp), init.ctypes.data_as(fp),
                 ys.ctypes.data_as(fp), ctypes.c_int32(T), ctypes.c_int32(d),
                 ctypes.c_int32(m))
    if rc != 1:
        carry = init.astype(np.float64)
        for t in range(T):
            carry = np.tanh(carry @ Wh.astype(np.float64)
                            + xseq[t].astype(np.float64) @ Wx.astype(np.float64))
            ys[t] = carry.astype(np.float32)
    return ys


def _apple_gpu_cf_serial_draft_f32() -> Any:
    """Phase-G Rung 1 serial-draft forLoop symbol. None when unavailable."""
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_cf_serial_draft_f32", None)
    if sym is None:
        return None
    fp = ctypes.POINTER(ctypes.c_float)
    ip = ctypes.POINTER(ctypes.c_int32)
    sym.argtypes = ([fp] * 12 + [ctypes.c_int32, ip, fp]
                    + [ctypes.c_int32] * 5 + [ctypes.c_float])
    sym.restype = ctypes.c_int32
    return sym


def apple_gpu_cf_serial_draft(embed, fc_in, ln1_all, ln2_all, wv_all, wo_all,
                              wg_all, wu_all, wd_all, snorm, lm_head, h_init,
                              root_token, T, L, d, ffn, V, eps, np) -> Any:
    """The Gumiho serial draft (``T`` autoregressive steps) as a single MPSGraph
    control-flow executable. Per-layer weights are packed ``[L, ...]``. Returns
    ``(tokens[T] int64, hiddens[T, d] f32)`` or ``None`` when the symbol is
    unavailable (caller falls back to the host serial head)."""
    sym = _apple_gpu_cf_serial_draft_f32()
    if sym is None:
        return None
    arrs = [np.ascontiguousarray(a, np.float32) for a in
            (embed, fc_in, ln1_all, ln2_all, wv_all, wo_all, wg_all, wu_all,
             wd_all, snorm, lm_head, h_init)]
    tokens = np.empty(int(T), np.int32)
    hiddens = np.empty((int(T), int(d)), np.float32)
    fp = ctypes.POINTER(ctypes.c_float)
    ip = ctypes.POINTER(ctypes.c_int32)
    rc = sym(*[a.ctypes.data_as(fp) for a in arrs],
             ctypes.c_int32(int(root_token)),
             tokens.ctypes.data_as(ip), hiddens.ctypes.data_as(fp),
             ctypes.c_int32(int(T)), ctypes.c_int32(int(L)), ctypes.c_int32(int(d)),
             ctypes.c_int32(int(ffn)), ctypes.c_int32(int(V)),
             ctypes.c_float(float(eps)))
    if rc != 1:
        return None
    return tokens.astype(np.int64), hiddens


def _apple_gpu_cf_while_generate_f32() -> Any:
    """Phase-G Rung 2 predicate-driven while-generate symbol. None when absent."""
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_cf_while_generate_f32", None)
    if sym is None:
        return None
    fp = ctypes.POINTER(ctypes.c_float)
    ip = ctypes.POINTER(ctypes.c_int32)
    sym.argtypes = [fp, fp, fp, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                    ip, ip, ctypes.c_int32, ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def apple_gpu_cf_while_generate(W, lm, h_init, start_token, eos_token, max_steps,
                                d, V, np) -> Any:
    """Greedy generation ``token = argmax((hidden = tanh(hidden @ W)) @ lm)``
    looped via a single MPSGraph ``while`` until the EOS token or ``max_steps``.
    Returns ``(tokens[list], n_generated)``. Falls back to a numpy while-loop."""
    W = np.ascontiguousarray(W, np.float32)
    lm = np.ascontiguousarray(lm, np.float32)
    h_init = np.ascontiguousarray(h_init, np.float32).reshape(1, -1)
    sym = _apple_gpu_cf_while_generate_f32()
    if sym is not None:
        toks = np.zeros(int(max_steps), np.int32)
        n = np.zeros(1, np.int32)
        fp = ctypes.POINTER(ctypes.c_float)
        ip = ctypes.POINTER(ctypes.c_int32)
        rc = sym(W.ctypes.data_as(fp), lm.ctypes.data_as(fp),
                 h_init.ctypes.data_as(fp), ctypes.c_int32(int(start_token)),
                 ctypes.c_int32(int(eos_token)), ctypes.c_int32(int(max_steps)),
                 toks.ctypes.data_as(ip), n.ctypes.data_as(ip),
                 ctypes.c_int32(int(d)), ctypes.c_int32(int(V)))
        if rc == 1:
            k = int(n[0])
            return [int(t) for t in toks[:k]], k
    # numpy fallback — same predicate (loop while step<max and last!=eos).
    h = h_init.astype(np.float64).reshape(-1)
    Wf, lmf = W.astype(np.float64), lm.astype(np.float64)
    out, last, step = [], int(start_token), 0
    while step < int(max_steps) and last != int(eos_token):
        h = np.tanh(h @ Wf)
        last = int(np.argmax(h @ lmf))
        out.append(last)
        step += 1
    return out, step


def apple_gpu_metal4_caps() -> dict:
    """Live Metal 4 capability probe (M0). Actually creates the Metal 4 objects
    on-device and reports which are usable on this machine. All ``False`` off
    Tahoe / non-Darwin. See docs/apple_gpu_metal4_adoption.md."""
    keys = ("command_queue", "command_allocator", "compiler", "tensor", "msl4")
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_metal4_probe", None)
    if sym is None:
        return {"available": False, **{k: False for k in keys}, "bits": 0}
    sym.argtypes = [ctypes.POINTER(ctypes.c_int32)]
    sym.restype = ctypes.c_int32
    caps = ctypes.c_int32(0)
    rc = sym(ctypes.byref(caps))
    bits = int(caps.value)
    # "available" = the full MTL4 lane is usable (all 5 core bits), not just any
    # single bit — matches the C probe + the doc (MTL4 + MTLTensor + MSL 4.0).
    _METAL4_FULL = 1 | 2 | 4 | 8 | 16
    return {
        "available": rc == 1 and bits == _METAL4_FULL,
        "command_queue": bool(bits & 1),
        "command_allocator": bool(bits & 2),
        "compiler": bool(bits & 4),
        "tensor": bool(bits & 8),
        "msl4": bool(bits & 16),
        "bits": bits,
    }


_SIMD_CAP_BITS = {
    "reduction": 1,        # simd_sum / simd_max / simd_prefix_*
    "shuffle": 2,          # simd_shuffle / simd_shuffle_xor
    "shuffle_and_fill": 4,  # simd_shuffle_and_fill_down / _up
    "simdgroup_barrier": 8,
}


def apple_gpu_simd_caps() -> dict:
    """SIMD-feature capability probe of the active GPU. Reports which SIMD-group
    intrinsics the device supports (``reduction``/``shuffle``/``shuffle_and_fill``/
    ``simdgroup_barrier``) — all True on Apple-Silicon (M-series); all False off
    Metal. Returns ``{available, <feature>: bool, ..., bits}``. Used to document /
    gate SIMD-reduction kernel paths."""
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_simd_caps", None)
    if sym is None:
        return {"available": False, **{k: False for k in _SIMD_CAP_BITS}, "bits": 0}
    sym.argtypes = []
    sym.restype = ctypes.c_int32
    bits = int(sym())
    return {
        "available": bool(bits & _SIMD_CAP_BITS["reduction"]),
        **{name: bool(bits & b) for name, b in _SIMD_CAP_BITS.items()},
        "bits": bits,
    }


def _apple_gpu_raw_handle(symname: str) -> int:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, symname, None)
    if sym is None:
        return 0
    sym.argtypes = []
    sym.restype = ctypes.c_void_p
    p = sym()
    return int(p) if p else 0


def apple_gpu_device_handle() -> int:
    """Interop escape hatch (cf. Mojo's ``metal_device(ctx)``): the raw
    ``id<MTLDevice>`` Tessera's runtime uses, as an integer pointer (0 off Metal).
    For advanced interop — build custom Metal/MPS work against the *same* device so
    it composes with Tessera's resident buffers (see ``DeviceTensor.mtl_buffer``).
    Tessera owns the lifetime; do not release it. Bridge into metal-cpp / PyObjC
    via the pointer value."""
    return _apple_gpu_raw_handle("tessera_apple_gpu_device_handle")


def apple_gpu_command_queue_handle() -> int:
    """The raw ``id<MTLCommandQueue>`` Tessera dispatches on, as an integer pointer
    (0 off Metal). Serialize any work you enqueue on it against Tessera's own use.
    See :func:`apple_gpu_device_handle`."""
    return _apple_gpu_raw_handle("tessera_apple_gpu_command_queue_handle")


def apple_gpu_matmul2d_dev(A: Any, B: Any, C: Any, *, bf16: bool = False) -> bool:
    """R0 — general device-resident matmul ``C = A @ B`` where A, B, C are
    :class:`DeviceTensor`s (A/B f16 or bf16 per ``bf16``, C f32, shapes M×K · K×N).
    *Both* operands stay resident (no host upload) — the both-resident complement
    to the M8 session's fixed-weight ``run_dev`` (e.g. attention ``Q @ Kᵀ`` where
    both are resident activations). Returns True if it ran on the matrix-unit lane.

    Note: like the R0 bridge generally, this is a *capability* (resident operands),
    not a throughput win on unified memory — see the integration review."""
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mtl4_matmul2d_dev", None)
    if sym is None:
        return False
    sym.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int32] * 4
    sym.restype = ctypes.c_int32
    M, K = int(A.shape[0]), int(A.shape[1])
    N = int(B.shape[1])
    rc = sym(A.handle, B.handle, C.handle, ctypes.c_int32(M), ctypes.c_int32(N),
             ctypes.c_int32(K), ctypes.c_int32(1 if bf16 else 0))
    return rc == 1


def apple_gpu_metal4_tensor_roundtrip(arr: Any, np: Any) -> Any:
    """Round-trip ``arr`` (f32/f16/bf16) through a native Metal 4 ``MTLTensor``
    of the same dtype and return it — proving the typed resource stores +
    retrieves data on this machine (M1). Falls back to a numpy copy when
    MTLTensor is unavailable, so the contract holds everywhere."""
    bf16 = _bfloat16_dtype()
    a = np.ascontiguousarray(arr)
    code = {np.dtype(np.float32): 0, np.dtype(np.float16): 1}.get(a.dtype, None)
    if code is None and bf16 is not None and a.dtype == np.dtype(bf16):
        code = 2
    if code is None:                                   # promote unknown dtypes
        a = np.ascontiguousarray(arr, np.float32)
        code = 0
    out = np.empty_like(a)
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_metal4_tensor_roundtrip", None)
    rc = 0
    if sym is not None:
        sym.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
        sym.restype = ctypes.c_int32
        rc = sym(a.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int32(int(a.size)), ctypes.c_int32(int(code)))
    if rc != 1:
        out = a.copy()
    return out


def _apple_gpu_mtl4_scan_f32() -> Any:
    """Metal 4 M2 MSL-loop scan symbol. None when unavailable."""
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mtl4_scan_f32", None)
    if sym is None:
        return None
    fp = ctypes.POINTER(ctypes.c_float)
    sym.argtypes = [fp, fp, fp, fp, fp, ctypes.c_int32, ctypes.c_int32,
                    ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def apple_gpu_mtl4_scan(Wh: Any, Wx: Any, xseq: Any, init: Any, np: Any):
    """Bounded scan ``carry_{i+1} = tanh(carry_i @ Wh + x_i @ Wx)`` run as a
    hand-written MSL kernel (native in-kernel for-loop) dispatched through the
    full Metal 4 command model. Returns ``(ys[T, d], ran_on_mtl4)``; falls back
    to a numpy scan when Metal 4 is unavailable. See
    docs/apple_gpu_metal4_adoption.md."""
    Wh = np.ascontiguousarray(Wh, np.float32)
    Wx = np.ascontiguousarray(Wx, np.float32)
    xseq = np.ascontiguousarray(xseq, np.float32)
    init = np.ascontiguousarray(init, np.float32).reshape(-1)
    T, m = xseq.shape
    d = Wh.shape[0]
    ys = np.empty((T, d), np.float32)
    sym = _apple_gpu_mtl4_scan_f32()
    fp = ctypes.POINTER(ctypes.c_float)
    rc = 0
    if sym is not None:
        rc = sym(Wh.ctypes.data_as(fp), Wx.ctypes.data_as(fp),
                 xseq.ctypes.data_as(fp), init.ctypes.data_as(fp),
                 ys.ctypes.data_as(fp), ctypes.c_int32(T), ctypes.c_int32(d),
                 ctypes.c_int32(m))
    if rc != 1:
        carry = init.astype(np.float64)
        for t in range(T):
            carry = np.tanh(carry @ Wh.astype(np.float64)
                            + xseq[t].astype(np.float64) @ Wx.astype(np.float64))
            ys[t] = carry.astype(np.float32)
    return ys, rc == 1


def _apple_gpu_mtl4_matmul_sg_f32() -> Any:
    """Metal 4 M3 cooperative-matrix matmul symbol. None when unavailable."""
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mtl4_matmul_sg_f32", None)
    if sym is None:
        return None
    fp = ctypes.POINTER(ctypes.c_float)
    sym.argtypes = [fp, fp, fp, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def apple_gpu_mtl4_matmul_sg(A: Any, B: Any, np: Any):
    """``C = A @ B`` via an MSL cooperative-matrix kernel (``simdgroup_matrix``,
    the GPU matrix-unit path) dispatched through the Metal 4 command model.
    ``M``, ``N``, ``K`` must be multiples of 8. Returns ``(C[M, N], ran_on_mtl4)``;
    falls back to numpy when Metal 4 / the 8-tile envelope doesn't apply. See
    docs/apple_gpu_metal4_adoption.md."""
    A = np.ascontiguousarray(A, np.float32)
    B = np.ascontiguousarray(B, np.float32)
    M, K = A.shape
    K2, N = B.shape
    C = np.empty((M, N), np.float32)
    rc = 0
    if K2 == K and M % 8 == 0 and N % 8 == 0 and K % 8 == 0:
        sym = _apple_gpu_mtl4_matmul_sg_f32()
        if sym is not None:
            fp = ctypes.POINTER(ctypes.c_float)
            rc = sym(A.ctypes.data_as(fp), B.ctypes.data_as(fp),
                     C.ctypes.data_as(fp), ctypes.c_int32(M), ctypes.c_int32(N),
                     ctypes.c_int32(K))
    if rc != 1:
        C = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
    return C, rc == 1


def _apple_gpu_mtl4_matmul2d_f16_sym() -> Any:
    """Metal 4 M6 MPP matmul2d fp16 tensor-op symbol. None when unavailable."""
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mtl4_matmul2d_f16", None)
    if sym is None:
        return None
    u16 = ctypes.POINTER(ctypes.c_uint16)
    fp = ctypes.POINTER(ctypes.c_float)
    sym.argtypes = [u16, u16, fp, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def apple_gpu_mtl4_matmul2d_f16(A: Any, B: Any, np: Any):
    """``C[f32] = A[f16] @ B[f16]`` via the MSL 4.0 cooperative ``tensor`` op
    (MetalPerformancePrimitives ``matmul2d``) on the GPU matrix units, dispatched
    through the Metal 4 command model with MTLTensor-bound arguments. Unlike the
    ``simdgroup_matrix`` f32 kernel (which tops out ~80% of MPS), this fp16 path
    *beats* MPS fp16 (~1.1-1.18x at N=1024-2048). Any ``M``/``N``/``K`` (matmul2d
    edge-checks partial tiles). Returns ``(C[M, N] float32, ran_on_mtl4)``; falls
    back to a numpy fp16 reference when Metal 4 is unavailable. See
    docs/apple_gpu_metal4_adoption.md (M6)."""
    A = np.ascontiguousarray(A, np.float16)
    B = np.ascontiguousarray(B, np.float16)
    M, K = A.shape
    K2, N = B.shape
    C = np.empty((M, N), np.float32)
    rc = 0
    if K2 == K:
        sym = _apple_gpu_mtl4_matmul2d_f16_sym()
        if sym is not None:
            u16 = ctypes.POINTER(ctypes.c_uint16)
            fp = ctypes.POINTER(ctypes.c_float)
            rc = sym(A.view(np.uint16).ctypes.data_as(u16),
                     B.view(np.uint16).ctypes.data_as(u16),
                     C.ctypes.data_as(fp), ctypes.c_int32(M), ctypes.c_int32(N),
                     ctypes.c_int32(K))
    if rc != 1:
        C = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float32)
    return C, rc == 1


def _apple_gpu_mtl4_matmul2d_sym(name: str, *, fused: bool) -> Any:
    """Loader for an MPP matmul2d symbol (plain or fused-epilogue). None when
    the symbol is missing."""
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, name, None)
    if sym is None:
        return None
    u16 = ctypes.POINTER(ctypes.c_uint16)
    fp = ctypes.POINTER(ctypes.c_float)
    i32 = ctypes.c_int32
    if fused:
        # (A, B, C, bias, act, M, N, K)
        sym.argtypes = [u16, u16, fp, fp, i32, i32, i32, i32]
    else:
        # (A, B, C, M, N, K)
        sym.argtypes = [u16, u16, fp, i32, i32, i32]
    sym.restype = i32
    return sym


def apple_gpu_mtl4_matmul2d_bf16(A: Any, B: Any, np: Any):
    """``C[f32] = A[bf16] @ B[bf16]`` via the MSL 4.0 cooperative ``tensor`` op
    (MetalPerformancePrimitives ``matmul2d``) on the GPU matrix units — the bf16
    sibling of :func:`apple_gpu_mtl4_matmul2d_f16`, same MTLTensor-bound MTL4 path
    and 64x64/4-SIMD-group kernel. Returns ``(C[M, N] float32, ran_on_mtl4)``;
    numpy fallback off Metal 4. See docs/apple_gpu_metal4_adoption.md (M6)."""
    bf16 = _bfloat16_dtype()
    if bf16 is None:
        raise RuntimeError("bf16 matmul2d requires the optional ml_dtypes package")
    A = np.ascontiguousarray(A, bf16)
    B = np.ascontiguousarray(B, bf16)
    M, K = A.shape
    K2, N = B.shape
    C = np.empty((M, N), np.float32)
    rc = 0
    if K2 == K:
        sym = _apple_gpu_mtl4_matmul2d_sym("tessera_apple_gpu_mtl4_matmul2d_bf16",
                                           fused=False)
        if sym is not None:
            u16 = ctypes.POINTER(ctypes.c_uint16)
            fp = ctypes.POINTER(ctypes.c_float)
            rc = sym(A.view(np.uint16).ctypes.data_as(u16),
                     B.view(np.uint16).ctypes.data_as(u16),
                     C.ctypes.data_as(fp), ctypes.c_int32(M), ctypes.c_int32(N),
                     ctypes.c_int32(K))
    if rc != 1:
        C = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float32)
    return C, rc == 1


_MTL4_EPILOGUE_ACT = {"none": 0, "relu": 1, "gelu": 2, "silu": 3}


def apple_gpu_mtl4_matmul2d_epilogue(A: Any, B: Any, np: Any, *, bias: Any = None,
                                     act: str = "none", dtype: str = "f16"):
    """Fused ``C[f32] = act(A @ B + bias)`` in one MPP ``matmul2d`` dispatch — the
    bias (per output column, length ``N``) and activation are applied IN-REGISTER
    on the float ``cooperative_tensor`` before the single store, so there is no
    extra device round-trip. ``dtype`` selects the f16/bf16 input kernel; ``act``
    is one of ``none``/``relu``/``gelu``/``silu``. Returns ``(C[M, N] float32,
    ran_on_mtl4)``; numpy fallback off Metal 4. See
    docs/apple_gpu_metal4_adoption.md (M7)."""
    if act not in _MTL4_EPILOGUE_ACT:
        raise ValueError(f"act must be one of {sorted(_MTL4_EPILOGUE_ACT)}; got {act!r}")
    act_code = _MTL4_EPILOGUE_ACT[act]
    if dtype == "bf16":
        in_dtype = _bfloat16_dtype()
        if in_dtype is None:
            raise RuntimeError("bf16 epilogue requires the optional ml_dtypes package")
        entry = "tessera_apple_gpu_mtl4_matmul2d_epilogue_bf16"
    elif dtype == "f16":
        in_dtype = np.float16
        entry = "tessera_apple_gpu_mtl4_matmul2d_epilogue_f16"
    else:
        raise ValueError(f"dtype must be 'f16' or 'bf16'; got {dtype!r}")
    A = np.ascontiguousarray(A, in_dtype)
    B = np.ascontiguousarray(B, in_dtype)
    M, K = A.shape
    K2, N = B.shape
    bias_arr = None
    if bias is not None:
        bias_arr = np.ascontiguousarray(bias, np.float32).reshape(-1)
        if bias_arr.shape[0] != N:
            raise ValueError(f"bias length must be N={N}; got {bias_arr.shape[0]}")
    C = np.empty((M, N), np.float32)
    rc = 0
    if K2 == K:
        sym = _apple_gpu_mtl4_matmul2d_sym(entry, fused=True)
        if sym is not None:
            u16 = ctypes.POINTER(ctypes.c_uint16)
            fp = ctypes.POINTER(ctypes.c_float)
            bias_ptr = bias_arr.ctypes.data_as(fp) if bias_arr is not None else None
            rc = sym(A.view(np.uint16).ctypes.data_as(u16),
                     B.view(np.uint16).ctypes.data_as(u16),
                     C.ctypes.data_as(fp), bias_ptr, ctypes.c_int32(act_code),
                     ctypes.c_int32(M), ctypes.c_int32(N), ctypes.c_int32(K))
    if rc != 1:
        C = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float32)
        if bias_arr is not None:
            C = C + bias_arr[None, :]
        C = _mtl4_epilogue_act_numpy(C, act_code, np)
    return C, rc == 1


def _mtl4_epilogue_act_numpy(x: Any, act_code: int, np: Any) -> Any:
    """numpy reference matching the in-kernel epilogue activations (M7)."""
    if act_code == 1:
        return np.maximum(0.0, x)
    if act_code == 2:  # gelu (tanh approximation, matches the MSL kernel)
        t = 0.7978845608028654 * (x + 0.044715 * x ** 3)
        return 0.5 * x * (1.0 + np.tanh(t))
    if act_code == 3:  # silu
        return x / (1.0 + np.exp(-x))
    return x


class AppleGPUMLPSession:
    """M8 — a resident-weight fused MLP-block session for the Metal 4 lane.

    The weight ``W`` (+ optional bias) is uploaded to the GPU **once** and kept
    resident; the pipeline, residency set, and command queue are reused across
    :meth:`run` calls. Each step only uploads the (small) activation ``X`` and
    dispatches one fused ``matmul2d`` epilogue — ``Y = act(X @ W + bias)``. This
    amortizes the per-call MTL4 overhead (re-uploading ``W``, re-committing
    residency) that otherwise makes the lane slower than MPS at decode (small-M)
    sizes. ``act`` ∈ {none, relu, gelu, silu}; ``dtype`` ∈ {f16, bf16}. Falls
    back to a numpy reference when Metal 4 is unavailable. Use as a context
    manager or call :meth:`close` to release the resident weights. See
    docs/apple_gpu_metal4_adoption.md (M8)."""

    def __init__(self, W: Any, np: Any, *, bias: Any = None, act: str = "none",
                 dtype: str = "f16"):
        if act not in _MTL4_EPILOGUE_ACT:
            raise ValueError(f"act must be one of {sorted(_MTL4_EPILOGUE_ACT)}; got {act!r}")
        self._np = np
        self._act_code = _MTL4_EPILOGUE_ACT[act]
        self._handle = None
        if dtype == "bf16":
            self._in_dtype = _bfloat16_dtype()
            if self._in_dtype is None:
                raise RuntimeError("bf16 session requires the optional ml_dtypes package")
            bf16_flag = 1
        elif dtype == "f16":
            self._in_dtype = np.float16
            bf16_flag = 0
        else:
            raise ValueError(f"dtype must be 'f16' or 'bf16'; got {dtype!r}")
        W = np.ascontiguousarray(W, self._in_dtype)
        self._K, self._N = int(W.shape[0]), int(W.shape[1])
        self._bias = None
        if bias is not None:
            self._bias = np.ascontiguousarray(bias, np.float32).reshape(-1)
            if self._bias.shape[0] != self._N:
                raise ValueError(f"bias length must be N={self._N}; got {self._bias.shape[0]}")
        # Keep host copies so the numpy fallback (and repr) stays correct.
        self._W = W
        create = _apple_gpu_mtl4_mlp_session_create_sym()
        if create is not None:
            u16 = ctypes.POINTER(ctypes.c_uint16)
            fp = ctypes.POINTER(ctypes.c_float)
            bias_ptr = self._bias.ctypes.data_as(fp) if self._bias is not None else None
            h = create(W.view(np.uint16).ctypes.data_as(u16), bias_ptr,
                       ctypes.c_int32(self._act_code), ctypes.c_int32(self._K),
                       ctypes.c_int32(self._N), ctypes.c_int32(bf16_flag))
            self._handle = h if h else None

    @property
    def ran_on_gpu(self) -> bool:
        return self._handle is not None

    def run(self, X: Any) -> Any:
        """``Y[M,N] (f32) = act(X[M,K] @ W + bias)`` for this step's ``X``."""
        np = self._np
        X = np.ascontiguousarray(X, self._in_dtype)
        M = int(X.shape[0])
        if int(X.shape[1]) != self._K:
            raise ValueError(f"X has K={X.shape[1]}; session expects K={self._K}")
        if self._handle is not None:
            run = _apple_gpu_mtl4_mlp_session_run_sym()
            if run is not None:
                Y = np.empty((M, self._N), np.float32)
                u16 = ctypes.POINTER(ctypes.c_uint16)
                fp = ctypes.POINTER(ctypes.c_float)
                rc = run(self._handle, X.view(np.uint16).ctypes.data_as(u16),
                         Y.ctypes.data_as(fp), ctypes.c_int32(M))
                if rc == 1:
                    return Y
        # numpy fallback
        Y = (X.astype(np.float32) @ self._W.astype(np.float32)).astype(np.float32)
        if self._bias is not None:
            Y = Y + self._bias[None, :]
        return _mtl4_epilogue_act_numpy(Y, self._act_code, np)

    def run_dev(self, X: Any, Y: Any = None) -> Any:
        """R0-bridge device-resident step: ``X`` is a :class:`DeviceTensor`
        already on the GPU (f16/bf16, shape ``[M, K]``); returns a resident
        :class:`DeviceTensor` ``Y[M,N] (f32) = act(X @ W + bias)`` with **no host
        round-trip** — X is not uploaded and Y is not downloaded, so a decode loop
        keeps the activation resident across steps. Allocates ``Y`` if not given.
        Falls back to a host compute (writing into ``Y``'s shared storage) when
        Metal 4 is unavailable. See docs/apple_gpu_metal4_adoption.md (M8/R0)."""
        np = self._np
        if not isinstance(X, DeviceTensor):
            raise TypeError("run_dev expects a DeviceTensor X (use DeviceTensor.from_numpy)")
        if len(X.shape) != 2 or int(X.shape[1]) != self._K:
            raise ValueError(f"X must be [M, {self._K}]; got shape {X.shape}")
        M = int(X.shape[0])
        if Y is None:
            Y = DeviceTensor.empty((M, self._N), np.float32)
            if Y is None:
                raise RuntimeError("device-tensor ABI unavailable for run_dev")
        elif not isinstance(Y, DeviceTensor) or tuple(Y.shape) != (M, self._N):
            raise ValueError(f"Y must be a DeviceTensor of shape ({M}, {self._N})")
        if self._handle is not None:
            run = _apple_gpu_mtl4_mlp_session_run_dev_sym()
            if run is not None:
                rc = run(self._handle, X.handle, Y.handle, ctypes.c_int32(M))
                if rc == 1:
                    return Y
        # host fallback — compute into Y's shared storage (stays "resident")
        Xh = X.numpy().astype(np.float32)
        Yh = Xh @ self._W.astype(np.float32)
        if self._bias is not None:
            Yh = Yh + self._bias[None, :]
        Yh = _mtl4_epilogue_act_numpy(Yh, self._act_code, np)
        Y.numpy()[...] = Yh.astype(np.float32)
        return Y

    def close(self) -> None:
        if self._handle is not None:
            destroy = _apple_gpu_mtl4_mlp_session_destroy_sym()
            if destroy is not None:
                destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def apple_gpu_mtl4_archive_enable(path: str) -> bool:
    """P4 — enable MTL4Archive pipeline persistence (opt-in). Loads an archive at
    ``path`` (if present) so matching MTL4 pipelines skip the MSL recompile on
    process start, and captures subsequently-built pipelines for a later
    :func:`apple_gpu_mtl4_archive_flush`. No effect on the default path; returns
    True if enabled (Metal 4 available). See docs/apple_backend_integration_review.md
    (P4)."""
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mtl4_archive_enable", None)
    if sym is None:
        return False
    sym.argtypes = [ctypes.c_char_p]
    sym.restype = ctypes.c_int32
    return bool(sym(str(path).encode("utf-8")))


def apple_gpu_mtl4_archive_flush() -> bool:
    """P4 — flush captured MTL4 pipelines to the enabled archive path. Call after
    warming up the kernels of interest. Returns True on success."""
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mtl4_archive_flush", None)
    if sym is None:
        return False
    sym.argtypes = []
    sym.restype = ctypes.c_int32
    return bool(sym())


def _apple_gpu_mtl4_mlp_session_create_sym() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mtl4_mlp_session_create", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    sym.restype = ctypes.c_void_p
    return sym


def _apple_gpu_mtl4_mlp_session_run_sym() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mtl4_mlp_session_run", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16),
                    ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def _apple_gpu_mtl4_mlp_session_run_dev_sym() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mtl4_mlp_session_run_dev", None)
    if sym is None:
        return None
    # (handle, X devtensor, Y devtensor, M) — X/Y are opaque void* handles.
    sym.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    sym.restype = ctypes.c_int32
    return sym


def _apple_gpu_mtl4_mlp_session_destroy_sym() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mtl4_mlp_session_destroy", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.c_void_p]
    sym.restype = None
    return sym


# ── M4 — capability-gated routing of real ops onto the Metal 4 lane ───────────
# OFF by default: the MTL4 cooperative-matrix kernel is *correct* but still
# slower than the tuned MPS matmul end-to-end. M5 added a register-blocked,
# vectorized 64x64 fast kernel (~2.8x the old 8x8-per-threadgroup kernel,
# ~80% of MPS in pure GPU-kernel time, tying MPS around N=1024), but MTL4's
# per-call dispatch + buffer-copy overhead keeps the end-to-end path at/above
# MPS. The MSL 4.0 `tensor` cooperative op (MPP matmul2d) has no f32 path under
# execution_simdgroups (fp16/bf16 only) so it is not the f32 answer. The
# mechanism is here so flipping it on is a one-liner if a future kernel clears
# MPS. See docs/apple_gpu_metal4_adoption.md (M5).
_MTL4_ROUTING_ENABLED = os.environ.get("TESSERA_APPLE_GPU_MTL4_ROUTE", "0") in ("1", "true", "True")
_MTL4_CAPS_CACHE: dict | None = None


def _mtl4_caps_cached() -> dict:
    """Probe Metal 4 capabilities once (the probe creates GPU objects, so it is
    not cheap to call per-op) and cache the result."""
    global _MTL4_CAPS_CACHE
    if _MTL4_CAPS_CACHE is None:
        _MTL4_CAPS_CACHE = apple_gpu_metal4_caps()
    return _MTL4_CAPS_CACHE


def set_apple_gpu_mtl4_routing(enabled: bool) -> None:
    """Enable/disable routing eligible ops (today: rank-2 f32 matmul with
    8-multiple M/N/K) onto the Metal 4 lane. Also settable at import via
    ``TESSERA_APPLE_GPU_MTL4_ROUTE=1``. See docs/apple_gpu_metal4_adoption.md."""
    global _MTL4_ROUTING_ENABLED
    _MTL4_ROUTING_ENABLED = bool(enabled)


def apple_gpu_mtl4_routing_enabled() -> bool:
    return _MTL4_ROUTING_ENABLED


def _mtl4_route_matmul_f32(a: Any, b: Any, np: Any) -> Any:
    """Capability + envelope gate: return the MTL4-routed matmul result, or
    ``None`` to fall back to the (default) MPS path."""
    if not _MTL4_ROUTING_ENABLED:
        return None
    if a.ndim != 2 or b.ndim != 2 or a.dtype != np.float32 or b.dtype != np.float32:
        return None
    M, K = a.shape
    K2, N = b.shape
    if K2 != K or M % 8 or N % 8 or K % 8:
        return None
    caps = _mtl4_caps_cached()
    if not (caps.get("command_queue") and caps.get("compiler")):
        return None
    C, ran = apple_gpu_mtl4_matmul_sg(a, b, np)
    return C if ran else None


# P5 — native bf16 matmul is the DEFAULT on apple_gpu (unlike the f32 lane, which
# stays opt-in). MPS has no native bf16 GEMM, so the legacy bf16 path converts to
# fp32 on the host; the MPP `matmul2d` tensor-op is ~10x faster, so it is the
# better default whenever Metal 4 is available. Toggle with
# TESSERA_APPLE_GPU_MTL4_BF16=0 / set_apple_gpu_mtl4_bf16_default(False) (e.g. to
# force the legacy path for comparison). See docs/apple_gpu_metal4_adoption.md (P5).
_MTL4_BF16_DEFAULT = os.environ.get("TESSERA_APPLE_GPU_MTL4_BF16", "1") not in ("0", "false", "False")


def set_apple_gpu_mtl4_bf16_default(enabled: bool) -> None:
    global _MTL4_BF16_DEFAULT
    _MTL4_BF16_DEFAULT = bool(enabled)


def apple_gpu_mtl4_bf16_default_enabled() -> bool:
    return _MTL4_BF16_DEFAULT


def _mtl4_route_matmul2d_bf16(a: Any, b: Any, np: Any) -> Any:
    """Default bf16 matmul → native MPP ``matmul2d`` (beats the fp32-conversion
    MPS fallback ~10x). Returns a bf16 result (the f32 accumulator cast back to
    bf16 to preserve the bf16-in/bf16-out contract), or ``None`` to fall back."""
    if not _MTL4_BF16_DEFAULT:
        return None
    bf16 = _bfloat16_dtype()
    if bf16 is None or a.dtype != bf16 or b.dtype != bf16 or a.ndim != 2 or b.ndim != 2:
        return None
    if a.shape[1] != b.shape[0]:
        return None
    caps = _mtl4_caps_cached()
    if not (caps.get("command_queue") and caps.get("compiler")):
        return None
    C, ran = apple_gpu_mtl4_matmul2d_bf16(a, b, np)  # f32 output
    return C.astype(bf16) if ran else None


# P7 (2026-06-01) — fp16 matmul routing onto the MTL4 tensor-op.
#
# Unlike bf16 (where MTL4 beats the MPS *fallback* across the board), fp16
# has a well-tuned MPS GEMM — MTL4 only clears it in ONE regime: the M==1
# GEMV decode step, where MPS's fp16 path is slow and MTL4 wins a robust
# 3.2-3.4x (measured, 3 trials, tight variance — see
# benchmarks/apple_gpu/benchmark_mtl4_matmul_routing.py). At M>=2 and on
# square shapes MPS wins (0.7-0.9x), so the default is strictly size-gated
# to M==1; everything else stays on MPS. ``TESSERA_APPLE_GPU_MTL4_F16``:
#   unset / "1" / "auto" → route M==1 only (the measured win)   [default]
#   "all"                → route every fp16 2-D matmul (benchmarking)
#   "0" / "false"        → never route fp16 (force MPS)
_MTL4_F16_MODE = os.environ.get("TESSERA_APPLE_GPU_MTL4_F16", "auto").lower()


def set_apple_gpu_mtl4_f16_mode(mode: str) -> None:
    """Set the fp16 MTL4 routing mode: 'auto' (M==1 only), 'all', or 'off'."""
    global _MTL4_F16_MODE
    _MTL4_F16_MODE = str(mode).lower()


def apple_gpu_mtl4_f16_mode() -> str:
    return _MTL4_F16_MODE


def _mtl4_route_matmul2d_f16(a: Any, b: Any, np: Any) -> Any:
    """fp16 matmul → native MPP ``matmul2d``, size-gated. Returns an fp16
    result (f32 accumulator cast back to fp16 to preserve the contract) when
    the gate fires, else ``None`` to fall back to MPS.

    Default mode ``auto`` routes ONLY the M==1 GEMV decode step (the one
    regime where MTL4 clears MPS, ~3.3x); ``all`` routes every fp16 2-D
    matmul (for benchmarking — regresses square shapes); ``off`` disables."""
    mode = _MTL4_F16_MODE
    if mode in ("0", "false", "off", "none"):
        return None
    if a.dtype != np.float16 or b.dtype != np.float16:
        return None
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        return None
    # Size gate: default ('auto'/'1') only routes the M==1 decode GEMV.
    if mode not in ("all",) and a.shape[0] != 1:
        return None
    caps = _mtl4_caps_cached()
    if not (caps.get("command_queue") and caps.get("compiler")):
        return None
    C, ran = apple_gpu_mtl4_matmul2d_f16(a, b, np)  # f32 output
    return C.astype(np.float16) if ran else None


def apple_gpu_msl_spec_accept(draft_paths: Any, target_greedy: Any, np: Any):
    """Phase-G Rung 3 — the dynamic speculative-verify control flow as one MSL
    kernel. ``draft_paths`` is ``[P, depth]`` candidate tokens; ``target_greedy``
    is ``[P, depth+1]`` the target's greedy token at each position along each
    path. Per path the kernel accepts draft tokens while they match (breaking at
    the first mismatch — a data-dependent trip count), keeps the longest accepted
    prefix, and returns the bonus (the target's correction). Returns
    ``(best_path, accepted_len, bonus, accepted_tokens)``. The C reference (and
    a numpy fallback) keep the contract off Metal."""
    draft = np.ascontiguousarray(draft_paths, np.int32)
    target = np.ascontiguousarray(target_greedy, np.int32)
    P, depth = draft.shape
    if target.shape != (P, depth + 1):
        raise ValueError(f"target_greedy must be [P, depth+1]={(P, depth + 1)}; "
                         f"got {target.shape}")
    out = np.empty(3 + depth, np.int32)
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_msl_spec_accept", None)
    rc = 0
    if sym is not None:
        ip = ctypes.POINTER(ctypes.c_int32)
        sym.argtypes = [ip, ip, ip, ctypes.c_int32, ctypes.c_int32]
        sym.restype = ctypes.c_int32
        rc = sym(draft.ctypes.data_as(ip), target.ctypes.data_as(ip),
                 out.ctypes.data_as(ip), ctypes.c_int32(P), ctypes.c_int32(depth))
    if rc != 1:
        best_path, best_len, best_bonus = 0, -1, 0
        for p in range(P):
            length = 0
            for i in range(depth):
                if int(draft[p, i]) == int(target[p, i]):
                    length += 1
                else:
                    break
            if length > best_len:
                best_len, best_path = length, p
                best_bonus = int(target[p, length])
        out[0], out[1], out[2] = best_path, best_len, best_bonus
        for i in range(depth):
            out[3 + i] = int(draft[best_path, i]) if i < best_len else -1
    accepted_len = int(out[1])
    return (int(out[0]), accepted_len, int(out[2]),
            [int(out[3 + i]) for i in range(accepted_len)])


def _apple_gpu_layer_norm_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_layer_norm_f32", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
    ]
    sym.restype = None
    return sym


def _apple_gpu_rmsnorm_gpu_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_rmsnorm_gpu_f32", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
    ]
    sym.restype = None
    return sym


def _apple_gpu_log_softmax_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_log_softmax_f32", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_mpsgraph_softmax_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_mpsgraph_softmax_f32", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_unary_numpy(op_name: str, x: Any, np: Any) -> Any:
    """Host reference matching apple_gpu_runtime.mm mpsg_unary_node."""
    f = x.astype(np.float32)
    return {
        "tessera.relu": lambda v: np.maximum(0.0, v),
        "tessera.sigmoid": lambda v: 1.0 / (1.0 + np.exp(-v)),
        "tessera.sigmoid_safe": lambda v: 1.0 / (1.0 + np.exp(-v)),
        "tessera.tanh": np.tanh,
        "tessera.softplus": lambda v: np.maximum(v, 0.0) + np.log1p(np.exp(-np.abs(v))),
        "tessera.silu": lambda v: v / (1.0 + np.exp(-v)),
        "tessera.exp": np.exp,
        "tessera.log": np.log,
        "tessera.sqrt": np.sqrt,
        "tessera.rsqrt": lambda v: 1.0 / np.sqrt(v),
        "tessera.neg": lambda v: -v,
        "tessera.negative": lambda v: -v,
        "tessera.abs": np.abs,
        "tessera.absolute": np.abs,
        # Batch 1 — float-output elementwise math.
        "tessera.sin": np.sin, "tessera.cos": np.cos, "tessera.tan": np.tan,
        "tessera.asin": np.arcsin, "tessera.acos": np.arccos, "tessera.atan": np.arctan,
        "tessera.sinh": np.sinh, "tessera.cosh": np.cosh,
        "tessera.erf": _np_erf, "tessera.erfc": lambda v: 1.0 - _np_erf(v),
        "tessera.expm1": np.expm1, "tessera.log1p": np.log1p,
        "tessera.reciprocal": lambda v: 1.0 / v, "tessera.sign": np.sign,
        "tessera.floor": np.floor, "tessera.ceil": np.ceil,
        "tessera.round": np.round, "tessera.trunc": np.trunc,
        # Batch 2 — unary predicates / logical / bitwise → f32 mask.
        "tessera.isfinite": lambda v: np.isfinite(v).astype(np.float32),
        "tessera.isinf": lambda v: np.isinf(v).astype(np.float32),
        "tessera.isnan": lambda v: np.isnan(v).astype(np.float32),
        "tessera.logical_not": lambda v: (v == 0.0).astype(np.float32),
        "tessera.bitwise_not": lambda v: (~v.astype(np.int32)).astype(np.float32),
    }[op_name](f)


def _np_erf(v: Any) -> Any:
    """erf without scipy — Abramowitz-Stegun 7.1.26 (matches MPSGraph/std::erf to
    ~1e-7, adequate for the CI/non-Darwin reference path)."""
    import numpy as _np
    s = _np.sign(v)
    a = _np.abs(v)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
               - 0.284496736) * t + 0.254829592) * t * _np.exp(-a * a)
    return (s * y).astype(_np.float32)


def _apple_gpu_dispatch_unary(op_name: str, operands: list[Any], np: Any) -> Any:
    """Elementwise unary via the MPSGraph lane. Shape-agnostic (flattened).
    f32 + f16 run natively on the GPU; bf16 upcasts to f32, runs, downcasts."""
    if len(operands) < 1:
        raise ValueError(f"{op_name!r} requires one operand")
    op = _APPLE_GPU_UNARY_OPCODES[op_name]
    x = np.asarray(operands[0])
    shape = x.shape
    n = int(x.size)
    bf16_dtype = _bfloat16_dtype()

    if x.dtype == np.float32:
        xf = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        sym = _apple_gpu_mpsgraph_unary_f32()
        if sym is not None:
            out = np.empty(n, dtype=np.float32)
            sym(ctypes.c_int32(op),
                xf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int64(n))
            return out.reshape(shape)
        return _apple_gpu_unary_numpy(op_name, x, np).reshape(shape)

    if x.dtype == np.float16:
        xf = np.ascontiguousarray(x, dtype=np.float16).reshape(-1)
        sym = _apple_gpu_mpsgraph_unary_f16()
        if sym is not None:
            out = np.empty(n, dtype=np.float16)
            sym(ctypes.c_int32(op),
                xf.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int64(n))
            return out.reshape(shape)
        return _apple_gpu_unary_numpy(op_name, x, np).astype(np.float16).reshape(shape)

    if bf16_dtype is not None and x.dtype == bf16_dtype:
        # Upcast -> GPU f32 -> downcast (mirrors the bf16 matmul path).
        x32 = np.ascontiguousarray(x.astype(np.float32)).reshape(-1)
        sym = _apple_gpu_mpsgraph_unary_f32()
        if sym is not None:
            out = np.empty(n, dtype=np.float32)
            sym(ctypes.c_int32(op),
                x32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int64(n))
            return out.reshape(shape).astype(bf16_dtype)
        return _apple_gpu_unary_numpy(op_name, x, np).astype(bf16_dtype).reshape(shape)

    return _apple_gpu_unary_numpy(op_name, x, np).astype(x.dtype).reshape(shape)


def _apple_gpu_binary_numpy(op_name: str, a: Any, b: Any, np: Any) -> Any:
    """Host reference matching apple_gpu_runtime.mm mpsg_binary_node."""
    return {
        "tessera.add": lambda x, y: x + y,
        "tessera.sub": lambda x, y: x - y,
        "tessera.mul": lambda x, y: x * y,
        "tessera.div": lambda x, y: x / y,
        "tessera.maximum": np.maximum,
        "tessera.minimum": np.minimum,
        "tessera.pow": np.power,
        "tessera.atan2": np.arctan2,
        "tessera.mod": lambda x, y: x - y * np.floor(x / y),
        "tessera.floor_div": lambda x, y: np.floor(x / y),
        "tessera.eq": lambda x, y: (x == y).astype(np.float32),
        "tessera.ne": lambda x, y: (x != y).astype(np.float32),
        "tessera.lt": lambda x, y: (x < y).astype(np.float32),
        "tessera.le": lambda x, y: (x <= y).astype(np.float32),
        "tessera.gt": lambda x, y: (x > y).astype(np.float32),
        "tessera.ge": lambda x, y: (x >= y).astype(np.float32),
        "tessera.logical_and": lambda x, y: ((x != 0) & (y != 0)).astype(np.float32),
        "tessera.logical_or": lambda x, y: ((x != 0) | (y != 0)).astype(np.float32),
        "tessera.logical_xor": lambda x, y: ((x != 0) ^ (y != 0)).astype(np.float32),
        "tessera.bitwise_and": lambda x, y: (x.astype(np.int32) & y.astype(np.int32)).astype(np.float32),
        "tessera.bitwise_or": lambda x, y: (x.astype(np.int32) | y.astype(np.int32)).astype(np.float32),
        "tessera.bitwise_xor": lambda x, y: (x.astype(np.int32) ^ y.astype(np.int32)).astype(np.float32),
    }[op_name](a, b)


def _apple_gpu_dispatch_mpsgraph_binary(op_name: str, operands: list[Any],
                                        kwargs: dict, np: Any) -> Any:
    """Elementwise binary via the MPSGraph lane. Broadcasts the two operands on
    the host, runs the f32 kernel element-wise (comparison ops → f32 0/1 mask).
    The scalar second-operand form (`add(x, scalar=s)`) is supported."""
    op = _APPLE_GPU_BINARY_OPCODES[op_name]
    a = np.asarray(operands[0], dtype=np.float32)
    if len(operands) >= 2:
        b = np.asarray(operands[1], dtype=np.float32)
    else:
        sc = kwargs.get("scalar", kwargs.get("other"))
        if sc is None:
            raise ValueError(f"{op_name!r} needs a second operand or a scalar= kwarg")
        b = np.asarray(sc, dtype=np.float32)
    a, b = np.broadcast_arrays(a, b)
    shape = a.shape
    n = int(a.size)
    af = np.ascontiguousarray(a).reshape(-1)
    bf = np.ascontiguousarray(b).reshape(-1)
    sym = _apple_gpu_mpsgraph_binary_f32()
    if sym is not None and n > 0:
        out = np.empty(n, dtype=np.float32)
        sym(ctypes.c_int32(op),
            af.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            bf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(n))
        return out.reshape(shape)
    return _apple_gpu_binary_numpy(op_name, a, b, np).reshape(shape)


def _apple_gpu_dispatch_clamp(op_name: str, operands: list[Any], kwargs: dict,
                              np: Any) -> Any:
    """clamp/clip(x, lo, hi) = maximum(minimum(x, hi), lo) — composed on the GPU
    binary lane (scalar bounds). Either bound may be None (one-sided)."""
    res = np.asarray(operands[0], dtype=np.float32)
    if len(operands) >= 3:
        lo, hi = operands[1], operands[2]
    else:
        lo = kwargs.get("min", kwargs.get("a_min"))
        hi = kwargs.get("max", kwargs.get("a_max"))
    if hi is not None:
        res = _apple_gpu_dispatch_mpsgraph_binary("tessera.minimum", [res], {"scalar": hi}, np)
    if lo is not None:
        res = _apple_gpu_dispatch_mpsgraph_binary("tessera.maximum", [res], {"scalar": lo}, np)
    return np.asarray(res, dtype=np.float32)


def _apple_gpu_dispatch_where(operands: list[Any], np: Any) -> Any:
    """where(cond, a, b) = cond*a + (1−cond)*b — composed on the GPU binary lane
    (cond as a 0/1 f32 mask)."""
    bn = _apple_gpu_dispatch_mpsgraph_binary
    c = np.asarray(operands[0], dtype=np.float32)
    a = np.asarray(operands[1], dtype=np.float32)
    b = np.asarray(operands[2], dtype=np.float32)
    c, a, b = np.broadcast_arrays(c, a, b)
    ca = bn("tessera.mul", [np.ascontiguousarray(c), np.ascontiguousarray(a)], {}, np)
    omc = bn("tessera.sub", [np.ascontiguousarray(c)], {"scalar": 1.0}, np)  # c − 1
    omc = bn("tessera.mul", [omc], {"scalar": -1.0}, np)                     # 1 − c
    cb = bn("tessera.mul", [omc, np.ascontiguousarray(b)], {}, np)
    return bn("tessera.add", [ca, cb], {}, np)


def _apple_gpu_dispatch_loss(op_name: str, operands: list[Any], kwargs: dict,
                             np: Any) -> Any:
    """Regression / CE losses composed entirely from the GPU opcode lanes (no
    dedicated kernel): each computes a per-element loss tensor via chained
    unary/binary/where ops, then reduces (mean/sum/none) on the GPU reduce lane.
    Falls back to the numpy reference for the integer-target cross-entropy gather
    (no GPU gather)."""
    import tessera.losses as _L
    reduction = kwargs.get("reduction", "mean")

    def u(op: str, x: Any) -> Any:
        return _apple_gpu_dispatch_unary(f"tessera.{op}", [np.asarray(x, np.float32)], np)

    def bb(op: str, x: Any, y: Any) -> Any:
        return _apple_gpu_dispatch_mpsgraph_binary(
            f"tessera.{op}", [np.asarray(x, np.float32), np.asarray(y, np.float32)], {}, np)

    def bs(op: str, x: Any, s: float) -> Any:
        return _apple_gpu_dispatch_mpsgraph_binary(
            f"tessera.{op}", [np.asarray(x, np.float32)], {"scalar": float(s)}, np)

    def reduce_all(loss: Any) -> Any:
        if reduction == "none":
            return np.asarray(loss, np.float32)
        opn = "tessera.mean" if reduction == "mean" else "tessera.reduce"
        return _apple_gpu_dispatch_reduce(opn, [np.asarray(loss, np.float32)], {"axis": None}, np)

    short = str(op_name)
    a = np.asarray(operands[0], np.float32)
    c = np.asarray(operands[1], np.float32) if len(operands) > 1 else None
    ln2 = float(np.log(2.0))

    if short in ("tessera.loss.mse", "tessera.loss.ddpm_noise_pred"):
        d = bb("sub", a, c)
        return reduce_all(bb("mul", d, d))
    if short == "tessera.loss.mae":
        return reduce_all(u("abs", bb("sub", a, c)))
    if short == "tessera.loss.vlb":
        return reduce_all(a)
    if short == "tessera.loss.log_cosh":
        err = bb("sub", a, c)
        l1 = u("log1p", u("exp", bs("mul", err, -2.0)))
        return reduce_all(bs("add", bb("add", err, l1), -ln2))
    if short == "tessera.loss.huber":
        delta = float(kwargs.get("delta", 1.0))
        err = bb("sub", a, c)
        ae = u("abs", err)
        quad = bs("mul", bb("mul", err, err), 0.5)
        lin = bs("mul", bs("add", ae, -0.5 * delta), delta)
        cond = bs("le", ae, delta)
        return reduce_all(_apple_gpu_dispatch_where([cond, quad, lin], np))
    if short == "tessera.loss.smooth_l1":
        beta = float(kwargs.get("beta", 1.0))
        err = u("abs", bb("sub", a, c))
        quad = bs("mul", bb("mul", err, err), 0.5 / beta)
        lin = bs("add", err, -0.5 * beta)
        cond = bs("lt", err, beta)
        return reduce_all(_apple_gpu_dispatch_where([cond, quad, lin], np))
    if short == "tessera.loss.binary_cross_entropy":
        relu_l = u("relu", a)
        lt_ = bb("mul", a, c)
        l1 = u("log1p", u("exp", bs("mul", u("abs", a), -1.0)))
        return reduce_all(bb("add", bb("sub", relu_l, lt_), l1))
    if short == "tessera.loss.cross_entropy":
        targets = np.asarray(operands[1])
        if targets.dtype.kind in "iu":
            return _L.cross_entropy_loss(a, targets, reduction=reduction)  # gather → host
        lp = _apple_gpu_dispatch_rowop("tessera.log_softmax", [a], {}, np)
        prod = bb("mul", targets, lp)
        s = _apple_gpu_dispatch_reduce("tessera.reduce", [np.asarray(prod, np.float32)], {"axis": -1}, np)
        return reduce_all(u("neg", s))

    def sum_last(x: Any) -> Any:
        return _apple_gpu_dispatch_reduce(
            "tessera.reduce", [np.asarray(x, np.float32)], {"axis": -1}, np)

    def clamp_lo(x: Any) -> Any:  # max(x, 1e-12), matches the loss reference
        return _apple_gpu_dispatch_clamp("tessera.clamp", [x], {"min": 1e-12}, np)

    if short == "tessera.loss.kl_divergence":
        # a = p_log_probs, c = q_probs.  sum(exp(p_log) * (p_log - log q)).
        p = u("exp", a)
        diff = bb("sub", a, u("log", clamp_lo(c)))
        return reduce_all(sum_last(bb("mul", p, diff)))
    if short == "tessera.loss.js_divergence":
        # a = p_probs, c = q_probs.  0.5*(KL(p||m) + KL(q||m)), m = (p+q)/2.
        m = bs("mul", bb("add", a, c), 0.5)
        lm = u("log", clamp_lo(m))
        kl_pm = sum_last(bb("mul", a, bb("sub", u("log", clamp_lo(a)), lm)))
        kl_qm = sum_last(bb("mul", c, bb("sub", u("log", clamp_lo(c)), lm)))
        return reduce_all(bs("mul", bb("add", kl_pm, kl_qm), 0.5))
    raise ValueError(f"apple_gpu loss dispatcher has no recipe for {op_name!r}")


def _apple_gpu_dispatch_norm(op_name: str, operands: list[Any], kwargs: dict,
                             np: Any) -> Any:
    """group_norm / instance_norm / weight_norm composed from the GPU rowop
    (layer_norm) + reduce lanes.  The normalized axes are folded to the last
    axis so the MPSGraph layer_norm row-op does the mean/var reduction on the
    GPU; the optional per-channel affine is a GPU mul/add with a materialized
    broadcast weight.  weight_norm uses the GPU sum-reduce lane for ||w||."""
    short = str(op_name)
    x = np.asarray(operands[0], np.float32)
    weight = operands[1] if len(operands) > 1 else kwargs.get("weight")
    bias = operands[2] if len(operands) > 2 else kwargs.get("bias")
    eps = float(kwargs.get("eps", 1e-5))

    def rownorm(folded: Any) -> Any:
        return np.asarray(
            _apple_gpu_dispatch_rowop("tessera.layer_norm", [folded], {"eps": eps}, np),
            np.float32)

    def affine(y: Any) -> Any:
        # weight/bias are per-channel [C]; broadcast over [N, C, *spatial].
        c = int(x.shape[1])
        shp = (1, c) + (1,) * (x.ndim - 2)
        if weight is not None:
            w = np.broadcast_to(np.asarray(weight, np.float32).reshape(shp), x.shape)
            y = _apple_gpu_dispatch_mpsgraph_binary(
                "tessera.mul", [np.ascontiguousarray(y), np.ascontiguousarray(w)], {}, np)
        if bias is not None:
            b = np.broadcast_to(np.asarray(bias, np.float32).reshape(shp), x.shape)
            y = _apple_gpu_dispatch_mpsgraph_binary(
                "tessera.add", [np.ascontiguousarray(np.asarray(y, np.float32)),
                                np.ascontiguousarray(b)], {}, np)
        return np.asarray(y, np.float32)

    if short == "tessera.instance_norm":
        if x.ndim < 3:
            raise ValueError(f"instance_norm expects rank >= 3; got {x.shape}")
        n, c = x.shape[:2]
        folded = np.ascontiguousarray(x.reshape(n * c, -1))
        y = rownorm(folded).reshape(x.shape)
        return affine(y)
    if short == "tessera.group_norm":
        num_groups = int(kwargs.get("num_groups", operands[1] if len(operands) > 1 else 1))
        n, c = x.shape[:2]
        if c % num_groups != 0:
            raise ValueError(f"channels {c} must be divisible by num_groups {num_groups}")
        folded = np.ascontiguousarray(x.reshape(n * num_groups, -1))
        y = rownorm(folded).reshape(x.shape)
        return affine(y)
    if short == "tessera.weight_norm":
        axis = int(kwargs.get("axis", 0))
        axis = axis if axis >= 0 else x.ndim + axis
        # Move `axis` to front, fold the rest to one column dim, sum-reduce w**2.
        moved = np.ascontiguousarray(np.moveaxis(x, axis, 0))
        rows = int(moved.shape[0])
        flat = moved.reshape(rows, -1)
        sq = _apple_gpu_dispatch_mpsgraph_binary(
            "tessera.mul", [np.ascontiguousarray(flat), np.ascontiguousarray(flat)], {}, np)
        ss = np.asarray(_apple_gpu_dispatch_reduce(
            "tessera.reduce", [np.asarray(sq, np.float32)], {"axis": -1}, np), np.float32)
        norm = np.sqrt(ss.reshape(rows, 1) + float(kwargs.get("eps", 1e-12)))
        out = (flat / norm).reshape(moved.shape)
        return np.moveaxis(out, 0, axis)
    raise ValueError(f"apple_gpu norm dispatcher has no recipe for {op_name!r}")


def _apple_gpu_rowop_numpy(kind: int, x: Any, eps: float, np: Any) -> Any:
    f = x.astype(np.float32)
    if kind == 0:  # layer_norm (unweighted)
        mu = f.mean(axis=-1, keepdims=True)
        var = f.var(axis=-1, keepdims=True)
        return (f - mu) / np.sqrt(var + eps)
    if kind == 1:  # rmsnorm (unweighted)
        return f / np.sqrt(np.mean(f * f, axis=-1, keepdims=True) + eps)
    # log_softmax
    m = f.max(axis=-1, keepdims=True)
    shifted = f - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _apple_gpu_dispatch_rowop(op_name: str, operands: list[Any], kwargs: dict,
                              np: Any) -> Any:
    """Row-wise op over the last axis via the MPSGraph lane: layer_norm /
    rmsnorm (unweighted, gamma=1 beta=0) and log_softmax. Any rank is folded
    to [rows, cols]. f32 native; f16/bf16 upcast -> GPU f32 -> downcast."""
    if len(operands) < 1:
        raise ValueError(f"{op_name!r} requires one operand")
    kind = _APPLE_GPU_ROWOP_KINDS[op_name]
    eps_default = 1e-6 if op_name == "tessera.rmsnorm_safe" else 1e-5
    eps = float(kwargs.get("eps", eps_default))
    x = np.asarray(operands[0])
    if x.ndim < 1 or x.shape[-1] < 1:
        return _apple_gpu_rowop_numpy(kind, x, eps, np).astype(x.dtype)
    shape = x.shape
    cols = int(shape[-1])
    rows = int(x.size // cols)
    out_dtype = x.dtype
    x32 = np.ascontiguousarray(x.astype(np.float32)).reshape(rows, cols)

    sym = None
    if kind == 0:
        sym = _apple_gpu_layer_norm_f32()
    elif kind == 1:
        sym = _apple_gpu_rmsnorm_gpu_f32()
    else:
        sym = _apple_gpu_log_softmax_f32()

    if sym is None:
        return _apple_gpu_rowop_numpy(kind, x, eps, np).astype(out_dtype)

    out = np.empty((rows, cols), dtype=np.float32)
    fp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    if kind == 0:
        gamma = np.ones(cols, dtype=np.float32)
        beta = np.zeros(cols, dtype=np.float32)
        sym(fp(x32), fp(gamma), fp(beta), fp(out),
            ctypes.c_int32(rows), ctypes.c_int32(cols), ctypes.c_float(eps))
    elif kind == 1:
        gamma = np.ones(cols, dtype=np.float32)
        sym(fp(x32), fp(gamma), fp(out),
            ctypes.c_int32(rows), ctypes.c_int32(cols), ctypes.c_float(eps))
    else:
        sym(fp(x32), fp(out), ctypes.c_int32(rows), ctypes.c_int32(cols))
    return out.reshape(shape).astype(out_dtype)


_TILED_SOFTMAX_N_CAP: Optional[int] = None


def _apple_threadgroup_tiled_softmax_n_cap() -> int:
    """P1 (2026-06-02) — feature-limit-derived N cap for the tiled
    matmul→softmax kernel, cached. Consults the live threadgroup-memory
    budget (``probe_apple_runtime_limits``) via the apple_target helper;
    falls back to the static per-arch floor (32 KB ⇒ 8192 fp32 scores)
    off Metal / on probe failure."""
    global _TILED_SOFTMAX_N_CAP
    if _TILED_SOFTMAX_N_CAP is None:
        from .compiler.apple_target import (
            apple_threadgroup_tiled_softmax_n_cap,
            probe_apple_runtime_limits,
        )
        try:
            limits = probe_apple_runtime_limits()
        except Exception:
            limits = None
        _TILED_SOFTMAX_N_CAP = apple_threadgroup_tiled_softmax_n_cap(
            runtime_limits=limits)
    return _TILED_SOFTMAX_N_CAP


def _apple_gpu_dispatch_silu_mul(operands: list[Any], np: Any) -> Any:
    """silu_mul(a, b) = silu(a) * b, via the MPSGraph binary lane. The runtime
    opcode 6 computes a' * silu(b'), so we pass (a'=b, b'=a) to get silu(a)*b.
    f32 native; f16/bf16 upcast -> GPU f32 -> downcast."""
    if len(operands) != 2:
        raise ValueError("silu_mul requires two operands (a, b)")
    a = np.asarray(operands[0])
    b = np.asarray(operands[1])
    shape = a.shape
    n = int(a.size)
    out_dtype = a.dtype
    sym = _apple_gpu_mpsgraph_binary_f32()

    def _ref() -> Any:
        af = a.astype(np.float32)
        bf = b.astype(np.float32)
        return (af / (1.0 + np.exp(-af))) * bf

    if sym is None:
        return _ref().astype(out_dtype).reshape(shape)

    # opcode 6 = first * silu(second); pass (b, a) -> b * silu(a) = silu(a)*b.
    first = np.ascontiguousarray(b.astype(np.float32)).reshape(-1)
    second = np.ascontiguousarray(a.astype(np.float32)).reshape(-1)
    out = np.empty(n, dtype=np.float32)
    fp = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    sym(ctypes.c_int32(6), fp(first), fp(second), fp(out), ctypes.c_int64(n))
    return out.reshape(shape).astype(out_dtype)


def _apple_gpu_dispatch_mpsgraph_softmax(x: Any, np: Any) -> Any:
    """Row softmax over the last axis via the MPSGraph lane — no N limit.
    Used to complete the f16/bf16 (and f32 N>8192) matmul->softmax chain by
    composing the GPU matmul with this epilogue. f32 native; f16/bf16 upcast
    -> GPU f32 -> downcast."""
    x = np.asarray(x)
    out_dtype = x.dtype
    if x.ndim < 1 or x.shape[-1] < 1:
        f = x.astype(np.float32)
        e = np.exp(f - f.max(axis=-1, keepdims=True))
        return (e / e.sum(axis=-1, keepdims=True)).astype(out_dtype)
    shape = x.shape
    cols = int(shape[-1])
    rows = int(x.size // cols)
    sym = _apple_gpu_mpsgraph_softmax_f32()
    if sym is None:
        f = x.astype(np.float32)
        e = np.exp(f - f.max(axis=-1, keepdims=True))
        return (e / e.sum(axis=-1, keepdims=True)).astype(out_dtype)
    x32 = np.ascontiguousarray(x.astype(np.float32)).reshape(rows, cols)
    out = np.empty((rows, cols), dtype=np.float32)
    fp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    sym(fp(x32), fp(out), ctypes.c_int32(rows), ctypes.c_int32(cols))
    return out.reshape(shape).astype(out_dtype)


def _apple_gpu_dispatch_matmul_softmax(operands: list[Any], np: Any) -> Any:
    """Phase 8.4.3 + 8.4.4.2 — dispatch a fused matmul -> softmax(axis=-1)
    chain through the apple_gpu runtime shim's purpose-built MSL kernel.
    Picks symbol by element type (f32, f16, bf16). Inputs outside the
    supported envelope (rank, mixed dtypes, N>256) fall back to a
    numpy-equivalent host computation for correctness."""

    if len(operands) != 2:
        raise ValueError("matmul_softmax fusion requires two operands (A, B)")
    a = np.asarray(operands[0])
    b = np.asarray(operands[1])

    # Phase 8.4.6 + native-half tiled — N upper bound depends on dtype. f32
    # has the threadgroup-tiled variant; f16/bf16 now also have native tiled
    # fused kernels (per-thread for N<=256, threadgroup-tiled above), so the
    # single-kernel envelope matches f32 when the tiled symbol is present.
    # Older builds without it stay at 256.
    #
    # P1 (2026-06-02) — the tiled cap is feature-limit-derived, not a magic
    # constant: the kernel holds one row of N fp32 scores in threadgroup
    # memory, so N_max = threadgroup_memory_budget // 4. On every current
    # Apple arch that is 32 KB // 4 = 8192 (preserving the old constant),
    # and it auto-scales on a higher-memory SKU.
    tiled_n_cap = _apple_threadgroup_tiled_softmax_n_cap()
    bf16_dtype = _bfloat16_dtype()
    if a.dtype == np.float32:
        n_max = tiled_n_cap
    elif a.dtype == np.float16:
        n_max = tiled_n_cap if _apple_gpu_matmul_softmax_tiled_f16() is not None else 256
    elif bf16_dtype is not None and a.dtype == bf16_dtype:
        n_max = tiled_n_cap if _apple_gpu_matmul_softmax_tiled_bf16() is not None else 256
    else:
        n_max = 256

    if not (
        a.ndim == 2 and b.ndim == 2
        and a.shape[1] == b.shape[0]
        and b.shape[1] <= n_max
        and a.dtype == b.dtype
    ):
        # Outside the single fused-kernel envelope (notably f16/bf16 with
        # N>256, where no tiled fused variant exists yet): compose the GPU
        # matmul with the MPSGraph softmax epilogue (no N limit) so the chain
        # still runs on-device instead of falling back to host numpy.
        if a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]:
            scores = _apple_gpu_dispatch_matmul("tessera.matmul", [a, b], np)
            return _apple_gpu_dispatch_mpsgraph_softmax(scores, np)
        scores = np.matmul(a, b)
        e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    M, K = a.shape
    N = b.shape[1]

    if a.dtype == np.float32:
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a, dtype=np.float32)
        if not b.flags.c_contiguous:
            b = np.ascontiguousarray(b, dtype=np.float32)
        out = np.zeros((M, N), dtype=np.float32)
        fused = _apple_gpu_matmul_softmax_f32()
        fused(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(M), ctypes.c_int32(N), ctypes.c_int32(K),
        )
        return out

    if a.dtype == np.float16:
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a, dtype=np.float16)
        if not b.flags.c_contiguous:
            b = np.ascontiguousarray(b, dtype=np.float16)
        out = np.zeros((M, N), dtype=np.float16)
        # Per-thread kernel for N<=256 (no threadgroup sync, faster);
        # threadgroup-tiled native kernel for larger N.
        fused_f16 = (_apple_gpu_matmul_softmax_f16() if N <= 256
                     else _apple_gpu_matmul_softmax_tiled_f16())
        if fused_f16 is not None:
            fused_f16(
                a.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                b.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int32(M), ctypes.c_int32(N), ctypes.c_int32(K),
            )
            return out
        scores = (a.astype(np.float32) @ b.astype(np.float32))
        e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        return (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float16)

    bf16_dtype = _bfloat16_dtype()
    if bf16_dtype is not None and a.dtype == bf16_dtype:
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a, dtype=bf16_dtype)
        if not b.flags.c_contiguous:
            b = np.ascontiguousarray(b, dtype=bf16_dtype)
        out = np.zeros((M, N), dtype=bf16_dtype)
        fused_bf16 = (_apple_gpu_matmul_softmax_bf16() if N <= 256
                      else _apple_gpu_matmul_softmax_tiled_bf16())
        if fused_bf16 is not None:
            fused_bf16(
                a.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                b.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int32(M), ctypes.c_int32(N), ctypes.c_int32(K),
            )
            return out
        scores = (a.astype(np.float32) @ b.astype(np.float32))
        e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        return (e / np.sum(e, axis=-1, keepdims=True)).astype(bf16_dtype)

    scores = np.matmul(a, b)
    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _apple_gpu_matmul_softmax_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = runtime.tessera_apple_gpu_matmul_softmax_f32
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,  # M
        ctypes.c_int32,  # N
        ctypes.c_int32,  # K
    ]
    sym.restype = None
    return sym


def _apple_gpu_matmul_softmax_f16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_matmul_softmax_f16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_matmul_softmax_bf16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_matmul_softmax_bf16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_matmul_softmax_tiled_f16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_matmul_softmax_tiled_f16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_matmul_softmax_tiled_bf16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_matmul_softmax_tiled_bf16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_dispatch_gelu_gpu(x: Any, np: Any) -> Any:
    """gelu(x) via the MPSGraph unary lane (op 5). Used as the matmul_gelu
    epilogue: the hand-written MSL gelu kernel overflows its tanh for large
    activations (|x| >~ 16 -> NaN), whereas MPSGraph's tanh node is robust.
    f32/f16 native; bf16 upcast -> GPU f32 -> downcast."""
    x = np.asarray(x)
    shape = x.shape
    n = int(x.size)
    bf16_dtype = _bfloat16_dtype()

    def _ref() -> Any:
        f = x.astype(np.float32)
        return 0.5 * f * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (f + 0.044715 * f ** 3)))

    if x.dtype == np.float32:
        sym = _apple_gpu_mpsgraph_unary_f32()
        if sym is None:
            return _ref().reshape(shape)
        xf = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        out = np.empty(n, dtype=np.float32)
        sym(ctypes.c_int32(5),
            xf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_int64(n))
        return out.reshape(shape)
    if x.dtype == np.float16:
        sym = _apple_gpu_mpsgraph_unary_f16()
        if sym is None:
            return _ref().astype(np.float16).reshape(shape)
        xf = np.ascontiguousarray(x, dtype=np.float16).reshape(-1)
        out = np.empty(n, dtype=np.float16)
        sym(ctypes.c_int32(5),
            xf.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_int64(n))
        return out.reshape(shape)
    if bf16_dtype is not None and x.dtype == bf16_dtype:
        sym = _apple_gpu_mpsgraph_unary_f32()
        if sym is None:
            return _ref().astype(bf16_dtype).reshape(shape)
        x32 = np.ascontiguousarray(x.astype(np.float32)).reshape(-1)
        out = np.empty(n, dtype=np.float32)
        sym(ctypes.c_int32(5),
            x32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_int64(n))
        return out.reshape(shape).astype(bf16_dtype)
    return _ref().astype(x.dtype).reshape(shape)


def _apple_gpu_matmul_gelu_fused_half(a: Any, b: Any, np: Any) -> Any:
    """Native single fused matmul -> gelu MSL kernel for f16/bf16, N<=256.
    Returns the result array, or None when not applicable (wrong dtype, rank,
    N>256, or the half symbol isn't exported by the loaded runtime) so the
    caller can fall through to the compose path."""
    bf16_dtype = _bfloat16_dtype()
    if not (a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
            and b.shape[1] <= 256 and a.dtype == b.dtype):
        return None
    if a.dtype == np.float16:
        fused = _apple_gpu_matmul_gelu_f16()
        dt = np.float16
    elif bf16_dtype is not None and a.dtype == bf16_dtype:
        fused = _apple_gpu_matmul_gelu_bf16()
        dt = bf16_dtype
    else:
        return None
    if fused is None:
        return None
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a, dtype=dt)
    if not b.flags.c_contiguous:
        b = np.ascontiguousarray(b, dtype=dt)
    M, K = a.shape
    N = b.shape[1]
    out = np.zeros((M, N), dtype=dt)
    fused(
        a.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        b.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        ctypes.c_int32(M), ctypes.c_int32(N), ctypes.c_int32(K),
    )
    return out


def _apple_gpu_dispatch_matmul_gelu(operands: list[Any], np: Any) -> Any:
    """Phase 8.4.7 + native-half — dispatch a fused matmul -> gelu chain
    through the apple_gpu runtime shim. f32 and (N<=256) f16/bf16 run as a
    single fused MSL kernel; large-N and dtype-less paths compose GPU matmul
    with the MPSGraph gelu epilogue; numpy is the final fallback."""

    if len(operands) != 2:
        raise ValueError("matmul_gelu fusion requires two operands (A, B)")
    a = np.asarray(operands[0])
    b = np.asarray(operands[1])
    if not (
        a.ndim == 2 and b.ndim == 2
        and a.shape[1] == b.shape[0]
        and b.shape[1] <= 256
        and a.dtype == np.float32 and b.dtype == np.float32
    ):
        # f16/bf16 with N<=256: native single fused MSL kernel when present
        # (half I/O, fp32 accumulators) — one dispatch instead of composing.
        native = _apple_gpu_matmul_gelu_fused_half(a, b, np)
        if native is not None:
            return native
        # Otherwise (large-N, or no native half symbol): compose the GPU
        # matmul (f16/bf16 native, any N) with the GPU gelu epilogue so
        # f16/bf16 and large-N still execute on-device. Both sub-dispatchers
        # degrade to numpy automatically when Metal is unavailable.
        if a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]:
            scores = _apple_gpu_dispatch_matmul("tessera.matmul", [a, b], np)
            return _apple_gpu_dispatch_gelu_gpu(scores, np)
        scores = np.matmul(a, b)
        return 0.5 * scores * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) *
                                              (scores + 0.044715 * scores ** 3)))
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a, dtype=np.float32)
    if not b.flags.c_contiguous:
        b = np.ascontiguousarray(b, dtype=np.float32)
    M, K = a.shape
    N = b.shape[1]
    out = np.zeros((M, N), dtype=np.float32)
    fused = _apple_gpu_matmul_gelu_f32()
    # Apple plan phase D — emit a unified JitBridgeRoute so the
    # generic-tensor lane's proof envelope matches GA/EBM/M7.
    import time as _time
    from tessera.compiler import jit_bridge as _bridge
    _t0 = _time.perf_counter_ns()
    fused(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(M), ctypes.c_int32(N), ctypes.c_int32(K),
    )
    _latency_ms = (_time.perf_counter_ns() - _t0) / 1e6
    _bridge.record_driver_route(
        op_name="matmul_gelu",
        target="apple_gpu",
        status="fused",
        symbol="tessera_apple_gpu_matmul_gelu_f32",
        latency_ms=_latency_ms,
        args_summary=_bridge.shaped_summary(a, b),
    )
    return out


def _apple_gpu_matmul_gelu_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = runtime.tessera_apple_gpu_matmul_gelu_f32
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_matmul_gelu_f16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_matmul_gelu_f16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_matmul_gelu_bf16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_matmul_gelu_bf16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_matmul_rmsnorm_fused_half(a: Any, b: Any, eps: float,
                                         np: Any) -> Any:
    """Native single fused matmul -> rmsnorm MSL kernel for f16/bf16, N<=256.
    Returns the result array, or None when not applicable so the caller can
    fall through to the compose path."""
    bf16_dtype = _bfloat16_dtype()
    if not (a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
            and b.shape[1] <= 256 and a.dtype == b.dtype):
        return None
    if a.dtype == np.float16:
        fused = _apple_gpu_matmul_rmsnorm_f16()
        dt = np.float16
    elif bf16_dtype is not None and a.dtype == bf16_dtype:
        fused = _apple_gpu_matmul_rmsnorm_bf16()
        dt = bf16_dtype
    else:
        return None
    if fused is None:
        return None
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a, dtype=dt)
    if not b.flags.c_contiguous:
        b = np.ascontiguousarray(b, dtype=dt)
    M, K = a.shape
    N = b.shape[1]
    out = np.zeros((M, N), dtype=dt)
    fused(
        a.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        b.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        ctypes.c_int32(M), ctypes.c_int32(N), ctypes.c_int32(K),
        ctypes.c_float(eps),
    )
    return out


def _apple_gpu_dispatch_matmul_rmsnorm(operands: list[Any], eps: float, np: Any) -> Any:
    """Phase 8.4.7 + native-half — dispatch a fused matmul -> rmsnorm chain
    through the apple_gpu runtime shim. f32 and (N<=256) f16/bf16 run as a
    single fused MSL kernel; large-N composes GPU matmul + MPSGraph epilogue.
    eps is passed in by the metadata-layer dispatcher (it knows the rmsnorm
    vs rmsnorm_safe default and any explicit override)."""

    if len(operands) != 2:
        raise ValueError("matmul_rmsnorm fusion requires two operands (A, B)")
    a = np.asarray(operands[0])
    b = np.asarray(operands[1])
    if not (
        a.ndim == 2 and b.ndim == 2
        and a.shape[1] == b.shape[0]
        and b.shape[1] <= 256
        and a.dtype == np.float32 and b.dtype == np.float32
    ):
        # f16/bf16 with N<=256: native single fused MSL kernel when present
        # (half I/O, fp32 accumulators) — one dispatch instead of composing.
        native = _apple_gpu_matmul_rmsnorm_fused_half(a, b, eps, np)
        if native is not None:
            return native
        # Otherwise (large-N, or no native half symbol): compose the GPU
        # matmul with the GPU rmsnorm epilogue (MPSGraph, any N) so f16/bf16
        # and large-N still execute on-device.
        if a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]:
            scores = _apple_gpu_dispatch_matmul("tessera.matmul", [a, b], np)
            return _apple_gpu_dispatch_rowop(
                "tessera.rmsnorm", [scores], {"eps": eps}, np
            )
        scores = np.matmul(a, b)
        rms = np.sqrt(np.mean(scores * scores, axis=-1, keepdims=True) + eps)
        return scores / rms
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a, dtype=np.float32)
    if not b.flags.c_contiguous:
        b = np.ascontiguousarray(b, dtype=np.float32)
    M, K = a.shape
    N = b.shape[1]
    out = np.zeros((M, N), dtype=np.float32)
    fused = _apple_gpu_matmul_rmsnorm_f32()
    fused(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(M), ctypes.c_int32(N), ctypes.c_int32(K),
        ctypes.c_float(eps),
    )
    return out


def _apple_gpu_matmul_rmsnorm_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = runtime.tessera_apple_gpu_matmul_rmsnorm_f32
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_float,
    ]
    sym.restype = None
    return sym


def _apple_gpu_matmul_rmsnorm_f16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_matmul_rmsnorm_f16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_float,
    ]
    sym.restype = None
    return sym


def _apple_gpu_matmul_rmsnorm_bf16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_matmul_rmsnorm_bf16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_float,
    ]
    sym.restype = None
    return sym


def _apple_gpu_dispatch_swiglu(x: Any, wg: Any, wu: Any, wd: Any, np: Any) -> Any:
    """Phase 8.4.8 (Stage 3 SwiGLU Performance Plan) — dispatch a fused
    SwiGLU MLP block (`silu(x @ Wg) * (x @ Wu) @ Wd`) through the apple_gpu
    runtime shim's `tessera_apple_gpu_swiglu_f32` symbol. f32 only this
    phase. Inputs outside the supported envelope (rank, dtype, H>256,
    Kout>256) fall back to a numpy host computation."""

    x = np.asarray(x)
    wg = np.asarray(wg)
    wu = np.asarray(wu)
    wd = np.asarray(wd)
    H = wg.shape[1] if wg.ndim == 2 else None
    Kout = wd.shape[1] if wd.ndim == 2 else None
    if not (
        x.ndim == 2 and wg.ndim == 2 and wu.ndim == 2 and wd.ndim == 2
        and wg.shape == wu.shape
        and x.shape[1] == wg.shape[0]
        and wd.shape[0] == wg.shape[1]
        and H is not None and Kout is not None
        and H <= 256 and Kout <= 256
        and x.dtype == np.float32 and wg.dtype == np.float32
        and wu.dtype == np.float32 and wd.dtype == np.float32
    ):
        gate = np.matmul(x, wg)
        up = np.matmul(x, wu)
        hidden = (gate / (1.0 + np.exp(-gate))) * up
        return np.matmul(hidden, wd)
    if not x.flags.c_contiguous:
        x = np.ascontiguousarray(x, dtype=np.float32)
    if not wg.flags.c_contiguous:
        wg = np.ascontiguousarray(wg, dtype=np.float32)
    if not wu.flags.c_contiguous:
        wu = np.ascontiguousarray(wu, dtype=np.float32)
    if not wd.flags.c_contiguous:
        wd = np.ascontiguousarray(wd, dtype=np.float32)
    M, K = x.shape
    H = wg.shape[1]
    Kout = wd.shape[1]
    out = np.zeros((M, Kout), dtype=np.float32)
    fused = _apple_gpu_swiglu_f32()
    fused(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wu.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wd.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(M), ctypes.c_int32(K),
        ctypes.c_int32(H), ctypes.c_int32(Kout),
    )
    return out


def _apple_gpu_swiglu_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = runtime.tessera_apple_gpu_swiglu_f32
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # X
        ctypes.POINTER(ctypes.c_float),  # Wg
        ctypes.POINTER(ctypes.c_float),  # Wu
        ctypes.POINTER(ctypes.c_float),  # Wd
        ctypes.POINTER(ctypes.c_float),  # O
        ctypes.c_int32, ctypes.c_int32,  # M, K
        ctypes.c_int32, ctypes.c_int32,  # H, Kout
    ]
    sym.restype = None
    return sym


def _apple_gpu_dispatch_matmul_softmax_matmul(operands: list[Any], np: Any) -> Any:
    """Phase 8.4.5 — dispatch a fused matmul -> softmax -> matmul chain
    (full attention block) through the apple_gpu runtime shim. Picks symbol
    by element type (f32/f16/bf16). Inputs outside the supported envelope
    fall back to a numpy-equivalent host computation."""

    if len(operands) != 3:
        raise ValueError("matmul_softmax_matmul fusion requires three operands (A, B, C)")
    a = np.asarray(operands[0])
    b = np.asarray(operands[1])
    c = np.asarray(operands[2])

    if not (
        a.ndim == 2 and b.ndim == 2 and c.ndim == 2
        and a.shape[1] == b.shape[0]
        and b.shape[1] == c.shape[0]
        and b.shape[1] <= 256
        and c.shape[1] <= 256
        and a.dtype == b.dtype == c.dtype
    ):
        scores = a @ b
        e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs = e / np.sum(e, axis=-1, keepdims=True)
        return probs @ c

    M, K = a.shape
    N = b.shape[1]
    P = c.shape[1]

    if a.dtype == np.float32:
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a, dtype=np.float32)
        if not b.flags.c_contiguous:
            b = np.ascontiguousarray(b, dtype=np.float32)
        if not c.flags.c_contiguous:
            c = np.ascontiguousarray(c, dtype=np.float32)
        out = np.zeros((M, P), dtype=np.float32)
        fused = _apple_gpu_matmul_softmax_matmul_f32()
        # Phase D (Apple plan, 2026-05-20): record the unified proof
        # route the same way GA/EBM/M7 manifest dispatch does, so
        # ``CompileReport.proof_routes`` carries a row for the
        # generic-tensor lane too.
        import time as _time
        from tessera.compiler import jit_bridge as _bridge
        _t0 = _time.perf_counter_ns()
        fused(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(M), ctypes.c_int32(K),
            ctypes.c_int32(N), ctypes.c_int32(P),
        )
        _latency_ms = (_time.perf_counter_ns() - _t0) / 1e6
        _bridge.record_driver_route(
            op_name="matmul_softmax_matmul",
            target="apple_gpu",
            status="fused",
            symbol="tessera_apple_gpu_matmul_softmax_matmul_f32",
            latency_ms=_latency_ms,
            args_summary=_bridge.shaped_summary(a, b, c),
        )
        return out

    if a.dtype == np.float16:
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a, dtype=np.float16)
        if not b.flags.c_contiguous:
            b = np.ascontiguousarray(b, dtype=np.float16)
        if not c.flags.c_contiguous:
            c = np.ascontiguousarray(c, dtype=np.float16)
        out = np.zeros((M, P), dtype=np.float16)
        fused_f16 = _apple_gpu_matmul_softmax_matmul_f16()
        if fused_f16 is not None:
            fused_f16(
                a.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                b.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                c.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int32(M), ctypes.c_int32(K),
                ctypes.c_int32(N), ctypes.c_int32(P),
            )
            return out
        scores = a.astype(np.float32) @ b.astype(np.float32)
        e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs = e / np.sum(e, axis=-1, keepdims=True)
        return (probs @ c.astype(np.float32)).astype(np.float16)

    bf16_dtype = _bfloat16_dtype()
    if bf16_dtype is not None and a.dtype == bf16_dtype:
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a, dtype=bf16_dtype)
        if not b.flags.c_contiguous:
            b = np.ascontiguousarray(b, dtype=bf16_dtype)
        if not c.flags.c_contiguous:
            c = np.ascontiguousarray(c, dtype=bf16_dtype)
        out = np.zeros((M, P), dtype=bf16_dtype)
        fused_bf16 = _apple_gpu_matmul_softmax_matmul_bf16()
        if fused_bf16 is not None:
            fused_bf16(
                a.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                b.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                c.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int32(M), ctypes.c_int32(K),
                ctypes.c_int32(N), ctypes.c_int32(P),
            )
            return out
        scores = a.astype(np.float32) @ b.astype(np.float32)
        e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs = e / np.sum(e, axis=-1, keepdims=True)
        return (probs @ c.astype(np.float32)).astype(bf16_dtype)

    scores = a @ b
    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs = e / np.sum(e, axis=-1, keepdims=True)
    return probs @ c


def _apple_gpu_matmul_softmax_matmul_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = runtime.tessera_apple_gpu_matmul_softmax_matmul_f32
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_matmul_softmax_matmul_f16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_matmul_softmax_matmul_f16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    sym.restype = None
    return sym


def _apple_gpu_matmul_softmax_matmul_bf16() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_matmul_softmax_matmul_bf16", None)
    if sym is None:
        return None
    sym.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    sym.restype = None
    return sym


_DEVTENSOR_API_CONFIGURED = False


def _apple_gpu_devtensor_api() -> Any:
    """Configure + return the device-tensor C ABI (R0). None when unavailable."""
    runtime = _load_apple_gpu_runtime()
    if getattr(runtime, "ts_dev_alloc", None) is None:
        return None
    global _DEVTENSOR_API_CONFIGURED
    if not _DEVTENSOR_API_CONFIGURED:
        vp, i64, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32
        runtime.ts_dev_alloc.argtypes = [i64]; runtime.ts_dev_alloc.restype = vp
        runtime.ts_dev_contents.argtypes = [vp]; runtime.ts_dev_contents.restype = vp
        runtime.ts_dev_nbytes.argtypes = [vp]; runtime.ts_dev_nbytes.restype = i64
        runtime.ts_dev_upload.argtypes = [vp, vp, i64]; runtime.ts_dev_upload.restype = None
        runtime.ts_dev_download.argtypes = [vp, vp, i64]; runtime.ts_dev_download.restype = None
        runtime.ts_dev_free.argtypes = [vp]; runtime.ts_dev_free.restype = None
        runtime.ts_dev_is_metal.argtypes = []; runtime.ts_dev_is_metal.restype = i32
        if getattr(runtime, "ts_dev_mtl_buffer", None) is not None:
            runtime.ts_dev_mtl_buffer.argtypes = [vp]; runtime.ts_dev_mtl_buffer.restype = vp
        if getattr(runtime, "ts_dev_cast", None) is not None:
            runtime.ts_dev_cast.argtypes = [vp, vp, i64, i32]; runtime.ts_dev_cast.restype = i32
        _DEVTENSOR_API_CONFIGURED = True
    return runtime


class DeviceTensor:
    """An opaque, GPU-resident tensor (R0).

    Wraps one shared (unified-memory) Metal buffer. On Apple Silicon
    ``[buf contents]`` is a CPU pointer to the *same* bytes the GPU sees, so
    after the one-time ``from_numpy`` copy there are **no further host↔device
    copies**: ``.numpy()`` returns a zero-copy view, and (from R1 on) a producer
    op's output can feed a consumer op without any host round-trip. On non-Apple
    hosts the handle is backed by plain host memory, so the surface is portable.

    Lifetime is explicit — call ``free()`` (or rely on ``__del__``). Holding a
    ``.numpy()`` view alive past ``free()`` is a use-after-free; copy first with
    ``.copy_to_host()`` if you need to outlive the handle."""

    __slots__ = ("_handle", "shape", "dtype", "_rt", "_freed", "_owns")

    def __init__(self, handle: Any, shape: Any, dtype: Any, rt: Any,
                 owns: bool = True) -> None:
        import numpy as _np
        self._handle = handle
        self.shape = tuple(int(s) for s in shape)
        self.dtype = _np.dtype(dtype)
        self._rt = rt
        self._freed = False
        self._owns = owns  # non-owning views (reshape_view) never free the buffer

    @property
    def nbytes(self) -> int:
        n = self.dtype.itemsize
        for s in self.shape:
            n *= s
        return int(n)

    @staticmethod
    def is_metal() -> bool:
        rt = _apple_gpu_devtensor_api()
        return bool(rt is not None and rt.ts_dev_is_metal() == 1)

    @classmethod
    def from_numpy(cls, arr: Any) -> "DeviceTensor | None":
        """Allocate a device tensor and copy ``arr`` into its shared storage
        once. Returns None when the device-tensor ABI is unavailable."""
        import numpy as _np
        rt = _apple_gpu_devtensor_api()
        if rt is None:
            return None
        a = _np.ascontiguousarray(arr)
        handle = rt.ts_dev_alloc(ctypes.c_int64(int(a.nbytes)))
        if not handle:
            return None
        rt.ts_dev_upload(handle, a.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int64(int(a.nbytes)))
        return cls(handle, a.shape, a.dtype, rt)

    @classmethod
    def empty(cls, shape: Any, dtype: Any) -> "DeviceTensor | None":
        import numpy as _np
        rt = _apple_gpu_devtensor_api()
        if rt is None:
            return None
        dt = _np.dtype(dtype)
        nbytes = dt.itemsize
        for s in shape:
            nbytes *= int(s)
        handle = rt.ts_dev_alloc(ctypes.c_int64(int(nbytes)))
        if not handle:
            return None
        return cls(handle, shape, dt, rt)

    def numpy(self) -> Any:
        """Zero-copy numpy view over the shared storage (no download). Valid
        only while the handle is alive."""
        import numpy as _np
        if self._freed:
            raise RuntimeError("DeviceTensor used after free()")
        ptr = self._rt.ts_dev_contents(self._handle)
        buf = (ctypes.c_byte * self.nbytes).from_address(int(ptr))
        return _np.frombuffer(memoryview(buf), dtype=self.dtype).reshape(self.shape)

    def copy_to_host(self) -> Any:
        """A standalone host copy that outlives the handle."""
        return self.numpy().copy()

    def reshape_view(self, *shape: Any) -> "DeviceTensor":
        """A **non-owning** view of the same device buffer with a new shape
        (same element count + dtype). Used to chain ops with shape changes
        *without* a host round-trip — e.g. a bmm output ``[H,1,N]`` viewed as a
        rowop input ``[H,N]`` — so the chain stays in one command buffer. The
        view never frees the buffer; the original owner does."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        n = 1
        for s in shape:
            n *= int(s)
        cur = 1
        for s in self.shape:
            cur *= s
        if n != cur:
            raise ValueError(f"reshape_view {self.shape} -> {shape}: element "
                             f"count mismatch ({cur} != {n})")
        return DeviceTensor(self._handle, shape, self.dtype, self._rt, owns=False)

    def prefix_view(self, n_rows: int) -> "DeviceTensor":
        """A **non-owning** view of the first ``n_rows`` rows of a 2-D+ buffer,
        shape ``(n_rows, *self.shape[1:])``. The prefix starts at offset 0 and is
        contiguous, so a growing resident KV cache can expose its populated
        window ``[current_seq, ...]`` with **no copy** — the foundation of R4's
        device-resident cache."""
        if not self.shape:
            raise ValueError("prefix_view requires a non-scalar tensor")
        n_rows = int(n_rows)
        if n_rows < 0 or n_rows > self.shape[0]:
            raise ValueError(f"prefix_view {n_rows} out of range [0, {self.shape[0]}]")
        return DeviceTensor(self._handle, (n_rows,) + self.shape[1:], self.dtype,
                            self._rt, owns=False)

    @property
    def handle(self) -> Any:
        """The raw `void*` handle (for handle-taking kernels in R1)."""
        return self._handle

    def mtl_buffer(self) -> int:
        """Interop escape hatch: the underlying ``id<MTLBuffer>`` as an integer
        pointer (0 if unavailable / non-Metal). Combined with
        :func:`apple_gpu_device_handle`, lets external Metal/MPS code operate on
        this resident tensor GPU-side. Tessera owns the buffer's lifetime — do not
        release it; valid only while this handle is alive."""
        if self._freed or not self._handle:
            return 0
        sym = getattr(self._rt, "ts_dev_mtl_buffer", None)
        if sym is None:
            return 0
        p = sym(self._handle)
        return int(p) if p else 0

    def cast_to(self, dtype: Any) -> "DeviceTensor":
        """Resident dtype cast (f32 ↔ f16/bf16) to a **new** DeviceTensor, entirely
        on the GPU — no host round-trip. The missing link for round-trip-free MLP
        stacking (M8 ``run_dev`` outputs f32; the next layer's matmul wants
        f16/bf16). Falls back to a host cast off Metal."""
        import numpy as _np
        if self._freed:
            raise RuntimeError("DeviceTensor used after free()")
        bf16 = _bfloat16_dtype()
        dst_dt = _np.dtype(dtype)
        src_dt = self.dtype
        _MODE = {  # (src, dst) -> kernel mode
            (_np.dtype(_np.float32), _np.dtype(_np.float16)): 0,
            (_np.dtype(_np.float16), _np.dtype(_np.float32)): 2,
        }
        if bf16 is not None:
            _MODE[(_np.dtype(_np.float32), _np.dtype(bf16))] = 1
            _MODE[(_np.dtype(bf16), _np.dtype(_np.float32))] = 3
        out = DeviceTensor.empty(self.shape, dst_dt)
        n = 1
        for s in self.shape:
            n *= int(s)
        mode = _MODE.get((src_dt, dst_dt))
        sym = getattr(self._rt, "ts_dev_cast", None)
        if out is not None and mode is not None and sym is not None and not self._freed:
            rc = sym(self._handle, out._handle, ctypes.c_int64(n), ctypes.c_int32(mode))
            if rc == 1:
                return out
        # host fallback — cast through numpy on the shared storage
        if out is None:
            out = DeviceTensor.empty(self.shape, dst_dt)
        assert out is not None  # empty() always returns a tensor here
        out.numpy()[...] = self.numpy().astype(dst_dt)
        return out

    def free(self) -> None:
        if not self._freed and self._owns and self._handle:
            self._rt.ts_dev_free(self._handle)
            self._freed = True
            self._handle = None
        elif not self._owns:
            self._freed = True
            self._handle = None

    def __del__(self) -> None:
        try:
            self.free()
        except Exception:
            pass

    def __repr__(self) -> str:
        loc = "metal" if DeviceTensor.is_metal() else "host"
        return (f"DeviceTensor(shape={self.shape}, dtype={self.dtype.name}, "
                f"{loc}, freed={self._freed})")


_BMM_DEV_CONFIGURED = False


def _apple_gpu_bmm_dev_f32() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_bmm_dev_f32", None)
    if sym is None:
        return None
    global _BMM_DEV_CONFIGURED
    if not _BMM_DEV_CONFIGURED:
        vp, i32 = ctypes.c_void_p, ctypes.c_int32
        sym.argtypes = [vp, vp, vp, i32, i32, i32, i32, i32]
        sym.restype = i32
        _BMM_DEV_CONFIGURED = True
    return sym


_ROWOP_DEV_CONFIGURED = False


def _apple_gpu_rowop_dev_sym() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_rowop_dev_f32", None)
    if sym is None:
        return None
    global _ROWOP_DEV_CONFIGURED
    if not _ROWOP_DEV_CONFIGURED:
        vp, i32, f32 = ctypes.c_void_p, ctypes.c_int32, ctypes.c_float
        sym.argtypes = [vp, vp, vp, i32, i32, i32, f32]
        sym.restype = i32
        _ROWOP_DEV_CONFIGURED = True
    return sym


def _apple_gpu_rowop_device(X: "DeviceTensor", kind: int,
                            gamma: "DeviceTensor | None" = None,
                            eps: float = 1e-6) -> "DeviceTensor | None":
    """Standalone (non-session) device-resident row op: kind 0 layer_norm,
    1 rmsnorm, 2 softmax, 3 log_softmax. ``X`` ``[rows, cols]`` -> resident
    ``[rows, cols]``; optional ``gamma`` ``[cols]``."""
    import numpy as _np
    if X.dtype != _np.float32 or len(X.shape) != 2:
        return None
    rows, cols = X.shape
    if gamma is not None and gamma.shape != (cols,):
        return None
    sym = _apple_gpu_rowop_dev_sym()
    if sym is None:
        return None
    out = DeviceTensor.empty((rows, cols), _np.float32)
    if out is None:
        return None
    rc = sym(X.handle, gamma.handle if gamma is not None else None, out.handle,
             ctypes.c_int32(int(kind)), ctypes.c_int32(rows),
             ctypes.c_int32(cols), ctypes.c_float(float(eps)))
    if rc != 1:
        out.free()
        return None
    return out


_GATHER_DEV_CONFIGURED = False


def _apple_gpu_gather_blocks_dev_sym() -> Any:
    runtime = _load_apple_gpu_runtime()
    sym = getattr(runtime, "tessera_apple_gpu_gather_blocks_dev_f32", None)
    if sym is None:
        return None
    global _GATHER_DEV_CONFIGURED
    if not _GATHER_DEV_CONFIGURED:
        vp, i32 = ctypes.c_void_p, ctypes.c_int32
        sym.argtypes = [vp, vp, vp, i32, i32, i32, i32]
        sym.restype = i32
        _GATHER_DEV_CONFIGURED = True
    return sym


def _apple_gpu_gather_blocks_device(pool: "DeviceTensor",
                                    block_table: "DeviceTensor",
                                    num_blocks: int, n: int, block_size: int,
                                    dim: int) -> "DeviceTensor | None":
    """R4 (block-paged) — gather ``n`` physical blocks (by ``block_table`` int32
    ids) from the resident pool ``[num_blocks, block_size, dim]`` into a
    contiguous resident window ``[n, block_size, dim]`` on-GPU. No host copy."""
    import numpy as _np
    sym = _apple_gpu_gather_blocks_dev_sym()
    if sym is None:
        return None
    out = DeviceTensor.empty((n, block_size, dim), _np.float32)
    if out is None:
        return None
    rc = sym(pool.handle, block_table.handle, out.handle,
             ctypes.c_int32(num_blocks), ctypes.c_int32(n),
             ctypes.c_int32(block_size), ctypes.c_int32(dim))
    if rc != 1:
        out.free()
        return None
    return out


def _apple_gpu_bmm_device(A: "DeviceTensor", B: "DeviceTensor",
                          b_broadcast: bool = False) -> "DeviceTensor | None":
    """R1 — device-resident batched matmul. Both inputs are ``DeviceTensor``s
    (shapes ``[batch, M, K]`` and ``[batch|1, K, N]``); the result is a new
    ``DeviceTensor`` ``[batch, M, N]`` that stays on-device — **no host upload
    or readback**, so it can feed the next op directly. f32 only. Returns None
    when the device path is unavailable."""
    import numpy as _np
    if A.dtype != _np.float32 or B.dtype != _np.float32:
        return None
    if len(A.shape) != 3 or len(B.shape) != 3:
        return None
    batch, M, K = A.shape
    bBatch, K2, N = B.shape
    if K2 != K or (bBatch != batch and bBatch != 1):
        return None
    bcast = (bBatch == 1 and batch != 1) or bool(b_broadcast)
    sym = _apple_gpu_bmm_dev_f32()
    if sym is None:
        return None
    out = DeviceTensor.empty((batch, M, N), _np.float32)
    if out is None:
        return None
    rc = sym(A.handle, B.handle, out.handle, ctypes.c_int32(batch),
             ctypes.c_int32(M), ctypes.c_int32(N), ctypes.c_int32(K),
             ctypes.c_int32(1 if bcast else 0))
    if rc != 1:
        out.free()
        return None
    return out


_ENC_API_CONFIGURED = False


def _apple_gpu_enc_api() -> Any:
    """Configure + return the R2 encode-session ABI. None when unavailable."""
    runtime = _load_apple_gpu_runtime()
    if getattr(runtime, "ts_enc_begin", None) is None:
        return None
    global _ENC_API_CONFIGURED
    if not _ENC_API_CONFIGURED:
        vp, i32, f32 = ctypes.c_void_p, ctypes.c_int32, ctypes.c_float
        runtime.ts_enc_begin.argtypes = []; runtime.ts_enc_begin.restype = vp
        runtime.ts_enc_commit_wait.argtypes = [vp]; runtime.ts_enc_commit_wait.restype = None
        runtime.tessera_apple_gpu_bmm_dev_f32_enc.argtypes = [vp, vp, vp, vp, i32, i32, i32, i32, i32]
        runtime.tessera_apple_gpu_bmm_dev_f32_enc.restype = i32
        runtime.tessera_apple_gpu_rowop_dev_f32_enc.argtypes = [vp, vp, vp, vp, i32, i32, i32, f32]
        runtime.tessera_apple_gpu_rowop_dev_f32_enc.restype = i32
        runtime.tessera_apple_gpu_gumbel_argmax_dev_f32_enc.argtypes = [vp, vp, vp, vp, i32, i32, f32]
        runtime.tessera_apple_gpu_gumbel_argmax_dev_f32_enc.restype = i32
        i64 = ctypes.c_int64
        runtime.tessera_apple_gpu_unary_dev_f32_enc.argtypes = [vp, vp, vp, i64, i32]
        runtime.tessera_apple_gpu_unary_dev_f32_enc.restype = i32
        runtime.tessera_apple_gpu_binary_dev_f32_enc.argtypes = [vp, vp, vp, vp, i64, i32]
        runtime.tessera_apple_gpu_binary_dev_f32_enc.restype = i32
        _ENC_API_CONFIGURED = True
    return runtime


class AppleGPUEncodeSession:
    """R2 — command-buffer batching. Encode a chain of device-resident ops into
    **one** command buffer and commit + wait **once**, removing the per-op
    CPU↔GPU sync that dominates small-batch decode.

    Encoded outputs are **deferred**: a returned ``DeviceTensor`` is only valid
    after ``commit()`` (or the ``with`` block exits). Use as a context manager::

        with AppleGPUEncodeSession() as s:
            c = s.bmm(a, b)        # encoded, not yet computed
            d = s.bmm(c, e)        # consumes c's buffer; one command buffer
        result = d.numpy()         # valid after the block commits + waits
    """

    def __init__(self) -> None:
        self._rt = _apple_gpu_enc_api()
        self._handle = None if self._rt is None else self._rt.ts_enc_begin()
        self._committed = False
        self._outputs: list = []

    @property
    def available(self) -> bool:
        return self._handle is not None

    def bmm(self, A: "DeviceTensor", B: "DeviceTensor",
            b_broadcast: bool = False) -> "DeviceTensor | None":
        import numpy as _np
        if self._handle is None or self._committed:
            return None
        if A.dtype != _np.float32 or B.dtype != _np.float32:
            return None
        if len(A.shape) != 3 or len(B.shape) != 3:
            return None
        batch, M, K = A.shape
        bBatch, K2, N = B.shape
        if K2 != K or (bBatch != batch and bBatch != 1):
            return None
        bcast = (bBatch == 1 and batch != 1) or bool(b_broadcast)
        out = DeviceTensor.empty((batch, M, N), _np.float32)
        if out is None:
            return None
        rc = self._rt.tessera_apple_gpu_bmm_dev_f32_enc(
            self._handle, A.handle, B.handle, out.handle, ctypes.c_int32(batch),
            ctypes.c_int32(M), ctypes.c_int32(N), ctypes.c_int32(K),
            ctypes.c_int32(1 if bcast else 0))
        if rc != 1:
            out.free()
            return None
        self._outputs.append(out)
        return out

    def rowop(self, X: "DeviceTensor", kind: int, gamma: "DeviceTensor | None" = None,
              eps: float = 1e-6) -> "DeviceTensor | None":
        """Encode a row op: kind 0 layer_norm, 1 rmsnorm, 2 softmax, 3
        log_softmax. ``X`` is ``[rows, cols]``; optional ``gamma`` is ``[cols]``."""
        import numpy as _np
        if self._handle is None or self._committed or X.dtype != _np.float32:
            return None
        if len(X.shape) != 2:
            return None
        rows, cols = X.shape
        if gamma is not None and (gamma.dtype != _np.float32 or gamma.shape != (cols,)):
            return None
        out = DeviceTensor.empty((rows, cols), _np.float32)
        if out is None:
            return None
        rc = self._rt.tessera_apple_gpu_rowop_dev_f32_enc(
            self._handle, X.handle, gamma.handle if gamma is not None else None,
            out.handle, ctypes.c_int32(int(kind)), ctypes.c_int32(rows),
            ctypes.c_int32(cols), ctypes.c_float(float(eps)))
        if rc != 1:
            out.free()
            return None
        self._outputs.append(out)
        return out

    def rmsnorm(self, X: "DeviceTensor", gamma: "DeviceTensor | None" = None,
                eps: float = 1e-6) -> "DeviceTensor | None":
        return self.rowop(X, 1, gamma, eps)

    def softmax(self, X: "DeviceTensor") -> "DeviceTensor | None":
        return self.rowop(X, 2)

    def gumbel(self, logits: "DeviceTensor", gumbel_noise: "DeviceTensor",
               inv_temp: float = 1.0) -> "DeviceTensor | None":
        """Encode a Gumbel-max sample: ``argmax(logits/T + gumbel_noise)`` per
        row. ``logits`` / ``gumbel_noise`` are ``[rows, vocab]`` f32; returns an
        int32 ``[rows]`` device tensor of token ids (valid after commit)."""
        import numpy as _np
        if self._handle is None or self._committed or logits.dtype != _np.float32:
            return None
        if len(logits.shape) != 2 or gumbel_noise.shape != logits.shape:
            return None
        rows, cols = logits.shape
        out = DeviceTensor.empty((rows,), _np.int32)
        if out is None:
            return None
        rc = self._rt.tessera_apple_gpu_gumbel_argmax_dev_f32_enc(
            self._handle, logits.handle, gumbel_noise.handle, out.handle,
            ctypes.c_int32(rows), ctypes.c_int32(cols),
            ctypes.c_float(float(inv_temp)))
        if rc != 1:
            out.free()
            return None
        self._outputs.append(out)
        return out

    def _unary(self, X: "DeviceTensor", op: int) -> "DeviceTensor | None":
        import numpy as _np
        if self._handle is None or self._committed or X.dtype != _np.float32:
            return None
        n = int(_np.prod(X.shape)) if X.shape else 1
        out = DeviceTensor.empty(X.shape, _np.float32)
        if out is None:
            return None
        rc = self._rt.tessera_apple_gpu_unary_dev_f32_enc(
            self._handle, X.handle, out.handle, ctypes.c_int64(n),
            ctypes.c_int32(int(op)))
        if rc != 1:
            out.free()
            return None
        self._outputs.append(out)
        return out

    def _binary(self, A: "DeviceTensor", B: "DeviceTensor",
                op: int) -> "DeviceTensor | None":
        import numpy as _np
        if self._handle is None or self._committed:
            return None
        if A.dtype != _np.float32 or B.dtype != _np.float32:
            return None
        n = int(_np.prod(A.shape)) if A.shape else 1
        if (int(_np.prod(B.shape)) if B.shape else 1) != n:
            return None        # same-shape elementwise (broadcast happens host-side)
        out = DeviceTensor.empty(A.shape, _np.float32)
        if out is None:
            return None
        rc = self._rt.tessera_apple_gpu_binary_dev_f32_enc(
            self._handle, A.handle, B.handle, out.handle, ctypes.c_int64(n),
            ctypes.c_int32(int(op)))
        if rc != 1:
            out.free()
            return None
        self._outputs.append(out)
        return out

    def relu(self, X: "DeviceTensor") -> "DeviceTensor | None":
        """Encode an elementwise ReLU (layout-agnostic, valid after commit)."""
        return self._unary(X, 0)

    def silu(self, X: "DeviceTensor") -> "DeviceTensor | None":
        """Encode an elementwise SiLU = ``x * sigmoid(x)``."""
        return self._unary(X, 4)

    def add(self, A: "DeviceTensor", B: "DeviceTensor") -> "DeviceTensor | None":
        """Encode an elementwise add (residuals, additive attention masks)."""
        return self._binary(A, B, 0)

    def mul(self, A: "DeviceTensor", B: "DeviceTensor") -> "DeviceTensor | None":
        """Encode an elementwise multiply."""
        return self._binary(A, B, 2)

    def silu_mul(self, A: "DeviceTensor", B: "DeviceTensor") -> "DeviceTensor | None":
        """Encode SwiGLU activation ``silu(A) * B`` (matches ``ops.silu_mul``)."""
        t = self.silu(A)
        return self._binary(t, B, 2) if t is not None else None

    def commit(self) -> None:
        """Commit the encoded command buffer and wait for completion. After
        this, every encoded output's storage holds its result."""
        if self._handle is not None and not self._committed:
            self._rt.ts_enc_commit_wait(self._handle)
            self._committed = True
            self._handle = None

    def __enter__(self) -> "AppleGPUEncodeSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.commit()


def _load_apple_gpu_runtime() -> ctypes.CDLL:
    """Phase 8.3: locate or compile the apple_gpu runtime shared library.

    Prefers a CMake-built `libTesseraAppleRuntime` (which already links the
    GPU shim alongside the CPU shim on Darwin), then falls back to compiling
    the .mm + .cpp pair on the fly. On non-Darwin only the .cpp stub is
    compiled so the symbol still exists with a portable reference path.
    """

    global _apple_gpu_runtime
    if _apple_gpu_runtime is not None:
        return _apple_gpu_runtime

    # Single-loader unification (2026-06-02): delegate to
    # ``_apple_gpu_dispatch`` so the ctypes / ``bind_symbol`` lane (apple_mlpkg,
    # apple_dylib, encode session) and this MPS/MSL execution lane share ONE
    # loaded runtime image. Previously each compiled + dlopen'd its own dylib
    # (versioned vs. unversioned) into the same cache dir, so a process that
    # used both lanes loaded the runtime twice — every ObjC class
    # (``TesseraMlpkgPipeline``, …) was defined twice, emitting a
    # duplicate-class warning. ``_apple_gpu_dispatch`` now owns the
    # env-var + CMake-build preference and the from-source build, returning a
    # single cached handle.
    from ._apple_gpu_dispatch import (
        apple_gpu_runtime as _shared_apple_gpu_runtime,
        apple_gpu_skip_reason as _shared_skip_reason,
    )
    handle = _shared_apple_gpu_runtime()
    if handle is not None:
        _apple_gpu_runtime = handle
        return _apple_gpu_runtime
    # The shared loader couldn't provide a runtime (non-Darwin without the
    # stub-built path, missing compiler, etc.). Fall through to the legacy
    # candidate scan below so the original diagnostics/behavior are preserved.
    candidates = []
    env = os.environ.get("TESSERA_APPLE_GPU_RUNTIME_LIB")
    if env:
        candidates.append(Path(env))
    root = Path(__file__).resolve().parents[2]
    candidates.extend([
        root / "build/src/compiler/codegen/Tessera_Apple_Backend/libTesseraAppleRuntime.dylib",
        root / "build/src/compiler/codegen/Tessera_Apple_Backend/libTesseraAppleRuntime.so",
    ])
    _ = _shared_skip_reason  # diagnostic available if needed
    for candidate in candidates:
        if candidate.exists():
            try:
                lib = ctypes.CDLL(str(candidate))
                # Only accept the prebuilt library when it exports the full
                # Phase 8.4.2 envelope. Older builds lack the softmax/gelu
                # (or earlier) symbols; falling through forces a rebuild.
                getattr(lib, "tessera_apple_gpu_mps_matmul_f32")
                getattr(lib, "tessera_apple_gpu_rope_f32")
                getattr(lib, "tessera_apple_gpu_flash_attn_f32")
                getattr(lib, "tessera_apple_gpu_softmax_f32")
                getattr(lib, "tessera_apple_gpu_gelu_f32")
                getattr(lib, "tessera_apple_gpu_matmul_softmax_f32")
                # Phase 8.4.4 — require fp16/bf16 matmul symbols too. Older
                # builds lack them; falling through forces a rebuild.
                getattr(lib, "tessera_apple_gpu_mps_matmul_f16")
                getattr(lib, "tessera_apple_gpu_mps_matmul_bf16")
                # Phase 8.4.4.1 — fp16/bf16 for the simple MSL kernels.
                getattr(lib, "tessera_apple_gpu_rope_f16")
                getattr(lib, "tessera_apple_gpu_rope_bf16")
                getattr(lib, "tessera_apple_gpu_softmax_f16")
                getattr(lib, "tessera_apple_gpu_softmax_bf16")
                getattr(lib, "tessera_apple_gpu_gelu_f16")
                getattr(lib, "tessera_apple_gpu_gelu_bf16")
                # Phase 8.4.4.2 — fp16/bf16 for fused matmul_softmax + flash_attn.
                getattr(lib, "tessera_apple_gpu_matmul_softmax_f16")
                getattr(lib, "tessera_apple_gpu_matmul_softmax_bf16")
                getattr(lib, "tessera_apple_gpu_flash_attn_f16")
                getattr(lib, "tessera_apple_gpu_flash_attn_bf16")
                # Phase 8.4.5 — 3-op fusion (full attention block).
                getattr(lib, "tessera_apple_gpu_matmul_softmax_matmul_f32")
                getattr(lib, "tessera_apple_gpu_matmul_softmax_matmul_f16")
                getattr(lib, "tessera_apple_gpu_matmul_softmax_matmul_bf16")
                # Phase 8.4.6 — threadgroup-tiled matmul_softmax_f32 (lifts N constraint).
                getattr(lib, "tessera_apple_gpu_matmul_softmax_tiled_f32")
                # Native-half tiled matmul_softmax (f16/bf16 large-N single kernel).
                getattr(lib, "tessera_apple_gpu_matmul_softmax_tiled_f16")
                getattr(lib, "tessera_apple_gpu_matmul_softmax_tiled_bf16")
                # Phase 8.4.7 — MLP block fusions (matmul -> gelu, matmul -> rmsnorm).
                getattr(lib, "tessera_apple_gpu_matmul_gelu_f32")
                getattr(lib, "tessera_apple_gpu_matmul_rmsnorm_f32")
                # Native-half MLP-block fusions (f16/bf16 single fused kernel).
                getattr(lib, "tessera_apple_gpu_matmul_gelu_f16")
                getattr(lib, "tessera_apple_gpu_matmul_gelu_bf16")
                getattr(lib, "tessera_apple_gpu_matmul_rmsnorm_f16")
                getattr(lib, "tessera_apple_gpu_matmul_rmsnorm_bf16")
                # Phase 8.4.8 — SwiGLU MLP-block fusion (Stage 3 of the
                # SwiGLU Performance Plan). f32 native MSL + f16/bf16
                # reference fallback today; native half MSL is a follow-up.
                getattr(lib, "tessera_apple_gpu_swiglu_f32")
                getattr(lib, "tessera_apple_gpu_swiglu_f16")
                getattr(lib, "tessera_apple_gpu_swiglu_bf16")
                # attention_variants_plan, LA-2 — linear / kernel-feature
                # attention. f32 only in v1; D_qk*D_v ≤ 256 envelope.
                getattr(lib, "tessera_apple_gpu_linear_attn_f32")
                # attention_variants_plan, MLA-2 — DeepSeek MLA decode.
                # Host-reference path today; absorb-K MSL kernel is a
                # follow-up.
                getattr(lib, "tessera_apple_gpu_mla_decode_f32")
                # attention_variants_plan, NSA-5 — DeepSeek Native Sparse
                # Attention. Host-reference path today; fully fused MSL
                # kernel is a follow-up.
                getattr(lib, "tessera_apple_gpu_native_sparse_attn_f32")
                # Stage 14 — PPO policy loss value executor. This is the
                # freshest dylib staleness sentinel for Apple GPU runtime
                # execution; a prebuilt runtime lacking it must be rejected so
                # `_apple_gpu_dispatch` can compile a fresh source dylib.
                getattr(lib, "tessera_apple_gpu_ppo_policy_loss_ex_f32")
                # R2 (cont.) — encoded flat elementwise unary/binary, so the
                # command-buffer session can express full transformer/MLP blocks.
                getattr(lib, "tessera_apple_gpu_unary_dev_f32_enc")
                getattr(lib, "tessera_apple_gpu_binary_dev_f32_enc")
                # Phase-G Rung 0 — control-flow scan via MPSGraph forLoop.
                getattr(lib, "tessera_apple_gpu_cf_scan_f32")
                # Phase-G Rung 1 — serial draft as a forLoop.
                getattr(lib, "tessera_apple_gpu_cf_serial_draft_f32")
                # Phase-G Rung 2 — predicate-driven while generation.
                getattr(lib, "tessera_apple_gpu_cf_while_generate_f32")
                # Metal 4 — capability probe (M0) + MTLTensor round-trip (M1).
                getattr(lib, "tessera_apple_gpu_metal4_probe")
                getattr(lib, "tessera_apple_gpu_metal4_tensor_roundtrip")
                # Metal 4 M2 — MSL-loop scan through the MTL4 command model.
                getattr(lib, "tessera_apple_gpu_mtl4_scan_f32")
                # Metal 4 M3 — cooperative-matrix matmul (simdgroup_matrix).
                getattr(lib, "tessera_apple_gpu_mtl4_matmul_sg_f32")
                # Metal 4 M6 — MPP matmul2d fp16 tensor-op (MSL 4.0 cooperative).
                getattr(lib, "tessera_apple_gpu_mtl4_matmul2d_f16")
                # Metal 4 M6/M7 — bf16 sibling + fused-epilogue (bias/act) kernels.
                getattr(lib, "tessera_apple_gpu_mtl4_matmul2d_bf16")
                getattr(lib, "tessera_apple_gpu_mtl4_matmul2d_epilogue_f16")
                getattr(lib, "tessera_apple_gpu_mtl4_matmul2d_epilogue_bf16")
                # Metal 4 M8 — resident-weight fused MLP-block session.
                getattr(lib, "tessera_apple_gpu_mtl4_mlp_session_create")
                getattr(lib, "tessera_apple_gpu_mtl4_mlp_session_run")
                getattr(lib, "tessera_apple_gpu_mtl4_mlp_session_destroy")
                # Metal 4 P4 — MTL4Archive pipeline persistence.
                getattr(lib, "tessera_apple_gpu_mtl4_archive_enable")
                getattr(lib, "tessera_apple_gpu_mtl4_archive_flush")
                # P8 — on-device conv (GPU im2col + matmul2d epilogue).
                getattr(lib, "tessera_apple_gpu_mtl4_conv2d_f16")
                getattr(lib, "tessera_apple_gpu_mtl4_conv2d_bf16")
                # Phase-G Rung 3 — dynamic speculative accept as one MSL kernel.
                getattr(lib, "tessera_apple_gpu_msl_spec_accept")
                _apple_gpu_runtime = lib
                return _apple_gpu_runtime
            except (OSError, AttributeError):
                continue

    built = _build_apple_gpu_runtime_shared(root)
    _apple_gpu_runtime = ctypes.CDLL(str(built))
    return _apple_gpu_runtime


def _build_apple_gpu_runtime_shared(root: Path) -> Path:
    backend_dir = root / "src/compiler/codegen/Tessera_Apple_Backend/runtime"
    if sys.platform == "darwin":
        sources = [backend_dir / "apple_gpu_runtime.mm"]
    else:
        sources = [backend_dir / "apple_gpu_runtime_stub.cpp"]
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"Apple GPU runtime source not found: {source}")
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        raise RuntimeError(
            "Apple GPU runtime library is not available; set "
            "TESSERA_APPLE_GPU_RUNTIME_LIB or install a C++ compiler"
        )
    out_dir = Path(tempfile.gettempdir()) / "tessera_apple_gpu_runtime"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    out = out_dir / f"libtessera_apple_gpu_runtime{suffix}"
    if out.exists() and all(
        out.stat().st_mtime >= source.stat().st_mtime for source in sources
    ):
        return out
    cmd = [cxx, "-std=c++17", "-shared", "-fPIC"]
    if sys.platform == "darwin":
        cmd.extend(["-fobjc-arc", "-x", "objective-c++"])
    cmd.extend([str(source) for source in sources])
    cmd.extend(["-o", str(out)])
    if sys.platform == "darwin":
        cmd.extend([
            "-framework", "Foundation",
            "-framework", "Metal",
            "-framework", "MetalPerformanceShaders",
            # 2026-05-29: MPSGraph-backed Tier-1 / long-tail execution lane.
            "-framework", "MetalPerformanceShadersGraph",
        ])
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (
            f"Apple GPU runtime compile failed (rc={exc.returncode}).\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{exc.stderr or ''}"
        )
        raise RuntimeError(detail) from exc
    return out


def _build_apple_cpu_runtime_shared(root: Path) -> Path:
    source = root / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_cpu_runtime.cpp"
    if not source.exists():
        raise FileNotFoundError(f"Apple CPU runtime source not found: {source}")
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        raise RuntimeError(
            "Apple CPU runtime library is not available; set "
            "TESSERA_APPLE_CPU_RUNTIME_LIB or install a C++ compiler"
        )
    out_dir = Path(tempfile.gettempdir()) / "tessera_apple_cpu_runtime"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    out = out_dir / f"libtessera_apple_cpu_runtime{suffix}"
    if out.exists() and out.stat().st_mtime >= source.stat().st_mtime:
        return out
    cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(out)]
    if sys.platform == "darwin":
        cmd.extend(["-framework", "Accelerate"])
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return out


def _bind_launch_args(args: Any, arg_names: list[str]) -> dict[str, Any]:
    if isinstance(args, Mapping):
        return dict(args)
    if args is None:
        args = ()
    if not isinstance(args, (list, tuple)):
        args = (args,)
    values = {name: value for name, value in zip(arg_names, args)}
    if len(values) < len(arg_names):
        missing = ", ".join(arg_names[len(values):])
        raise ValueError(f"launch args missing value(s): {missing}")
    return values


def _as_numpy(value: Any) -> Any:
    import numpy as np

    if hasattr(value, "numpy") and callable(value.numpy):
        return np.asarray(value.numpy())
    if hasattr(value, "_data"):
        return np.asarray(value._data)
    return np.asarray(value)


def _execute_runtime_cpu_op(op_name: str, operands: list[Any], kwargs: dict[str, Any], np: Any) -> Any:
    if op_name in {"tessera.matmul", "tessera.gemm"}:
        return np.matmul(operands[0], operands[1])
    if op_name in {"tessera.conv2d_nhwc", "tessera.conv2d"}:
        bias = operands[2] if len(operands) > 2 else kwargs.get("bias", None)
        return _runtime_conv2d_nhwc(np, operands[0], operands[1], bias=bias, stride=kwargs.get("stride", 1), padding=kwargs.get("padding", 0))
    if op_name == "tessera.relu":
        return np.maximum(0, operands[0])
    if op_name == "tessera.sigmoid":
        return 1.0 / (1.0 + np.exp(-operands[0]))
    if op_name == "tessera.sin":
        return np.sin(operands[0])
    if op_name == "tessera.tanh":
        return np.tanh(operands[0])
    if op_name == "tessera.add":
        rhs = operands[1] if len(operands) > 1 else kwargs.get("scalar", 0.0)
        return np.asarray(operands[0]) + rhs
    if op_name == "tessera.mul":
        rhs = operands[1] if len(operands) > 1 else kwargs.get("scalar", 1.0)
        return np.asarray(operands[0]) * rhs
    if op_name == "tessera.softmax":
        x = operands[0]
        sm_axis = int(kwargs.get("axis", -1))
        e = np.exp(x - np.max(x, axis=sm_axis, keepdims=True))
        return e / np.sum(e, axis=sm_axis, keepdims=True)
    if op_name == "tessera.reduce":
        if str(kwargs.get("op", "sum")) != "sum":
            raise ValueError("runtime CPU reduce only supports op='sum'")
        # ``axis`` can be None (reduce over all dims), an int, or
        # a tuple of ints; widen the local variable accordingly.
        axis_raw = kwargs.get("axis", None)
        red_axis: Any = int(axis_raw) if axis_raw is not None else None
        return np.sum(operands[0], axis=red_axis, keepdims=bool(kwargs.get("keepdims", False)))
    if op_name in {"tessera.rmsnorm", "tessera.rmsnorm_safe"}:
        x = np.asarray(operands[0])
        eps = float(kwargs.get("eps", 1e-5 if op_name == "tessera.rmsnorm" else 1e-6))
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    if op_name == "tessera.rope":
        return _runtime_rope(np, operands[0], operands[1])
    if op_name == "tessera.flash_attn":
        return _runtime_flash_attn(np, operands[0], operands[1], operands[2], kwargs)
    if op_name == "tessera.transpose":
        axes = kwargs.get("axes", None)
        if isinstance(axes, list):
            axes = tuple(axes)
        return np.transpose(operands[0], axes)
    if op_name == "tessera.adam":
        param, grad, moment1, moment2 = operands
        beta1 = float(kwargs.get("beta1", 0.9))
        beta2 = float(kwargs.get("beta2", 0.999))
        lr = float(kwargs.get("lr", 1e-3))
        eps = float(kwargs.get("eps", 1e-8))
        step = int(kwargs.get("step", 1))
        new_m = beta1 * moment1 + (1.0 - beta1) * grad
        new_v = beta2 * moment2 + (1.0 - beta2) * (grad * grad)
        m_hat = new_m / (1.0 - beta1**step)
        v_hat = new_v / (1.0 - beta2**step)
        return param - lr * m_hat / (np.sqrt(v_hat) + eps), new_m, new_v
    # Type-widen so mypy accepts the assignment in the failure branch.
    GRAPH_OP_TO_SPEC: Any
    tessera: Any
    try:
        from tessera.compiler.op_catalog import GRAPH_OP_TO_SPEC
        import tessera
    except Exception:
        GRAPH_OP_TO_SPEC = {}
        tessera = None
    spec = GRAPH_OP_TO_SPEC.get(op_name) if GRAPH_OP_TO_SPEC else None
    if spec is not None and tessera is not None:
        return tessera.ops.registry.dispatch(spec.public_name, *operands, prefer_runtime=False, **kwargs)
    raise ValueError(f"unsupported runtime CPU op {op_name!r}")


def _runtime_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _runtime_conv2d_nhwc(np: Any, x: Any, weight: Any, *, bias: Any = None, stride: Any = 1, padding: Any = 0) -> Any:
    x = np.asarray(x)
    weight = np.asarray(weight)
    stride_h, stride_w = _runtime_pair(stride)
    pad_h, pad_w = _runtime_pair(padding)
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError("conv2d runtime artifact expects NHWC input and HWIO weights")
    x_pad = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    batch, in_h, in_w, _ = x_pad.shape
    k_h, k_w, _, out_c = weight.shape
    out_h = (in_h - k_h) // stride_h + 1
    out_w = (in_w - k_w) // stride_w + 1
    out = np.zeros((batch, out_h, out_w, out_c), dtype=np.result_type(x, weight))
    for i in range(out_h):
        for j in range(out_w):
            window = x_pad[:, i * stride_h:i * stride_h + k_h, j * stride_w:j * stride_w + k_w, :]
            out[:, i, j, :] = np.tensordot(window, weight, axes=([1, 2, 3], [0, 1, 2]))
    if bias is not None:
        out = out + np.asarray(bias)
    return out


def _runtime_flash_attn(np: Any, q: Any, k: Any, v: Any, kwargs: Mapping[str, Any]) -> Any:
    q = np.asarray(q)
    k = np.asarray(k)
    v = np.asarray(v)
    scale = kwargs.get("scale", None)
    scale = 1.0 / np.sqrt(q.shape[-1]) if scale is None else float(scale)
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    if bool(kwargs.get("causal", False)):
        q_len, k_len = scores.shape[-2], scores.shape[-1]
        mask = np.triu(np.ones((q_len, k_len), dtype=bool), k=1 + max(k_len - q_len, 0))
        scores = np.where(mask, -np.inf, scores)
    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = e / np.sum(e, axis=-1, keepdims=True)
    return np.matmul(weights, v)


def _runtime_rope(np: Any, x: Any, theta: Any) -> Any:
    x = np.asarray(x)
    theta = np.asarray(theta)
    if x.shape[-1] % 2 != 0:
        raise ValueError("rope requires an even innermost dimension")
    even = x[..., 0::2]
    odd = x[..., 1::2]
    if theta.shape[-1] == x.shape[-1]:
        theta = theta[..., 0::2]
    out = np.empty_like(x)
    out[..., 0::2] = even * np.cos(theta) - odd * np.sin(theta)
    out[..., 1::2] = even * np.sin(theta) + odd * np.cos(theta)
    return out
