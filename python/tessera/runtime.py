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
    return {
        "name": cap.name,
        "available": cap.available,
        "executable": cap.executable,
        "reason": cap.reason,
        "dtypes": list(cap.dtypes),
        "features": list(cap.features),
    }


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
    return RuntimeArtifact(
        graph_ir=str(module_ir),
        metadata={"target": target or "cpu", "options": options or {}, "runtime_status": "artifact_only"},
        abi_signature=f"tessera.runtime.v1.{target or 'cpu'}",
    )


def load_artifact(path_or_bytes: str | bytes | os.PathLike[str] | RuntimeArtifact) -> RuntimeArtifact:
    if isinstance(path_or_bytes, RuntimeArtifact):
        return path_or_bytes
    if isinstance(path_or_bytes, (str, os.PathLike)) and Path(path_or_bytes).exists():
        return RuntimeArtifact.from_json(Path(path_or_bytes).read_text(encoding="utf-8"))
    if isinstance(path_or_bytes, bytes):
        return RuntimeArtifact.from_json(path_or_bytes)
    if isinstance(path_or_bytes, str):
        return RuntimeArtifact.from_json(path_or_bytes)
    raise TypeError(f"unsupported artifact input: {type(path_or_bytes)!r}")


def launch(kernel: RuntimeArtifact, args: Any, stream: Any = None) -> dict[str, Any]:
    """Launch executable CPU artifacts or return a structured non-success result."""
    global _last_profile
    start_ns = time.perf_counter_ns()
    artifact = load_artifact(kernel)
    metadata = artifact.metadata or {}
    target = str(metadata.get("target", "cpu"))
    cap = backend_capabilities(target)
    if (
        target == "apple_cpu"
        and metadata.get("executable") is True
        and metadata.get("compiler_path") == "apple_cpu_accelerate"
    ):
        try:
            output = _execute_apple_cpu_accelerate_artifact(artifact, args)
        except Exception as exc:
            _last_profile = RuntimeProfile(launch_overhead_ms=0.0)
            telemetry = make_event(
                "runtime.launch",
                source="runtime",
                op="artifact_launch",
                arch="apple_cpu",
                kernel_id=str(metadata.get("kernel_id", "apple_cpu_accelerate")),
                graph_hash=artifact.artifact_hash,
                status="invalid_artifact",
                metadata={"compiler_path": "apple_cpu_accelerate", "execution_kind": "native_cpu", "reason": str(exc)},
            )
            return {
                "ok": False,
                "runtime_status": "invalid_artifact",
                "compiler_path": "apple_cpu_accelerate",
                "execution_kind": "native_cpu",
                "artifact_hash": artifact.artifact_hash,
                "reason": str(exc),
                "telemetry": telemetry,
            }
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        _last_profile = RuntimeProfile(cpu_wall_ms=elapsed_ms, launch_overhead_ms=elapsed_ms)
        telemetry = make_event(
            "runtime.launch",
            source="runtime",
            op="artifact_launch",
            arch="apple_cpu",
            kernel_id=str(metadata.get("kernel_id", "apple_cpu_accelerate")),
            graph_hash=artifact.artifact_hash,
            latency_ms=elapsed_ms,
            status="ok",
            metadata={"compiler_path": "apple_cpu_accelerate", "execution_mode": "cpu_accelerate", "execution_kind": "native_cpu"},
        )
        return {
            "ok": True,
            "runtime_status": "success",
            "compiler_path": "apple_cpu_accelerate",
            "execution_kind": "native_cpu",
            "artifact_hash": artifact.artifact_hash,
            "output": output,
            "telemetry": telemetry,
            "profile": {
                "cpu_wall_ms": elapsed_ms,
                "launch_overhead_ms": elapsed_ms,
            },
        }
    if (
        target == "apple_gpu"
        and metadata.get("executable") is True
        and metadata.get("compiler_path") == "apple_gpu_mps"
    ):
        try:
            output = _execute_apple_gpu_mps_artifact(artifact, args)
        except Exception as exc:
            _last_profile = RuntimeProfile(launch_overhead_ms=0.0)
            telemetry = make_event(
                "runtime.launch",
                source="runtime",
                op="artifact_launch",
                arch="apple_gpu",
                kernel_id=str(metadata.get("kernel_id", "apple_gpu_mps")),
                graph_hash=artifact.artifact_hash,
                status="invalid_artifact",
                metadata={"compiler_path": "apple_gpu_mps", "execution_kind": "native_gpu", "reason": str(exc)},
            )
            return {
                "ok": False,
                "runtime_status": "invalid_artifact",
                "compiler_path": "apple_gpu_mps",
                "execution_kind": "native_gpu",
                "artifact_hash": artifact.artifact_hash,
                "reason": str(exc),
                "telemetry": telemetry,
            }
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        _last_profile = RuntimeProfile(cpu_wall_ms=elapsed_ms, launch_overhead_ms=elapsed_ms)
        telemetry = make_event(
            "runtime.launch",
            source="runtime",
            op="artifact_launch",
            arch="apple_gpu",
            kernel_id=str(metadata.get("kernel_id", "apple_gpu_mps")),
            graph_hash=artifact.artifact_hash,
            latency_ms=elapsed_ms,
            status="ok",
            metadata={"compiler_path": "apple_gpu_mps", "execution_mode": "metal_runtime", "execution_kind": "native_gpu"},
        )
        return {
            "ok": True,
            "runtime_status": "success",
            "compiler_path": "apple_gpu_mps",
            "execution_kind": "native_gpu",
            "artifact_hash": artifact.artifact_hash,
            "output": output,
            "telemetry": telemetry,
            "profile": {
                "cpu_wall_ms": elapsed_ms,
                "launch_overhead_ms": elapsed_ms,
            },
        }
    if target != "cpu":
        _last_profile = RuntimeProfile(launch_overhead_ms=0.0)
        telemetry = make_event(
            "runtime.launch",
            source="runtime",
            op="artifact_launch",
            arch=target,
            kernel_id=str(metadata.get("kernel_id", "artifact")),
            graph_hash=artifact.artifact_hash,
            status="unimplemented" if cap.available else "missing_backend",
            metadata={
                "compiler_path": str(metadata.get("compiler_path", "artifact_only")),
                "execution_kind": str(metadata.get("execution_kind", "artifact_only")),
                "reason": f"{target} generated artifact execution is not wired to the runtime ABI yet",
            },
        )
        return {
            "ok": False,
            "runtime_status": "unimplemented" if cap.available else "missing_backend",
            "compiler_path": str(metadata.get("compiler_path", "artifact_only")),
            "execution_kind": str(metadata.get("execution_kind", "artifact_only")),
            "artifact_hash": artifact.artifact_hash,
            "reason": f"{target} generated artifact execution is not wired to the runtime ABI yet",
            "telemetry": telemetry,
        }
    if metadata.get("executable") is True and metadata.get("execution_kind") == "native_cpu":
        try:
            output = _execute_native_cpu_artifact(artifact, args)
        except Exception as exc:
            try:
                output = _execute_jit_cpu_artifact(artifact, args)
            except Exception:
                output = None
            if output is not None:
                elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
                _last_profile = RuntimeProfile(cpu_wall_ms=elapsed_ms, launch_overhead_ms=elapsed_ms)
                telemetry = make_event(
                    "runtime.launch",
                    source="runtime",
                    op="artifact_launch",
                    arch="cpu",
                    kernel_id=str(metadata.get("kernel_id", "jit_cpu_numpy")),
                    graph_hash=artifact.artifact_hash,
                    latency_ms=elapsed_ms,
                    status="ok",
                    metadata={
                        "compiler_path": str(metadata.get("compiler_path", "jit_cpu_numpy")),
                        "execution_kind": "reference_cpu",
                        "native_fallback_reason": str(exc),
                    },
                )
                return {
                    "ok": True,
                    "runtime_status": "success",
                    "compiler_path": str(metadata.get("compiler_path", "jit_cpu_numpy")),
                    "execution_kind": "reference_cpu",
                    "artifact_hash": artifact.artifact_hash,
                    "output": output,
                    "telemetry": telemetry,
                    "profile": {
                        "cpu_wall_ms": elapsed_ms,
                        "launch_overhead_ms": elapsed_ms,
                    },
                }
            _last_profile = RuntimeProfile(launch_overhead_ms=0.0)
            telemetry = make_event(
                "runtime.launch",
                source="runtime",
                op="artifact_launch",
                arch="cpu",
                kernel_id=str(metadata.get("kernel_id", "native_cpu_gemm")),
                graph_hash=artifact.artifact_hash,
                status="invalid_artifact",
                metadata={"compiler_path": str(metadata.get("compiler_path", "jit_cpu_numpy")), "execution_kind": "native_cpu", "reason": str(exc)},
            )
            return {
                "ok": False,
                "runtime_status": "invalid_artifact",
                "compiler_path": str(metadata.get("compiler_path", "jit_cpu_numpy")),
                "execution_kind": "native_cpu",
                "artifact_hash": artifact.artifact_hash,
                "reason": str(exc),
                "telemetry": telemetry,
            }
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        _last_profile = RuntimeProfile(cpu_wall_ms=elapsed_ms, launch_overhead_ms=elapsed_ms)
        telemetry = make_event(
            "runtime.launch",
            source="runtime",
            op="artifact_launch",
            arch="cpu",
            kernel_id=str(metadata.get("kernel_id", "native_cpu_gemm")),
            graph_hash=artifact.artifact_hash,
            latency_ms=elapsed_ms,
            status="ok",
            metadata={"compiler_path": str(metadata.get("compiler_path", "jit_cpu_numpy")), "execution_kind": "native_cpu"},
        )
        return {
            "ok": True,
            "runtime_status": "success",
            "compiler_path": str(metadata.get("compiler_path", "jit_cpu_numpy")),
            "execution_kind": "native_cpu",
            "artifact_hash": artifact.artifact_hash,
            "output": output,
            "telemetry": telemetry,
            "profile": {
                "cpu_wall_ms": elapsed_ms,
                "launch_overhead_ms": elapsed_ms,
            },
        }

    if metadata.get("executable") is True and metadata.get("compiler_path") == "jit_cpu_numpy":
        try:
            output = _execute_jit_cpu_artifact(artifact, args)
        except Exception as exc:
            _last_profile = RuntimeProfile(launch_overhead_ms=0.0)
            telemetry = make_event(
                "runtime.launch",
                source="runtime",
                op="artifact_launch",
                arch="cpu",
                kernel_id=str(metadata.get("kernel_id", "jit_cpu_numpy")),
                graph_hash=artifact.artifact_hash,
                status="invalid_artifact",
                metadata={"compiler_path": "jit_cpu_numpy", "execution_kind": "reference_cpu", "reason": str(exc)},
            )
            return {
                "ok": False,
                "runtime_status": "invalid_artifact",
                "compiler_path": "jit_cpu_numpy",
                "execution_kind": "reference_cpu",
                "artifact_hash": artifact.artifact_hash,
                "reason": str(exc),
                "telemetry": telemetry,
            }
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        _last_profile = RuntimeProfile(cpu_wall_ms=elapsed_ms, launch_overhead_ms=elapsed_ms)
        telemetry = make_event(
            "runtime.launch",
            source="runtime",
            op="artifact_launch",
            arch="cpu",
            kernel_id=str(metadata.get("kernel_id", "jit_cpu_numpy")),
            graph_hash=artifact.artifact_hash,
            latency_ms=elapsed_ms,
            status="ok",
            metadata={"compiler_path": "jit_cpu_numpy", "execution_kind": "reference_cpu"},
        )
        return {
            "ok": True,
            "runtime_status": "success",
            "compiler_path": "jit_cpu_numpy",
            "execution_kind": "reference_cpu",
            "artifact_hash": artifact.artifact_hash,
            "output": output,
            "telemetry": telemetry,
            "profile": {
                "cpu_wall_ms": elapsed_ms,
                "launch_overhead_ms": elapsed_ms,
            },
        }

    _last_profile = RuntimeProfile(launch_overhead_ms=0.0)
    reason = str(metadata.get("reason", "Generated artifact is not executable by the runtime"))
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
        },
    )
    return {
        "ok": False,
        "runtime_status": "unsupported" if cap.available else "missing_backend",
        "compiler_path": str(metadata.get("compiler_path", "artifact_only")),
        "execution_kind": str(metadata.get("execution_kind", "artifact_only")),
        "artifact_hash": artifact.artifact_hash,
        "reason": reason,
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


_APPLE_CPU_ACCELERATE_OPS = frozenset({"tessera.matmul", "tessera.gemm"})


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
_APPLE_GPU_MPSGRAPH_OPS = (
    frozenset(_APPLE_GPU_UNARY_OPCODES)
    | frozenset(_APPLE_GPU_ROWOP_KINDS)
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
}
_APPLE_GPU_REDUCTION_OPS = frozenset(_APPLE_GPU_REDUCE_OPS)
# 2026-05-30 — Tier-3 convolutions: conv2d via the MPSGraph convolution2D node
# (NHWC/HWIO); conv3d via im2col + a GPU MPSGraph batched matmul (NDHWC/DHWIO).
_APPLE_GPU_CONV_OPS = frozenset({"tessera.conv2d", "tessera.conv3d"})
_APPLE_GPU_RUNTIME_OPS = (
    _APPLE_GPU_MPS_OPS | _APPLE_GPU_MSL_OPS | _APPLE_GPU_MPSGRAPH_OPS
    | _APPLE_GPU_PROJECTION_OPS | _APPLE_GPU_REDUCTION_OPS | _APPLE_GPU_CONV_OPS
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
            return np.std(x, axis=axis, keepdims=keepdims, ddof=ddof)
        if kind == "arg":
            r = (np.argmax if op == 0 else np.argmin)(x, axis=axis)
            return np.expand_dims(r, axis) if (keepdims and axis is not None) else r
        return (np.cumsum if op == 0 else np.cumprod)(x, axis=axis)

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
    }[op_name](f)


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
    # has the threadgroup-tiled variant (capped at 8192); f16/bf16 now also
    # have native tiled fused kernels (per-thread for N<=256, threadgroup-
    # tiled up to 8192), so the single-kernel envelope matches f32 when the
    # tiled symbol is present. Older builds without it stay at 256.
    bf16_dtype = _bfloat16_dtype()
    if a.dtype == np.float32:
        n_max = 8192
    elif a.dtype == np.float16:
        n_max = 8192 if _apple_gpu_matmul_softmax_tiled_f16() is not None else 256
    elif bf16_dtype is not None and a.dtype == bf16_dtype:
        n_max = 8192 if _apple_gpu_matmul_softmax_tiled_bf16() is not None else 256
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
    candidates = []
    env = os.environ.get("TESSERA_APPLE_GPU_RUNTIME_LIB")
    if env:
        candidates.append(Path(env))
    root = Path(__file__).resolve().parents[2]
    candidates.extend([
        root / "build/src/compiler/codegen/Tessera_Apple_Backend/libTesseraAppleRuntime.dylib",
        root / "build/src/compiler/codegen/Tessera_Apple_Backend/libTesseraAppleRuntime.so",
    ])
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
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
