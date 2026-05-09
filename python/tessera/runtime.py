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


_APPLE_GPU_MPS_OPS = frozenset({"tessera.matmul", "tessera.gemm"})
_APPLE_GPU_MSL_OPS = frozenset({
    "tessera.rope",
    "tessera.flash_attn",
    "tessera.softmax",
    "tessera.softmax_safe",
    "tessera.gelu",
})
_APPLE_GPU_RUNTIME_OPS = _APPLE_GPU_MPS_OPS | _APPLE_GPU_MSL_OPS


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

    # Phase 8.4.3 — fused chain dispatch. Detect the matmul -> softmax
    # pattern at the metadata layer and route to the fused kernel.
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
    """Phase 8.4.1: dispatch a single rank-3 f32 flash-attention forward
    through the apple_gpu runtime shim's custom MSL kernel. Inputs outside
    the supported envelope (rank, dtype, head_dim > 256) fall back to the
    numpy reference path used by the default `cpu` target.
    """

    if len(operands) < 3:
        raise ValueError(f"{op_name!r} requires Q, K, V operands")
    q = np.asarray(operands[0])
    k = np.asarray(operands[1])
    v = np.asarray(operands[2])

    rank3_fast_path = (
        q.dtype == np.float32 and k.dtype == np.float32 and v.dtype == np.float32
        and q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        and q.shape[0] == k.shape[0] == v.shape[0]   # B
        and k.shape[1] == v.shape[1]                  # Sk
        and q.shape[2] == k.shape[2] == v.shape[2]   # D
        and q.shape[2] <= 256
    )
    if not rank3_fast_path:
        return _runtime_flash_attn(np, q, k, v, kwargs)

    if not q.flags.c_contiguous:
        q = np.ascontiguousarray(q, dtype=np.float32)
    if not k.flags.c_contiguous:
        k = np.ascontiguousarray(k, dtype=np.float32)
    if not v.flags.c_contiguous:
        v = np.ascontiguousarray(v, dtype=np.float32)

    B, Sq, D = q.shape
    Sk = k.shape[1]
    scale = kwargs.get("scale", None)
    scale = (1.0 / float(np.sqrt(D))) if scale is None else float(scale)
    causal = 1 if bool(kwargs.get("causal", False)) else 0

    out = np.zeros((B, Sq, D), dtype=np.float32)
    flash_attn = _apple_gpu_flash_attn_f32()
    flash_attn(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(B),
        ctypes.c_int32(Sq),
        ctypes.c_int32(Sk),
        ctypes.c_int32(D),
        ctypes.c_float(scale),
        ctypes.c_int32(causal),
    )
    return out


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


def _apple_gpu_dispatch_matmul_softmax(operands: list[Any], np: Any) -> Any:
    """Phase 8.4.3 — dispatch a fused matmul -> softmax(axis=-1) chain
    through the apple_gpu runtime shim's purpose-built MSL kernel. Inputs
    outside the supported envelope (rank, dtype, N>256) fall back to a
    numpy-equivalent host computation for correctness."""

    if len(operands) != 2:
        raise ValueError("matmul_softmax fusion requires two operands (A, B)")
    a = np.asarray(operands[0])
    b = np.asarray(operands[1])
    rank2_fast_path = (
        a.dtype == np.float32 and b.dtype == np.float32
        and a.ndim == 2 and b.ndim == 2
        and a.shape[1] == b.shape[0]
        and b.shape[1] <= 256
    )
    if not rank2_fast_path:
        # Numpy reference — same algorithm; the runtime's stub also takes
        # this path on non-Darwin builds.
        scores = np.matmul(a, b)
        e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a, dtype=np.float32)
    if not b.flags.c_contiguous:
        b = np.ascontiguousarray(b, dtype=np.float32)
    M, K = a.shape
    N = b.shape[1]
    out = np.zeros((M, N), dtype=np.float32)
    fused = _apple_gpu_matmul_softmax_f32()
    fused(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(M),
        ctypes.c_int32(N),
        ctypes.c_int32(K),
    )
    return out


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
        axis = int(kwargs.get("axis", -1))
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)
    if op_name == "tessera.reduce":
        if str(kwargs.get("op", "sum")) != "sum":
            raise ValueError("runtime CPU reduce only supports op='sum'")
        axis = kwargs.get("axis", None)
        if axis is not None:
            axis = int(axis)
        return np.sum(operands[0], axis=axis, keepdims=bool(kwargs.get("keepdims", False)))
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
