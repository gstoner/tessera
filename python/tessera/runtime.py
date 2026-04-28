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
import os
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, List, Optional


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
        return (1, 0, 0)

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Initialise the runtime and all available backends."""
        if self._mock_mode:
            self._backend.init()
        else:
            self._check("tsrInit", self._lib.tsrInit())
        self._initialized = True

    def shutdown(self) -> None:
        """Release all runtime resources."""
        if self._mock_mode:
            self._backend.shutdown()
        else:
            self._check("tsrShutdown", self._lib.tsrShutdown())
        self._initialized = False

    # ------------------------------------------------------------------
    # Device enumeration
    # ------------------------------------------------------------------

    def get_device_count(self) -> int:
        if self._mock_mode:
            return self._backend.get_device_count()
        count = ctypes.c_int(0)
        self._check("tsrGetDeviceCount",
                    self._lib.tsrGetDeviceCount(ctypes.byref(count)))
        return count.value

    def get_device(self, index: int) -> Any:
        if self._mock_mode:
            return self._backend.get_device(index)
        handle = ctypes.c_void_p(0)
        self._check("tsrGetDevice",
                    self._lib.tsrGetDevice(index, ctypes.byref(handle)))
        return handle

    def get_device_props(self, dev: Any) -> DeviceProps:
        if self._mock_mode:
            return self._backend.get_device_props(dev)
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
        return DeviceProps(
            kind=DeviceKind(p.kind),
            name=p.name.decode(),
            logical_tile_threads_max=p.logical_tile_threads_max,
            concurrent_tiles_hint=p.concurrent_tiles_hint,
        )

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def malloc(self, dev: Any, size: int) -> Any:
        if self._mock_mode:
            return self._backend.malloc(dev, size)
        handle = ctypes.c_void_p(0)
        self._check("tsrMalloc",
                    self._lib.tsrMalloc(dev, size, ctypes.byref(handle)))
        return handle

    def free(self, buf: Any) -> None:
        if self._mock_mode:
            self._backend.free(buf)
        else:
            self._check("tsrFree", self._lib.tsrFree(buf))

    def memset(self, buf: Any, value: int, size: int) -> None:
        if self._mock_mode:
            self._backend.memset(buf, value, size)
        else:
            self._check("tsrMemset", self._lib.tsrMemset(buf, value, size))

    def memcpy(self, dst: Any, src: Any, size: int,
               kind: MemcpyKind = MemcpyKind.DEVICE_TO_DEVICE) -> None:
        if self._mock_mode:
            self._backend.memcpy(dst, src, size, kind)
        else:
            self._check("tsrMemcpy",
                        self._lib.tsrMemcpy(dst, src, size, int(kind)))

    def map(self, buf: Any) -> bytes:
        if self._mock_mode:
            return self._backend.map(buf)
        host_ptr = ctypes.c_void_p(0)
        size = ctypes.c_size_t(0)
        self._check("tsrMap",
                    self._lib.tsrMap(buf, ctypes.byref(host_ptr),
                                     ctypes.byref(size)))
        if not host_ptr.value or not size.value:
            return b""
        return (ctypes.c_char * size.value).from_address(host_ptr.value).raw

    def unmap(self, buf: Any) -> None:
        if self._mock_mode:
            self._backend.unmap(buf)
        else:
            self._check("tsrUnmap", self._lib.tsrUnmap(buf))

    # ------------------------------------------------------------------
    # Streams & events
    # ------------------------------------------------------------------

    def create_stream(self, dev: Any) -> Any:
        if self._mock_mode:
            return self._backend.create_stream(dev)
        h = ctypes.c_void_p(0)
        self._check("tsrCreateStream",
                    self._lib.tsrCreateStream(dev, ctypes.byref(h)))
        return h

    def destroy_stream(self, stream: Any) -> None:
        if self._mock_mode:
            self._backend.destroy_stream(stream)
        else:
            self._check("tsrDestroyStream",
                        self._lib.tsrDestroyStream(stream))

    def stream_sync(self, stream: Any) -> None:
        if self._mock_mode:
            self._backend.stream_sync(stream)
        else:
            self._check("tsrStreamSynchronize",
                        self._lib.tsrStreamSynchronize(stream))

    def create_event(self, dev: Any) -> Any:
        if self._mock_mode:
            return self._backend.create_event(dev)
        h = ctypes.c_void_p(0)
        self._check("tsrCreateEvent",
                    self._lib.tsrCreateEvent(dev, ctypes.byref(h)))
        return h

    def record_event(self, event: Any, stream: Any) -> None:
        if self._mock_mode:
            self._backend.record_event(event, stream)
        else:
            self._check("tsrRecordEvent",
                        self._lib.tsrRecordEvent(event, stream))

    def wait_event(self, event: Any, stream: Any) -> None:
        if self._mock_mode:
            self._backend.wait_event(event, stream)
        else:
            self._check("tsrWaitEvent",
                        self._lib.tsrWaitEvent(event, stream))

    def event_sync(self, event: Any) -> None:
        if self._mock_mode:
            self._backend.event_sync(event)
        else:
            self._check("tsrEventSynchronize",
                        self._lib.tsrEventSynchronize(event))

    def destroy_event(self, event: Any) -> None:
        if self._mock_mode:
            self._backend.destroy_event(event)
        else:
            self._check("tsrDestroyEvent",
                        self._lib.tsrDestroyEvent(event))

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

    def get_last_error(self) -> str:
        if self._mock_mode:
            return self._backend.get_last_error()
        err = self._lib.tsrGetLastError()
        return err.decode() if isinstance(err, bytes) else (err or "")

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
