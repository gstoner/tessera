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
import sys
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, List, Mapping, Optional


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
    target = target.lower()
    backends = available_backends()
    if target not in backends:
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
    return BackendCapability(
        name=target,
        available=True,
        executable=False,
        reason=f"{target} device discovered, but generated artifact execution is not wired yet",
        dtypes=(),
        features=("memory", "events"),
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
    if target != "cpu":
        _last_profile = RuntimeProfile(launch_overhead_ms=0.0)
        return {
            "ok": False,
            "runtime_status": "unimplemented" if cap.available else "missing_backend",
            "compiler_path": str(metadata.get("compiler_path", "artifact_only")),
            "artifact_hash": artifact.artifact_hash,
            "reason": f"{target} generated artifact execution is not wired to the runtime ABI yet",
        }
    if metadata.get("executable") is True and metadata.get("compiler_path") == "jit_cpu_numpy":
        try:
            output = _execute_jit_cpu_artifact(artifact, args)
        except Exception as exc:
            _last_profile = RuntimeProfile(launch_overhead_ms=0.0)
            return {
                "ok": False,
                "runtime_status": "invalid_artifact",
                "compiler_path": "jit_cpu_numpy",
                "artifact_hash": artifact.artifact_hash,
                "reason": str(exc),
            }
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        _last_profile = RuntimeProfile(cpu_wall_ms=elapsed_ms, launch_overhead_ms=elapsed_ms)
        return {
            "ok": True,
            "runtime_status": "success",
            "compiler_path": "jit_cpu_numpy",
            "artifact_hash": artifact.artifact_hash,
            "output": output,
            "profile": {
                "cpu_wall_ms": elapsed_ms,
                "launch_overhead_ms": elapsed_ms,
            },
        }

    _last_profile = RuntimeProfile(launch_overhead_ms=0.0)
    return {
        "ok": False,
        "runtime_status": "unsupported" if cap.available else "missing_backend",
        "compiler_path": str(metadata.get("compiler_path", "artifact_only")),
        "artifact_hash": artifact.artifact_hash,
        "reason": str(metadata.get("reason", "Generated artifact is not executable by the runtime")),
    }


def get_last_profile() -> RuntimeProfile:
    return _last_profile


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
    if op_name == "tessera.relu":
        return np.maximum(0, operands[0])
    if op_name == "tessera.sigmoid":
        return 1.0 / (1.0 + np.exp(-operands[0]))
    if op_name == "tessera.sin":
        return np.sin(operands[0])
    if op_name == "tessera.softmax":
        x = operands[0]
        axis = int(kwargs.get("axis", -1))
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)
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
    raise ValueError(f"unsupported runtime CPU op {op_name!r}")
