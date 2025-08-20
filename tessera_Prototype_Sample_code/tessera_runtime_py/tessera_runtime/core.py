# Tessera Runtime Python Stub
# - ctypes wrapper facade + in-process mock for prototyping
from __future__ import annotations
import os, sys, ctypes as ct
from enum import IntEnum
from typing import Optional, Tuple, Sequence

__all__ = [
    "Status", "DType", "MeshAxes", "LaunchConfig",
    "init", "shutdown", "get_version",
    "mesh_create", "mesh_destroy",
    "stream_create", "stream_destroy", "stream_synchronize",
    "event_create", "event_destroy", "event_record", "stream_wait_event",
    "malloc", "free", "host_alloc", "host_free", "memcpy_async",
    "module_load", "module_unload", "kernel_get", "launch",
    "all_reduce", "reduce_scatter", "all_gather", "broadcast",
    "set_numerics_policy", "get_numerics_policy",
]

# ----------------------------- Enums & Structs -----------------------------

class Status(IntEnum):
    OK = 0
    ERR_INVALID_VALUE = 1
    ERR_OUT_OF_MEMORY = 2
    ERR_NOT_INITIALIZED = 3
    ERR_LAUNCH_FAILED = 4
    ERR_ARCH_MISMATCH = 5
    ERR_UNSUPPORTED = 6
    ERR_COLLECTIVE_MISMATCH = 7
    ERR_DETERMINISM_VIOLATION = 8
    ERR_TIMEOUT = 9
    ERR_ABI_VERSION_MISMATCH = 10
    ERR_CACHE_CORRUPT = 11

class DType(IntEnum):
    FP8_E4M3 = 1
    FP8_E5M2 = 2
    FP16 = 3
    BF16 = 4
    FP32 = 5
    FP64 = 6
    INT8 = 7
    INT16 = 8
    INT32 = 9
    INT64 = 10
    BOOL = 11
    COMPLEX64 = 12
    COMPLEX128 = 13

class MeshAxes(ct.Structure):
    _fields_ = [("tp", ct.c_int), ("pp", ct.c_int), ("dp", ct.c_int), ("ep", ct.c_int)]

class LaunchConfig(ct.Structure):
    _fields_ = [
        ("grid",  ct.c_int * 3),
        ("block", ct.c_int * 3),
        ("shmemBytes", ct.c_size_t),
        ("flags", ct.c_uint),
    ]

# Flags
LAUNCH_DEFAULT       = 0
LAUNCH_DETERMINISTIC = 1 << 0
LAUNCH_CAPTURE       = 1 << 1
LAUNCH_PERSISTENT    = 1 << 2
LAUNCH_LOW_LATENCY   = 1 << 3

# Opaque handles as void*
Handle = ct.c_void_p
Context = Handle
Mesh    = Handle
Stream  = Handle
Event   = Handle
Module  = Handle
Kernel  = Handle
Profiler= Handle

# Numerics policy
class NumericsPolicy(ct.Structure):
    _fields_ = [
        ("stableReductions", ct.c_int),
        ("deterministic",    ct.c_int),
        ("rngStateless",     ct.c_int),
        ("allowFastMath",    ct.c_int),
    ]

# ------------------------------- Loader -----------------------------------

def _load_lib() -> Optional[ct.CDLL]:
    names = [
        os.environ.get("TESSERA_RUNTIME_LIB"),
        "libtessera.so", "libtessera.dylib", "tessera.dll",
    ]
    for n in names:
        if not n: 
            continue
        try:
            return ct.CDLL(n)
        except OSError:
            continue
    return None

_lib = _load_lib()

# ---------------------------- ctypes Signatures ----------------------------

def _bind_signatures(lib: ct.CDLL):
    # keep minimal for prototypes; extend as needed
    lib.tessInit.argtypes  = [ct.POINTER(Context)]
    lib.tessInit.restype   = ct.c_int
    lib.tessShutdown.argtypes = [Context]
    lib.tessShutdown.restype  = ct.c_int

    lib.tessGetVersion.argtypes = [ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
    lib.tessGetVersion.restype  = ct.c_int

    lib.tessMeshCreate.argtypes = [Context, ct.POINTER(ct.c_int), ct.c_int, MeshAxes, ct.POINTER(Mesh)]
    lib.tessMeshCreate.restype  = ct.c_int
    lib.tessMeshDestroy.argtypes= [Mesh]
    lib.tessMeshDestroy.restype = ct.c_int

    lib.tessStreamCreate.argtypes = [Mesh, ct.c_int, ct.POINTER(Stream)]
    lib.tessStreamCreate.restype  = ct.c_int
    lib.tessStreamDestroy.argtypes= [Stream]
    lib.tessStreamDestroy.restype = ct.c_int
    lib.tessStreamSynchronize.argtypes = [Stream]
    lib.tessStreamSynchronize.restype  = ct.c_int

    lib.tessEventCreate.argtypes = [ct.POINTER(Event)]
    lib.tessEventCreate.restype  = ct.c_int
    lib.tessEventDestroy.argtypes= [Event]
    lib.tessEventDestroy.restype  = ct.c_int
    lib.tessEventRecord.argtypes = [Event, Stream]
    lib.tessEventRecord.restype  = ct.c_int
    lib.tessStreamWaitEvent.argtypes = [Stream, Event]
    lib.tessStreamWaitEvent.restype  = ct.c_int

    lib.tessMalloc.argtypes = [Mesh, ct.c_size_t, ct.POINTER(Handle)]
    lib.tessMalloc.restype  = ct.c_int
    lib.tessFree.argtypes   = [Mesh, Handle]
    lib.tessFree.restype    = ct.c_int

    lib.tessModuleLoad.argtypes = [Context, ct.c_void_p, ct.c_size_t, ct.POINTER(Module)]
    lib.tessModuleLoad.restype  = ct.c_int
    lib.tessModuleUnload.argtypes = [Module]
    lib.tessModuleUnload.restype  = ct.c_int
    lib.tessKernelGet.argtypes   = [Module, ct.c_char_p, ct.POINTER(Kernel)]
    lib.tessKernelGet.restype    = ct.c_int

    lib.tessLaunch.argtypes = [Kernel, Mesh, LaunchConfig, ct.c_void_p, ct.c_size_t, Stream]
    lib.tessLaunch.restype  = ct.c_int

    lib.tessSetNumericsPolicy.argtypes = [Context, ct.POINTER(NumericsPolicy)]
    lib.tessSetNumericsPolicy.restype  = ct.c_int
    lib.tessGetNumericsPolicy.argtypes = [Context, ct.POINTER(NumericsPolicy)]
    lib.tessGetNumericsPolicy.restype  = ct.c_int

if _lib:
    _bind_signatures(_lib)

# ------------------------------- Mock -------------------------------------

class _Mock:
    def __init__(self):
        self._ctx = Context(1)
        self._meshes = set()
        self._streams = set()
        self._events  = set()
        self._modules = {}
        self._kernels = {}
        self._allocs  = set()
        self._policy = NumericsPolicy(1,1,1,0)

    # API
    def tessInit(self, out):
        out[0] = self._ctx
        return Status.OK

    def tessShutdown(self, ctx):
        return Status.OK

    def tessGetVersion(self, maj, minr, pat):
        maj[0], minr[0], pat[0] = 1, 0, 0
        return Status.OK

    def tessMeshCreate(self, ctx, devIds, n, axes, out):
        m = Mesh(len(self._meshes)+2)
        self._meshes.add(int(m.value))
        out[0] = m
        return Status.OK

    def tessMeshDestroy(self, mesh):
        self._meshes.discard(int(mesh.value))
        return Status.OK

    def tessStreamCreate(self, mesh, prio, out):
        s = Stream(len(self._streams)+100)
        self._streams.add(int(s.value))
        out[0] = s
        return Status.OK

    def tessStreamDestroy(self, stream):
        self._streams.discard(int(stream.value))
        return Status.OK

    def tessStreamSynchronize(self, stream):
        return Status.OK

    def tessEventCreate(self, out):
        e = Event(len(self._events)+200)
        self._events.add(int(e.value))
        out[0] = e
        return Status.OK

    def tessEventDestroy(self, ev):
        self._events.discard(int(ev.value))
        return Status.OK

    def tessEventRecord(self, ev, stream):
        return Status.OK

    def tessStreamWaitEvent(self, stream, ev):
        return Status.OK

    def tessMalloc(self, mesh, bytes_, out):
        ptr = Handle(len(self._allocs)+0xDEADBEEF)
        self._allocs.add(int(ptr.value))
        out[0] = ptr
        return Status.OK

    def tessFree(self, mesh, ptr):
        self._allocs.discard(int(ptr.value))
        return Status.OK

    def tessModuleLoad(self, ctx, img, sz, out):
        mod = Module(len(self._modules)+300)
        self._modules[int(mod.value)] = img
        out[0] = mod
        return Status.OK

    def tessModuleUnload(self, mod):
        self._modules.pop(int(mod.value), None)
        return Status.OK

    def tessKernelGet(self, mod, name, out):
        key = (int(mod.value), name.decode() if isinstance(name, bytes) else str(name))
        ker = Kernel(len(self._kernels)+400)
        self._kernels[int(ker.value)] = key
        out[0] = ker
        return Status.OK

    def tessLaunch(self, ker, mesh, cfg, argbuf, argsize, stream):
        # Mock: accept anything, pretend success
        return Status.OK

    def tessSetNumericsPolicy(self, ctx, pol):
        self._policy = pol.contents
        return Status.OK

    def tessGetNumericsPolicy(self, ctx, pol):
        pol.contents = self._policy
        return Status.OK

_mock = _Mock() if _lib is None else None

# ------------------------------- API --------------------------------------

def _chk(st: int):
    if st != Status.OK:
        raise RuntimeError(f"Tessera error: {Status(st).name}")
    return st

def init() -> Context:
    ctx = Context()
    if _lib: _chk(_lib.tessInit(ct.byref(ctx)))
    else:    _chk(_mock.tessInit(ct.byref(ctx)))
    return ctx

def shutdown(ctx: Context) -> None:
    if _lib: _chk(_lib.tessShutdown(ctx))
    else:    _chk(_mock.tessShutdown(ctx))

def get_version() -> Tuple[int,int,int]:
    a=b=c=ct.c_int()
    if _lib: _chk(_lib.tessGetVersion(ct.byref(a), ct.byref(b), ct.byref(c)))
    else:    _chk(_mock.tessGetVersion(ct.byref(a), ct.byref(b), ct.byref(c)))
    return a.value, b.value, c.value

def mesh_create(ctx: Context, device_ids: Sequence[int], axes: MeshAxes) -> Mesh:
    arr = (ct.c_int * len(device_ids))(*device_ids)
    mesh = Mesh()
    if _lib: _chk(_lib.tessMeshCreate(ctx, arr, len(device_ids), axes, ct.byref(mesh)))
    else:    _chk(_mock.tessMeshCreate(ctx, arr, len(device_ids), axes, ct.byref(mesh)))
    return mesh

def mesh_destroy(mesh: Mesh) -> None:
    if _lib: _chk(_lib.tessMeshDestroy(mesh))
    else:    _chk(_mock.tessMeshDestroy(mesh))

def stream_create(mesh: Mesh, priority: int = 0) -> Stream:
    s = Stream()
    if _lib: _chk(_lib.tessStreamCreate(mesh, priority, ct.byref(s)))
    else:    _chk(_mock.tessStreamCreate(mesh, priority, ct.byref(s)))
    return s

def stream_destroy(stream: Stream) -> None:
    if _lib: _chk(_lib.tessStreamDestroy(stream))
    else:    _chk(_mock.tessStreamDestroy(stream))

def stream_synchronize(stream: Stream) -> None:
    if _lib: _chk(_lib.tessStreamSynchronize(stream))
    else:    _chk(_mock.tessStreamSynchronize(stream))

def event_create() -> Event:
    e = Event()
    if _lib: _chk(_lib.tessEventCreate(ct.byref(e)))
    else:    _chk(_mock.tessEventCreate(ct.byref(e)))
    return e

def event_destroy(ev: Event) -> None:
    if _lib: _chk(_lib.tessEventDestroy(ev))
    else:    _chk(_mock.tessEventDestroy(ev))

def event_record(ev: Event, stream: Stream) -> None:
    if _lib: _chk(_lib.tessEventRecord(ev, stream))
    else:    _chk(_mock.tessEventRecord(ev, stream))

def stream_wait_event(stream: Stream, ev: Event) -> None:
    if _lib: _chk(_lib.tessStreamWaitEvent(stream, ev))
    else:    _chk(_mock.tessStreamWaitEvent(stream, ev))

def malloc(mesh: Mesh, nbytes: int) -> Handle:
    ptr = Handle()
    if _lib: _chk(_lib.tessMalloc(mesh, nbytes, ct.byref(ptr)))
    else:    _chk(_mock.tessMalloc(mesh, nbytes, ct.byref(ptr)))
    return ptr

def free(mesh: Mesh, ptr: Handle) -> None:
    if _lib: _chk(_lib.tessFree(mesh, ptr))
    else:    _chk(_mock.tessFree(mesh, ptr))

def host_alloc(nbytes: int) -> Handle:
    ptr = Handle()
    # Using global vs pinned enum not exposed here for simplicity
    if _lib:
        _lib.tessHostAlloc.argtypes = [ct.c_size_t, ct.POINTER(Handle), ct.c_int]
        _lib.tessHostAlloc.restype  = ct.c_int
        _chk(_lib.tessHostAlloc(nbytes, ct.byref(ptr), 1))
    else:
        ptr = Handle(0xBEEF)
    return ptr

def host_free(ptr: Handle) -> None:
    if _lib:
        _lib.tessHostFree.argtypes = [Handle]
        _lib.tessHostFree.restype  = ct.c_int
        _chk(_lib.tessHostFree(ptr))
    else:
        pass

def memcpy_async(dst: Handle, src: Handle, nbytes: int, kind: str, stream: Stream) -> None:
    kmap = {"h2d":0, "d2h":1, "d2d":2}
    kval = kmap[kind.lower()]
    if _lib:
        _lib.tessMemcpyAsync.argtypes = [Handle, Handle, ct.c_size_t, ct.c_int, Stream]
        _lib.tessMemcpyAsync.restype  = ct.c_int
        _chk(_lib.tessMemcpyAsync(dst, src, nbytes, kval, stream))
    else:
        # mock: no-op
        pass

def module_load(ctx: Context, image: bytes) -> Module:
    mod = Module()
    if _lib:
        buf = ct.create_string_buffer(image)
        _chk(_lib.tessModuleLoad(ctx, ct.cast(buf, ct.c_void_p), len(image), ct.byref(mod)))
    else:
        _chk(_mock.tessModuleLoad(ctx, None, len(image), ct.byref(mod)))
    return mod

def module_unload(mod: Module) -> None:
    if _lib: _chk(_lib.tessModuleUnload(mod))
    else:    _chk(_mock.tessModuleUnload(mod))

def kernel_get(mod: Module, name: str) -> Kernel:
    ker = Kernel()
    if _lib: _chk(_lib.tessKernelGet(mod, name.encode(), ct.byref(ker)))
    else:    _chk(_mock.tessKernelGet(mod, name.encode(), ct.byref(ker)))
    return ker

def launch(ker: Kernel, mesh: Mesh, cfg: LaunchConfig, arg_buffer: bytes, stream: Stream) -> None:
    if _lib:
        buf = ct.create_string_buffer(arg_buffer)
        _chk(_lib.tessLaunch(ker, mesh, cfg, ct.cast(buf, ct.c_void_p), len(arg_buffer), stream))
    else:
        _chk(_mock.tessLaunch(ker, mesh, cfg, None, len(arg_buffer), stream))

def set_numerics_policy(ctx: Context, *, stable_reductions=True, deterministic=True, rng_stateless=True, allow_fastmath=False) -> None:
    pol = NumericsPolicy(int(stable_reductions), int(deterministic), int(rng_stateless), int(allow_fastmath))
    if _lib: _chk(_lib.tessSetNumericsPolicy(ctx, ct.byref(pol)))
    else:    _chk(_mock.tessSetNumericsPolicy(ctx, ct.byref(pol)))

def get_numerics_policy(ctx: Context) -> NumericsPolicy:
    pol = NumericsPolicy()
    if _lib: _chk(_lib.tessGetNumericsPolicy(ctx, ct.byref(pol)))
    else:    _chk(_mock.tessGetNumericsPolicy(ctx, ct.byref(pol)))
    return pol

# --------------------------- Arg Buffer Helpers ---------------------------

import struct
def pack_args(fmt: str, *values) -> bytes:
    """
    Pack a kernel parameter buffer using little-endian layout.
    Align each field to min(sizeof(field), 8) and pad final to 8 bytes.

    Example:
        # Q,K,V,O pointers (u64), B,H,L,D int32, scale float
        buf = pack_args('<QQQQiiiif', Q, K, V, O, B, H, L, D, scale)
    """
    data = struct.pack(fmt, *values)
    # round up to 8
    if len(data) % 8 != 0:
        data += b"\x00" * (8 - (len(data) % 8))
    return data

# ------------------------------- Example ----------------------------------

def _example():
    ctx = init()
    axes = MeshAxes(2, 1, 4, 0)
    mesh = mesh_create(ctx, list(range(8)), axes)
    s = stream_create(mesh)

    mod  = module_load(ctx, b"\\xCA\\xFE\\xBA\\xBE")    # mock image
    kern = kernel_get(mod, "flash_attention_fused")

    cfg = LaunchConfig((256,1,1), (128,1,1), 0, LAUNCH_DETERMINISTIC)
    Q=K=V=O=ct.c_void_p(0x1000); B=8; H=16; L=4096; D=256; scale=1.0
    argbuf = pack_args('<QQQQiiiif', Q.value, K.value, V.value, O.value, B, H, L, D, scale)

    launch(kern, mesh, cfg, argbuf, s)
    stream_synchronize(s)

    stream_destroy(s); module_unload(mod); mesh_destroy(mesh); shutdown(ctx)

if __name__ == "__main__":
    _example()
