"""Synchronous native ownership of verified SM120 matmul packages."""
from __future__ import annotations
import atexit
import copy
import ctypes as ct
import os
import threading
import time
import warnings
import weakref
from typing import Any
import numpy as np


class HostView(ct.Structure):
    _fields_ = [("data", ct.c_void_p), ("bytes", ct.c_size_t),
                ("dtype", ct.c_int32), ("rank", ct.c_int32),
                ("shape", ct.c_int64 * 2), ("strides", ct.c_int64 * 2)]


_DRIVER = None
# (handle, context identity) of owners whose close was refused because the
# context they were prepared in could not be proven live. Native memory in a
# destroyed context was already released by the driver with that context.
_forgotten_handles: list[tuple[int, int]] = []
# handle -> (lib, context identity, pid) of owners released while their
# context was not provably live. That context may be alive but not current, so
# they wait for a later close in it and are forgotten only at interpreter exit.
_deferred: dict[int, tuple[Any, int, int]] = {}
_deferred_lock = threading.Lock()
_exiting = False


def _driver():
    global _DRIVER
    if _DRIVER is None:
        driver = ct.CDLL("libcuda.so.1")
        driver.cuCtxGetCurrent.argtypes = [ct.POINTER(ct.c_void_p)]
        driver.cuCtxGetId.argtypes = [ct.c_void_p, ct.POINTER(ct.c_ulonglong)]
        driver.cuDeviceGetCount.argtypes = [ct.POINTER(ct.c_int)]
        driver.cuDeviceGet.argtypes = [ct.POINTER(ct.c_int), ct.c_int]
        driver.cuDevicePrimaryCtxGetState.argtypes = [ct.c_int, ct.POINTER(ct.c_uint), ct.POINTER(ct.c_int)]
        driver.cuDevicePrimaryCtxRetain.argtypes = [ct.POINTER(ct.c_void_p), ct.c_int]
        driver.cuDevicePrimaryCtxRelease_v2.argtypes = [ct.c_int]
        _DRIVER = driver
    return _DRIVER


def current_context_identity():
    """Identity (cuCtxGetId) of the calling thread's current CUDA context."""
    driver = _driver()
    context, identity = ct.c_void_p(), ct.c_ulonglong()
    if driver.cuCtxGetCurrent(ct.byref(context)) or not context.value:
        raise RuntimeError("prepared NVIDIA owner has no current CUDA context")
    if driver.cuCtxGetId(context, ct.byref(identity)):
        raise RuntimeError("prepared NVIDIA owner cannot read its CUDA context identity")
    return identity.value


def _live_context_identities():
    """Identities of contexts provably alive: the current one and active primaries.

    A destroyed context's handle cannot be probed safely (the driver
    dereferences it), so liveness is established only from handles the driver
    hands out now. Identities are unique for the life of the process.
    """
    driver = _driver()
    live = set()
    context, identity = ct.c_void_p(), ct.c_ulonglong()
    if (not driver.cuCtxGetCurrent(ct.byref(context)) and context.value
            and not driver.cuCtxGetId(context, ct.byref(identity))):
        live.add(identity.value)
    count = ct.c_int()
    if driver.cuDeviceGetCount(ct.byref(count)):
        return live
    for ordinal in range(count.value):
        device, flags, active = ct.c_int(), ct.c_uint(), ct.c_int()
        if (driver.cuDeviceGet(ct.byref(device), ordinal)
                or driver.cuDevicePrimaryCtxGetState(device, ct.byref(flags), ct.byref(active))
                or not active.value):
            continue
        # Retaining an active primary context only bumps its refcount.
        primary = ct.c_void_p()
        if driver.cuDevicePrimaryCtxRetain(ct.byref(primary), device):
            continue
        try:
            if not driver.cuCtxGetId(primary, ct.byref(identity)):
                live.add(identity.value)
        finally:
            driver.cuDevicePrimaryCtxRelease_v2(device)
    return live


def _forget(lib, handle, identity):
    _forgotten_handles.append((handle, identity))
    forget = getattr(lib, "tessera_nvidia_matmul_forget", None)
    if forget is not None:
        forget.argtypes = [ct.c_uint64]
        forget.restype = ct.c_int
        forget(handle)


def close_in_live_context(lib, handle, identity, pid):
    """Close a native owner only inside a context that is provably alive.

    Native close pushes the owner's context to free its modules and memory;
    against a destroyed context that push/pop segfaults inside the driver.
    Liveness is only provable for the current context and active primaries, so
    an owner of any other context is deferred, not forgotten: a live but
    non-current context still holds its memory. ``close_deferred`` closes it
    once that context is current again, and at interpreter exit any owner
    still deferred is forgotten -- recorded in ``_forgotten_handles`` and
    removed from the native registry without running its destructor, which
    would otherwise push a possibly dead context when the registry is
    destroyed. A forked child never touches the driver: the owner belongs to
    its parent.
    """
    if os.getpid() != pid:
        return 0
    try:
        live = identity in _live_context_identities()
    except Exception:
        live = False
    if live:
        status = lib.tessera_nvidia_matmul_close(handle)
        close_deferred()
        return status
    with _deferred_lock:
        if _exiting:
            _forget(lib, handle, identity)
        else:
            _deferred[handle] = (lib, identity, pid)
    warnings.warn(f"prepared NVIDIA owner {handle} not closed: CUDA context {identity} "
                  "is not provably live; deferred until a close in that context or exit",
                  ResourceWarning, stacklevel=2)
    return 0


def close_deferred(*, final=False):
    """Close deferred owners whose context is live now; with ``final``, forget the rest."""
    with _deferred_lock:
        pending = {handle: entry for handle, entry in _deferred.items()
                   if entry[2] == os.getpid()}
        if not pending:
            return
        try:
            live = _live_context_identities()
        except Exception:
            live = set()
        for handle, (lib, identity, _) in pending.items():
            if identity in live:
                del _deferred[handle]
                lib.tessera_nvidia_matmul_close(handle)
            elif final:
                del _deferred[handle]
                _forget(lib, handle, identity)


def deferred_handles():
    with _deferred_lock:
        return tuple((handle, entry[1]) for handle, entry in _deferred.items())


@atexit.register
def _close_deferred_at_exit():
    # Finalizers that run after this (weakref.finalize's own exit hook may be
    # registered either side of it) forget directly instead of deferring.
    global _exiting
    with _deferred_lock:
        _exiting = True
    close_deferred(final=True)


def forgotten_handles():
    return tuple(_forgotten_handles)


class PreparedMatmulCall:
    def __init__(self, compiled, artifact, module, frontend_graph):
        from .native_artifact import LaunchGeometry, OrderingSemantics, WorkspaceRequirement
        image, descriptor = artifact.native_image, artifact.launch_descriptor
        if not compiled.executable or image is None or descriptor is None:
            raise ValueError("prepared matmul requires an executable compiler package")
        descriptor.validate_image(image)
        if (image.target != "nvidia_sm120" or image.architecture != "sm_120a"
                or compiled.launch_descriptor != descriptor or compiled.native_image != image):
            raise ValueError("prepared matmul differs from canonical compile result")
        stages = [compiled.bundle.graph, compiled.bundle.schedule, compiled.bundle.tile,
                  compiled.bundle.target_ir, compiled.bundle.backend]
        if (compiled.bundle.schedule.producer != "tessera-opt.tessera-graph-to-schedule"
                or any(b.input_digest != a.output_digest
                       for a, b in zip(stages[:-1], stages[1:], strict=True))):
            raise ValueError("prepared matmul lacks adjacent native compiler ancestry")
        provenance = descriptor.provenance
        if (provenance.get("route") != "canonical_scheduled_tile_consumer"
                or provenance.get("physical_route") != "typed_fragment_global"
                or provenance.get("dynamic_shape_bounds") is not None
                or descriptor.geometry != LaunchGeometry(policy="sm120_scheduled_typed_16x8_mn")
                or descriptor.ordering != OrderingSemantics(
                    ordered_submission=True, residency="none", synchronization=("completion",))
                or descriptor.workspace != WorkspaceRequirement()
                or descriptor.dynamic_local_memory_bytes or descriptor.dynamic_local_memory_expression):
            raise ValueError("prepared matmul requires static typed-view synchronous ABI")
        self._initialize(artifact, module, frontend_graph)
        self.compiled = compiled

    def _initialize(self, artifact, module, frontend_graph, *, dynamic=False, binding_names=None):
        from tessera import runtime as rt
        image, descriptor = artifact.native_image, artifact.launch_descriptor
        provenance = descriptor.provenance
        declared = sorted(descriptor.scalars, key=lambda item: item.ordinal)
        scalar_names = ("M","N","K","LDA","LDB","LDD") if dynamic else ("M","N","K")
        if (tuple(item.name for item in declared) != scalar_names
                or any(item.dtype != "int64" for item in declared)):
            raise ValueError("prepared matmul scalar ABI mismatch")
        m, n, k = provenance["shape"]
        storage = {"f16": 2, "bf16": 3}.get(provenance.get("storage"))
        if storage is None:
            raise ValueError("prepared matmul requires FP16/BF16 storage")
        self.bindings = tuple(sorted(descriptor.buffers, key=lambda item: item.ordinal))
        epilogue = provenance["epilogue"]
        bias, residual = bool(epilogue["bias"]), bool(epilogue["residual"])
        output = self.bindings[-1]
        expected_shapes: list[tuple[int, ...]] = [(m, k), (k, n)]
        expected_types = [storage, storage]
        if bias:
            expected_shapes.append((n,)); expected_types.append(1)
        if residual:
            expected_shapes.append((m, n)); expected_types.append(1)
        expected_shapes.append((m, n)); expected_types.append(2 if output.dtype == "fp16" else 1)
        if (len(self.bindings) != len(expected_shapes) or output.direction != "output"
                or output.dtype not in {"fp16", "fp32"}
                or any(item.direction != "input" for item in self.bindings[:-1])):
            raise ValueError("prepared matmul tensor ABI mismatch")
        guards = {(g.binding,g.dimension): (g.value,g.predicate) for g in descriptor.shape_guards}
        if len(guards) != len(descriptor.shape_guards):
            raise ValueError("prepared matmul requires unique shape guards")
        self.dynamic_axes = (
            guards.get((self.bindings[0].name,0),(0,""))[1] == "max",
            guards.get((self.bindings[1].name,1),(0,""))[1] == "max",
            guards.get((self.bindings[0].name,1),(0,""))[1] == "max")
        if bool(any(self.dynamic_axes)) != bool(dynamic):
            raise ValueError("prepared matmul dynamic axis declaration mismatch")
        dm,dn,dk = self.dynamic_axes
        dynamic_shapes: list[tuple[bool, ...]] = [(dm,dk),(dk,dn)]
        if bias:dynamic_shapes.append((dn,))
        if residual:dynamic_shapes.append((dm,dn))
        dynamic_shapes.append((dm,dn))
        for item, shape, dtype, axes in zip(
                self.bindings,expected_shapes,expected_types,dynamic_shapes,strict=True):
            if (item.rank != len(shape)
                    or {"fp32": 1, "fp16": 2, "bf16": 3}.get(item.dtype) != dtype
                    or tuple(guards.get((item.name,axis)) for axis in range(item.rank)) !=
                    tuple((size,"max" if bounded else "eq") for size,bounded in zip(shape,axes,strict=True))):
                raise ValueError("prepared matmul guard/type projection mismatch")
        expected_layouts = ["row_major", provenance["b_layout"]]
        expected_layouts.extend(["row_major"] * (len(self.bindings) - 2))
        if dynamic:
            from .nvidia_native import (SM120_STRIDED_F16_ABI,SM120_STRIDED_BF16_ABI,
                SM120_STRIDED_ROW_B_F16_ABI,SM120_STRIDED_ROW_B_BF16_ABI)
            row=provenance["b_layout"]=="row_major"
            expected_abi=(SM120_STRIDED_ROW_B_F16_ABI if storage==2 else SM120_STRIDED_ROW_B_BF16_ABI) if row else (
                SM120_STRIDED_F16_ABI if storage==2 else SM120_STRIDED_BF16_ABI)
            if descriptor.abi_id!=expected_abi or ("_row_rhs_kernel" in descriptor.entry_symbol)!=row:
                raise ValueError("prepared dynamic RHS ABI/entry differs")
            if provenance["b_layout"] not in {"row_major","col_major"} or provenance.get("dynamic_shape_bounds") != [m,n,k]:
                raise ValueError("prepared dynamic matmul requires checked storage bounds")
            expected_layouts=["strided" if item.rank==2 else "row_major" for item in self.bindings]
        if [item.layout for item in self.bindings] != expected_layouts:
            raise ValueError("prepared matmul layout projection mismatch")
        if binding_names is None:
            names = [arg.name for arg in module.functions[0].args]
        else:
            names = list(binding_names)
            if (module is not None or len(names)!=len(self.bindings)-1 or
                    len(set(names))!=len(names) or
                    set(names)!={item.name for item in self.bindings[:-1]}):
                raise ValueError("prepared descriptor input binding names differ")
        self.input_positions = tuple(names.index(item.name) for item in self.bindings[:-1])
        self.output_shape = (m, n)
        self.output_dtype = np.float16 if output.dtype == "fp16" else np.float32
        lib = rt._load_nvidia_ptx_launch()
        if lib is None or not hasattr(lib, "tessera_nvidia_matmul_prepare"):
            raise ValueError("prepared matmul runtime unavailable")
        self.lib: ct.CDLL = lib
        lib.tessera_nvidia_matmul_prepare.argtypes = [
            ct.c_void_p, ct.c_size_t, ct.c_char_p, ct.POINTER(ct.c_int64),
            ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.POINTER(ct.c_uint64)]
        lib.tessera_nvidia_matmul_prepare.restype = ct.c_int
        lib.tessera_nvidia_matmul_invoke.argtypes = [ct.c_uint64, ct.POINTER(HostView), ct.c_size_t]
        lib.tessera_nvidia_matmul_invoke.restype = ct.c_int
        lib.tessera_nvidia_matmul_close.argtypes = [ct.c_uint64]
        lib.tessera_nvidia_matmul_close.restype = ct.c_int
        lib.tessera_nvidia_matmul_last_error.argtypes = []
        lib.tessera_nvidia_matmul_last_error.restype = ct.c_char_p
        image_buffer = ct.create_string_buffer(image.payload)
        dims = (ct.c_int64 * 3)(m, n, k)
        handle = ct.c_uint64()
        self._check(lib.tessera_nvidia_matmul_prepare(
            image_buffer, len(image.payload), descriptor.entry_symbol.encode(), dims,
            storage, int(bias), int(residual), int(provenance["b_layout"] == "row_major"),
            int(output.dtype == "fp16"), ct.byref(handle)))
        self.handle = handle.value
        if dynamic:
            try:
                setter = lib.tessera_nvidia_matmul_set_dynamic_axes
                setter.argtypes = [ct.c_uint64,ct.c_int]
                setter.restype = ct.c_int
                self._check(setter(self.handle,sum((1<<axis) for axis,enabled in
                                                   enumerate(self.dynamic_axes) if enabled)))
            except Exception:
                lib.tessera_nvidia_matmul_close(self.handle)
                raise
        try:
            self.context_identity = current_context_identity()
        except Exception:
            lib.tessera_nvidia_matmul_close(self.handle)
            raise
        self.pid = os.getpid()
        self._finalizer = weakref.finalize(self, close_in_live_context, lib, self.handle,
                                           self.context_identity, self.pid)
        self.frontend_snapshot = copy.deepcopy(frontend_graph)
        self.descriptor_snapshot = copy.deepcopy(descriptor)
        self.artifact = artifact
        self.receipt_fields = dict(
            ok=True, execution_kind="native_gpu", compiler_path="canonical_native_descriptor",
            native_call_binding="prepared_cpp_matmul", runtime_status="executed",
            image_digest=image.image_digest, launch_descriptor_digest=descriptor.descriptor_digest,
            artifact_hash=artifact.artifact_hash)

    def _check(self, status):
        if status:
            reason = self.lib.tessera_nvidia_matmul_last_error()
            raise RuntimeError(reason.decode() if reason else "prepared matmul native failure")

    def matches(self, frontend_graph):
        return (self._finalizer.alive and self.pid == os.getpid()
                and frontend_graph == self.frontend_snapshot
                and self.artifact.launch_descriptor == self.descriptor_snapshot)

    def close(self):
        if self.pid != os.getpid():
            raise ValueError("prepared matmul cannot cross fork")
        self._finalizer()

    def __call__(self, ordered):
        if self.pid != os.getpid():
            raise ValueError("prepared matmul cannot cross fork")
        if not self._finalizer.alive:
            raise ValueError("prepared matmul is closed")
        if self.artifact.launch_descriptor != self.descriptor_snapshot:
            raise ValueError("prepared matmul descriptor changed")
        start = time.perf_counter_ns()
        output_shape = self.output_shape
        if any(self.dynamic_axes):
            lhs,rhs = (ordered[i] for i in self.input_positions[:2])
            if not isinstance(lhs,np.ndarray) or not isinstance(rhs,np.ndarray) or lhs.ndim!=2 or rhs.ndim!=2:
                raise TypeError("prepared dynamic matmul expects rank-two arrays")
            output_shape=(lhs.shape[0],rhs.shape[1])
        output = np.empty(output_shape, self.output_dtype)
        arrays = tuple(ordered[i] for i in self.input_positions) + (output,)
        views = (HostView * len(arrays))()
        for view, array in zip(views, arrays, strict=True):
            if not isinstance(array, np.ndarray) or not 1 <= array.ndim <= 2:
                raise TypeError("prepared matmul expects rank-one/two host arrays")
            view.data, view.bytes, view.rank = array.ctypes.data, array.nbytes, array.ndim
            view.dtype = {"float32": 1, "float16": 2, "bfloat16": 3}.get(array.dtype.name, 0)
            for axis, (shape, stride) in enumerate(zip(array.shape, array.strides, strict=True)):
                view.shape[axis], view.strides[axis] = shape, stride
        self._check(self.lib.tessera_nvidia_matmul_invoke(self.handle, views, len(arrays)))
        elapsed = (time.perf_counter_ns() - start) / 1e6
        from tessera import runtime as rt
        rt._last_profile = rt.RuntimeProfile(launch_overhead_ms=elapsed, kernel_elapsed_ms=None)
        return output, dict(self.receipt_fields, output=output, elapsed_ms=elapsed)
