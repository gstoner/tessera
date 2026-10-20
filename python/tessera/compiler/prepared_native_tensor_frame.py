"""Thin typed ABI binding for compiler-projected native tensor program frames.

This module marshals storage and launch records. It builds no Graph or kernel
body. Source admission and public compiler packaging are separate owners.
"""
from __future__ import annotations

import copy
import ctypes as ct
import os
import threading

import numpy as np

from .prepared_nvidia_matmul import HostView

_STORAGE = {"f32": (1, np.dtype("float32")), "f16": (2, np.dtype("float16"))}


class NativeFrameStep(ct.Structure):
    _fields_ = [
        ("image", ct.c_void_p), ("image_bytes", ct.c_size_t), ("entry", ct.c_char_p),
        ("input_count", ct.c_uint32), ("scalar_count", ct.c_uint32),
        ("buffers", ct.c_uint32 * 9), ("scalars", ct.c_int64 * 8),
        ("grid", ct.c_uint32 * 3), ("block", ct.c_uint32 * 3),
    ]


def _storage(name):
    if name == "bf16":
        import ml_dtypes

        return 3, np.dtype(ml_dtypes.bfloat16)
    if name not in _STORAGE:
        raise ValueError("native frame storage is not registered in this ABI")
    return _STORAGE[name]


def _integers(values, *, limit, count=None):
    if (not isinstance(values, list) or (count is not None and len(values) != count)
            or any(type(value) is not int or not 0 <= value < limit for value in values)):
        raise ValueError("native frame integer extent differs")
    return values


def validate_frame(frame):
    """Validate transport structure; C++ verifies initialization and ownership."""
    keys = {"schema", "target", "architecture", "function", "buffers", "roots", "stages", "output"}
    if (not isinstance(frame, dict) or set(frame) != keys
            or frame["schema"] != "tessera.native.tensor_program_frame.v1"
            or frame["target"] != "nvidia_sm120" or frame["architecture"] != "sm_120"
            or not isinstance(frame["function"], str)):
        raise ValueError("native tensor frame schema/target differs")
    buffers = frame["buffers"]
    if not isinstance(buffers, list) or not 2 <= len(buffers) <= 128:
        raise ValueError("native frame buffer extent differs")
    arena = 0
    for index, spec in enumerate(buffers):
        if (not isinstance(spec, dict) or set(spec) != {"id", "role", "shape", "storage", "bytes"}
                or type(spec["id"]) is not int or spec["id"] != index
                or spec["role"] not in {"input", "private", "output"}):
            raise ValueError("native frame buffer identity/role differs")
        _, dtype = _storage(spec["storage"])
        shape = _integers(spec["shape"], limit=2**31, count=2)
        if (not all(shape) or type(spec["bytes"]) is not int
                or spec["bytes"] != shape[0] * shape[1] * dtype.itemsize
                or spec["bytes"] >= 2**63):
            raise ValueError("native frame physical buffer bytes differ")
        arena = (arena + spec["bytes"] + 255) & ~255
        if arena >= 2**64:
            raise ValueError("native frame aligned arena exceeds its 64-bit capacity")
    roots = _integers(frame["roots"], limit=len(buffers))
    if roots != [i for i, spec in enumerate(buffers) if spec["role"] == "input"] or not roots:
        raise ValueError("native frame source argument roles differ")
    outputs = [i for i, spec in enumerate(buffers) if spec["role"] == "output"]
    if type(frame["output"]) is not int or outputs != [frame["output"]]:
        raise ValueError("native frame result role differs")
    stages = frame["stages"]
    if not isinstance(stages, list) or not 1 <= len(stages) <= 64:
        raise ValueError("native frame stage extent differs")
    for stage in stages:
        if (not isinstance(stage, dict)
                or set(stage) != {"member", "entry", "inputs", "output", "scalars", "geometry"}
                or not isinstance(stage["member"], str) or not isinstance(stage["entry"], str)
                or not stage["entry"] or "\0" in stage["entry"]):
            raise ValueError("native frame stage fields differ")
        inputs = _integers(stage["inputs"], limit=len(buffers))
        # Kernel scalar parameters are signed i64 in the native frame ABI.
        # Shape axes and CUDA grids retain their separate narrower bounds.
        scalars = _integers(stage["scalars"], limit=2**63)
        geometry = _integers(stage["geometry"], limit=2**32, count=6)
        if (not 1 <= len(inputs) <= 8 or len(scalars) > 8 or not all(scalars)
                or not all(geometry) or type(stage["output"]) is not int
                or not 0 <= stage["output"] < len(buffers)):
            raise ValueError("native frame launch extent differs")
        if (geometry[0] >= 2**31 or geometry[1] > 65535 or geometry[2] > 65535
                or geometry[3] > 1024 or geometry[4] > 1024 or geometry[5] > 64
                or geometry[3] * geometry[4] * geometry[5] > 1024):
            raise ValueError("native frame launch geometry exceeds CUDA limits")


def _descriptor(spec, data=None, strides=None):
    storage, dtype = _storage(spec["storage"])
    rows, columns = spec["shape"]
    return HostView(data, spec["bytes"], storage, 2, (ct.c_int64 * 2)(rows, columns),
                    (ct.c_int64 * 2)(*(strides or (columns * dtype.itemsize, dtype.itemsize))))


# A failed native close retains the provider and borrowed storage until context
# teardown permits explicit retry. No automatic retry changes the caller's context.
_quarantine: dict[int, "PreparedNativeTensorFrame"] = {}


class PreparedNativeTensorFrame:
    """Retained C++ execution of an already compiler-projected static frame."""

    def __init__(self, frame, images, *, provider=None):
        validate_frame(frame)
        if len(images) != len(frame["stages"]) or any(not isinstance(image, bytes) for image in images):
            raise ValueError("native frame images differ from its member count")
        if provider is None:
            from tessera import runtime as rt

            provider = rt._load_nvidia_ptx_launch()
        if provider is None or not hasattr(provider, "tessera_nvidia_native_frame_prepare_v1"):
            raise RuntimeError("native frame requires its updated native provider")
        self._frame = copy.deepcopy(frame)
        self.lib = provider
        self.pid, self.thread = os.getpid(), threading.current_thread()
        self.handle = 0
        self._borrowed: tuple[object, ...] = ()
        self._streams: tuple[int, ...] = ()
        self._failed_borrows: tuple[object, ...] = ()
        self._configure()
        specs = frame["buffers"]
        descriptors = (HostView * len(specs))(*(_descriptor(spec) for spec in specs))
        roles = (ct.c_uint8 * len(specs))(*({"input": 0, "private": 1, "output": 2}[s["role"]] for s in specs))
        steps = (NativeFrameStep * len(images))()
        retained = []
        for index, (stage, image) in enumerate(zip(frame["stages"], images, strict=True)):
            retained.append(ct.create_string_buffer(image))
            step = steps[index]
            step.image, step.image_bytes, step.entry = ct.addressof(retained[-1]), len(image), stage["entry"].encode()
            step.input_count, step.scalar_count = len(stage["inputs"]), len(stage["scalars"])
            for ordinal, value in enumerate([*stage["inputs"], stage["output"]]):
                step.buffers[ordinal] = value
            for ordinal, value in enumerate(stage["scalars"]):
                step.scalars[ordinal] = value
            step.grid[:] = stage["geometry"][:3]
            step.block[:] = stage["geometry"][3:]
        handle = ct.c_uint64()
        self._check(self.lib.tessera_nvidia_native_frame_prepare_v1(
            descriptors, roles, len(specs), steps, len(steps), ct.byref(handle)))
        self.handle = handle.value

    @property
    def frame(self):
        return copy.deepcopy(self._frame)

    def _configure(self):
        lib = self.lib
        def bind(name, args):
            function = getattr(lib, "tessera_nvidia_native_frame_" + name + "_v1")
            function.argtypes, function.restype = args, ct.c_int

        views = ct.POINTER(HostView)
        bind("prepare", [views, ct.POINTER(ct.c_uint8), ct.c_size_t,
                         ct.POINTER(NativeFrameStep), ct.c_size_t, ct.POINTER(ct.c_uint64)])
        bind("invoke", [ct.c_uint64, views, ct.c_size_t, views, ct.c_size_t])
        bind("resident", [ct.c_uint64, views, ct.c_size_t, ct.POINTER(ct.c_uint64), ct.c_size_t,
                          views, ct.c_size_t])
        bind("profile", [ct.c_uint64, ct.c_int, ct.c_int, ct.POINTER(ct.c_float), ct.c_size_t,
                         ct.POINTER(ct.c_float)])
        bind("stats", [ct.c_uint64, ct.POINTER(ct.c_uint64), ct.c_size_t])
        bind("close", [ct.c_uint64])
        lib.tessera_nvidia_matmul_last_error.argtypes = []
        lib.tessera_nvidia_matmul_last_error.restype = ct.c_char_p

    def _check(self, status):
        if status:
            message = self.lib.tessera_nvidia_matmul_last_error()
            raise RuntimeError(message.decode() if message else "native tensor frame failed")

    def _owner(self):
        if os.getpid() != self.pid or threading.current_thread() is not self.thread:
            raise RuntimeError("native tensor frame cannot cross process/thread ownership")
        if not self.handle:
            raise RuntimeError("native tensor frame is closed")

    def _root_views(self, roots, resident):
        if len(roots) != len(self._frame["roots"]):
            raise ValueError("native frame root count differs")
        views, streams = [], []
        for value, index in zip(roots, self._frame["roots"], strict=True):
            spec = self._frame["buffers"][index]
            _, expected_dtype = _storage(spec["storage"])
            if resident:
                interface = value.__cuda_array_interface__
                shape = interface["shape"]
                # Reuse the established CUDA-interface storage interpretation.
                from .resident_nvidia_tensor import _cuda_metadata_dtype

                if not isinstance(interface, dict) or interface.get("version") != 3:
                    raise ValueError("native resident frame requires CUDA interface version three")
                actual_dtype = _cuda_metadata_dtype(value, interface)
                if actual_dtype != expected_dtype:
                    raise ValueError("native resident frame dtype differs")
                strides = interface.get("strides")
                data = interface.get("data")
                if (not isinstance(data, (tuple, list)) or len(data) != 2
                        or type(data[0]) is not int or not 0 < data[0] < 2**64
                        or type(data[1]) is not bool):
                    raise ValueError("native resident frame pointer metadata differs")
                pointer = data[0]
                stream = interface.get("stream")
                if type(stream) is not int or not 0 < stream < 2**64:
                    raise ValueError("native resident frame requires an explicit producer stream")
                streams.append(stream)
            else:
                if not isinstance(value, np.ndarray) or value.dtype != expected_dtype:
                    raise ValueError("native host frame dtype differs")
                shape, strides, pointer = value.shape, value.strides, value.ctypes.data
            if strides is not None and (
                    not isinstance(strides, (tuple, list)) or len(strides) != 2
                    or any(type(value) is not int or not 0 < value < 2**63 for value in strides)):
                raise ValueError("native frame strides differ")
            if tuple(shape) != tuple(spec["shape"]):
                raise ValueError("native frame root shape differs")
            views.append(_descriptor(spec, pointer, strides))
        return (HostView * len(views))(*views), streams

    def __call__(self, roots, *, output=None):
        self._owner()
        resident = any(hasattr(value, "__cuda_array_interface__") for value in roots)
        if resident and not all(hasattr(value, "__cuda_array_interface__") for value in roots):
            raise ValueError("native frame roots must share host or resident storage")
        inputs, streams = self._root_views(roots, resident)
        spec = self._frame["buffers"][self._frame["output"]]
        _, dtype = _storage(spec["storage"])
        if output is None:
            output = np.empty(spec["shape"], dtype=dtype)
        if (not isinstance(output, np.ndarray) or tuple(output.shape) != tuple(spec["shape"])
                or output.dtype != dtype or not output.flags.writeable):
            raise ValueError("native frame output shape/storage differs")
        out = _descriptor(spec, output.ctypes.data, output.strides)
        if resident:
            tokens = (ct.c_uint64 * len(streams))(*streams)
            status = self.lib.tessera_nvidia_native_frame_resident_v1(
                self.handle, inputs, len(inputs), tokens, len(tokens), ct.byref(out), 1)
        else:
            status = self.lib.tessera_nvidia_native_frame_invoke_v1(
                self.handle, inputs, len(inputs), ct.byref(out), 1)
        if status:
            if resident:
                self._failed_borrows = (*self._failed_borrows, *roots)
            self._check(status)
        self._borrowed = tuple(roots) if resident else ()
        self._streams = tuple(streams)
        return output

    def profile(self, repeats=100, *, captured=False):
        self._owner()
        if type(repeats) is not int or type(captured) is not bool:
            raise ValueError("native frame profile arguments differ")
        if self._borrowed:
            _, streams = self._root_views(self._borrowed, True)
            if tuple(streams) != self._streams:
                raise ValueError("native frame producer streams changed; rebind before profiling")
        times = (ct.c_float * len(self._frame["stages"]))()
        program = ct.c_float()
        self._check(self.lib.tessera_nvidia_native_frame_profile_v1(
            self.handle, repeats, int(captured), times, len(times), ct.byref(program)))
        return {"stage_ms": list(times), "program_ms": program.value,
                "measurement": "cuda_graph_events" if captured else "cuda_events"}

    def stats(self):
        self._owner()
        values = (ct.c_uint64 * 6)()
        self._check(self.lib.tessera_nvidia_native_frame_stats_v1(self.handle, values, len(values)))
        return dict(zip(("capacity", "allocations", "modules", "stages", "generation", "bytes"),
                        values, strict=True))

    def close(self):
        if not self.handle:
            return
        self._owner()
        self._check(self.lib.tessera_nvidia_native_frame_close_v1(self.handle))
        self.handle = 0
        self._borrowed = self._failed_borrows = ()
        _quarantine.pop(id(self), None)

    def __enter__(self):
        self._owner()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "handle", 0):
            try:
                self.close()
            except Exception:
                _quarantine[id(self)] = self
