"""Thin typed ABI binding for compiler-projected native tensor program frames.

This module marshals storage and launch records. It builds no Graph or kernel
body. Source admission and public compiler packaging are separate owners.
"""
from __future__ import annotations

import copy
import ctypes as ct
import os
import threading
from pathlib import Path
import tempfile

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


class NativeFrameRowPlan(ct.Structure):
    _fields_ = [("scalar_multipliers", ct.c_int64 * 8),
                ("grid_multipliers", ct.c_int64 * 3),
                ("grid_divisors", ct.c_int64 * 3)]


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
    bounded = isinstance(frame, dict) and frame.get("schema") == "tessera.native.tensor_program_frame.v2"
    shaped = isinstance(frame, dict) and frame.get("schema") == "tessera.native.tensor_program_frame.v3"
    keys = {"schema", "target", "architecture", "function", "buffers", "roots", "stages", "output"}
    if shaped:
        keys |= {"shape_program", "shape_bindings", "shape_variables"}
    if bounded:
        keys |= {"row_bound", "row_root", "row_buffers"}
    if (not isinstance(frame, dict) or set(frame) != keys
            or frame["schema"] not in {"tessera.native.tensor_program_frame.v1", "tessera.native.tensor_program_frame.v2", "tessera.native.tensor_program_frame.v3"}
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
    if bounded:
        flags = _integers(frame["row_buffers"], limit=2, count=len(buffers))
        bound, root = frame["row_bound"], frame["row_root"]
        if (type(bound) is not int or not 0 < bound < 2**31
                or type(root) is not int or root not in roots or flags[root] != 1
                or any(flag and spec["shape"][0] != bound for flag, spec in zip(flags, buffers, strict=True))):
            raise ValueError("bounded native frame row binding/capacity differs")
    stages = frame["stages"]
    if not isinstance(stages, list) or not 1 <= len(stages) <= 64:
        raise ValueError("native frame stage extent differs")
    if shaped:
        from .native_tensor_shape import _validate_program

        _validate_program(frame["shape_program"])
        variables, bindings = frame["shape_variables"], frame["shape_bindings"]
        if (not isinstance(variables, list) or not 1 <= len(variables) <= 2 * len(roots)
                or not isinstance(bindings, list) or len(bindings) != len(buffers)):
            raise ValueError("native shape dimension binding extent differs")
        for variable in variables:
            if (not isinstance(variable, list) or len(variable) != 3
                    or any(type(value) is not int for value in variable)
                    or not 0 <= variable[0] < len(roots) or variable[1] not in (0, 1)
                    or not 0 < variable[2] < 2**31):
                raise ValueError("native shape variable binding differs")
        for binding, buffer in zip(bindings, buffers, strict=True):
            if (not isinstance(binding, list) or len(binding) != 2
                    or any(type(symbol) is not int or not -1 <= symbol < len(variables) for symbol in binding)
                    or any(symbol >= 0 and variables[symbol][2] != buffer["shape"][axis]
                           for axis, symbol in enumerate(binding))):
                raise ValueError("native shape buffer capacity binding differs")
        for symbol, (argument, axis, _) in enumerate(variables):
            if bindings[roots[argument]][axis] != symbol:
                raise ValueError("native shape variable does not originate at its declared root")
        if (frame["shape_program"]["root_words"] != 2 * len(roots)
                or frame["shape_program"]["plan_words"] != 2 * len(buffers) + 11 * len(stages)):
            raise ValueError("native shape frame plan extent differs from selected ABI")
    for stage in stages:
        stage_keys = {"member", "entry", "inputs", "output", "scalars", "geometry"}
        if bounded:
            stage_keys |= {"scalar_row_multipliers", "grid_row_multipliers", "grid_row_divisors"}
        if (not isinstance(stage, dict)
                or set(stage) != stage_keys
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
        if bounded:
            scalar_rows = _integers(stage["scalar_row_multipliers"], limit=2**63, count=len(scalars))
            grid_rows = _integers(stage["grid_row_multipliers"], limit=2**63, count=3)
            divisors = _integers(stage["grid_row_divisors"], limit=2**63, count=3)
            for multiple, capacity in zip(scalar_rows, scalars, strict=True):
                if multiple and (bound * multiple >= 2**63 or bound * multiple != capacity):
                    raise ValueError("bounded native scalar expression differs from capacity")
            for multiple, divisor, capacity in zip(grid_rows, divisors, geometry[:3], strict=True):
                if (not divisor or (not multiple and divisor != 1)
                        or bound * multiple >= 2**63
                        or (multiple and (bound * multiple + divisor - 1) // divisor != capacity)):
                    raise ValueError("bounded native grid expression differs from capacity")


def _descriptor(spec, data=None, strides=None):
    storage, dtype = _storage(spec["storage"])
    rows, columns = spec["shape"]
    return HostView(data, spec["bytes"], storage, 2, (ct.c_int64 * 2)(rows, columns),
                    (ct.c_int64 * 2)(*(strides or spec.get("strides") or (columns * dtype.itemsize, dtype.itemsize))))


# A failed native close retains the provider and borrowed storage until context
# teardown permits explicit retry. No automatic retry changes the caller's context.
_quarantine: dict[int, "PreparedNativeTensorFrame"] = {}


class PreparedNativeTensorFrame:
    """Retained C++ execution of an already compiler-projected static frame."""

    def __init__(self, frame, images, *, provider=None, shape_companion=None):
        validate_frame(frame)
        if len(images) != len(frame["stages"]) or any(not isinstance(image, bytes) for image in images):
            raise ValueError("native frame images differ from its member count")
        if provider is None:
            from tessera import runtime as rt

            provider = rt._load_nvidia_ptx_launch()
        self._bounded = frame["schema"] == "tessera.native.tensor_program_frame.v2"
        self._shaped = frame["schema"] == "tessera.native.tensor_program_frame.v3"
        prepare_name = ("tessera_nvidia_native_frame_prepare_v3" if self._shaped else
                        "tessera_nvidia_native_frame_prepare_v2" if self._bounded else
                        "tessera_nvidia_native_frame_prepare_v1")
        if (provider is None or not hasattr(provider, prepare_name)
                or (self._shaped and not hasattr(provider, "tessera_nvidia_native_frame_shapes_v3"))):
            raise RuntimeError("native frame requires its updated native provider")
        self._shape_companion, self._shape_library = shape_companion, None
        if self._shaped:
            from .native_tensor_shape import NativeShapeCompanion

            if not isinstance(shape_companion, NativeShapeCompanion):
                raise ValueError("native shape frame requires its compiled shape companion")
            shape_companion.validate(frame["shape_program"])
            with tempfile.TemporaryDirectory(prefix="tessera-bound-shape-") as tmp:
                library = Path(tmp) / "shape.so"
                library.write_bytes(shape_companion.image)
                self._shape_library = ct.CDLL(str(library))
        elif shape_companion is not None:
            raise ValueError("static/row frame cannot carry a general shape companion")
        self._frame = copy.deepcopy(frame)
        self._active = copy.deepcopy(frame["buffers"])
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
        if self._shaped:
            function = getattr(self._shape_library, shape_companion.entry)
            function.argtypes = [ct.POINTER(ct.c_int64), ct.POINTER(ct.c_int64)]
            function.restype = ct.c_int32
            status = self.lib.tessera_nvidia_native_frame_prepare_v3(
                descriptors, roles, len(specs), steps, len(steps), ct.cast(function, ct.c_void_p),
                shape_companion.root_words, shape_companion.plan_words, ct.byref(handle))
        elif self._bounded:
            flags = (ct.c_uint8 * len(specs))(*frame["row_buffers"])
            plans = (NativeFrameRowPlan * len(steps))()
            for plan, stage in zip(plans, frame["stages"], strict=True):
                for index, multiple in enumerate(stage["scalar_row_multipliers"]):
                    plan.scalar_multipliers[index] = multiple
                plan.grid_multipliers[:] = stage["grid_row_multipliers"]
                plan.grid_divisors[:] = stage["grid_row_divisors"]
            status = self.lib.tessera_nvidia_native_frame_prepare_v2(
                descriptors, roles, len(specs), steps, len(steps), flags,
                frame["row_root"], frame["row_bound"], plans, ct.byref(handle))
        else:
            status = self.lib.tessera_nvidia_native_frame_prepare_v1(
                descriptors, roles, len(specs), steps, len(steps), ct.byref(handle))
        self._check(status)
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
        if self._bounded:
            function = lib.tessera_nvidia_native_frame_prepare_v2
            function.argtypes, function.restype = [
                views, ct.POINTER(ct.c_uint8), ct.c_size_t, ct.POINTER(NativeFrameStep), ct.c_size_t,
                ct.POINTER(ct.c_uint8), ct.c_uint32, ct.c_int64,
                ct.POINTER(NativeFrameRowPlan), ct.POINTER(ct.c_uint64)], ct.c_int
        if self._shaped:
            prepare = lib.tessera_nvidia_native_frame_prepare_v3
            prepare.argtypes, prepare.restype = [
                views, ct.POINTER(ct.c_uint8), ct.c_size_t, ct.POINTER(NativeFrameStep), ct.c_size_t,
                ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.POINTER(ct.c_uint64)], ct.c_int
            query = lib.tessera_nvidia_native_frame_shapes_v3
            query.argtypes, query.restype = [ct.c_uint64, ct.POINTER(ct.c_int64), ct.c_size_t,
                                             views, ct.c_size_t], ct.c_int
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

    def _active_specs(self, roots):
        specs = copy.deepcopy(self._frame["buffers"])
        if self._shaped:
            if len(roots) != len(self._frame["roots"]):
                raise ValueError("native frame root count differs")
            dimensions: list[int] = []
            for value in roots:
                interface = getattr(value, "__cuda_array_interface__", None)
                shape = interface.get("shape") if isinstance(interface, dict) else getattr(value, "shape", None)
                if (not isinstance(shape, (tuple, list)) or len(shape) != 2
                        or any(type(axis) is not int or not -(2**63) <= axis < 2**63 for axis in shape)):
                    raise ValueError("native shape frame requires rank-two i64 root dimensions")
                dimensions.extend(shape)
            words = (ct.c_int64 * len(dimensions))(*dimensions)
            native = (HostView * len(specs))()
            self._check(self.lib.tessera_nvidia_native_frame_shapes_v3(
                self.handle, words, len(words), native, len(native)))
            for spec, descriptor in zip(specs, native, strict=True):
                spec["shape"] = list(descriptor.shape)
                spec["bytes"] = descriptor.bytes
                spec["strides"] = list(descriptor.strides)
            return specs
        if not self._bounded:
            return specs
        if len(roots) != len(self._frame["roots"]):
            raise ValueError("native frame root count differs")
        ordinal = self._frame["roots"].index(self._frame["row_root"])
        value = roots[ordinal]
        shape = value.__cuda_array_interface__.get("shape") if hasattr(value, "__cuda_array_interface__") else getattr(value, "shape", None)
        if (not isinstance(shape, (tuple, list)) or len(shape) != 2
                or type(shape[0]) is not int or not 0 < shape[0] <= self._frame["row_bound"]):
            raise ValueError("bounded native frame active rows exceed capacity")
        # Generic view marshaling only. Native C++ derives and validates every
        # stage scalar/grid from the compiler's expression plan.
        for flag, spec in zip(self._frame["row_buffers"], specs, strict=True):
            if flag:
                _, dtype = _storage(spec["storage"])
                spec["shape"][0] = shape[0]
                spec["bytes"] = shape[0] * spec["shape"][1] * dtype.itemsize
        return specs

    def _root_views(self, roots, resident, specs=None):
        if len(roots) != len(self._frame["roots"]):
            raise ValueError("native frame root count differs")
        views, streams = [], []
        specs = self._active if specs is None else specs
        for value, index in zip(roots, self._frame["roots"], strict=True):
            spec = specs[index]
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
        active = self._active_specs(roots)
        inputs, streams = self._root_views(roots, resident, active)
        spec = active[self._frame["output"]]
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
        self._active = active
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

    def _release_shape_library(self):
        library = getattr(self, "_shape_library", None)
        if library is not None:
            import _ctypes

            # The native frame has retired every call before successful close.
            # Failed close retains this code image in the existing quarantine.
            if os.name == "nt":
                getattr(_ctypes, "FreeLibrary")(library._handle)
            else:
                _ctypes.dlclose(library._handle)
            self._shape_library = None

    def close(self):
        if not self.handle:
            self._release_shape_library()
            return
        self._owner()
        self._check(self.lib.tessera_nvidia_native_frame_close_v1(self.handle))
        self.handle = 0
        self._release_shape_library()
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
        else:
            self._release_shape_library()
