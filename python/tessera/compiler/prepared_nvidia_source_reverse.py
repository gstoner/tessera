"""Binding for the native composed source pullback; no Python member launches."""
import ctypes as ct
import hashlib
import os
import re
import threading
import weakref

import numpy as np

from .native_sm120_source_reverse import export_native_source_reverse
from .prepared_nvidia_active_source import ProgramStep
from .prepared_nvidia_matmul import HostView
from .prepared_nvidia_tensor_jvp import _view


class PreparedSourceReverse:
    def __init__(self, source):
        from .nvidia_native import _compile_tile_ir
        self.native = export_native_source_reverse(source)
        self._plan = self.native.manifest
        self._members = []
        for graph, schedule, tile, target, native in self.native.lower_members():
            entries = re.findall(r'\bllvm.func @(\w+)\(', tile)
            if len(entries) != 1:
                raise ValueError("source reverse member entry differs")
            lowered, ptx, resources, compiler, toolchain, _, _ = _compile_tile_ir(tile, entries[0])
            self._members.append(dict(graph_ir=graph, schedule_ir=schedule, tile_ir=tile,
                                      target_ir=target, native_ir=lowered, entry=entries[0], ptx=ptx,
                                      image_digest=hashlib.sha256(ptx.encode()).hexdigest(),
                                      resources=dict(resources), compiler=compiler, toolchain=toolchain))
        from .portable_nvidia_source_reverse import seal_members
        self._members = seal_members(self.native.plan_json, self._members)
        self._bind()

    @classmethod
    def from_manifest(cls, data):
        from .portable_nvidia_source_reverse import restore
        native, members = restore(data)
        owner = cls.__new__(cls)
        owner.native, owner._members = native, members
        owner._plan = native.manifest
        owner._bind()
        return owner

    def manifest(self):
        import copy
        self._guard()
        return copy.deepcopy(self._portable_manifest)

    def _bind(self):
        from tessera import runtime as rt
        from .portable_nvidia_source_reverse import manifest
        self._portable_manifest = manifest(self.native.plan_json, self._members)
        lib = rt._load_nvidia_ptx_launch()
        signatures = {
            "prepare": [ct.POINTER(HostView), ct.c_size_t, ct.POINTER(ct.c_int64),
                        ct.POINTER(ProgramStep), ct.c_size_t, ct.POINTER(ct.c_uint64)],
            "invoke": [ct.c_uint64, ct.POINTER(HostView), ct.c_size_t,
                       ct.POINTER(ct.c_uint64), ct.c_size_t],
            "profile": [ct.c_uint64, ct.POINTER(HostView), ct.c_size_t,
                        ct.POINTER(ct.c_uint64), ct.c_size_t, ct.c_int, ct.c_int,
                        ct.POINTER(ct.c_float), ct.c_size_t, ct.POINTER(ct.c_float)],
            "close": [ct.c_uint64],
        }
        if lib is None or any(not hasattr(lib, f"tessera_nvidia_source_reverse_{key}_v1")
                              for key in signatures):
            raise RuntimeError("source reverse requires an updated native provider")
        self.lib = lib
        for key, args in signatures.items():
            fn = getattr(lib, f"tessera_nvidia_source_reverse_{key}_v1")
            fn.argtypes, fn.restype = args, ct.c_int
        lib.tessera_nvidia_matmul_last_error.restype = ct.c_char_p
        metadata = (HostView * 5)()
        for i, buffer in enumerate(self._plan["buffers"]):
            rows, cols = buffer["shape"]
            width = 4 if buffer["storage"] == "f32" else 2
            metadata[i] = HostView(None, buffer["bytes"],
                                  {"f32": 1, "f16": 2, "bf16": 3}[buffer["storage"]],
                                  2, (ct.c_int64 * 2)(rows, cols),
                                  (ct.c_int64 * 2)(cols * width, width))
        roles = (ct.c_int64 * 3)(*self._plan["role_indices"])
        source_buffer = self._plan["buffers"][roles[0]]
        m, k = source_buffer["shape"]
        n = self._plan["buffers"][roles[1]]["shape"][1]
        images = [ct.create_string_buffer(record["ptx"].encode()) for record in self._members]
        stages = (ProgramStep * 2)()
        for i, (record, step) in enumerate(zip(self._members, self._plan["steps"], strict=True)):
            stages[i] = ProgramStep(ct.addressof(images[i]), len(record["ptx"].encode()),
                                   record["entry"].encode(), (ct.c_int64 * 2)(*step["inputs"]),
                                   (ct.c_int64 * 2)(step["outputs"][0], -1), 2, 1,
                                   (ct.c_int64 * 3)(m, k, n if i == 0 else 0),
                                   1, 3 if i == 0 else 4, metadata[roles[0]].dtype, 0)
        handle = ct.c_uint64()
        self._check(lib.tessera_nvidia_source_reverse_prepare_v1(
            metadata, 5, roles, stages, 2, ct.byref(handle)))
        self.handle = handle.value
        self.pid = os.getpid()
        self.lock = threading.RLock()
        self.finalizer = weakref.finalize(self, lib.tessera_nvidia_source_reverse_close_v1, self.handle)
        self.receipt: dict[str, object] = dict(execution_kind="native_gpu", evidence_target="nvidia_sm120",
                            compiler_path="nvidia_sm120_source_reverse_compiled",
                            native_call_binding="prepared_cpp_source_reverse_program",
                            source_graph_ir_digest=hashlib.sha256(self.native.source_ir.encode()).hexdigest(),
                            image_digests=tuple(record["image_digest"] for record in self._members))

        from .nvidia_source_reverse_runtime import register
        register(self)

    def _check(self, status):
        if status:
            raise RuntimeError(self.lib.tessera_nvidia_matmul_last_error().decode())

    def _guard(self):
        if os.getpid() != self.pid:
            raise RuntimeError("source reverse owner cannot cross fork")
        if not self.finalizer.alive:
            raise RuntimeError("source reverse owner is closed")

    def _frame(self, values):
        if len(values) != 3:
            raise ValueError("source reverse requires two primal roots and one FP32 cotangent")
        resident = [hasattr(value, "__cuda_array_interface__") for value in values]
        streams = None
        if any(resident):
            if not all(resident):
                raise ValueError("source reverse roots require one placement")
            from .resident_nvidia_tensor import cuda_frontend_specs, ordered_resident_views
            specs = cuda_frontend_specs(values, ranks=(2,))
            for (shape, dtype), buffer in zip(specs, self._plan["buffers"][:3], strict=True):
                if list(shape) != buffer["shape"] or dtype.name != {
                        "f16": "float16", "bf16": "bfloat16", "f32": "float32"}[buffer["storage"]]:
                    raise ValueError("source reverse root shape/storage differs")
            projected, declared = ordered_resident_views(values, None, writable_from=3)
            roots = tuple(projected)
            streams = (ct.c_uint64 * 3)(*declared)
        else:
            if any(not isinstance(v, np.ndarray) or not v.dtype.isnative or not v.flags.c_contiguous
                   for v in values):
                raise ValueError("source reverse roots require compact native-endian ndarrays")
            roots = tuple(_view(value) for value in values)
        import ml_dtypes
        output_dtype = np.float16 if self._plan["buffers"][4]["storage"] == "f16" else ml_dtypes.bfloat16
        output = np.empty(self._plan["buffers"][4]["shape"], output_dtype)
        return output, (HostView * 4)(*roots, _view(output)), streams

    def __call__(self, values):
        from .native_tensor_result import input_scope
        self._guard()
        with input_scope(values), self.lock:
            self._guard()
            output, views, streams = self._frame(values)
            self._check(self.lib.tessera_nvidia_source_reverse_invoke_v1(
                self.handle, views, 4, streams, 3 if streams is not None else 0))
            self.receipt.update(input_binding="cuda_resident_roots" if streams is not None else "host_roots",
                                root_uploads=0 if streams is not None else 3)
            return (output,)

    def profile(self, values, *, repeats=32, captured=True):
        from .native_tensor_result import input_scope
        self._guard()
        if type(repeats) is not int or not 0 < repeats <= (4096 if captured else 1000000) or type(captured) is not bool:
            raise ValueError("source reverse profile options differ")
        with input_scope(values), self.lock:
            self._guard()
            output, views, streams = self._frame(values)
            stages, total = (ct.c_float * 2)(), ct.c_float()
            self._check(self.lib.tessera_nvidia_source_reverse_profile_v1(
                self.handle, views, 4, streams, 3 if streams is not None else 0,
                repeats, int(captured), stages, 2, ct.byref(total)))
            return (output,), {"stage_ms": tuple(stages), "program_ms": total.value,
                               "captured": captured, "repeats": repeats}

    def close(self):
        self._guard()
        with self.lock:
            self._guard()
            self.finalizer()
