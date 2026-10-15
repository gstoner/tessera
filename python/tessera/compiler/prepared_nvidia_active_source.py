"""Checked host binding for compiler-owned source-active normalization JVP.

This binding launches a composed C++ program. It performs no derivative math
and never launches individual members from Python.
"""
import copy
import ctypes as ct
import hashlib
import os
import re
import weakref

import numpy as np

from .native_sm120_active_source import export_native_active_source_jvp, validate_plan
from .prepared_nvidia_matmul import HostView
from .prepared_nvidia_tensor_jvp import _view


class ProgramStep(ct.Structure):
    _fields_ = [
        ("image", ct.c_void_p), ("image_bytes", ct.c_size_t), ("entry", ct.c_char_p),
        ("inputs", ct.c_int64 * 2), ("outputs", ct.c_int64 * 2),
        ("input_count", ct.c_int64), ("output_count", ct.c_int64),
        ("dimensions", ct.c_int64 * 3), ("schema", ct.c_int32),
        ("kind", ct.c_int32), ("storage", ct.c_int32), ("reserved", ct.c_int32),
    ]


def candidate(module, wrt_indices):
    """Select the named Graph; native MLIR owns semantic admission."""
    if len(module.functions) != 1:
        return False
    fn = module.functions[0]
    if len(fn.args) != 2 or len(fn.body) != 3:
        return False
    norm, relu, product = fn.body
    if (norm.op_name != "tessera.rmsnorm" or relu.op_name != "tessera.relu"
            or product.op_name != "tessera.matmul" or len(norm.operands) != 1
            or len(relu.operands) != 1 or len(product.operands) != 2):
        return False
    roots = ["%" + a.name for a in fn.args]
    return (norm.operands[0] in roots and product.operands[1] in roots
            and norm.operands[0] != product.operands[1]
            and tuple(wrt_indices) == (roots.index(norm.operands[0]),))


class PreparedActiveSourceJVP:
    def __init__(self, source):
        from .nvidia_native import _compile_tile_ir

        self.native = export_native_active_source_jvp(source)
        self._plan = self.native.manifest
        self._members = []
        cache = {}
        for graph, schedule, tile, target, native in self.native.lower_members():
            if graph not in cache:
                entries = re.findall(r'\bllvm.func @(\w+)\(', tile)
                if len(entries) != 1:
                    raise ValueError("source-active member entry differs")
                lowered, ptx, resources, compiler, toolchain, libraries, state = _compile_tile_ir(tile, entries[0])
                cache[graph] = {
                    "graph_ir": graph, "schedule_ir": schedule, "tile_ir": tile,
                    "target_ir": target, "native_ir": lowered, "entry": entries[0],
                    "ptx": ptx, "image_digest": hashlib.sha256(ptx.encode()).hexdigest(),
                    "resources": dict(resources), "compiler": compiler, "toolchain": toolchain,
                }
            self._members.append(cache[graph])
        from .portable_nvidia_active_source import seal_members
        self._members = seal_members(self.native.plan_json, self._members)
        self._bind()

    @classmethod
    def from_manifest(cls, data):
        from .portable_nvidia_active_source import restore
        native, _, members = restore(data)
        owner = cls.__new__(cls)
        owner.native = native
        owner._plan = native.manifest
        owner._members = members
        owner._bind()
        return owner

    def manifest(self):
        self._guard()
        return copy.deepcopy(self._portable_manifest)

    @property
    def plan(self):
        # Callers receive a fresh view, never mutable owner metadata.
        return validate_plan(self.native.plan_json)

    def _check(self, code):
        if code:
            raise RuntimeError(self.lib.tessera_nvidia_matmul_last_error().decode())

    def _bind(self):
        from tessera import runtime as rt

        from .portable_nvidia_active_source import manifest
        self._portable_manifest = manifest(self.native.plan_json, self._members)
        plan = validate_plan(self.native.plan_json)
        if len(self._members) != 3:
            raise ValueError("source-active compiled member count differs")
        for graph, record in zip(plan['member_graphs'], self._members, strict=True):
            if (record['graph_ir'] != graph
                    or hashlib.sha256(record['ptx'].encode()).hexdigest() != record['image_digest']):
                raise ValueError("source-active compiled member witness differs")
        lib = rt._load_nvidia_ptx_launch()
        signatures = {
            'prepare': [ct.POINTER(HostView), ct.c_size_t, ct.POINTER(ct.c_int64), ct.POINTER(ProgramStep), ct.c_size_t, ct.POINTER(ct.c_uint64)],
            'invoke': [ct.c_uint64, ct.POINTER(HostView), ct.c_size_t],
            'profile': [ct.c_uint64, ct.c_int, ct.c_int, ct.POINTER(ct.c_float), ct.c_size_t, ct.POINTER(ct.c_float), ct.POINTER(HostView), ct.c_size_t],
            'scratch_stats': [ct.c_uint64, ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t)],
            'close': [ct.c_uint64],
        }
        if lib is None or any(not hasattr(lib, 'tessera_nvidia_active_source_program_' + key) for key in signatures):
            raise RuntimeError("source-active program requires updated native provider")
        self.lib = lib
        for key, args in signatures.items():
            fn = getattr(lib, 'tessera_nvidia_active_source_program_' + key)
            fn.argtypes, fn.restype = args, ct.c_int
        lib.tessera_nvidia_matmul_last_error.restype = ct.c_char_p
        metadata = (HostView * 7)()
        for i, buffer in enumerate(plan['buffers']):
            rows, cols = buffer['shape']
            code = {'f32': 1, 'f16': 2, 'bf16': 3}[buffer['storage']]
            width = 4 if code == 1 else 2
            metadata[i] = HostView(None, buffer['bytes'], code, 2,
                (ct.c_int64 * 2)(rows, cols), (ct.c_int64 * 2)(cols * width, width))
        roles = (ct.c_int64 * 3)(*plan['role_indices'])
        m, k = plan['buffers'][roles[0]]['shape']
        n = plan['buffers'][roles[1]]['shape'][1]
        steps = (ProgramStep * 3)()
        images = {}
        for i, (step, record) in enumerate(zip(plan['steps'], self._members, strict=True)):
            digest = record['image_digest']
            if digest not in images:
                images[digest] = ct.create_string_buffer(record['ptx'].encode())
            outputs = [*step['outputs'], *([-1] if i else [])]
            steps[i] = ProgramStep(ct.addressof(images[digest]), len(record['ptx'].encode()), record['entry'].encode(),
                (ct.c_int64 * 2)(*step['inputs']), (ct.c_int64 * 2)(*outputs), 2, len(step['outputs']),
                (ct.c_int64 * 3)(*([m, n, k] if i else [m, k, 0])), 1, 1 if i else 2,
                metadata[roles[0]].dtype, 0)
        handle = ct.c_uint64()
        self._check(lib.tessera_nvidia_active_source_program_prepare(metadata, 7, roles, steps, 3, ct.byref(handle)))
        self.handle = handle.value
        self._pid = os.getpid()
        self._finalizer = weakref.finalize(self, lib.tessera_nvidia_active_source_program_close, self.handle)
        self.receipt = {
            'family': 'tensor_active_source_jvp_program', 'execution_mode': 'cuda_runtime',
            'execution_kind': 'native_gpu', 'evidence_target': 'nvidia_sm120',
            'compiler_path': 'nvidia_sm120_active_source_jvp_compiled',
            'native_call_binding': 'prepared_cpp_active_source_program',
            'source_graph_ir_digest': hashlib.sha256(plan['source_graph_ir'].encode()).hexdigest(),
            'paired_jvp_ir_digest': hashlib.sha256(plan['paired_graph_ir'].encode()).hexdigest(),
            'image_digests': tuple(record['image_digest'] for record in self._members),
        }

        from .nvidia_tensor_jvp_runtime import register
        register(self)

    def _guard(self):
        if os.getpid() != self._pid or not self._finalizer.alive:
            raise RuntimeError("source-active owner is closed or belongs to another process")

    def _frame(self, values):
        self._guard()
        if len(values) != 3 or any(not isinstance(a, np.ndarray) for a in values):
            raise ValueError("source-active program requires three ndarray roots")
        roots = tuple(np.ascontiguousarray(a) for a in values)
        plan = self._plan
        for i, root in enumerate(roots):
            b = plan['buffers'][i]
            if list(root.shape) != b['shape'] or root.dtype.name != {'f16': 'float16', 'bf16': 'bfloat16'}[b['storage']]:
                raise ValueError("source-active root shape/storage differs")
        outputs = tuple(np.empty(plan['buffers'][i]['shape'], np.float32) for i in (5, 6))
        views = (HostView * 5)(*[_view(a) for a in (*roots, *outputs)])
        return roots, outputs, views

    def __call__(self, values):
        roots, outputs, views = self._frame(values)
        self._check(self.lib.tessera_nvidia_active_source_program_invoke(self.handle, views, 5))
        return outputs

    def profile(self, values, *, repeats=20, captured=True):
        """Refresh same-address roots outside device timing, then profile the program."""
        if type(repeats) is not int or not 0 < repeats < 2**31 or type(captured) is not bool:
            raise ValueError("source-active profile options differ")
        roots, outputs, views = self._frame(values)
        stage, program = (ct.c_float * 3)(), ct.c_float()
        self._check(self.lib.tessera_nvidia_active_source_program_profile(self.handle, repeats, int(captured),
            stage, 3, ct.byref(program), views, 5))
        return outputs, {'stage_ms': tuple(stage), 'program_ms': program.value}

    def scratch_stats(self):
        self._guard()
        capacity, allocations = ct.c_size_t(), ct.c_size_t()
        self._check(self.lib.tessera_nvidia_active_source_program_scratch_stats(self.handle, ct.byref(capacity), ct.byref(allocations)))
        return capacity.value, allocations.value

    def close(self):
        self._guard()
        self._finalizer()
