"""Marshal a compiler-authored paired tensor program to one sealed C++ owner.

No Graph/Tile constructors, derivative formulas or per-member Python launches.
"""

from __future__ import annotations
import copy
import ctypes as ct
import hashlib
from dataclasses import asdict
import os
import weakref
import numpy as np
from .native_sm120_tensor_program import export_native_sm120_tensor_jvp
from .prepared_nvidia_matmul import HostView


class ProgramStep(ct.Structure):
    _fields_ = [
        ("image", ct.c_void_p),
        ("image_bytes", ct.c_size_t),
        ("entry", ct.c_char_p),
        ("inputs", ct.c_int64 * 2),
        ("input_count", ct.c_int64),
        ("output", ct.c_int64),
        ("kind", ct.c_int32),
        ("mapping", ct.c_int32),
    ]


def _view(value):
    dtype = {"float32": 1, "float16": 2, "bfloat16": 3}.get(value.dtype.name)
    if dtype is None or value.ndim != 2:
        raise ValueError("paired tensor host storage differs")
    return HostView(
        value.ctypes.data, value.nbytes, dtype, 2, (ct.c_int64 * 2)(*value.shape), (ct.c_int64 * 2)(*value.strides)
    )


def candidate(module, wrt_indices):
    """Route selection only; native MLIR validates arithmetic and the ABI."""
    if len(module.functions) != 1:
        return False
    fn = module.functions[0]
    if len(fn.args) != 2 or len(fn.body) < 2:
        return False
    product = fn.body[-1]
    if product.op_name != "tessera.matmul" or len(product.operands) != 2:
        return False
    prefix = fn.body[:-1]
    if any(
        op.op_name not in {"tessera.rmsnorm", "tessera.layer_norm", "tessera.softmax", "tessera.relu"}
        or len(op.operands) != 1
        for op in prefix
    ):
        return False
    roots = ["%" + arg.name for arg in fn.args]
    return product.operands[1] in roots and tuple(wrt_indices) == (roots.index(product.operands[1]),)


class PreparedTensorJVP:
    """Static native producer coefficient shared by primal and RHS tangent."""

    def __init__(self, source):
        from . import nvidia_native as native

        self.native = export_native_sm120_tensor_jvp(source)
        self.plan = self.native.manifest
        artifacts = self.native.scheduled_members()
        self.members = tuple(
            (
                native.package_scheduled_matmul
                if step["operation"] == "tessera.matmul"
                else native.package_scheduled_kernel
            )(artifact, pipeline_name="tessera-nvidia-pipeline-sm120")
            for step, artifact in zip(self.plan["steps"], artifacts, strict=True)
        )
        self.artifacts = artifacts
        self._bind(source)

    def _bind(self, source):
        from tessera import runtime as rt
        from .portable_nvidia_tensor_jvp import validate_members

        artifacts = self.artifacts
        validate_members(self.native.plan_json, artifacts, self.members)
        self._snapshot = (self.native.plan_json, copy.deepcopy(self.members), copy.deepcopy(self.plan))
        lib = rt._load_nvidia_ptx_launch()
        symbols = ("prepare", "invoke", "profile", "close", "scratch_stats")
        if lib is None or any(not hasattr(lib, "tessera_nvidia_tensor_program_" + name) for name in symbols):
            raise ValueError("paired tensor program requires an updated native provider")
        self.lib = lib
        lib.tessera_nvidia_tensor_program_prepare.argtypes = [
            ct.POINTER(HostView),
            ct.c_size_t,
            ct.POINTER(ct.c_int64),
            ct.POINTER(ProgramStep),
            ct.c_size_t,
            ct.POINTER(ct.c_uint64),
        ]
        lib.tessera_nvidia_tensor_program_prepare.restype = ct.c_int
        lib.tessera_nvidia_tensor_program_invoke.argtypes = [ct.c_uint64, ct.POINTER(HostView), ct.c_size_t]
        lib.tessera_nvidia_tensor_program_invoke.restype = ct.c_int
        lib.tessera_nvidia_tensor_program_profile.argtypes = [
            ct.c_uint64,
            ct.c_int,
            ct.c_int,
            ct.POINTER(ct.c_float),
            ct.c_size_t,
            ct.POINTER(ct.c_float),
            ct.POINTER(HostView),
            ct.c_size_t,
        ]
        lib.tessera_nvidia_tensor_program_profile.restype = ct.c_int
        lib.tessera_nvidia_tensor_program_close.argtypes = [ct.c_uint64]
        lib.tessera_nvidia_tensor_program_close.restype = ct.c_int
        lib.tessera_nvidia_tensor_program_scratch_stats.argtypes = [
            ct.c_uint64,
            ct.POINTER(ct.c_size_t),
            ct.POINTER(ct.c_size_t),
        ]
        lib.tessera_nvidia_tensor_program_scratch_stats.restype = ct.c_int
        lib.tessera_nvidia_matmul_last_error.argtypes = []
        lib.tessera_nvidia_matmul_last_error.restype = ct.c_char_p
        buffers = []
        widths = {"f16": 2, "bf16": 2, "f32": 4}
        for buffer in self.plan["buffers"]:
            rows, cols = buffer["shape"]
            width = widths[buffer["storage"]]
            buffers.append(
                HostView(
                    None,
                    buffer["bytes"],
                    {"f32": 1, "f16": 2, "bf16": 3}[buffer["storage"]],
                    2,
                    (ct.c_int64 * 2)(rows, cols),
                    (ct.c_int64 * 2)(cols * width, width),
                )
            )
        steps = []
        images = []
        for step, artifact, package in zip(self.plan["steps"], artifacts, self.members, strict=True):
            package.descriptor.validate_image(package.image)
            image = ct.create_string_buffer(package.image.payload)
            images.append(image)
            matmul = step["operation"] == "tessera.matmul"
            mapping = 0 if matmul else {"serial": 0, "cooperative_128": 1, "flat_128": 2}[artifact.schedule]
            ids = step["inputs"]
            steps.append(
                ProgramStep(
                    ct.cast(image, ct.c_void_p),
                    len(package.image.payload),
                    package.descriptor.entry_symbol.encode(),
                    (ct.c_int64 * 2)(ids[0], ids[1] if matmul else -1),
                    len(ids),
                    step["outputs"][0],
                    int(matmul),
                    mapping,
                )
            )
        self.pid = os.getpid()
        handle = ct.c_uint64()
        self._check(
            lib.tessera_nvidia_tensor_program_prepare(
                (HostView * len(buffers))(*buffers),
                len(buffers),
                (ct.c_int64 * 3)(*self.plan["role_indices"]),
                (ProgramStep * len(steps))(*steps),
                len(steps),
                ct.byref(handle),
            )
        )
        self.handle = handle.value
        self._finalizer = weakref.finalize(self, lib.tessera_nvidia_tensor_program_close, self.handle)
        self.source = source

        def digest(text):
            return hashlib.sha256(text.encode()).hexdigest()

        self.receipt = {
            "execution_kind": "native_gpu",
            "compiler_path": "nvidia_sm120_tensor_jvp_compiled",
            "execution_mode": "cuda_runtime",
            "evidence_target": "nvidia_sm120",
            "family": "tensor_rhs_jvp_program",
            "native_call_binding": "prepared_cpp_tensor_program",
            "native_program_digest": digest(self.native.plan_json),
            "source_graph_ir_digest": digest(self.plan["source_graph_ir"]),
            "paired_jvp_ir_digest": digest(self.plan["paired_graph_ir"]),
            "schedule_program_digest": digest("".join(a.schedule_ir for a in artifacts)),
            "tile_program_digest": digest("".join(a.tile_ir for a in artifacts)),
            "image_digests": [p.image.image_digest for p in self.members],
        }

    def manifest(self):
        """Serialize trusted native images and witnesses; no device pointers."""
        from .nvidia_tensor_rhs import NvidiaNormRhsProgram
        from .portable_nvidia_tensor_jvp import digest
        self._guard()
        data = {
            "schema": "tessera.nvidia.tensor_rhs_jvp.v1",
            "plan_json": self.native.plan_json,
            "artifacts": [asdict(a) for a in self.artifacts],
            "members": [NvidiaNormRhsProgram.runtime_artifact(p).to_dict() for p in self.members],
        }
        return copy.deepcopy({**data, "contract_digest": digest(data)})

    @classmethod
    def from_manifest(cls, data):
        """Prepare serialized paired images without a compiler invocation."""
        from .portable_nvidia_tensor_jvp import restore
        native, artifacts, members = restore(data)
        result = cls.__new__(cls)
        result.native = native
        result.plan = native.manifest
        result.artifacts = artifacts
        result.members = members
        result._bind(result.plan["source_graph_ir"])
        return result

    def _check(self, status):
        if status:
            reason = self.lib.tessera_nvidia_matmul_last_error()
            raise RuntimeError(reason.decode() if reason else "native paired tensor program failed")

    def _guard(self):
        if self.pid != os.getpid():
            raise ValueError("paired tensor owner cannot cross fork")
        if not self._finalizer.alive:
            raise ValueError("paired tensor owner is closed")
        if (self.native.plan_json, self.members, self.plan) != self._snapshot:
            raise ValueError("paired tensor program metadata changed")

    def __call__(self, values):
        self._guard()
        if len(values) != 3 or any(not isinstance(v, np.ndarray) for v in values):
            raise ValueError("paired tensor program requires three host ndarray roots")
        inputs = tuple(np.ascontiguousarray(value) for value in values)
        shape = tuple(self.plan["buffers"][self.plan["outputs"][0]]["shape"])
        outputs = tuple(np.empty(shape, dtype=np.float32) for _ in range(2))
        views = (HostView * 5)(*(_view(value) for value in (*inputs, *outputs)))
        self._check(self.lib.tessera_nvidia_tensor_program_invoke(self.handle, views, 5))
        return outputs

    def profile(self, *, repeats=128, captured=True):
        self._guard()
        if type(repeats) is not int or not 0 < repeats <= (4096 if captured else 1000000) or type(captured) is not bool:
            raise ValueError("paired tensor profiling count/mode differs")
        shape = tuple(self.plan["buffers"][self.plan["outputs"][0]]["shape"])
        outputs = tuple(np.empty(shape, dtype=np.float32) for _ in range(2))
        views = (HostView * 2)(*(_view(value) for value in outputs))
        times = (ct.c_float * len(self.members))()
        program = ct.c_float()
        self._check(
            self.lib.tessera_nvidia_tensor_program_profile(
                self.handle, repeats, int(captured), times, len(times), ct.byref(program), views, 2
            )
        )
        return {
            "timing_domain": "cuda_graph_batch" if captured else "native_enqueue_event",
            "program_ms": program.value,
            "stage_ms": list(times),
            "outputs": outputs,
        }

    def scratch_stats(self):
        self._guard()
        capacity = ct.c_size_t()
        allocations = ct.c_size_t()
        self._check(
            self.lib.tessera_nvidia_tensor_program_scratch_stats(self.handle, ct.byref(capacity), ct.byref(allocations))
        )
        return {"capacity_bytes": capacity.value, "allocation_count": allocations.value}

    def close(self):
        if self.pid != os.getpid():
            raise ValueError("paired tensor owner cannot cross fork")
        self._finalizer()
