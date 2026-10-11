"""Retained native ownership for compiler-generated ordinary SM120 attention."""
from __future__ import annotations

import copy
import ctypes as ct
import math
import os
import threading
import time
import weakref

import numpy as np


def input_signature(values):
    from .resident_nvidia_tensor import cuda_frontend_specs
    resident = any(hasattr(value, "__cuda_array_interface__") for value in values)
    if resident:
        if not all(hasattr(value, "__cuda_array_interface__") for value in values):
            raise ValueError("attention requires all resident roots or all host roots")
        specs = cuda_frontend_specs(values, ranks=(4,))
    else:
        if not all(isinstance(value, np.ndarray) for value in values):
            raise TypeError("attention requires tensor roots")
        specs = tuple((tuple(value.shape), value.dtype) for value in values)
    return tuple((str(dtype), tuple(shape)) for shape, dtype in specs)


class PreparedAttentionForward:
    def __init__(self, compiled, artifact, module):
        from .native_artifact import LaunchDescriptor, NativeImageArtifact, OrderingSemantics, WorkspaceRequirement
        from .nvidia_native import (
            _attention_contract, _attention_lse_contract,
            SM120_ATTN_LSE_F32_ABI, SM120_ATTN_LSE_BIAS_F32_ABI,
            SM120_ATTN_LSE_BCAST_F32_ABI, SM120_ATTN_F16_ABI,
            SM120_ATTN_BF16_ABI, SM120_ATTN_F32_ABI,
            SM120_ATTN_BIAS_F16_ABI, SM120_ATTN_BIAS_BF16_ABI,
            SM120_ATTN_BIAS_F32_ABI,
            SM120_ATTN_F16_RESULT_ABI, SM120_ATTN_BF16_RESULT_ABI,
            SM120_ATTN_BIAS_F16_RESULT_ABI, SM120_ATTN_BIAS_BF16_RESULT_ABI,
        )
        image, descriptor = artifact.native_image, artifact.launch_descriptor
        if not compiled.executable or image is None or descriptor is None:
            raise ValueError("prepared attention requires an executable compiler package")
        image = NativeImageArtifact.from_dict(image.to_dict())
        descriptor = LaunchDescriptor.from_dict(descriptor.to_dict())
        descriptor.validate_image(image)
        if (image.target != "nvidia_sm120" or image.architecture != "sm_120a"
                or compiled.native_image != image or compiled.launch_descriptor != descriptor):
            raise ValueError("prepared attention differs from canonical compiler package")
        stages = [compiled.bundle.graph, compiled.bundle.schedule, compiled.bundle.tile,
                  compiled.bundle.target_ir, compiled.bundle.backend]
        if (compiled.bundle.schedule.producer != "tessera-opt.tessera-graph-to-schedule"
                or any(b.input_digest != a.output_digest
                       for a, b in zip(stages[:-1], stages[1:], strict=True))):
            raise ValueError("prepared attention lacks adjacent native compiler ancestry")
        self.saved = descriptor.geometry.policy == "sm120_attention_lse_thread_per_output_128"
        if descriptor.geometry.policy not in {
            "sm120_attention_thread_per_output_128", "sm120_attention_lse_thread_per_output_128",
        } or descriptor.dynamic_local_memory_bytes or descriptor.dynamic_local_memory_expression or (
            descriptor.ordering != OrderingSemantics(
                ordered_submission=True, residency="none", synchronization=("completion",))
            or descriptor.workspace != WorkspaceRequirement(alignment=4 if self.saved else 1)
        ):
            raise ValueError("prepared attention launch geometry differs")
        contract = _attention_lse_contract(module) if self.saved else _attention_contract(module)
        allowed = ({SM120_ATTN_LSE_F32_ABI, SM120_ATTN_LSE_BIAS_F32_ABI, SM120_ATTN_LSE_BCAST_F32_ABI}
                   if self.saved else {SM120_ATTN_F16_ABI, SM120_ATTN_BF16_ABI,
                                      SM120_ATTN_F32_ABI, SM120_ATTN_BIAS_F16_ABI,
                                      SM120_ATTN_BIAS_BF16_ABI, SM120_ATTN_BIAS_F32_ABI,
                                      SM120_ATTN_F16_RESULT_ABI, SM120_ATTN_BF16_RESULT_ABI,
                                      SM120_ATTN_BIAS_F16_RESULT_ABI, SM120_ATTN_BIAS_BF16_RESULT_ABI})
        shape_facts = descriptor.provenance.get("shape")
        if not isinstance(shape_facts, (tuple, list)) or len(shape_facts) != 7:
            raise ValueError("prepared attention requires seven dimensions")
        dims: tuple[int, ...] = tuple(shape_facts)
        if contract is None or descriptor.abi_id not in allowed or dims != contract[1]:
            raise ValueError("prepared attention differs from typed frontend contract")
        if any(type(d) is not int or not 0 < d <= 65536 for d in dims):
            raise ValueError("prepared attention dimensions exceed native bounds")
        self.dims = dims
        b, hq, hkv, sq, sk, d, dv = dims
        bindings = tuple(sorted(descriptor.buffers, key=lambda item: item.ordinal))
        outputs = 2 if self.saved else 1
        inputs = len(bindings) - outputs
        if inputs not in (3, 4):
            raise ValueError("prepared attention input arity differs")
        self.storage = {"fp32": 1, "fp16": 2, "bf16": 3}.get(bindings[0].dtype)
        if self.storage is None or (self.saved and self.storage != 1):
            raise ValueError("prepared attention storage differs")
        self.output_dtype = bindings[inputs].dtype
        self.output_storage = {"fp32": 1, "fp16": 2, "bf16": 3}.get(self.output_dtype)
        if (self.output_storage is None or self.output_dtype != module.functions[0].result_types[0].dtype
                or (self.saved and self.output_storage != 1)
                or (self.output_storage != 1 and self.output_storage != self.storage)):
            raise ValueError("prepared attention result storage differs")
        if not self.saved:
            expected_abi = ({
                "fp16": SM120_ATTN_BIAS_F16_RESULT_ABI, "bf16": SM120_ATTN_BIAS_BF16_RESULT_ABI,
            } if inputs == 4 else {
                "fp16": SM120_ATTN_F16_RESULT_ABI, "bf16": SM120_ATTN_BF16_RESULT_ABI,
            }).get(self.output_dtype) if self.output_storage != 1 else ({
                "fp16": SM120_ATTN_BIAS_F16_ABI, "bf16": SM120_ATTN_BIAS_BF16_ABI, "fp32": SM120_ATTN_BIAS_F32_ABI,
            } if inputs == 4 else {
                "fp16": SM120_ATTN_F16_ABI, "bf16": SM120_ATTN_BF16_ABI, "fp32": SM120_ATTN_F32_ABI,
            })[bindings[0].dtype]
            if descriptor.abi_id != expected_abi:
                raise ValueError("prepared attention result ABI differs")
        self.bias_shape = None
        if inputs == 4:
            bias_guards = {g.dimension: g.value for g in descriptor.shape_guards
                           if g.binding == bindings[3].name and g.predicate == "eq"}
            if set(bias_guards) != set(range(4)):
                raise ValueError("prepared attention bias requires static dimensions")
            self.bias_shape = tuple(bias_guards[axis] for axis in range(4))
        shapes: tuple[tuple[int, ...], ...] = ((b, hq, sq, d), (b, hkv, sk, d), (b, hkv, sk, dv))
        if self.bias_shape:
            shapes += (self.bias_shape,)
        self.output_shapes = ((b, hq, sq, dv),) + (((b, hq, sq),) if self.saved else ())
        types = (bindings[0].dtype,) * 3 + (("fp32",) if inputs == 4 else ()) + (self.output_dtype,) + (("fp32",) if self.saved else ())
        guards = {(g.binding, g.dimension): (g.value, g.predicate) for g in descriptor.shape_guards}
        if len(guards) != len(descriptor.shape_guards):
            raise ValueError("prepared attention requires unique shape guards")
        for index, (binding, shape, dtype) in enumerate(zip(bindings, shapes + self.output_shapes, types, strict=True)):
            if (binding.rank != len(shape) or binding.dtype != dtype or binding.layout != "row_major"
                    or binding.direction != ("input" if index < inputs else "output")
                    or tuple(guards.get((binding.name, axis)) for axis in range(len(shape))) !=
                    tuple((extent, "eq") for extent in shape)):
                raise ValueError("prepared attention buffer projection differs")
        declared = sorted(descriptor.scalars, key=lambda item: item.ordinal)
        names: tuple[str, ...] = ("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv")
        self.bias_scalars = len(declared) == 11
        if self.bias_scalars:
            names += ("BiasB", "BiasH", "BiasQ", "BiasK")
        if (tuple(item.name for item in declared) != names
                or any(item.dtype != "int64" for item in declared)
                or (self.bias_scalars and self.bias_shape is None)):
            raise ValueError("prepared attention scalar projection differs")
        frontend = tuple(arg.name for arg in module.functions[0].args)
        if len(frontend) != inputs or set(frontend) != {item.name for item in bindings[:inputs]}:
            raise ValueError("prepared attention frontend bindings differ")
        self.positions = tuple(frontend.index(item.name) for item in bindings[:inputs])
        self.shapes = shapes
        self.input_dtypes = types[:inputs]
        self.compiled, self.artifact = compiled, artifact
        self.descriptor, self.image = descriptor, image
        self.module = copy.deepcopy(module)
        self.pid, self.handle, self.closed = os.getpid(), 0, False
        self.lock = threading.RLock()
        self.lib: ct.CDLL | None = None
        self.last_device_ms = None
        self._finalizer: weakref.finalize | None = None

    def matches(self, module):
        return module == self.module and self.artifact.launch_descriptor == self.descriptor

    def _library(self) -> ct.CDLL:
        if self.lib is None:
            raise RuntimeError("native attention forward runtime is uninitialized")
        return self.lib

    def _check(self, status):
        if status:
            reason = self._library().tessera_nvidia_attention_forward_last_error()
            raise RuntimeError(reason.decode() if reason else "prepared attention failed")

    def _prepare(self):
        from tessera.runtime import _load_nvidia_ptx_launch
        self.lib = _load_nvidia_ptx_launch()
        if self.lib is None or not hasattr(self.lib, "tessera_nvidia_attention_forward_prepare"):
            raise RuntimeError("native attention forward owner unavailable")
        P, S, L, U, I = ct.c_void_p, ct.c_size_t, ct.c_int64, ct.c_uint64, ct.c_int
        self.lib.tessera_nvidia_attention_forward_prepare.argtypes = [
            P, S, ct.c_char_p, ct.POINTER(L), ct.POINTER(L), I, I, I, I, ct.POINTER(U),
        ]
        self.lib.tessera_nvidia_attention_forward_prepare.restype = I
        self.lib.tessera_nvidia_attention_forward_invoke.argtypes = [
            U, ct.POINTER(P), ct.POINTER(S), S, ct.POINTER(U), S,
            ct.POINTER(P), ct.POINTER(S), S, ct.POINTER(ct.c_float),
        ]
        self.lib.tessera_nvidia_attention_forward_invoke.restype = I
        self.lib.tessera_nvidia_attention_forward_close.argtypes = [U]
        self.lib.tessera_nvidia_attention_forward_close.restype = I
        self.lib.tessera_nvidia_attention_forward_last_error.argtypes = []
        self.lib.tessera_nvidia_attention_forward_last_error.restype = ct.c_char_p
        payload = ct.create_string_buffer(self.image.payload)
        handle = U()
        bias = (L * 4)(*self.bias_shape) if self.bias_shape else None
        self._check(self.lib.tessera_nvidia_attention_forward_prepare(
            payload, len(self.image.payload), self.descriptor.entry_symbol.encode(),
            (L * 7)(*self.dims), bias, self.storage, self.output_storage, self.saved, self.bias_scalars, ct.byref(handle)))
        self.handle = handle.value
        self._finalizer = weakref.finalize(self, _retire, self.lib, self.handle, self.pid)

    def __call__(self, ordered):
        if self.pid != os.getpid():
            raise ValueError("prepared attention cannot cross fork")
        with self.lock:
            if self.closed:
                raise ValueError("prepared attention is closed")
            if (self.artifact.launch_descriptor != self.descriptor or
                    self.artifact.native_image != self.image):
                raise ValueError("prepared attention package changed")
            self.last_device_ms = None
            signature = input_signature(ordered)
            if len(signature) != len(self.positions):
                raise ValueError("prepared attention input count differs")
            values = tuple(ordered[index] for index in self.positions)
            signature = tuple(signature[index] for index in self.positions)
            expected_types = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
            if signature != tuple((expected_types[dtype], shape)
                                  for dtype, shape in zip(self.input_dtypes, self.shapes, strict=True)):
                raise ValueError("prepared attention input storage differs")
            resident = hasattr(values[0], "__cuda_array_interface__")
            if not resident:
                values = tuple(np.ascontiguousarray(value) for value in values)
            start = time.perf_counter()
            if not self.handle:
                self._prepare()
            if self.output_dtype == "bf16":
                import ml_dtypes
                output_storage = ml_dtypes.bfloat16
            else:
                output_storage = np.float16 if self.output_dtype == "fp16" else np.float32
            outputs = tuple(np.empty(shape, output_storage if index == 0 else np.float32)
                            for index, shape in enumerate(self.output_shapes))
            if resident:
                interfaces = tuple(value.__cuda_array_interface__ for value in values)
                pointers = (ct.c_void_p * len(values))(*(item["data"][0] for item in interfaces))
                streams = (ct.c_uint64 * len(values))(*(item["stream"] for item in interfaces))
            else:
                pointers = (ct.c_void_p * len(values))(*(value.ctypes.data for value in values))
                streams = None
            sizes = (ct.c_size_t * len(values))(*(math.prod(shape) * (2 if dtype in ("fp16", "bf16") else 4)
                      for shape, dtype in zip(self.shapes, self.input_dtypes, strict=True)))
            destinations = (ct.c_void_p * len(outputs))(*(value.ctypes.data for value in outputs))
            lengths = (ct.c_size_t * len(outputs))(*(value.nbytes for value in outputs))
            elapsed = ct.c_float()
            self._check(self._library().tessera_nvidia_attention_forward_invoke(
                self.handle, pointers, sizes, len(values), streams, len(values) if resident else 0,
                destinations, lengths, len(outputs), ct.byref(elapsed)))
            self.last_device_ms = elapsed.value
            receipt = dict(ok=True, execution_kind="native_gpu", compiler_path="canonical_native_descriptor",
                           native_call_binding="prepared_cpp_attention_forward", runtime_status="executed",
                           host_preparation="native_ordered_resident_snapshot" if resident else "native_pinned_host_snapshot",
                           entry=self.descriptor.entry_symbol, abi_id=self.descriptor.abi_id,
                           image_digest=self.image.image_digest,
                           launch_descriptor_digest=self.descriptor.descriptor_digest,
                           artifact_hash=self.artifact.artifact_hash, native_device_ms=elapsed.value,
                           elapsed_ms=(time.perf_counter() - start) * 1000)
            return (outputs if self.saved else outputs[0]), receipt

    def close(self):
        if self.pid != os.getpid():
            raise ValueError("prepared attention cannot cross fork")
        with self.lock:
            if self.closed:
                return
            if self.handle:
                self._check(self._library().tessera_nvidia_attention_forward_close(self.handle))
            if self._finalizer is not None:
                self._finalizer.detach()
            self.handle, self.closed = 0, True


def _retire(library, handle, pid):
    if os.getpid() == pid:
        library.tessera_nvidia_attention_forward_close(handle)
