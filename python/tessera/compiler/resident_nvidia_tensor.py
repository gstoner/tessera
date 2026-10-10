"""Native submission of checked resident producer-to-matmul packages."""
from __future__ import annotations
import copy
import ctypes as ct
import os
import weakref
import numpy as np
from .prepared_nvidia_matmul import HostView


class ResidentTensorCall:
    """Retain compiler images; caller buffers stay live until native completion."""
    def __init__(self, program, *, producer_chain=()):
        from tessera import runtime as rt
        program.validate()
        from dataclasses import replace
        stages = tuple(producer_chain) or (program.producer,)
        if stages[-1] != program.producer:
            raise ValueError("resident chain final producer differs from the checked edge")
        for stage in stages:
            replace(program, producer=stage).validate()
        self.producer_chain = stages
        self.chain_snapshot = copy.deepcopy(stages)
        self.program = program
        self.snapshot = copy.deepcopy(program)
        self.pid = os.getpid()
        self.lib = lib = rt._load_nvidia_ptx_launch()
        if lib is None or not hasattr(lib, "tessera_nvidia_matmul_invoke_resident"):
            raise RuntimeError("native resident tensor runtime unavailable")
        lib.tessera_nvidia_matmul_prepare.argtypes = [
            ct.c_void_p, ct.c_size_t, ct.c_char_p, ct.POINTER(ct.c_int64),
            ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.POINTER(ct.c_uint64)]
        lib.tessera_nvidia_matmul_prepare.restype = ct.c_int
        lib.tessera_nvidia_matmul_set_dynamic_axes.argtypes = [ct.c_uint64, ct.c_int]
        lib.tessera_nvidia_matmul_set_dynamic_axes.restype = ct.c_int
        lib.tessera_nvidia_matmul_attach_producer.argtypes = [
            ct.c_uint64, ct.c_void_p, ct.c_size_t, ct.c_char_p, ct.c_int]
        lib.tessera_nvidia_matmul_attach_producer.restype = ct.c_int
        lib.tessera_nvidia_matmul_invoke_resident.argtypes = [
            ct.c_uint64, ct.POINTER(HostView), ct.c_size_t, ct.c_void_p]
        lib.tessera_nvidia_matmul_invoke_resident.restype = ct.c_int
        lib.tessera_nvidia_matmul_close.argtypes = [ct.c_uint64]
        lib.tessera_nvidia_matmul_close.restype = ct.c_int
        lib.tessera_nvidia_matmul_last_error.restype = ct.c_char_p
        consumer = program.consumer
        provenance = consumer.descriptor.provenance
        epilogue = provenance["epilogue"]
        output = program._binding(consumer, program.output_name, "output")
        image = ct.create_string_buffer(consumer.image.payload)
        dims = (ct.c_int64 * 3)(program.m, program.n, program.k)
        handle = ct.c_uint64()
        self._check(lib.tessera_nvidia_matmul_prepare(
            image, len(consumer.image.payload), consumer.descriptor.entry_symbol.encode(),
            dims, 2 if program.dtype == "fp16" else 3,
            int(epilogue["bias"]), int(epilogue["residual"]),
            int(provenance["b_layout"] == "row_major"),
            int(output.dtype == "fp16"), ct.byref(handle)))
        self.handle = handle.value
        self._finalizer = weakref.finalize(self, lib.tessera_nvidia_matmul_close, self.handle)
        try:
            axes = int(program.dynamic_m) | (int(program.dynamic_n) << 1) | (int(program.dynamic_k) << 2)
            if axes:
                self._check(lib.tessera_nvidia_matmul_set_dynamic_axes(self.handle, axes))
            for index, producer in enumerate(stages):
                attach = (lib.tessera_nvidia_matmul_attach_producer if index == 0
                          else lib.tessera_nvidia_matmul_append_producer)
                attach.argtypes = [ct.c_uint64, ct.c_void_p, ct.c_size_t, ct.c_char_p, ct.c_int]
                attach.restype = ct.c_int
                image = ct.create_string_buffer(producer.image.payload)
                self._check(attach(
                    self.handle, image, len(producer.image.payload),
                    producer.descriptor.entry_symbol.encode(),
                    int(producer.descriptor.provenance["schedule"] == "cooperative_128")))
        except Exception:
            self.close()
            raise

        receipts = []
        for package in (*stages, self.program.consumer):
            artifact = rt.RuntimeArtifact(
                metadata={"target": "nvidia_sm120"}, native_image=package.image,
                launch_descriptor=package.descriptor, tile_ir=package.tile_ir,
                target_ir=package.target_ir)
            receipts.append(dict(
                ok=True, execution_kind="native_gpu", runtime_status="executed",
                compiler_path="canonical_scheduled_tile_consumer",
                native_call_binding="prepared_cpp_resident_tensor_matmul",
                image_digest=package.image.image_digest,
                launch_descriptor_digest=package.descriptor.descriptor_digest,
                artifact_hash=artifact.artifact_hash))
        self.component_receipts = tuple(receipts)
    def _check(self, status):
        if status:
            assert self.lib is not None
            reason = self.lib.tessera_nvidia_matmul_last_error()
            raise RuntimeError(reason.decode() if reason else "native resident tensor failure")

    def close(self):
        if self.pid != os.getpid():
            raise ValueError("native resident tensor cannot cross fork")
        self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def invoke(self, buffers, edge, *, stream):
        """Consumer-ordered device buffers, with producer source in LHS slot."""
        from tessera import runtime as rt
        if self.pid != os.getpid() or not self._finalizer.alive:
            raise ValueError("native resident tensor is closed or belongs to another process")
        if self.program != self.snapshot or self.producer_chain != self.chain_snapshot:
            raise ValueError("native resident tensor package changed")
        values = tuple(buffers) + (edge,)
        interfaces = [value.__cuda_array_interface__ for value in values]
        rt._validate_nvidia_cuda_buffer_streams(interfaces, stream)
        views = (HostView * len(values))()
        for ordinal, (view, value, interface) in enumerate(zip(views, values, interfaces, strict=True)):
            shape = tuple(interface["shape"])
            dtype = np.dtype(value.dtype)
            if len(shape) not in {1, 2}:
                raise ValueError("native resident tensor requires rank-one/two buffers")
            if interface["data"][1] and ordinal >= len(values) - 2:
                raise ValueError("native resident tensor requires writable output and edge buffers")
            strides = interface["strides"]
            if strides is None:
                strides = (dtype.itemsize,) if len(shape) == 1 else (shape[1] * dtype.itemsize, dtype.itemsize)
            view.data = int(interface["data"][0])
            view.bytes = int(np.prod(shape)) * dtype.itemsize
            view.dtype = {"float16": 2, "bfloat16": 3, "float32": 1}.get(dtype.name, 0)
            view.rank = len(shape)
            for axis in range(len(shape)):
                view.shape[axis], view.strides[axis] = shape[axis], strides[axis]
        assert self.lib is not None
        self._check(self.lib.tessera_nvidia_matmul_invoke_resident(
            self.handle, views, len(values), ct.c_void_p(stream)))
        return tuple(dict(receipt) for receipt in self.component_receipts)
