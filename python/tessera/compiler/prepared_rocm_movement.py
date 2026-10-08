"""Prepared native host binding for already compiled static ROCm movement.

Graph/Schedule/Tile/native image compilation precedes preparation. This module
marshals host storage metadata; it generates no GPU body or physical schedule.
"""
from __future__ import annotations

import copy
import ctypes as ct
import os
import time
import weakref

import numpy as np


class HostView(ct.Structure):
    _fields_ = [("data", ct.c_void_p), ("bytes", ct.c_size_t),
                ("dtype", ct.c_int32), ("rank", ct.c_int32),
                ("shape", ct.c_int64 * 4), ("strides", ct.c_int64 * 4)]


class PreparedMovementCall:
    def __init__(self, compiled, module, artifact, contract, *, ordered=None):
        from tessera import runtime as rt
        from .native_artifact import BufferBinding, ShapeGuard, ScalarArgument, LaunchGeometry, OrderingSemantics, WorkspaceRequirement
        from .rocm_native import GFX_PAGED_KV_F32_ABI, GFX_PAGED_KV_STRIDED_F32_ABI, GFX_MOE_DISPATCH_F32_ABI
        image, descriptor = artifact.native_image, artifact.launch_descriptor
        if not compiled.executable or image is None or descriptor is None:
            raise ValueError("prepared movement requires a complete executable compiler package")
        descriptor.validate_image(image)
        arch = image.target.removeprefix("rocm_")
        if arch not in {"gfx1151", "gfx1201"} or image.architecture != arch:
            raise ValueError("prepared movement needs its exact ROCm image target")
        strided = descriptor.abi_id == GFX_PAGED_KV_STRIDED_F32_ABI
        family = 2 if strided else 0 if descriptor.abi_id == GFX_PAGED_KV_F32_ABI else 1
        if family == 1 and (descriptor.abi_id != GFX_MOE_DISPATCH_F32_ABI or arch != "gfx1151"):
            raise ValueError("prepared movement has an unsupported ABI")
        stages = [compiled.bundle.graph, compiled.bundle.schedule, compiled.bundle.tile,
                  compiled.bundle.target_ir, compiled.bundle.backend]
        if (compiled.bundle.schedule.producer != "tessera-opt.tessera-graph-to-schedule"
                or any(b.input_digest != a.output_digest for a, b in zip(stages[:-1], stages[1:], strict=True))):
            raise ValueError("prepared movement lacks the adjacent native compiler chain")
        names, dimensions = contract[:3], contract[3]
        shapes: tuple[tuple[int, ...], ...]
        scalar_names: tuple[str, ...]
        if family in {0, 2}:
            p, lp, page, h, d, start, tokens = dimensions
            shapes = ((p, page, h, d), (lp,), (tokens, h, d))
            scalar_names = ("P", "LP", "PageSize", "H", "D", "Start", "Tokens")
            policy = arch + ("_paged_kv_strided_256" if strided else "_paged_kv_direct_256")
            if strided:
                scalar_names += ("StrideP", "StridePage", "StrideH", "StrideD")
        else:
            t, slots, h = dimensions
            shapes = ((t, h), (slots,), (slots, h))
            scalar_names = ("T", "S", "H")
            policy = "gfx1151_moe_dispatch_direct_256"
        expected_buffers = tuple(BufferBinding(i, name, "output" if i == 2 else "input",
            "int32" if i == 1 else "fp32", len(shape), "strided" if strided and i == 0 else "row_major", 4)
            for i, (name, shape) in enumerate(zip(names, shapes, strict=True)))
        expected_guards = tuple(ShapeGuard(name, axis, "eq", extent)
            for name, shape in zip(names, shapes, strict=True) for axis, extent in enumerate(shape))
        expected_scalars = tuple(ScalarArgument(3+i, name, "int64") for i, name in enumerate(scalar_names))
        if (descriptor.buffers != expected_buffers or descriptor.shape_guards != expected_guards
                or descriptor.scalars != expected_scalars
                or descriptor.geometry != LaunchGeometry(policy=policy)
                or descriptor.ordering != OrderingSemantics(ordered_submission=True, residency="none",
                                                            synchronization=("completion",))
                or descriptor.workspace != WorkspaceRequirement()
                or descriptor.dynamic_local_memory_bytes or descriptor.dynamic_local_memory_expression
                or tuple(descriptor.provenance.get("shape", ())) != dimensions):
            raise ValueError("prepared movement descriptor differs from its static native tensor ABI")
        self._page_strides = None
        native_dimensions = dimensions
        if strided:
            if ordered is None:
                raise ValueError("strided preparation requires the actual host storage view")
            from .paged_host_span import checked_page_span
            position = next(i for i, arg in enumerate(module.functions[0].args) if arg.name == names[0])
            _, self._page_strides = checked_page_span(ordered[position])
            native_dimensions = (*dimensions, *self._page_strides)
        lib = rt._load_rocm_native_movement_runtime()
        if lib is None or not hasattr(lib, "tessera_rocm_movement_prepare"):
            raise ValueError("prepared movement requires the matching native runtime")
        lib.tessera_rocm_movement_prepare.argtypes = [
            ct.c_void_p, ct.c_size_t, ct.c_char_p, ct.c_char_p, ct.c_int,
            ct.POINTER(ct.c_int64), ct.c_size_t, ct.POINTER(ct.c_uint64)]
        lib.tessera_rocm_movement_prepare.restype = ct.c_int
        lib.tessera_rocm_movement_invoke.argtypes = [
            ct.c_uint64, ct.POINTER(HostView), ct.c_size_t, ct.c_int]
        lib.tessera_rocm_movement_invoke.restype = ct.c_int
        lib.tessera_rocm_movement_close.argtypes = [ct.c_uint64]
        lib.tessera_rocm_movement_close.restype = ct.c_int
        handle = ct.c_uint64()
        payload = ct.create_string_buffer(image.payload)
        dims = (ct.c_int64 * len(native_dimensions))(*native_dimensions)
        rc = lib.tessera_rocm_movement_prepare(payload, len(image.payload),
            descriptor.entry_symbol.encode(), arch.encode(), family, dims, len(native_dimensions), ct.byref(handle))
        if rc:
            raise RuntimeError(f"native movement preparation failed rc={rc}")
        self._lib, self._handle = lib, handle.value
        self._finalizer = weakref.finalize(self, lib.tessera_rocm_movement_close, handle.value)
        self._sealed_artifact_hash = artifact.artifact_hash
        self.graph_snapshot = copy.deepcopy(module)
        self.compiled, self.artifact = compiled, artifact
        argument_names = [arg.name for arg in module.functions[0].args]
        self.input_positions = tuple(argument_names.index(name) for name in names[:2])
        self.output_shape = shapes[2]
        self.receipt_fields = dict(ok=True, execution_kind="native_gpu",
            compiler_path="canonical_native_descriptor", native_call_binding="prepared_cpp_movement",
            image_digest=image.image_digest, launch_descriptor_digest=descriptor.descriptor_digest,
            artifact_hash=artifact.artifact_hash, runtime_status="executed")

    def resident(self):
        from .resident_rocm_movement import ResidentMovementCall
        return ResidentMovementCall(self)

    def matches(self, module, ordered=None):
        if not self._finalizer.alive or module != self.graph_snapshot:
            return False
        if self._page_strides is not None and ordered is not None:
            from .paged_host_span import checked_page_span
            _, strides = checked_page_span(ordered[self.input_positions[0]])
            return strides == self._page_strides
        return True

    def close(self):
        self._finalizer()

    def __call__(self, ordered):
        start = time.perf_counter_ns()
        if not self._finalizer.alive:
            raise ValueError("prepared native movement call is closed")
        inputs = tuple(ordered[i] for i in self.input_positions)
        output = np.empty(self.output_shape, dtype=np.float32)
        arrays = (*inputs, output)
        views = (HostView * 3)()
        for role, (view, array) in enumerate(zip(views, arrays, strict=True)):
            if not isinstance(array, np.ndarray) or array.ndim > 4:
                raise TypeError("prepared movement requires host tensor arrays of rank at most four")
            physical_bytes = array.nbytes
            if role == 0 and self._page_strides is not None:
                from .paged_host_span import checked_page_span
                physical_bytes, strides = checked_page_span(array)
                if strides != self._page_strides:
                    raise ValueError("page strides differ from the prepared native owner")
            view.data, view.bytes, view.rank = array.ctypes.data, physical_bytes, array.ndim
            view.dtype = 1 if array.dtype == np.dtype("float32") else 2 if array.dtype == np.dtype("int32") else 0
            for axis, (shape, stride) in enumerate(zip(array.shape, array.strides, strict=True)):
                view.shape[axis], view.strides[axis] = shape, stride
        reuse = os.environ.get("TESSERA_ROCM_MOVEMENT_STAGING_REUSE", "1").lower() not in {"0", "off", "false"}
        rc = self._lib.tessera_rocm_movement_invoke(self._handle, views, 3, int(reuse))
        if rc:
            raise RuntimeError(f"native prepared movement invocation failed rc={rc}")
        from tessera import runtime as rt
        elapsed = (time.perf_counter_ns() - start) / 1e6
        rt._last_profile = rt.RuntimeProfile(launch_overhead_ms=elapsed, kernel_elapsed_ms=None)
        return output, dict(self.receipt_fields, output=output, elapsed_ms=elapsed)
