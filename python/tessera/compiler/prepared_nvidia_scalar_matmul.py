"""Checked metadata binding for native ordered-FP32 scalar contractions."""
from __future__ import annotations

from collections import OrderedDict
import ctypes as ct
import copy
import time
import hashlib
import os
import re
import threading
import weakref
from typing import TYPE_CHECKING, Any

from .prepared_nvidia_matmul import HostView

if TYPE_CHECKING:
    from tessera.runtime import RuntimeArtifact
    from .canonical_compile import CompileResult
    from .graph_ir import GraphIRModule
    from .native_artifact import LaunchDescriptor
    from .nvidia_native import NVIDIANativePackage

_TYPES = {"fp32": 1, "fp16": 2, "bf16": 3}
_MLIR = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}


def checked_plan(image, descriptor):
    """Validate the complete static ABI before loading a CUDA provider."""
    from .native_artifact import LaunchGeometry, OrderingSemantics, WorkspaceRequirement
    from .nvidia_native import SM120_SCALAR_FMA_ABI
    descriptor.validate_image(image)
    if (image.target != "nvidia_sm120" or image.architecture != "sm_120a"
            or descriptor.abi_id != SM120_SCALAR_FMA_ABI
            or descriptor.geometry != LaunchGeometry(policy="sm120_scheduled_typed_16x8_mn")
            or descriptor.ordering != OrderingSemantics(
                ordered_submission=True, residency="none", synchronization=("completion",))
            or descriptor.workspace != WorkspaceRequirement()
            or descriptor.dynamic_local_memory_bytes or descriptor.dynamic_local_memory_expression):
        raise ValueError("scalar matmul requires its static completed native ABI")
    info = descriptor.provenance
    dimensions = info.get("shape")
    types = info.get("scalar_types")
    orientations = tuple(info.get(name) for name in ("transposeA", "transposeB"))
    if (info.get("route") != "canonical_scheduled_scalar_fma"
            or info.get("physical_route") != "ordered_fp32_scalar_fma"
            or not isinstance(dimensions, (tuple, list)) or len(dimensions) != 3
            or any(type(x) is not int or not 0 < x <= 2**31 - 1 for x in dimensions)
            or dimensions[0] > 65535 * 16
            or not isinstance(types, (tuple, list)) or len(types) != 3
            or any(type(x) is not str or x not in _TYPES for x in types)
            or "fp32" not in types[:2]
            or any(type(x) is not bool for x in orientations)):
        raise ValueError("scalar matmul precision/orientation contract differs")
    m, n, k = dimensions
    ta, tb = orientations
    shapes = ((k, m) if ta else (m, k), (n, k) if tb else (k, n), (m, n))
    bindings = tuple(sorted(descriptor.buffers, key=lambda item: item.ordinal))
    if len(bindings) != 3 or len({item.name for item in bindings}) != 3:
        raise ValueError("scalar matmul requires distinct A/B/output bindings")
    for ordinal, (binding, dtype) in enumerate(zip(bindings, types, strict=True)):
        if (binding.ordinal != ordinal or binding.dtype != dtype or binding.rank != 2
                or binding.layout != "row_major"
                or binding.alignment != (4 if dtype == "fp32" else 2)
                or binding.direction != ("output" if ordinal == 2 else "input")):
            raise ValueError("scalar matmul per-role buffer ABI differs")
    guards = {(g.binding, g.dimension, g.predicate, g.value) for g in descriptor.shape_guards}
    expected = {(binding.name, axis, "eq", size)
                for binding, shape in zip(bindings, shapes, strict=True)
                for axis, size in enumerate(shape)}
    if len(descriptor.shape_guards) != 6 or guards != expected:
        raise ValueError("scalar matmul physical shape guards differ")
    if [(s.ordinal, s.name, s.dtype) for s in sorted(descriptor.scalars, key=lambda s: s.ordinal)] != [
            (3, "M", "int64"), (4, "N", "int64"), (5, "K", "int64")]:
        raise ValueError("scalar matmul requires exactly M/N/K scalar roles")
    if not re.fullmatch(r"nvidia_sm120_scheduled_matmul_\w+_scalar_fma_kernel", descriptor.entry_symbol):
        raise ValueError("scalar matmul entry does not match the native strategy")
    return tuple(dimensions), tuple(types), shapes, orientations, bindings


class PreparedScalarMatmulCall:
    def __init__(self, image, descriptor):
        from tessera import runtime as rt
        dimensions, types, self.shapes, orientations, self.bindings = checked_plan(image, descriptor)
        lib = rt._load_nvidia_ptx_launch()
        if lib is None or not hasattr(lib, "tessera_nvidia_scalar_matmul_prepare_v1"):
            raise RuntimeError("scalar matmul requires an updated native provider")
        self.lib = lib
        lib.tessera_nvidia_scalar_matmul_prepare_v1.argtypes = [
            ct.c_void_p, ct.c_size_t, ct.c_char_p, ct.POINTER(ct.c_int64),
            ct.POINTER(ct.c_int32), ct.c_int, ct.c_int, ct.POINTER(ct.c_uint64)]
        lib.tessera_nvidia_scalar_matmul_prepare_v1.restype = ct.c_int
        lib.tessera_nvidia_scalar_matmul_invoke_resident_v1.argtypes = [
            ct.c_uint64, ct.POINTER(HostView), ct.c_size_t,
            ct.POINTER(ct.c_uint64), ct.c_size_t]
        lib.tessera_nvidia_scalar_matmul_invoke_resident_v1.restype = ct.c_int
        lib.tessera_nvidia_matmul_invoke.argtypes = [ct.c_uint64, ct.POINTER(HostView), ct.c_size_t]
        lib.tessera_nvidia_matmul_invoke.restype = ct.c_int
        lib.tessera_nvidia_matmul_close.argtypes = [ct.c_uint64]
        lib.tessera_nvidia_matmul_close.restype = ct.c_int
        lib.tessera_nvidia_matmul_last_error.argtypes = []
        lib.tessera_nvidia_matmul_last_error.restype = ct.c_char_p
        encoded = ct.create_string_buffer(image.payload)
        dims = (ct.c_int64 * 3)(*dimensions)
        codes = (ct.c_int32 * 3)(*(_TYPES[x] for x in types))
        handle = ct.c_uint64()
        self._check(lib.tessera_nvidia_scalar_matmul_prepare_v1(
            encoded, len(image.payload), descriptor.entry_symbol.encode(), dims, codes,
            *map(int, orientations), ct.byref(handle)))
        self.handle, self.pid = handle.value, os.getpid()
        self.lock = threading.RLock()
        self.finalizer = weakref.finalize(self, lib.tessera_nvidia_matmul_close, self.handle)

    def _check(self, status):
        if status:
            message = self.lib.tessera_nvidia_matmul_last_error()
            raise RuntimeError(message.decode() if message else "native scalar contraction failed")

    def _guard(self):
        if os.getpid() != self.pid:
            raise RuntimeError("scalar matmul cannot cross fork")
        if not self.finalizer.alive:
            raise ValueError("scalar matmul owner is closed")

    def invoke(self, values, stream=None):
        from .prepared_nvidia_tensor_jvp import _view
        from .resident_nvidia_tensor import cuda_frontend_specs, ordered_resident_views
        self._guard()
        with self.lock:
            self._guard()
            if len(values) != 3:
                raise ValueError("scalar matmul requires A/B/output")
            resident = [hasattr(value, "__cuda_array_interface__") for value in values]
            if any(resident):
                if not all(resident):
                    raise ValueError("scalar matmul requires one placement for all buffers")
                cuda_frontend_specs(values, ranks=(2,))
                output_stream = values[2].__cuda_array_interface__["stream"]
                views, producers = ordered_resident_views(
                    values, output_stream if stream is None else stream, writable_from=2)
                streams = (ct.c_uint64 * 3)(*producers, output_stream)
                self._check(self.lib.tessera_nvidia_scalar_matmul_invoke_resident_v1(
                    self.handle, views, 3, streams, 3))
            else:
                import numpy as np
                if any(not isinstance(value, np.ndarray) or not value.dtype.isnative
                       or not value.flags.c_contiguous for value in values):
                    raise ValueError("scalar host buffers require native-endian compact ndarrays")
                if not values[2].flags.writeable:
                    raise ValueError("scalar host output requires writable storage")
                views = (HostView * 3)(*(_view(value) for value in values))
                self._check(self.lib.tessera_nvidia_matmul_invoke(self.handle, views, 3))
            return values[2]

    def profile_host(self, values, *, repeats=64, captured=False):
        """Refresh host roots, then measure native CUDA-event launch windows."""
        import numpy as np
        from .prepared_nvidia_tensor_jvp import _view
        self._guard()
        if (type(repeats) is not int or type(captured) is not bool or
                not 0 < repeats <= (4096 if captured else 1000000)):
            raise ValueError("scalar profiling repeat/capture contract differs")
        if any(hasattr(value, "__cuda_array_interface__") for value in values):
            raise ValueError("scalar host profiling requires explicit host buffers")
        with self.lock:
            self._guard()
            self.invoke(values)
            output = np.empty(self.shapes[2], dtype=values[2].dtype)
            view = _view(output)
            symbol = ("tessera_nvidia_matmul_profile_captured" if captured
                      else "tessera_nvidia_matmul_profile")
            function = getattr(self.lib, symbol)
            function.argtypes = [ct.c_uint64, ct.c_int, ct.POINTER(ct.c_float),
                                 ct.c_size_t, ct.POINTER(ct.c_float), ct.POINTER(HostView)]
            function.restype = ct.c_int
            stage, program = (ct.c_float * 1)(), ct.c_float()
            self._check(function(self.handle, repeats, stage, 1, ct.byref(program), ct.byref(view)))
            return output, {"kernel_stage_ms": float(stage[0]), "program_ms": program.value,
                            "captured": captured, "repeats": repeats}

    def close(self):
        if os.getpid() != self.pid:
            raise RuntimeError("scalar matmul cannot cross fork")
        with self.lock:
            if self.finalizer.alive:
                self._check(self.finalizer())


class PreparedScalarMatmulFrontendCall(PreparedScalarMatmulCall):
    """Canonical frontend identity attached to the native allocation owner."""
    input_positions: tuple[int, ...]
    artifact: RuntimeArtifact
    compiled: CompileResult
    frontend_snapshot: GraphIRModule
    descriptor_snapshot: LaunchDescriptor
    output_type: str
    receipt_fields: dict[str, Any]

    @classmethod
    def from_compiled(cls, compiled, artifact, module, frontend_graph):
        if (not compiled.executable or compiled.native_image != artifact.native_image
                or compiled.launch_descriptor != artifact.launch_descriptor):
            raise ValueError("scalar matmul differs from its canonical compile result")
        stages = [compiled.bundle.graph, compiled.bundle.schedule, compiled.bundle.tile,
                  compiled.bundle.target_ir, compiled.bundle.backend]
        if (compiled.bundle.schedule.producer != "tessera-opt.tessera-graph-to-schedule"
                or any(b.input_digest != a.output_digest
                       for a, b in zip(stages[:-1], stages[1:], strict=True))):
            raise ValueError("scalar matmul lacks adjacent native compiler ancestry")
        self = cls(artifact.native_image, artifact.launch_descriptor)
        try:
            names = [arg.name for arg in module.functions[0].args]
            if len(names) != 2 or set(names) != {b.name for b in self.bindings[:2]}:
                raise ValueError("scalar matmul frontend root bindings differ")
            self.input_positions = tuple(names.index(b.name) for b in self.bindings[:2])
            self.artifact, self.compiled = artifact, compiled
            self.frontend_snapshot = copy.deepcopy(frontend_graph)
            self.descriptor_snapshot = copy.deepcopy(artifact.launch_descriptor)
            self.output_type = self.bindings[-1].dtype
            self.receipt_fields = dict(
                ok=True, execution_kind="native_gpu", compiler_path="canonical_native_descriptor",
                native_call_binding="prepared_cpp_scalar_matmul", runtime_status="executed",
                image_digest=artifact.native_image.image_digest,
                launch_descriptor_digest=artifact.launch_descriptor.descriptor_digest,
                artifact_hash=artifact.artifact_hash)
            return self
        except BaseException:
            self.close()
            raise

    def matches(self, frontend_graph):
        return (self.finalizer.alive and self.pid == os.getpid()
                and frontend_graph == self.frontend_snapshot
                and self.artifact.launch_descriptor == self.descriptor_snapshot)

    def __call__(self, ordered):
        import numpy as np
        from tessera import runtime as rt
        self._guard()
        if self.artifact.launch_descriptor != self.descriptor_snapshot:
            raise ValueError("scalar matmul descriptor changed")
        if self.output_type == "bf16":
            import ml_dtypes
            dtype = np.dtype(ml_dtypes.bfloat16)
        else:
            dtype = np.dtype("float32" if self.output_type == "fp32" else "float16")
        start = time.perf_counter_ns()
        output = np.empty(self.shapes[2], dtype)
        self.invoke(tuple(ordered[i] for i in self.input_positions) + (output,))
        elapsed = (time.perf_counter_ns() - start) / 1e6
        rt._last_profile = rt.RuntimeProfile(launch_overhead_ms=elapsed, kernel_elapsed_ms=None)
        return output, dict(self.receipt_fields, output=output, elapsed_ms=elapsed)


_cache: OrderedDict[tuple[int, str, str], PreparedScalarMatmulCall] = OrderedDict()
_cache_pid = os.getpid()
_cache_lock = threading.RLock()


def submit(image, descriptor, buffers, scalars, stream):
    """Synchronous descriptor replay; Python performs metadata binding only."""
    dimensions, _, _, _, bindings = checked_plan(image, descriptor)
    if (any(type(scalars.get(key)) is not int for key in ("M", "N", "K")) or
            tuple(scalars.get(key) for key in ("M", "N", "K")) != dimensions):
        raise ValueError("scalar matmul runtime dimensions differ from the sealed contract")
    if os.getpid() != _cache_pid:
        raise RuntimeError("retained scalar matmul cannot cross fork")
    from tessera import runtime as rt
    lib = rt._load_nvidia_ptx_launch()
    if lib is None:
        raise RuntimeError("scalar matmul native provider unavailable")
    query = lib.tessera_nvidia_matmul_context_identity
    query.argtypes, query.restype = [ct.POINTER(ct.c_uint64)], ct.c_int
    identity = ct.c_uint64()
    if query(ct.byref(identity)):
        raise RuntimeError("scalar matmul requires the owning SM120 context")
    key = (identity.value, image.image_digest, descriptor.descriptor_digest)
    with _cache_lock:
        owner = _cache.get(key)
        if owner is None:
            owner = PreparedScalarMatmulCall(image, descriptor)
            _cache[key] = owner
            if len(_cache) > 8:
                _, retired = _cache.popitem(last=False)
                retired.close()
        _cache.move_to_end(key)
        return owner.invoke(tuple(buffers[item.name] for item in bindings), stream)


def package(artifact: Any, *, pipeline_name: str) -> NVIDIANativePackage:
    """Package the actual native Graph/Schedule/Tile product without rebuilding IR."""
    from .native_artifact import (
        BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint,
        NativeImageArtifact, OrderingSemantics, ResourceRecord, ScalarArgument, ShapeGuard)
    from .nvidia_native import NVIDIANativePackage, SM120_SCALAR_FMA_ABI, _compile_tile_ir
    from .scheduled_matmul import verify_matmul_projection
    artifact.validate()
    verify_matmul_projection(artifact)
    if (artifact.target != "nvidia_sm120" or artifact.architecture != "sm_120"
            or any((artifact.dynamic_m, artifact.dynamic_n, artifact.dynamic_k))
            or artifact.bias_name or artifact.residual_name or artifact.activation != "none"
            or artifact.accum != "f32" or artifact.b_layout != "row_major"):
        raise ValueError("scalar matmul packaging requires its plain static contract")
    types = (artifact.a_dtype, artifact.b_dtype, artifact.output_dtype)
    if any(x not in _TYPES for x in types) or "fp32" not in types[:2]:
        raise ValueError("scalar matmul input or output dtype differs")
    records = re.findall(r"(?m)^\s*tile\.scalar_matmul_kernel[^\n]+", artifact.tile_ir)
    if len(records) != 1:
        raise ValueError("scalar matmul requires one typed native launch")
    record = records[0]
    for key, dtype in zip(("a_element_type", "b_element_type", "output_element_type"), types, strict=True):
        if f"{key} = {_MLIR[dtype]}" not in record:
            raise ValueError("scalar matmul Tile storage differs from descriptor")
    def orientation(name):
        values = re.findall(name + r" = (true|false)", record)
        if len(values) != 1:
            raise ValueError("scalar matmul Tile orientation is missing")
        return values[0] == "true"
    ta, tb = orientation("transpose_a"), orientation("transpose_b")
    m, n, k = artifact.m, artifact.n, artifact.k
    shapes = ((k, m) if ta else (m, k), (n, k) if tb else (k, n), (m, n))
    names = (artifact.a_name, artifact.b_name, artifact.output_name)
    lowered, ptx, metrics, compiler, toolchain, libraries, state = _compile_tile_ir(
        artifact.tile_ir, artifact.function_name)
    image = NativeImageArtifact(
        target="nvidia_sm120", architecture="sm_120a", pipeline_name=pipeline_name,
        compiler_fingerprint=compiler, toolchain_fingerprint=toolchain,
        target_ir_digest=hashlib.sha256(lowered.encode()).hexdigest(),
        binary_format="ptx", payload=ptx.encode("ascii"),
        entry_points=(NativeEntryPoint(artifact.function_name, SM120_SCALAR_FMA_ABI),),
        compile_state=state, device_libraries=libraries,
        resource_record=ResourceRecord(provenance="native ordered scalar Schedule/Tile; ptxas sm_120a",
                                       metrics=metrics))
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=artifact.function_name,
        abi_id=SM120_SCALAR_FMA_ABI,
        buffers=tuple(BufferBinding(i, name, "output" if i == 2 else "input",
                                   dtype, 2, "row_major", 4 if dtype == "fp32" else 2)
                      for i, (name, dtype) in enumerate(zip(names, types, strict=True))),
        scalars=tuple(ScalarArgument(i + 3, name, "int64") for i, name in enumerate(("M", "N", "K"))),
        shape_guards=tuple(ShapeGuard(name, axis, "eq", size)
                           for name, shape in zip(names, shapes, strict=True)
                           for axis, size in enumerate(shape)),
        geometry=LaunchGeometry(policy="sm120_scheduled_typed_16x8_mn"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none",
                                   synchronization=("completion",)),
        provenance={"work_item": "W1.1", "sync_key": "W11-MIXED-ADJOINT-CONTRACTION-20261018",
                    "shape": [m, n, k], "scalar_types": list(types),
                    "transposeA": ta, "transposeB": tb,
                    "route": "canonical_scheduled_scalar_fma",
                    "physical_route": "ordered_fp32_scalar_fma",
                    "graph_ir_digest": artifact.graph_digest,
                    "schedule_digest": artifact.schedule_digest,
                    "schedule_ir_digest": artifact.schedule_ir_digest,
                    "tile_ir_digest": artifact.tile_digest})
    checked_plan(image, descriptor)
    return NVIDIANativePackage(artifact.tile_ir, lowered, ptx, image, descriptor)
