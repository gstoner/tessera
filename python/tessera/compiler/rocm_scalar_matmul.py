"""Checked metadata and native ownership for ordered mixed FP32 contractions.

Graph/Schedule/Tile/Target/LLVM arithmetic is emitted by native compiler passes.
This module binds that product to the existing C++ program owner ABI.
"""
from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
import hashlib
import json
import math
import re

from .native_artifact import (
    BufferBinding, LaunchDescriptor, LaunchGeometry, NativeEntryPoint,
    NativeImageArtifact, OrderingSemantics, ScalarArgument, ShapeGuard,
    WorkspaceRequirement,
)

SCALAR_MATMUL_ABI = "tessera.rocm.scalar_matmul.a_b_o_m_n_k.mixed_fp32_fma.v1"
_STORAGE = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}


def checked_plan(image, descriptor):
    """Seal precision, static physical shapes, launch and ownership together."""
    descriptor.validate_image(image)
    info = descriptor.provenance
    dims, types = info.get("shape"), info.get("scalar_types")
    orientation = tuple(info.get(name) for name in ("transposeA", "transposeB"))
    if (image.architecture not in {"gfx1151", "gfx1201"} or
            image.target != "rocm_" + image.architecture or
            image.pipeline_name != "tessera-rocm-executable" or
            image.binary_format != "hsaco" or
            descriptor.abi_id != SCALAR_MATMUL_ABI or
            info.get("route") != "canonical_scheduled_scalar_fma" or
            info.get("physical_route") != "ordered_fp32_scalar_fma" or
            not isinstance(dims, (tuple, list)) or len(dims) != 3 or
            any(type(v) is not int or not 0 < v <= 2**31 - 1 for v in dims) or
            dims[0] > 65535 * 16 or
            not isinstance(types, (tuple, list)) or len(types) != 3 or
            any(type(v) is not str or v not in _STORAGE for v in types) or
            types[:2].count("fp32") != 1 or
            any(type(v) is not bool for v in orientation)):
        raise ValueError("ROCm scalar matmul precision/architecture contract differs")
    m, n, k = dims
    ta, tb = orientation
    shapes = ((k, m) if ta else (m, k), (n, k) if tb else (k, n), (m, n))
    if (descriptor.geometry != LaunchGeometry(grid=((n + 7)//8, (m + 15)//16, 1),
                                             workgroup=(32, 1, 1)) or
            descriptor.ordering != OrderingSemantics(
                ordered_submission=True, residency="none", synchronization=("completion",)) or
            descriptor.workspace != WorkspaceRequirement() or
            descriptor.dynamic_local_memory_bytes or descriptor.dynamic_local_memory_expression):
        raise ValueError("ROCm scalar matmul launch/ownership contract differs")
    bindings = tuple(sorted(descriptor.buffers, key=lambda b: b.ordinal))
    if len(bindings) != 3 or len({b.name for b in bindings}) != 3:
        raise ValueError("ROCm scalar matmul requires distinct A/B/output bindings")
    for ordinal, (binding, dtype) in enumerate(zip(bindings, types, strict=True)):
        if (binding.ordinal != ordinal or binding.dtype != dtype or binding.rank != 2 or
                binding.layout != "row_major" or
                binding.alignment != (4 if dtype == "fp32" else 2) or
                binding.direction != ("output" if ordinal == 2 else "input")):
            raise ValueError("ROCm scalar matmul per-role storage ABI differs")
    guards = {(g.binding, g.dimension, g.predicate, g.value) for g in descriptor.shape_guards}
    expected = {(binding.name, axis, "eq", size)
                for binding, shape in zip(bindings, shapes, strict=True)
                for axis, size in enumerate(shape)}
    if len(descriptor.shape_guards) != 6 or guards != expected:
        raise ValueError("ROCm scalar matmul shape guards differ")
    if [(s.ordinal, s.name, s.dtype) for s in sorted(descriptor.scalars, key=lambda s: s.ordinal)] != [
            (3, "M", "int64"), (4, "N", "int64"), (5, "K", "int64")]:
        raise ValueError("ROCm scalar matmul requires exactly M/N/K scalar roles")
    if not re.fullmatch(r"tessera_rocm_scalar_matmul_[0-9a-f]{16}", descriptor.entry_symbol):
        raise ValueError("ROCm scalar matmul entry differs from the native strategy")
    target = info.get("native_target_ir")
    schedule = info.get("native_schedule_ir")
    for key in ("graph_digest", "schedule_digest", "schedule_ir_digest", "tile_ir_digest"):
        value = info.get(key)
        if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("ROCm scalar ancestry digest differs")
    if (type(target) is not str or not target or
            hashlib.sha256(target.encode()).hexdigest() != image.target_ir_digest or
            type(schedule) is not str or not schedule or
            hashlib.sha256(schedule.encode()).hexdigest() != info["schedule_ir_digest"]):
        raise ValueError("ROCm scalar retained native recipe differs")
    records = re.findall(r'\btessera_rocm\.scalar_matmul\s*\{([^{}]*)\}', target)
    keys = re.findall(r'shape_key = "M=(\d+);N=(\d+);K=(\d+);dtype=(\w+)"', schedule)
    if (len(records) != 1 or len(keys) != 1 or tuple(map(int, keys[0][:3])) != tuple(dims) or
            keys[0][3] != _STORAGE[types[0]] or schedule.count(info["schedule_digest"]) != 3):
        raise ValueError("ROCm scalar dimensions disagree with the native Schedule")
    def field(name, pattern):
        values = re.findall(r'(?:^|,\s*)' + re.escape(name) + r' = (' + pattern + r')(?=,|\s*$)',
                            records[0].strip())
        if len(values) != 1:
            raise ValueError("ROCm scalar native Target field differs: " + name)
        return values[0]
    if (field("arch", r'"[^"]+"') != '"' + image.architecture + '"' or
            field("name", r'"[^"]+"') != '"' + descriptor.entry_symbol + '"' or
            tuple(field(name, r"f32|f16|bf16") for name in
                  ("a_element_type", "b_element_type", "output_element_type")) !=
                  tuple(_STORAGE[dtype] for dtype in types) or
            tuple(field(name, r"true|false") == "true" for name in
                  ("transpose_a", "transpose_b")) != orientation):
        raise ValueError("ROCm scalar descriptor disagrees with the native Target recipe")
    return tuple(dims), tuple(types), shapes, orientation, bindings


def package_scheduled_scalar(artifact, *, pipeline_name):
    from .rocm_native import ROCMNativePackage, _compile_shape_free_tile_ir, _directive_symbol
    from .scheduled_matmul import verify_matmul_projection

    artifact.validate()
    verify_matmul_projection(artifact)
    if (pipeline_name != "tessera-rocm-executable" or artifact.target != "rocm" or
            artifact.architecture not in {"gfx1151", "gfx1201"} or
            f'physical_contract = "rocm_{artifact.architecture}_scalar_fma_v1"' not in artifact.schedule_ir or
            artifact.bias_name is not None or artifact.residual_name is not None or
            artifact.activation != "none" or
            artifact.dynamic_m or artifact.dynamic_n or artifact.dynamic_k or artifact.split_k != 1):
        raise ValueError("ROCm scalar packaging requires its static native Schedule contract")
    target, backend, payload, compiler, toolchain, libraries, state = _compile_shape_free_tile_ir(
        artifact.tile_ir, family="scalar_matmul", architecture=artifact.architecture)
    entry = _directive_symbol(target, "tessera_rocm.scalar_matmul")
    image = NativeImageArtifact(
        target="rocm_" + artifact.architecture, architecture=artifact.architecture,
        pipeline_name=pipeline_name, compiler_fingerprint=compiler, toolchain_fingerprint=toolchain,
        target_ir_digest=hashlib.sha256(target.encode()).hexdigest(),
        binary_format="hsaco", payload=payload,
        entry_points=(NativeEntryPoint(entry, SCALAR_MATMUL_ABI),),
        compile_state=state, device_libraries=libraries)
    types = (artifact.a_dtype, artifact.b_dtype, artifact.output_dtype)
    # The native Schedule states orientation; no Python physical decision.
    records = re.findall(r' = schedule.matmul %\w+ \{([^{}]*)\}', artifact.schedule_ir)
    if len(records) != 1:
        raise ValueError("ROCm scalar Schedule orientation requires one native decision")
    flags = []
    for name in ("transposeA", "transposeB"):
        values = re.findall(r'(?:^|, )' + name + r' = (true|false)(?:,|$)', records[0])
        if len(values) > 1:
            raise ValueError("ROCm scalar Schedule orientation is ambiguous")
        flags.append(values == ["true"])
    m, n, k = artifact.m, artifact.n, artifact.k
    shapes = ((k, m) if flags[0] else (m, k), (n, k) if flags[1] else (k, n), (m, n))
    names = (artifact.a_name, artifact.b_name, artifact.output_name)
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest, entry_symbol=entry, abi_id=SCALAR_MATMUL_ABI,
        buffers=tuple(BufferBinding(i, name, "input" if i < 2 else "output", dtype, 2,
                                    "row_major", 4 if dtype == "fp32" else 2)
                      for i, (name, dtype) in enumerate(zip(names, types, strict=True))),
        scalars=tuple(ScalarArgument(3 + i, name, "int64") for i, name in enumerate(("M", "N", "K"))),
        shape_guards=tuple(ShapeGuard(name, axis, "eq", size)
                          for name, shape in zip(names, shapes, strict=True)
                          for axis, size in enumerate(shape)),
        geometry=LaunchGeometry(grid=((n + 7)//8, (m + 15)//16, 1), workgroup=(32, 1, 1)),
        ordering=OrderingSemantics(ordered_submission=True, residency="none", synchronization=("completion",)),
        provenance={"route": "canonical_scheduled_scalar_fma", "physical_route": "ordered_fp32_scalar_fma",
                    "shape": [m, n, k], "scalar_types": list(types),
                    "transposeA": flags[0], "transposeB": flags[1],
                    "native_target_ir": target, "native_schedule_ir": artifact.schedule_ir,
                    "graph_digest": artifact.graph_digest, "schedule_digest": artifact.schedule_digest,
                    "schedule_ir_digest": artifact.schedule_ir_digest, "tile_ir_digest": artifact.tile_digest})
    checked_plan(image, descriptor)
    return ROCMNativePackage(artifact.tile_ir, target, backend, image, descriptor)


def validate_runtime_artifact(artifact):
    image, descriptor = artifact.native_image, artifact.launch_descriptor
    checked_plan(image, descriptor)
    for key, text in (("graph_digest", artifact.graph_ir),
                      ("schedule_ir_digest", artifact.schedule_ir),
                      ("tile_ir_digest", artifact.tile_ir)):
        if not text or hashlib.sha256(text.encode()).hexdigest() != descriptor.provenance.get(key):
            raise ValueError("ROCm scalar artifact stage lineage differs: " + key)
    if (not artifact.target_ir or
            hashlib.sha256(artifact.target_ir.encode()).hexdigest() != image.target_ir_digest):
        raise ValueError("ROCm scalar artifact Target lineage differs")


@dataclass(frozen=True)
class ScalarOwnerContract:
    image: NativeImageArtifact
    descriptor: LaunchDescriptor

    def validate(self):
        return checked_plan(self.image, self.descriptor)

    @property
    def program_json(self):
        dims, types, shapes, _, _ = self.validate()
        return json.dumps({"kind": "scalar_matmul", "architecture": self.image.architecture,
                           "shape": dims, "types": types, "physical_shapes": shapes})


def prepare(image, descriptor, inputs):
    """Bind host roots to an existing native C++ image/storage owner."""
    from tessera import runtime as rt
    from .native_scaled_program import (
        PreparedScaledProgram, _BoundScaledABI, _Buffer, _Step, _StorageSpec, _storage_numpy_dtype,
    )
    from copy import deepcopy
    descriptor = deepcopy(descriptor)
    contract = ScalarOwnerContract(image, descriptor)
    contract.validate()
    if rt._rocm_live_arch() != image.architecture:
        raise ValueError("ROCm scalar package requires its exact owning architecture")
    library = rt._load_rocm_native_movement_runtime()
    if library is None:
        raise RuntimeError("ROCm scalar package requires the native C++ program owner")

    class Owner(PreparedScaledProgram):
        _architecture = image.architecture.encode()

        def _bind_abi(self, package):
            dims, types, shapes, _, _ = package.validate()
            storage = tuple(_StorageSpec(shape, _STORAGE[dtype],
                            math.prod(shape) * (4 if dtype == "fp32" else 2),
                            _storage_numpy_dtype(_STORAGE[dtype]))
                            for shape, dtype in zip(shapes, types, strict=True))
            buffers = (_Buffer * 3)(*(_Buffer(row.bytes, math.prod(row.shape),
                            -1 if i < 2 else 0, 0 if i < 2 else 1, 0 if i < 2 else 2, 0)
                            for i, row in enumerate(storage)))
            blob = ct.create_string_buffer(image.payload)
            entry = descriptor.entry_symbol.encode()
            step = _Step(ct.cast(blob, ct.c_void_p), len(image.payload), entry, 2, 3,
                         (ct.c_uint32 * 6)(0, 1), 2,
                         (ct.c_uint32 * 6)(*descriptor.geometry.grid, *descriptor.geometry.workgroup),
                         (ct.c_int64 * 8)(*dims))
            return _BoundScaledABI(buffers, (_Step * 1)(step), (blob,), (entry,),
                                   2, storage, (2,), ())

    return Owner(contract, inputs, runtime_library=library._name)


def submit(image, descriptor, buffers, scalars, stream):
    """Completed host-buffer replay; no tracing, compilation or tensor arithmetic."""
    import numpy as np
    from copy import deepcopy
    descriptor = deepcopy(descriptor)
    dims, _, shapes, _, bindings = checked_plan(image, descriptor)
    if stream not in (None, 0):
        raise ValueError("ROCm scalar package uses its native completion stream")
    if set(scalars) != {"M", "N", "K"} or any(type(scalars.get(name)) is not int for name in ("M", "N", "K")) or tuple(
            scalars[name] for name in ("M", "N", "K")) != dims:
        raise ValueError("ROCm scalar runtime dimensions differ from its sealed contract")
    if set(buffers) != {b.name for b in bindings}:
        raise ValueError("ROCm scalar runtime buffer roles differ")
    values = tuple(buffers[b.name] for b in bindings)
    for value, binding, shape in zip(values, bindings, shapes, strict=True):
        if not isinstance(value, np.ndarray) or not value.dtype.isnative or not value.flags.c_contiguous:
            raise ValueError("ROCm scalar host buffers require native compact ndarray storage")
        expected_dtype = "bfloat16" if binding.dtype == "bf16" else (
            "float16" if binding.dtype == "fp16" else "float32")
        if value.shape != shape or str(value.dtype) != expected_dtype:
            raise ValueError("ROCm scalar host buffer type/shape differs")
    if not values[2].flags.writeable or any(np.shares_memory(v, values[2]) for v in values[:2]):
        raise ValueError("ROCm scalar output must be writable and independent")
    with prepare(image, descriptor, values[:2]) as owner:
        generation, _ = owner.invoke()
        np.copyto(values[2], owner.read(generation)[0], casting="no")
    return values[2]
