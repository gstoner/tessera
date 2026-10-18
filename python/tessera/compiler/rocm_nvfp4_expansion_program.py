"""Prepared bindings for native source-preserving NVFP4 Graph programs.

The native compiler outlines the Graph and owns numerical lowering. Python
marshals its SSA plan and existing member descriptors into the native owner.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence, cast

import ml_dtypes
import numpy as np

from .native_nvfp4_program import export_native_nvfp4_program
from .native_scaled_program import (
    PreparedScaledProgram, _BoundScaledABI, _Buffer, _StorageSpec,
)
from .nvfp4_expansion import checked_expansion_inputs, nvfp4_expansion_policy
from .rocm_fp8_blockscale import (
    BlockScaleProgram, BlockScaleShape, check_blockscale_target_ir, package_blockscale,
)
from .rocm_nvfp4_expand_native import NVFP4ExpansionPackage, package_nvfp4_expansion
from .rocm_native import ROCMNativePackage
from .scheduled_matmul import find_tessera_opt, run_tessera_opt


class _Rows(C.Structure):
    _fields_ = [("capacity_m", C.c_int64), ("slot_count", C.c_uint32),
                ("slots", C.c_uint32 * 7), ("row_bytes", C.c_uint64 * 7),
                ("consumer_step", C.c_uint32), ("m_scalar", C.c_uint32),
                ("grid_axis", C.c_uint32), ("tile_m", C.c_uint32)]


class _MultiOutputStep(C.Structure):
    _fields_ = [
        ("image", C.c_void_p), ("image_bytes", C.c_uint64), ("entry", C.c_char_p),
        ("input_count", C.c_uint32), ("scalar_count", C.c_uint32),
        ("inputs", C.c_uint32 * 6), ("output_count", C.c_uint32),
        ("outputs", C.c_uint32 * 7), ("geometry", C.c_uint32 * 6),
        ("scalars", C.c_int64 * 8),
    ]


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class NativeNVFP4ExpansionProgram:
    program_json: str
    graph_ir: str
    producer: NVFP4ExpansionPackage
    consumer: ROCMNativePackage
    consumer_graph_ir: str
    consumer_schedule_ir: str
    consumer_checked_target_ir: str
    binding_digest: str
    activation: ROCMNativePackage | None = None
    activation_graph_ir: str = ""
    activation_schedule_ir: str = ""

    def _identity(self):
        parts = [
            self.program_json, self.graph_ir, self.producer.graph_ir,
            self.producer.schedule_ir, self.producer.native.tile_ir,
            self.producer.native.target_ir, self.producer.native.image.image_digest,
            self.producer.native.descriptor.descriptor_digest,
            self.consumer_graph_ir, self.consumer_schedule_ir, self.consumer_checked_target_ir,
            self.consumer.tile_ir, self.consumer.target_ir,
            self.consumer.image.image_digest, self.consumer.descriptor.descriptor_digest,
        ]
        if self.activation is not None:
            parts.extend([self.activation_graph_ir, self.activation_schedule_ir,
                          self.activation.tile_ir, self.activation.target_ir,
                          self.activation.image.image_digest,
                          self.activation.descriptor.descriptor_digest])
        return _digest(json.dumps(parts, separators=(",", ":")))

    def validate(self):
        if self.binding_digest != self._identity():
            raise ValueError("native expansion program member binding differs")
        self.producer.validate()
        p = json.loads(self.program_json)
        roles = p["role_indices"]
        if (not isinstance(roles, list) or len(roles) != 5
                or any(type(i) is not int for i in roles) or sorted(roles) != list(range(5))):
            raise ValueError("native expansion original argument roles differ")
        has_activation = p.get("schema") == "tessera.native.nvfp4_program.v5"
        if has_activation != (self.activation is not None):
            raise ValueError("native expansion activation schema/member differs")
        if not has_activation and (self.activation_graph_ir or self.activation_schedule_ir):
            raise ValueError("two-member expansion cannot carry an unbound activation witness")
        if has_activation and any(key in p for key in ("active_m", "m_bound", "row_frame", "original_graph_ir")):
            raise ValueError("static activation program cannot reuse a bounded two-member frame")
        members = [self.producer.graph_ir, self.consumer_graph_ir]
        operations = ["tessera.nvfp4_expand", "tessera.scaled_matmul"]
        inputs = [roles[:3], [roles[3], 5, roles[4], 6]]
        outputs = [[5, 6], [7]]
        if has_activation:
            members.append(self.activation_graph_ir)
            operations.append("tessera.silu_mul")
            inputs.append([7]); outputs.append([8])
        if (p.get("schema") not in {"tessera.native.nvfp4_program.v3", "tessera.native.nvfp4_program.v4", "tessera.native.nvfp4_program.v5"}
                or p["source_graph_ir"] != self.graph_ir
                or p["member_graphs"] != members
                or len(roles) != 5 or any(type(i) is not int for i in roles)
                or sorted(roles) != list(range(5)) or len(p["buffers"]) != (9 if has_activation else 8)
                or p["output"] != (8 if has_activation else 7) or len(p["steps"]) != len(operations)
                or [s["operation"] for s in p["steps"]] != operations
                or [s["inputs"] for s in p["steps"]] != inputs
                or [s["outputs"] for s in p["steps"]] != outputs
                or [s["step"] for s in p["steps"]] != list(range(len(operations)))):
            raise ValueError("native expansion program SSA frame differs")
        provenance = self.producer.native.descriptor.provenance
        n, k = cast(Sequence[int], provenance["shape_nk"])
        activation = p["buffers"][roles[3]]
        m = activation["shape"][0]
        bounded = p["schema"] == "tessera.native.nvfp4_program.v4"
        if bounded:
            if (type(p.get("m_bound")) is not int or p["m_bound"] != m
                    or type(p.get("active_m")) is not int or not 0 < p["active_m"] <= m
                    or not isinstance(p.get("original_graph_ir"), str) or not p["original_graph_ir"]
                    or p.get("row_frame") != {
                        "input_slots": roles[3:], "output_slot": 7,
                        "row_strides_bytes": [k, 4*(k//16), n*(2 if p["buffers"][7]["storage"] == "bf16" else 4)],
                        "consumer_step": 1}):
                raise ValueError("native expansion bounded row frame differs")
            if (self.consumer.descriptor.provenance.get("arbitrary_m") is not True
                    or self.consumer.descriptor.provenance.get("lds_whole_m") is not False):
                raise ValueError("native expansion bounded image requires arbitrary M guards")
        out = p["buffers"][7]["storage"]
        shape = BlockScaleShape(m, n, k, 16, 1, "nk", out)
        code_storage = p["buffers"][roles[0]]["storage"]
        if code_storage not in {"i8", "ui8"}:
            raise ValueError("native expansion source codes require byte storage")
        specs = [
            ((n, k // 2), code_storage), ((n, k // 16), "f8E4M3FN"),
            ((len(cast(Sequence[int], provenance["row_offsets"])) - 1,), "f64"),
            ((m, k), "f8E4M3FN"), ((m, k // 16), "f32"),
            ((n, k), "f8E4M3FN"), ((k // 16, n), "f32"), ((m, n), out),
        ]
        if has_activation:
            if out != "bf16" or n % 2:
                raise ValueError("native expansion activation requires an even BF16 parent")
            specs.append(((m, n // 2), "bf16"))
        ordered = list(specs)
        for role, slot in enumerate(roles):
            ordered[slot] = specs[role]
        ordered[5:] = specs[5:]
        reads = [-1] * len(specs)
        writes = [-1] * len(specs)
        for i, step in enumerate(p["steps"]):
            for slot in step["inputs"] + step["outputs"]:
                reads[slot] = i
            for slot in step["outputs"]:
                writes[slot] = i
        reads[p["output"]] = len(operations)
        sizes = {"ui8": 1, "i8": 1, "f8E4M3FN": 1, "f64": 8, "f32": 4, "bf16": 2}
        for i, (row, spec) in enumerate(zip(p["buffers"], ordered, strict=True)):
            dimensions, storage = spec
            if (row["id"] != i or row["shape"] != list(dimensions) or row["storage"] != storage
                    or row["bytes"] != math.prod(dimensions) * sizes[storage]
                    or row["ownership"] != ("readonly_input" if i < 5 else "returned_output" if i == p["output"] else "private_scratch")
                    or row["first_write"] != writes[i]
                    or row["last_read"] != reads[i]):
                raise ValueError("native expansion program buffer lifetime/storage differs")
        native = self.consumer
        native.descriptor.validate_image(native.image)
        if (native.image.target != "rocm_gfx1201"
                or native.image.architecture != "gfx1201"
                or native.image.target_ir_digest != _digest(native.target_ir)
                or native.descriptor.provenance["shape"] != [m, n, k]
                or native.descriptor.provenance["scale_k"] != 16
                or native.descriptor.provenance["scale_n"] != 1):
            raise ValueError("native expansion product target differs")
        checked = check_blockscale_target_ir(shape, native.tile_ir, self.consumer_checked_target_ir)
        from .native_artifact import LaunchGeometry, ScalarArgument
        from .rocm_fp8_blockscale import PACKAGE_ABIS
        from .rocm_native import _directive_symbol
        descriptor = native.descriptor
        if (descriptor.abi_id != PACKAGE_ABIS[("nk", out)]
                or descriptor.entry_symbol != _directive_symbol(native.target_ir, "tessera_rocm.scaled_wmma_gemm")
                or descriptor.geometry != LaunchGeometry(policy="rocm_wmma_macro_tile_grid")
                or descriptor.scalars != tuple(ScalarArgument(i + 5, name, "int64") for i, name in enumerate(("M", "N", "K")))
                or descriptor.provenance["macro_tile"] != [checked["block_m"], checked["block_n"]]
                or descriptor.provenance["workgroup"] != [32 * checked["warps"], 1, 1]
                or descriptor.provenance["macro_k"] != checked["macro_k"]
                or descriptor.provenance["staging"] != checked["staging"]):
            raise ValueError("native expansion product launch geometry/ABI differs")

        if self.activation is not None:
            from .rocm_parent_view_native import (
                PARENT_VIEW_BF16_ABI, _frame, validate_parent_view_runtime_artifact,
            )
            from tessera.runtime import RuntimeArtifact
            activation = self.activation
            artifact = RuntimeArtifact(graph_ir=self.activation_graph_ir,
                schedule_ir=self.activation_schedule_ir, tile_ir=activation.tile_ir,
                target_ir=activation.target_ir, native_image=activation.image,
                launch_descriptor=activation.descriptor,
                metadata={"target": "rocm_gfx1201", "compiler_path": "rocm_parent_view_native_descriptor"})
            validate_parent_view_runtime_artifact(artifact)
            _, frame = _frame(activation.target_ir)
            if (activation.descriptor.abi_id != PARENT_VIEW_BF16_ABI or
                    frame["parent_shape"] != [m, n] or frame["output_shape"] != [m, n // 2]):
                raise ValueError("native expansion activation shape/storage edge differs")

    @property
    def native_stages(self) -> tuple[ROCMNativePackage, ...]:
        stages = (self.producer.native, self.consumer)
        return stages if self.activation is None else (*stages, self.activation)

    def prepare(self, inputs, *, runtime_library):
        return PreparedNVFP4ExpansionProgram(self, inputs, runtime_library=runtime_library)


def package_native_nvfp4_expansion_program(graph):
    if isinstance(graph, str):
        source = graph
    else:
        from copy import deepcopy
        graph = deepcopy(graph)
        for key, value in (("tessera.target", "rocm_gfx1201"), ("tessera.arch", "gfx1201")):
            expected = json.dumps(value)
            if graph.module_attrs.get(key, expected) != expected:
                raise ValueError("native expansion frontend target differs")
            graph.module_attrs[key] = expected
        source = graph.to_mlir(target="rocm_gfx1201", canonical=True)
    program = export_native_nvfp4_program(source)
    plan = program.manifest
    if plan["schema"] not in {"tessera.native.nvfp4_program.v3", "tessera.native.nvfp4_program.v4", "tessera.native.nvfp4_program.v5"}:
        raise ValueError("prepared expansion requires the native source-preserving program")
    bounded = plan["schema"] == "tessera.native.nvfp4_program.v4"
    producer = package_nvfp4_expansion(program.project_member(0))
    consumer_graph = program.project_member(1)
    roles = plan["role_indices"]
    m, k = plan["buffers"][roles[3]]["shape"]
    n = plan["buffers"][5]["shape"][0]
    output = plan["buffers"][7]["storage"]
    shape = BlockScaleShape(m, n, k, 16, 1, "nk", output)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("native expansion product requires the matching compiler")
    schedule = run_tessera_opt(tool, consumer_graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    consumer = package_blockscale(BlockScaleProgram(
        shape, "w8a8_blockscale", consumer_graph, schedule, tile), arbitrary_m=bounded)
    # Retain the static checked Target witness separately from the native
    # shape-independent image projection. Neither is reconstructed in Python.
    from .rocm_native import _run_opt, _shape_free_target_ir
    from .rocm_pipeline import ROCMExecutablePipeline, ROCMInputLevel, ROCMOutputLevel
    config = ROCMExecutablePipeline(family="matmul", arch="gfx1201", input_level=ROCMInputLevel.TILE)
    checked_target = _run_opt(tool, tile, config.pass_pipeline(output=ROCMOutputLevel.TARGET))
    check_blockscale_target_ir(shape, tile, checked_target)
    descriptor = consumer.descriptor
    family = "scaled_matmul" if descriptor.provenance["runtime_shape_image"] else "scaled_matmul_lds"
    runtime_k = descriptor.provenance["lds_runtime_k"]
    if not isinstance(runtime_k, bool):
        raise ValueError("native expansion image projection policy differs")
    projection = _shape_free_target_ir(
        checked_target, family=family, directive="tessera_rocm.scaled_wmma_gemm",
        runtime_k=runtime_k, arbitrary_m=bounded)
    if projection != consumer.target_ir:
        raise ValueError("native expansion consumer image projection differs")
    fields = (program.plan_json, plan["source_graph_ir"], producer, consumer, consumer_graph, schedule, checked_target)
    activation, activation_graph, activation_schedule = None, "", ""
    if plan["schema"] == "tessera.native.nvfp4_program.v5":
        from .rocm_parent_view_native import ParentViewRecipe, package_parent_view_recipe
        activation_graph = program.project_member(2)
        activation_schedule = run_tessera_opt(tool, activation_graph, "--tessera-graph-to-schedule")
        activation_tile = run_tessera_opt(tool, activation_schedule, "--tessera-schedule-to-tile")
        activation = package_parent_view_recipe(
            ParentViewRecipe(activation_graph, activation_schedule, activation_tile),
            pipeline_name="tessera-rocm-executable")
    tail = (activation, activation_graph, activation_schedule)
    unsealed = NativeNVFP4ExpansionProgram(*fields, "", *tail)
    package = NativeNVFP4ExpansionProgram(*fields, unsealed._identity(), *tail)
    package.validate()
    return package


class PreparedNVFP4ExpansionProgram(PreparedScaledProgram):
    _prepare_symbol = "tessera_rocm_program_prepare_multi_output"
    _step_type = _MultiOutputStep

    def _bind_abi(self, package):
        package.validate()
        plan = json.loads(package.program_json)
        rows = plan["buffers"]
        self._bounded = plan["schema"] == "tessera.native.nvfp4_program.v4"
        self._active_m = plan.get("active_m", rows[plan["role_indices"][3]]["shape"][0])
        self._capacity_m = rows[plan["role_indices"][3]]["shape"][0]
        if self._bounded:
            self._prepare_symbol = "tessera_rocm_program_prepare_multi_output_rows"
            frame = plan["row_frame"]
            bm = package.consumer.descriptor.provenance["macro_tile"][0]
            self._rows = _Rows(self._capacity_m, 3,
                (C.c_uint32 * 7)(*frame["input_slots"], frame["output_slot"]),
                (C.c_uint64 * 7)(*frame["row_strides_bytes"]),
                frame["consumer_step"], 0, 1, bm)
        self._roles = tuple(plan["role_indices"])
        self._offsets = tuple(package.producer.native.descriptor.provenance["row_offsets"])
        native = package.native_stages
        blobs = tuple(C.create_string_buffer(p.image.payload) for p in native)
        entries = tuple(p.descriptor.entry_symbol.encode() for p in native)
        steps = (_MultiOutputStep * len(native))()
        m, k = rows[self._roles[3]]["shape"]
        n = rows[5]["shape"][0]
        for i, p in enumerate(native):
            step, spec = steps[i], plan["steps"][i]
            step.image, step.image_bytes = C.cast(blobs[i], C.c_void_p), len(p.image.payload)
            step.entry = entries[i]
            step.input_count, step.output_count = len(spec["inputs"]), len(spec["outputs"])
            step.inputs[:step.input_count], step.outputs[:step.output_count] = spec["inputs"], spec["outputs"]
            if i == 0:
                grid, workgroup = p.descriptor.geometry.grid, p.descriptor.geometry.workgroup
                if grid is None or workgroup is None:
                    raise ValueError("native expansion producer requires explicit launch geometry")
                step.geometry[:] = (*grid, *workgroup)
            elif i == 1:
                bm, bn = cast(Sequence[int], p.descriptor.provenance["macro_tile"])
                step.geometry[:] = ((n + bn - 1) // bn, (m + bm - 1) // bm, 1,
                                    *cast(Sequence[int], p.descriptor.provenance["workgroup"]))
                step.scalar_count = 3
                step.scalars[:3] = (m, n, k)
            else:
                info = cast(dict, p.descriptor.provenance["native_parent_view"])
                frame = json.loads(info["member_json"])
                step.geometry[:] = frame["geometry"]
                step.scalar_count = len(frame["scalars"])
                step.scalars[:step.scalar_count] = frame["scalars"]
        dtypes = {"ui8": np.dtype("uint8"), "i8": np.dtype("uint8"), "f8E4M3FN": np.dtype("uint8"),
                  "f64": np.dtype("float64"), "f32": np.dtype("float32"),
                  "bf16": np.dtype(ml_dtypes.bfloat16)}
        contracts = (_Buffer * len(rows))(*[
            _Buffer(b["bytes"], math.prod(b["shape"]), b["first_write"], b["last_read"],
                    0 if b["ownership"] == "readonly_input" else 2 if b["ownership"] == "returned_output" else 1, 0)
            for b in rows])
        return _BoundScaledABI(contracts, steps, blobs, entries, 5,
            tuple(_StorageSpec(tuple(b["shape"]), b["storage"], b["bytes"], dtypes[b["storage"]]) for b in rows),
            (plan["output"],), ())

    def _prepare_tail(self):
        if not self._bounded:
            return (), ()
        return (C.POINTER(_Rows), C.c_int64), (C.byref(self._rows), self._active_m)

    def _storage(self, active_m):
        if not self._bounded:
            return self._binding.storage
        result = list(self._binding.storage)
        for slot in (*self._roles[3:], 7):
            spec = result[slot]
            result[slot] = _StorageSpec((active_m, *spec.shape[1:]), spec.storage,
                spec.bytes // self._capacity_m * active_m, spec.numpy_dtype)
        return tuple(result)

    def _input_m(self, inputs):
        if len(inputs) != 5:
            raise ValueError("native expansion program requires five roots")
        shape = np.shape(inputs[self._roles[3]])
        if len(shape) != 2 or not 0 < shape[0] <= self._capacity_m:
            raise ValueError("native expansion active M exceeds capacity")
        return shape[0]

    def _inputs_at(self, inputs, active_m):
        if len(inputs) != 5:
            raise ValueError("native expansion program requires five roots")
        codes, scales, globals_ = (inputs[i] for i in self._roles[:3])
        checked_expansion_inputs(codes, scales, globals_, row_offsets=self._offsets,
                                 numeric_policy=nvfp4_expansion_policy(), _scale_bounds=self._nvfp4_scale_bounds())
        return self._inputs_for_storage(inputs, self._storage(active_m))

    def _inputs(self, inputs):
        active_m = self._input_m(inputs) if self._bounded else self._active_m
        arrays = self._inputs_at(inputs, active_m)
        self._active_m = active_m
        return arrays

    @property
    def active_m(self):
        return self._active_m

    def _update_rows(self, active_m, slots, arrays):
        from .native_scaled_program import _status
        update = self.lib.tessera_rocm_program_update_rows
        update.argtypes = [C.c_uint64, C.c_int64, C.c_uint32, C.POINTER(C.c_uint32),
                           C.POINTER(C.c_void_p), C.POINTER(C.c_uint64)]
        update.restype = C.c_int
        count = len(slots)
        _status(update(self.handle, active_m, count, (C.c_uint32 * count)(*slots),
                       (C.c_void_p * count)(*[a.ctypes.data for a in arrays]),
                       (C.c_uint64 * count)(*[a.nbytes for a in arrays])))
        self._active_m = active_m

    def update(self, inputs):
        if not self._bounded:
            return super().update(inputs)
        active_m = self._input_m(inputs)
        arrays, _, _ = self._inputs_at(inputs, active_m)
        self._update_rows(active_m, tuple(range(5)), arrays)

    def rows_info(self):
        if not self._bounded:
            raise ValueError("static expansion has no bounded row frame")
        from .native_scaled_program import _status
        info = self.lib.tessera_rocm_program_rows_info
        info.argtypes = [C.c_uint64, C.POINTER(C.c_int64), C.POINTER(C.c_int64),
                         C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]
        info.restype = C.c_int
        capacity, active, size, count = C.c_int64(), C.c_int64(), C.c_uint64(), C.c_uint64()
        _status(info(self.handle, C.byref(capacity), C.byref(active), C.byref(size), C.byref(count)))
        return capacity.value, active.value, size.value, count.value

    def read(self, generation, *, batched=True):
        output_slot = self._binding.outputs[0]
        contract = self._storage(self._active_m)[output_slot]
        output = np.empty(contract.shape, dtype=contract.numpy_dtype)
        from .native_scaled_program import _status
        _status(self.lib.tessera_rocm_program_read(
            self.handle, output_slot, generation, C.c_void_p(output.ctypes.data), output.nbytes))
        return (output,)


    def _invoke_range(self, first, count, *, repeats=1, timed=False):
        from .native_scaled_program import _status
        if type(repeats) is not int or not 0 < repeats <= 1048576:
            raise ValueError("native repetition count is invalid")
        invoke = self.lib.tessera_rocm_program_invoke_range
        invoke.argtypes = [C.c_uint64, C.c_uint32, C.c_uint32, C.c_uint32,
                           C.POINTER(C.c_uint64), C.POINTER(C.c_float)]
        invoke.restype = C.c_int
        generation, elapsed = C.c_uint64(), C.c_float()
        _status(invoke(self.handle, first, count, repeats, C.byref(generation),
                       C.byref(elapsed) if timed else None))
        return generation.value, float(elapsed.value) if timed else None

    def import_weights(self, *, timed=False):
        """Expand the copied weights once; no returned product is published."""
        return self._invoke_range(0, 1, timed=timed)

    def invoke_consumer(self, *, repeats=1, timed=False):
        """Use resident expanded weights; native readiness checks precede launch."""
        return self._invoke_range(1, len(self._binding.steps)-1, repeats=repeats, timed=timed)

    def update_activations(self, activation, scales):
        from .native_scaled_program import _status, pack_host_view
        slots = self._roles[3:]
        shape = np.shape(activation)
        active_m = shape[0] if len(shape) == 2 else 0
        if self._bounded and not 0 < active_m <= self._capacity_m:
            raise ValueError("native expansion active M exceeds capacity")
        specs = self._storage(active_m) if self._bounded else self._binding.storage
        arrays = []
        for slot, value in zip(slots, (activation, scales), strict=True):
            contract = specs[slot]
            array = np.asarray(value)
            if contract.storage == "f8E4M3FN" and array.dtype == np.dtype(ml_dtypes.float8_e4m3fn):
                array = array.view(np.uint8)
            if (array.dtype != contract.numpy_dtype or array.shape != contract.shape
                    or array.nbytes != contract.bytes):
                raise ValueError("native expansion activation storage/shape differs")
            arrays.append(pack_host_view(self.lib, array))
        if self._bounded:
            return self._update_rows(active_m, slots, arrays)
        update = self.lib.tessera_rocm_program_update_slots
        update.argtypes = [C.c_uint64, C.c_uint32, C.POINTER(C.c_uint32),
                           C.POINTER(C.c_void_p), C.POINTER(C.c_uint64)]
        update.restype = C.c_int
        _status(update(self.handle, 2, (C.c_uint32 * 2)(*slots),
                       (C.c_void_p * 2)(*[a.ctypes.data for a in arrays]),
                       (C.c_uint64 * 2)(*[a.nbytes for a in arrays])))
