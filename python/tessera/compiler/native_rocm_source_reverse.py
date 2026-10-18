"""Native ROCm source pullback packages and checked C++ frame ownership.

The compiler derives the AD graph, outlines SSA, and emits both member ABIs.
Python validates/marshals these records; it builds no numerical kernel.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import ctypes as ct
from dataclasses import dataclass
import hashlib
import json
import math
import re

from .native_artifact import NativeImageArtifact, LaunchDescriptor
from .native_sm120_source_reverse import export_native_source_reverse, validate_plan
from .native_scaled_program import _manifest
from .rocm_scalar_matmul import checked_plan, package_native_scalar_ir

SCHEMA = "tessera.rocm.source_reverse_program.v1"


def _sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


@dataclass(frozen=True)
class RocmSourceReverseProgram:
    program_json: str
    scalar_image: NativeImageArtifact
    scalar_descriptor: LaunchDescriptor
    row_image: bytes
    row_frame_json: str
    stages: tuple[tuple[str, ...], ...]
    stage_digests: tuple[tuple[str, ...], ...]
    row_image_digest: str
    compiler_identity: str
    row_compiler: str
    row_toolchain: str
    program_digest: str

    @property
    def manifest(self):
        return validate_plan(self.program_json, target_family="rocm")

    @property
    def images(self):
        return self.scalar_image.payload, self.row_image

    @property
    def frames(self):
        return (_manifest(self.stages[0][2], "tessera.native.scalar_matmul_frame_json",
                          expected_schema="tessera.native.rocm_scalar_matmul_frame.v1"),
                json.loads(self.row_frame_json))

    def validate(self) -> None:
        from .rocm_native import _extract_hsaco
        if _sha(self.program_json) != self.program_digest:
            raise ValueError("ROCm source reverse outline identity differs")
        plan = self.manifest
        if (len(self.stages) != 2 or any(len(row) != 5 for row in self.stages) or
                self.stage_digests != tuple(tuple(_sha(text) for text in row) for row in self.stages) or
                any(self.stages[i][0] != plan["member_graphs"][i] for i in range(2)) or
                not all(type(v) is str and v for v in
                        (self.compiler_identity, self.row_compiler, self.row_toolchain))):
            raise ValueError("ROCm source reverse member stage lineage differs")
        if (not self.row_image.startswith(b"\x7fELF") or _sha(self.row_image) != self.row_image_digest or
                tuple(_extract_hsaco(row[4]) for row in self.stages) != self.images):
            raise ValueError("ROCm source reverse native images differ from retained backend IR")
        _, types, shapes, _, _ = checked_plan(self.scalar_image, self.scalar_descriptor)
        if (self.scalar_image.architecture != plan["architecture"] or
                self.scalar_descriptor.provenance["graph_digest"] != _sha(self.stages[0][0]) or
                self.scalar_descriptor.provenance["tile_ir_digest"] != _sha(self.stages[0][2]) or
                self.scalar_descriptor.provenance["native_schedule_ir"] != self.stages[0][1] or
                self.scalar_descriptor.provenance["native_target_ir"] != self.stages[0][3]):
            raise ValueError("ROCm source reverse contraction provenance differs")
        slots = plan["steps"][0]["inputs"] + plan["steps"][0]["outputs"]
        storage = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}
        for dtype, shape, slot in zip(types, shapes, slots, strict=True):
            expected = plan["buffers"][slot]
            if storage[dtype] != expected["storage"] or shape != tuple(expected["shape"]):
                raise ValueError("ROCm source reverse contraction SSA binding differs")
        scalar, row = self.frames
        grid, workgroup = self.scalar_descriptor.geometry.grid, self.scalar_descriptor.geometry.workgroup
        if grid is None or workgroup is None:
            raise ValueError("ROCm source reverse requires fixed native member geometry")
        if (scalar["geometry"] != [*grid, *workgroup] or
                scalar["scalars"] != self.scalar_descriptor.provenance["shape"]):
            raise ValueError("ROCm source reverse contraction launch frame differs")
        fields = {"schema", "row_schema", "architecture", "entry", "abi", "buffers",
                  "inputs", "output", "scalars", "geometry", "source_graph_ir",
                  "source_graph_sha256", "source_hash", "artifact_hash", "tile_body_hash",
                  "target_body_hash"}
        source = plan["buffers"][plan["role_indices"][0]]
        rows, columns = source["shape"]
        if (not isinstance(row, dict) or set(row) != fields or
                row["schema"] != "tessera.rocm.native_source_row.v1" or
                type(row["row_schema"]) is not int or row["row_schema"] != 7 or
                row["architecture"] != plan["architecture"] or
                row["abi"] != "flattened_rank1_memrefs_then_i64" or
                row["inputs"] != [0, 1] or any(type(v) is not int for v in row["inputs"]) or
                type(row["output"]) is not int or row["output"] != 2 or
                not isinstance(row["scalars"], list) or any(type(v) is not int for v in row["scalars"]) or
                not isinstance(row["geometry"], list) or any(type(v) is not int for v in row["geometry"]) or
                row["scalars"] != [rows, columns] or
                row["geometry"] != [(rows + 127)//128, 1, 1, 128, 1, 1] or
                not isinstance(row["buffers"], list) or len(row["buffers"]) != 3 or
                not isinstance(row["entry"], str) or
                not re.fullmatch(r"tessera_native_row_[0-9a-f]{16}", row["entry"]) or
                row["source_graph_ir"].strip() != self.stages[1][0].strip() or
                _sha(row["source_graph_ir"]) != row["source_graph_sha256"] or
                _manifest(self.stages[1][3], "tessera.rocm.source_row_member_json",
                          expected_schema="tessera.rocm.native_source_row.v1") != row or
                _manifest(self.stages[1][4], "tessera.rocm.source_row_member_json",
                          expected_schema="tessera.rocm.native_source_row.v1") != row):
            raise ValueError("ROCm source reverse native row frame differs")
        for key in ("source_hash", "artifact_hash", "tile_body_hash", "target_body_hash"):
            if type(row[key]) is not str or not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise ValueError("ROCm source reverse row seal differs")
        for ordinal, slot in enumerate(plan["steps"][1]["inputs"] + plan["steps"][1]["outputs"]):
            expected = plan["buffers"][slot]
            record = row["buffers"][ordinal]
            if (not isinstance(record, dict) or
                    any(type(record.get(k)) is not int for k in ("ordinal", "elements", "bytes")) or
                    not isinstance(record.get("shape"), list) or
                    any(type(v) is not int for v in record["shape"]) or record != {
                    "ordinal": ordinal, "shape": expected["shape"], "storage": expected["storage"],
                    "elements": math.prod(expected["shape"]), "bytes": expected["bytes"],
                    "direction": "input" if ordinal < 2 else "output"}):
                raise ValueError("ROCm source reverse row SSA binding differs")

    def to_json(self):
        self.validate()
        value = {
            "schema": SCHEMA, "program_json": self.program_json, "program_digest": self.program_digest,
            "scalar_image": self.scalar_image.to_dict(), "scalar_descriptor": self.scalar_descriptor.to_dict(),
            "row_image": base64.b64encode(self.row_image).decode(), "row_frame_json": self.row_frame_json,
            "stages": self.stages, "stage_digests": self.stage_digests,
            "row_image_digest": self.row_image_digest, "compiler_identity": self.compiler_identity,
            "row_compiler": self.row_compiler, "row_toolchain": self.row_toolchain}
        value["record_digest"] = _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return json.dumps(value, sort_keys=True)

    @classmethod
    def from_json(cls, text):
        value = json.loads(text)
        fields = {"schema", "program_json", "program_digest", "record_digest", "scalar_image", "scalar_descriptor", "row_image", "row_frame_json",
                  "stages", "stage_digests", "row_image_digest", "compiler_identity", "row_compiler", "row_toolchain"}
        if not isinstance(value, dict) or set(value) != fields or value["schema"] != SCHEMA:
            raise ValueError("ROCm source reverse portable schema differs")
        record_digest = value.pop("record_digest")
        if record_digest != _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)):
            raise ValueError("ROCm source reverse portable record integrity differs")
        result = cls(value["program_json"], NativeImageArtifact.from_dict(value["scalar_image"]),
                     LaunchDescriptor.from_dict(value["scalar_descriptor"]),
                     base64.b64decode(value["row_image"], validate=True), value["row_frame_json"],
                     tuple(tuple(row) for row in value["stages"]), tuple(tuple(row) for row in value["stage_digests"]),
                     value["row_image_digest"], value["compiler_identity"], value["row_compiler"], value["row_toolchain"], value["program_digest"])
        result.validate()
        return result

    def validate_inputs(self, inputs) -> None:
        import numpy as np
        from .native_scaled_program import _storage_numpy_dtype
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 3:
            raise ValueError("ROCm source reverse requires three typed original roots")
        for value, row in zip(inputs, self.manifest["buffers"][:3], strict=True):
            array = np.asarray(value)
            if (array.dtype != _storage_numpy_dtype(row["storage"]) or not array.dtype.isnative or
                    array.shape != tuple(row["shape"]) or array.nbytes != row["bytes"]):
                raise ValueError("ROCm source reverse original root type/shape differs")

    def prepare(self, inputs, *, runtime_library=None):
        from tessera import runtime as rt
        from .native_scaled_program import (
            PreparedScaledProgram, _BoundScaledABI, _Buffer, _Step, _StorageSpec, _storage_numpy_dtype,
        )
        package = deepcopy(self)
        package.validate()
        package.validate_inputs(inputs)
        plan = package.manifest
        if rt._rocm_live_arch() != plan["architecture"]:
            raise ValueError("ROCm source reverse requires its exact owning architecture")
        if runtime_library is None:
            library = rt._load_rocm_native_movement_runtime()
            if library is None:
                raise RuntimeError("ROCm source reverse requires the native C++ program owner")
            runtime_library = library._name

        class Owner(PreparedScaledProgram):
            _architecture = plan["architecture"].encode()

            def _bind_abi(self, program):
                program.validate()
                metadata = program.manifest
                storage = tuple(_StorageSpec(tuple(row["shape"]), row["storage"], row["bytes"],
                                              _storage_numpy_dtype(row["storage"]))
                                for row in metadata["buffers"])
                buffers = (_Buffer * len(storage))(*(
                    _Buffer(row["bytes"], math.prod(row["shape"]), row["first_write"],
                            row["last_read"], {"readonly_input": 0, "private_scratch": 1, "returned_output": 2}[row["ownership"]], 0)
                    for row in metadata["buffers"]))
                blobs = tuple(ct.create_string_buffer(image) for image in program.images)
                entries = (program.scalar_descriptor.entry_symbol.encode(), program.frames[1]["entry"].encode())
                steps = []
                for index, (step, frame) in enumerate(zip(metadata["steps"], program.frames, strict=True)):
                    steps.append(_Step(ct.cast(blobs[index], ct.c_void_p), len(program.images[index]), entries[index],
                                       len(step["inputs"]), len(frame["scalars"]), (ct.c_uint32 * 6)(*step["inputs"]),
                                       step["outputs"][0], (ct.c_uint32 * 6)(*frame["geometry"]),
                                       (ct.c_int64 * 8)(*frame["scalars"])))
                return _BoundScaledABI(buffers, (_Step * 2)(*steps), blobs, entries, 3, storage, (4,), ())

        return Owner(package, inputs, runtime_library=runtime_library)


def package_source_reverse(source, *, tool=None):
    """Compile the original frontend Graph and consume native member/frame exports."""
    from .rocm_native import _compile_native_tile_ir
    native = export_native_source_reverse(source, tool=tool, target_family="rocm")
    members = native.lower_members()
    architecture = native.manifest["architecture"]
    scalar = package_native_scalar_ir(*members[0][:3], architecture=architecture)
    target, backend, payload, compiler, toolchain, _, _ = _compile_native_tile_ir(
        members[1][2], directive="tessera_rocm.row_region_kernel",
        family="source_reverse_row", architecture=architecture)
    frame = _manifest(target, "tessera.rocm.source_row_member_json",
                      expected_schema="tessera.rocm.native_source_row.v1")
    stages = (tuple(members[0][:3]) + (scalar.target_ir, scalar.backend_ir),
              tuple(members[1][:3]) + (target, backend))
    result = RocmSourceReverseProgram(native.plan_json, scalar.image, scalar.descriptor,
        payload, json.dumps(frame, sort_keys=True), stages, tuple(tuple(_sha(text) for text in row) for row in stages),
        _sha(payload), native.compiler_identity, compiler, toolchain, _sha(native.plan_json))
    result.validate()
    return result
