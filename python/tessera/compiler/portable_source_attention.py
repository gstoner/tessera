"""Portable compiled source-attention programs; no compiler required at replay.

Content digests detect corruption, not authenticity of arbitrary supplied code.
"""
from __future__ import annotations

import copy
import hashlib
import json
import keyword
import math
import re

SCHEMA = "tessera.nvidia.source_attention_program.v1"
COMPILER_PATH = "nvidia_sm120_source_attention_compiled"
_FIELDS = ("graph_ir", "schedule_ir", "tile_ir", "target_ir", "backend_ir")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def validate_plan(plan):
    keys = {"schema", "kind", "execution_state", "target", "architecture", "root",
            "source_graph_ir", "source_hash", "outlined_graph_ir", "role_indices",
            "buffers", "steps", "outputs", "member_graphs"}
    if (not isinstance(plan, dict) or set(plan) != keys
            or plan["schema"] != "tessera.native.source_program.v1"
            or plan["kind"] != "norm_attention_saved" or plan["execution_state"] != "artifact_only"
            or plan["target"] != "nvidia_sm120" or plan["architecture"] != "sm_120"):
        raise ValueError("source attention native plan schema or target differs")
    roles = plan["role_indices"]
    if (not isinstance(roles, list) or len(roles) != 3
            or any(type(i) is not int for i in roles) or sorted(roles) != [0, 1, 2]
            or not isinstance(plan["root"], str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.$]*", plan["root"])
            or any(not isinstance(plan[k], str) or not plan[k]
                   for k in ("source_graph_ir", "outlined_graph_ir"))
            or plan["source_hash"] != sha(plan["source_graph_ir"])
            or not isinstance(plan["member_graphs"], list) or len(plan["member_graphs"]) != 2
            or any(not isinstance(g, str) or not g for g in plan["member_graphs"])):
        raise ValueError("source attention native Graph or root roles differ")
    if (not isinstance(plan["outputs"], list) or len(plan["outputs"]) != 2
            or any(type(i) is not int for i in plan["outputs"])
            or sorted(plan["outputs"]) != [4, 5]):
        raise ValueError("source attention result order differs")
    expected_steps = [
        {"step": 0, "member": plan["root"] + "__member_0", "inputs": [roles[0]], "outputs": [3]},
        {"step": 1, "member": plan["root"] + "__member_1", "inputs": [3, roles[1], roles[2]],
         "outputs": [4, 5]}]
    if (plan["steps"] != expected_steps or any(
            type(step["step"]) is not int or any(type(i) is not int for i in
                                                (*step["inputs"], *step["outputs"]))
            for step in plan["steps"])):
        raise ValueError("source attention native member edges differ")
    buffers = plan["buffers"]
    if not isinstance(buffers, list) or len(buffers) != 6:
        raise ValueError("source attention buffer count differs")
    for i, buffer in enumerate(buffers):
        if (not isinstance(buffer, dict) or set(buffer) != {
                "id", "shape", "storage", "bytes", "ownership", "first_write", "last_read"}
                or type(buffer["id"]) is not int or buffer["id"] != i
                or not isinstance(buffer["shape"], list) or len(buffer["shape"]) != (3 if i == 5 else 4)
                or any(type(d) is not int or not 0 < d <= 65536 for d in buffer["shape"])
                or buffer["storage"] not in ({"f32"} if i == 5 else {"f16", "bf16"})):
            raise ValueError("source attention physical buffer schema differs")
        size = math.prod(buffer["shape"]) * (4 if i == 5 else 2)
        write = -1 if i < 3 else 0 if i == 3 else 1
        read = (0 if i == roles[0] else 1) if i < 3 else 1 if i == 3 else 2
        ownership = "readonly_input" if i < 3 else "private_scratch" if i == 3 else "returned_output"
        if (type(buffer["bytes"]) is not int or buffer["bytes"] != size or size >= 2**63
                or type(buffer["first_write"]) is not int or buffer["first_write"] != write
                or type(buffer["last_read"]) is not int or buffer["last_read"] != read
                or buffer["ownership"] != ownership):
            raise ValueError("source attention allocation lifetime differs")
    q, k, v = (buffers[i] for i in roles)
    b, hq, sq, d = q["shape"]
    bk, hkv, sk, dk = k["shape"]
    bv, hv, sv, dv = v["shape"]
    if (b != bk or b != bv or hq % hkv or hkv != hv or sk != sv or d != dk
            or any(buffer["storage"] != q["storage"] for buffer in buffers[:5])
            or buffers[3]["shape"] != q["shape"]
            or buffers[4]["shape"] != [b, hq, sq, dv] or buffers[5]["shape"] != [b, hq, sq]):
        raise ValueError("source attention composed storage or dimensions differ")
    return (b, hq, hkv, sq, sk, d, dv)


def seal(program, artifacts, packages, argument_names):
    plan = program.manifest
    records = []
    for artifact, package in zip(artifacts, packages, strict=True):
        records.append(dict(
            graph_ir=artifact.graph_ir, schedule_ir=artifact.schedule_ir,
            tile_ir=artifact.tile_ir, target_ir=package.target_ir,
            backend_ir=package.backend_ir, image=package.image.to_dict(),
            descriptor=package.descriptor.to_dict()))
    data = dict(schema=SCHEMA, plan=plan, members=records, argument_names=list(argument_names))
    data["contract_digest"] = digest(data)
    restore(data)
    return data



def _bindings(descriptor, shapes, dtypes, names, inputs, scalar_names, geometry, alignment):
    from .native_artifact import LaunchGeometry, OrderingSemantics, WorkspaceRequirement
    bindings = sorted(descriptor.buffers, key=lambda value: value.ordinal)
    if len(bindings) != len(shapes):
        raise ValueError("source attention member buffer arity differs")
    for i, (binding, shape, dtype, name) in enumerate(zip(bindings, shapes, dtypes, names, strict=True)):
        if (binding.ordinal != i or binding.name != name or binding.dtype != dtype
                or binding.rank != len(shape) or binding.layout != "row_major"
                or binding.direction != ("input" if i < inputs else "output")
                or binding.alignment != (4 if dtype == "fp32" else 2)):
            raise ValueError("source attention member physical binding differs")
    expected = {(name, axis): (extent, "eq") for name, shape in zip(names, shapes, strict=True)
                for axis, extent in enumerate(shape)}
    guards = {(guard.binding, guard.dimension): (guard.value, guard.predicate)
              for guard in descriptor.shape_guards}
    if guards != expected or len(guards) != len(descriptor.shape_guards):
        raise ValueError("source attention member static guards differ")
    scalars = sorted(descriptor.scalars, key=lambda value: value.ordinal)
    if (len(scalars) != len(scalar_names) or any(
            value.ordinal != len(bindings) + i or value.name != name or value.dtype != "int64"
            for i, (value, name) in enumerate(zip(scalars, scalar_names, strict=True)))
            or descriptor.geometry != LaunchGeometry(policy=geometry)
            or descriptor.ordering != OrderingSemantics(
                ordered_submission=True, residency="none", synchronization=("completion",))
            or descriptor.workspace != WorkspaceRequirement(alignment=alignment)
            or descriptor.dynamic_local_memory_bytes or descriptor.dynamic_local_memory_expression):
        raise ValueError("source attention member launch contract differs")

def restore(data):
    from .native_artifact import NativeImageArtifact, LaunchDescriptor
    from .nvidia_native import NVIDIANativePackage, SM120_NORM_F16_ABI, SM120_NORM_BF16_ABI
    from .typed_attention_checkpoint import validate_typed_saved_lse_descriptor

    if (not isinstance(data, dict) or set(data) != {
            "schema", "plan", "members", "argument_names", "contract_digest"}
            or data["schema"] != SCHEMA
            or data["contract_digest"] != digest({k: v for k, v in data.items() if k != "contract_digest"})):
        raise ValueError("source attention portable integrity differs")
    data = copy.deepcopy(data)
    plan = data["plan"]
    dims = validate_plan(plan)
    names = data["argument_names"]
    if (not isinstance(names, list) or len(names) != 3
            or any(not isinstance(n, str) or not n.isidentifier() or keyword.iskeyword(n) for n in names)
            or len(set(names)) != 3):
        raise ValueError("source attention original argument names differ")
    records = data["members"]
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("source attention executable member count differs")
    packages = []
    for i, record in enumerate(records):
        if (not isinstance(record, dict) or set(record) != {*_FIELDS, "image", "descriptor"}
                or any(not isinstance(record[key], str) or not record[key] for key in _FIELDS)
                or record["graph_ir"] != plan["member_graphs"][i]):
            raise ValueError("source attention retained member stages differ")
        image = NativeImageArtifact.from_dict(record["image"])
        descriptor = LaunchDescriptor.from_dict(record["descriptor"])
        descriptor.validate_image(image)
        if (image.target != "nvidia_sm120" or image.architecture != "sm_120a"
                or image.binary_format != "ptx" or image.payload.decode("ascii") != record["backend_ir"]
                or image.target_ir_digest != sha(record["target_ir"])
                or re.findall(r"\.visible\s+\.entry\s+([A-Za-z0-9_]+)", record["backend_ir"])
                    != [descriptor.entry_symbol]
                or not re.search(r"\.target\s+sm_120a\b", record["backend_ir"])):
            raise ValueError("source attention native image/entry differs")
        for field in ("graph_ir", "schedule_ir", "tile_ir"):
            if descriptor.provenance.get(field + "_digest") != sha(record[field]):
                raise ValueError("source attention member provenance differs")
        schedule_hashes = re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', record["tile_ir"])
        if schedule_hashes != [descriptor.provenance.get("schedule_digest")]:
            raise ValueError("source attention native Schedule seal differs")
        storage = plan["buffers"][3]["storage"]
        if i == 0:
            from .nvidia_native import _mlir_f32_bits
            norm_shape = plan["buffers"][3]["shape"]
            dtype = {"f16": "fp16", "bf16": "bf16"}[storage]
            _bindings(descriptor, (norm_shape, norm_shape), (dtype, dtype),
                      ("input_0", "output_0"), 1, ("Rows", "Columns"),
                      "sm120_norm_" + str(descriptor.provenance.get("schedule")) + "_rows", 1)
            epsilon = re.findall(r'tessera.norm_epsilon = ([^ ]+) : f32', record["tile_ir"])
            if (len(epsilon) != 1 or
                    _mlir_f32_bits(epsilon[0]) != _mlir_f32_bits(str(descriptor.provenance.get("epsilon")))):
                raise ValueError("source attention normalization epsilon differs")
            policy = descriptor.provenance
            mode = policy.get("schedule")
            expected_entry = "tessera_tile_norm_rmsnorm_" + storage + (
                "_" if mode == "serial" else "_cooperative_128_") + schedule_hashes[0][:10]
            if (descriptor.abi_id != {"f16": SM120_NORM_F16_ABI, "bf16": SM120_NORM_BF16_ABI}[storage]
                    or mode not in {"serial", "cooperative_128"} or descriptor.entry_symbol != expected_entry
                    or policy.get("shape") != plan["buffers"][3]["shape"]
                    or policy.get("storage") != storage or policy.get("accum") != "f32"
                    or policy.get("kind") != "rmsnorm" or policy.get("axis") != -1
                    or policy.get("keepdims") is not False
                    or record["tile_ir"].count("tile.norm_kernel") != 1
                    or "tessera.rmsnorm" not in record["graph_ir"]):
                raise ValueError("source attention normalization contract differs")
        else:
            dtype = {"f16": "fp16", "bf16": "bf16"}[storage]
            buffers = plan["buffers"]
            shapes = [buffers[index]["shape"] for index in (*plan["role_indices"], 4, 5)]
            _bindings(descriptor, shapes, (dtype, dtype, dtype, dtype, "fp32"),
                      ("input_0", "input_1", "input_2", "output_0", "output_1"), 3,
                      ("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv"),
                      "sm120_attention_lse_thread_per_output_128", 4)
            validate_typed_saved_lse_descriptor(descriptor)
            if (descriptor.provenance["shape"] != list(dims)
                    or descriptor.provenance["bias"] or descriptor.provenance.get("shape_bounds")
                    or descriptor.provenance["output"] != storage
                    or record["tile_ir"].count("tile.attention_kernel") != 1
                    or 'lse_checkpoint = "saved"' not in record["tile_ir"]
                    or "tessera.flash_attn" not in record["graph_ir"]):
                raise ValueError("source attention checkpoint contract differs")
        packages.append(NVIDIANativePackage(record["tile_ir"], record["target_ir"],
                                            record["backend_ir"], image, descriptor))
    return data, plan, tuple(packages), dims
