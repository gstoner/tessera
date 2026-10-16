"""Portable paired RHS JVP images, distinct from primal tensor schemas.

These are trusted executable artifacts. Digests detect accidental corruption;
they do not authenticate arbitrary native code from another party.
"""
import copy
import hashlib
import json
from pathlib import Path

from .native_sm120_tensor_program import NativeSM120TensorGraph
from .scheduled_kernel import ScheduledKernelArtifact
from .scheduled_matmul import ScheduledMatmulArtifact


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_plan(text):
    plan = json.loads(text)
    keys = {"schema", "kind", "source_graph_ir", "paired_graph_ir", "root", "role_indices",
            "buffers", "steps", "output", "outputs", "member_graphs"}
    if (not isinstance(plan, dict) or set(plan) != keys
            or plan["schema"] != "tessera.native.sm120_tensor_program.v7"
            or plan["kind"] != "paired_jvp"):
        raise ValueError("paired tensor native schema differs")
    roles, steps, buffers = plan["role_indices"], plan["steps"], plan["buffers"]
    if (not isinstance(roles, list) or len(roles) != 3
            or any(type(v) is not int for v in roles)
            or sorted(roles[:2]) != [0, 1] or roles[2] != 2
            or not isinstance(steps, list) or not 3 <= len(steps) <= 64
            or not isinstance(buffers, list) or len(buffers) != 3 + len(steps)):
        raise ValueError("paired tensor roots/step count differs")
    count = len(steps)
    expected_outputs = [count + 1, count + 2]
    if (plan["outputs"] != expected_outputs or type(plan["output"]) is not int
            or plan["output"] != expected_outputs[0]
            or any(type(v) is not int for v in plan["outputs"])
            or not isinstance(plan["root"], str) or not plan["root"]
            or any(not isinstance(plan[k], str) or not plan[k]
                   for k in ("source_graph_ir", "paired_graph_ir"))
            or not isinstance(plan["member_graphs"], list)
            or len(plan["member_graphs"]) != count
            or any(not isinstance(g, str) or not g for g in plan["member_graphs"])):
        raise ValueError("paired tensor output/Graph witnesses differ")
    for i, step in enumerate(steps):
        inputs = ([roles[0]] if i == 0 else [i + 2]) if i < count - 2 else [count, roles[1] if i == count - 2 else 2]
        operations = {"tessera.rmsnorm", "tessera.layer_norm", "tessera.softmax", "tessera.relu"} if i < count - 2 else {"tessera.matmul"}
        if (not isinstance(step, dict)
                or set(step) != {"step", "operation", "member", "inputs", "outputs"}
                or type(step["step"]) is not int or step["step"] != i
                or step["operation"] not in operations
                or step["member"] != plan["root"] + "__tensor_member_" + str(i)
                or step["inputs"] != inputs or step["outputs"] != [i + 3]
                or any(type(v) is not int for v in [*step["inputs"], *step["outputs"]])):
            raise ValueError("paired tensor member SSA differs")
    for i, b in enumerate(buffers):
        if (not isinstance(b, dict)
                or set(b) != {"id", "bytes", "shape", "storage", "ownership", "first_write", "last_read"}
                or type(b["id"]) is not int or b["id"] != i
                or not isinstance(b["shape"], list) or len(b["shape"]) != 2
                or any(type(v) is not int or not 0 < v < 2**31 for v in b["shape"])
                or b["storage"] not in {"f16", "bf16", "f32"}):
            raise ValueError("paired tensor buffer fields differ")
        size = (4 if b["storage"] == "f32" else 2) * b["shape"][0] * b["shape"][1]
        write = -1 if i < 3 else i - 3
        read = count if i in expected_outputs else max([s["step"] for s in steps if i in s["inputs"]] + [write])
        ownership = "readonly_input" if i < 3 else "returned_output" if i in expected_outputs else "private_scratch"
        if (type(b["bytes"]) is not int or b["bytes"] != size or size > 2**63 - 1
                or type(b["first_write"]) is not int or b["first_write"] != write
                or type(b["last_read"]) is not int or b["last_read"] != read
                or b["ownership"] != ownership):
            raise ValueError("paired tensor buffer lifetime/bytes differ")
    a, b = buffers[roles[0]], buffers[roles[1]]
    if (a["storage"] not in {"f16", "bf16"} or b["storage"] != a["storage"]
            or b["shape"][0] != a["shape"][1]
            or buffers[2]["shape"] != b["shape"] or buffers[2]["storage"] != b["storage"]
            or any(row["shape"] != a["shape"] or row["storage"] != a["storage"] for row in buffers[3:count + 1])
            or any(buffers[i]["shape"] != [a["shape"][0], b["shape"][1]] or buffers[i]["storage"] != "f32" for i in expected_outputs)):
        raise ValueError("paired tensor storage/shape differs")
    return plan


def validate_members(text, artifacts, members):
    from .nvidia_native import NVIDIANativeTensorProgram
    plan = validate_plan(text)
    if len(artifacts) != len(plan["steps"]) or len(members) != len(artifacts):
        raise ValueError("paired tensor member count differs")
    for i, (artifact, package) in enumerate(zip(artifacts, members, strict=True)):
        artifact.validate()
        package.descriptor.validate_image(package.image)
        p = package.descriptor.provenance
        if (artifact.target != "nvidia_sm120" or artifact.architecture != "sm_120"
                or artifact.graph_ir != plan["member_graphs"][i]
                or artifact.tile_ir != package.tile_ir
                or package.image.target != "nvidia_sm120" or package.image.architecture != "sm_120a"
                or package.image.target_ir_digest != hashlib.sha256(package.target_ir.encode()).hexdigest()
                or p.get("route") != "canonical_scheduled_tile_consumer"
                or p.get("graph_ir_digest") != artifact.graph_digest
                or p.get("schedule_ir_digest") != artifact.schedule_ir_digest
                or p.get("tile_ir_digest") != artifact.tile_digest
                or p.get("schedule_digest") != artifact.schedule_digest):
            raise ValueError("paired tensor member provenance differs")
    # Reuse the established shape/storage/layout/ABI validator for every unary
    # member against both consumers. This is validation only, never execution.
    for i, a in enumerate(artifacts[:-2]):
        for j in (-2, -1):
            c = artifacts[j]
            if not isinstance(a, ScheduledKernelArtifact) or not isinstance(c, ScheduledMatmulArtifact):
                raise ValueError("paired tensor member kind differs")
            operation = plan["steps"][i]["operation"]
            expected_kind = {"tessera.rmsnorm": "rmsnorm", "tessera.layer_norm": "layernorm",
                             "tessera.softmax": "softmax", "tessera.relu": "relu"}[operation]
            if (a.kind != expected_kind or members[i].descriptor.provenance.get("kind") != a.kind
                    or members[i].descriptor.provenance.get("schedule") != a.schedule
                    or members[i].descriptor.provenance.get("epsilon") != a.epsilon
                    or c.bias_name is not None or c.residual_name is not None
                    or c.activation != "none" or c.output_dtype != "fp32"
                    or c.dynamic_m or c.dynamic_n or c.dynamic_k or c.b_layout != "row_major"):
                raise ValueError("paired tensor member semantics differ")
            NVIDIANativeTensorProgram(members[i], members[j], a.input_name, a.output_name,
                                     c.a_name, c.b_name, c.output_name, c.m, c.k, c.n, c.a_dtype).validate()
            source = plan["buffers"][plan["role_indices"][0]]
            rhs = plan["buffers"][plan["role_indices"][1]]
            if (list(a.input_shape) != source["shape"] or list(a.output_shape) != source["shape"]
                    or a.storage != source["storage"]
                    or (c.m, c.k, c.n) != (source["shape"][0], source["shape"][1], rhs["shape"][1])):
                raise ValueError("paired tensor member dimensions differ")
    return plan


def restore(data):
    from tessera import runtime as rt
    from .nvidia_native import NVIDIANativePackage
    keys = {"schema", "plan_json", "artifacts", "members", "contract_digest"}
    if (not isinstance(data, dict) or set(data) != keys
            or data["schema"] != "tessera.nvidia.tensor_rhs_jvp.v1"
            or digest({k: v for k, v in data.items() if k != "contract_digest"}) != data["contract_digest"]):
        raise ValueError("portable paired tensor schema/integrity differs")
    data = copy.deepcopy(data)
    plan = validate_plan(data["plan_json"])
    if (not isinstance(data["artifacts"], list) or not isinstance(data["members"], list)
            or len(data["artifacts"]) != len(plan["steps"]) or len(data["members"]) != len(plan["steps"])):
        raise ValueError("portable paired tensor stage count differs")
    artifacts, members = [], []
    for step, record, item in zip(plan["steps"], data["artifacts"], data["members"], strict=True):
        record = dict(record)
        artifact: ScheduledKernelArtifact | ScheduledMatmulArtifact
        if step["operation"] == "tessera.matmul":
            artifact = ScheduledMatmulArtifact(**record)
        else:
            for name in ("input_shape", "output_shape", "input_names"):
                record[name] = tuple(record[name])
            artifact = ScheduledKernelArtifact(**record)
        a = rt.RuntimeArtifact.from_dict(item)
        if a.native_image is None or a.launch_descriptor is None or item.get("artifact_hash") != a.artifact_hash:
            raise ValueError("portable paired tensor component integrity differs")
        artifacts.append(artifact)
        members.append(NVIDIANativePackage(a.tile_ir, a.target_ir, "", a.native_image, a.launch_descriptor))
    validate_members(data["plan_json"], artifacts, members)
    # The compiler is deliberately unavailable on restored witnesses; replay
    # consumes only the captured members, never project_member/scheduled_members.
    native = NativeSM120TensorGraph(plan["source_graph_ir"], "", data["plan_json"], Path("/unavailable/compiler"), "", True)
    return native, tuple(artifacts), tuple(members)
