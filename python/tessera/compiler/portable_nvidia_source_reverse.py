"""Portable native source-reverse records, validated without a compiler.

Integrity hashes detect mutation, not authenticity of arbitrary supplied code.
"""
import copy
import hashlib
from pathlib import Path
import re

from .native_sm120_source_reverse import NativeSourceReverseProgram, validate_plan
from .portable_nvidia_tensor_jvp import digest

SCHEMA = "tessera.nvidia.source_reverse_program.v1"
COMPILER_PATH = "nvidia_sm120_source_reverse_compiled"
_FIELDS = ("graph_ir", "schedule_ir", "tile_ir", "target_ir", "native_ir")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _abi(plan, index, entry):
    source = plan["buffers"][plan["role_indices"][0]]
    rhs = plan["buffers"][plan["role_indices"][1]]
    m, k = source["shape"]
    n = rhs["shape"][1]
    step = plan["steps"][index]
    return dict(schema="tessera.nvidia.source_reverse_step.v1", step_bytes=112,
                kind="mixed_matmul" if index == 0 else "row_pullback",
                entry=entry, inputs=step["inputs"], outputs=step["outputs"],
                dimensions=[m, k, n if index == 0 else 0], storage=source["storage"],
                block_size=32 if index == 0 else 128,
                parameters=["ptr"] * 3 + ["i64"] * (3 if index == 0 else 2))


def _provenance(plan_json, plan, record):
    return {"program_digest": sha(plan_json),
            "source_graph_ir_digest": sha(plan["source_graph_ir"]),
            "paired_graph_ir_digest": sha(plan["paired_graph_ir"]),
            **{name + "_digest": sha(record[name]) for name in _FIELDS}}


def seal_members(plan_json, records):
    plan = validate_plan(plan_json)
    result = []
    for i, original in enumerate(records):
        record = copy.deepcopy(original)
        record["abi"] = _abi(plan, i, record["entry"])
        record["provenance"] = _provenance(plan_json, plan, record)
        record["member_digest"] = digest(record)
        result.append(record)
    validate_members(plan_json, result)
    return tuple(result)


def validate_members(plan_json, records):
    plan = validate_plan(plan_json)
    if not isinstance(records, (tuple, list)) or len(records) != 2:
        raise ValueError("source reverse member count differs")
    keys = {*_FIELDS, "entry", "ptx", "image_digest", "resources", "compiler",
            "toolchain", "abi", "provenance", "member_digest"}
    for i, record in enumerate(records):
        if (not isinstance(record, dict) or set(record) != keys
                or any(not isinstance(record[n], str) or not record[n] for n in (
                    *_FIELDS, "entry", "ptx", "image_digest", "compiler", "toolchain", "member_digest"))
                or digest({k: v for k, v in record.items() if k != "member_digest"}) != record["member_digest"]):
            raise ValueError("source reverse member integrity differs")
        if (record["graph_ir"] != plan["member_graphs"][i]
                or record["image_digest"] != sha(record["ptx"])
                or record["provenance"] != _provenance(plan_json, plan, record)
                or digest(record["abi"]) != digest(_abi(plan, i, record["entry"]))):
            raise ValueError("source reverse member provenance/ABI differs")
        resources = record["resources"]
        if (not isinstance(resources, dict) or set(resources) != {
                "registers_per_thread", "static_shared_memory_bytes", "spill_store_bytes", "spill_load_bytes"}
                or any(type(v) is not int or v < 0 for v in resources.values())):
            raise ValueError("source reverse resource metadata differs")
        entry = record["entry"]
        if (re.findall(r"\.visible\s+\.entry\s+([A-Za-z0-9_]+)", record["ptx"]) != [entry]
                or not re.search(r"\.target\s+sm_120a\b", record["ptx"])
                or re.findall(r"\bllvm.func @(\w+)\(", record["native_ir"]) != [entry]):
            raise ValueError("source reverse native entry/target differs")
        if i == 0:
            if (entry != plan["steps"][0]["member"] + "_scalar_fma_kernel"
                    or "schedule.matmul" not in record["schedule_ir"]
                    or 'physical_contract = "nvidia_sm120_scalar_fma_v1"' not in record["schedule_ir"]
                    or record["tile_ir"].count("tile.scalar_matmul_kernel") != 1
                    or "transpose_b = true" not in record["tile_ir"]
                    or "a_element_type = f32" not in record["tile_ir"]):
                raise ValueError("source reverse mixed contraction witness differs")
        else:
            graph_hash = re.findall(r'body_hash = "([0-9a-f]{64})"', record["graph_ir"])
            source_hash = re.findall(r'source_hash = "([0-9a-f]{64})"', record["tile_ir"])
            tile_hash = re.findall(r'(?<![\w_])body_hash = "([0-9a-f]{64})"', record["tile_ir"])
            parent_hash = re.findall(r'tile_body_hash = "([0-9a-f]{64})"', record["target_ir"])
            if (len(graph_hash) != 1 or source_hash != graph_hash or len(tile_hash) != 1
                    or parent_hash != tile_hash or entry != "tessera_native_row_" + graph_hash[0][:16]
                    or "schema = 6" not in record["tile_ir"]
                    or "schedule.artifact" not in record["schedule_ir"]
                    or "tile.row_region_kernel" not in record["tile_ir"]
                    or "tessera_nvidia.row_region_kernel" not in record["target_ir"]):
                raise ValueError("source reverse row seal/profile differs")
    return plan


def manifest(plan_json, records):
    validate_members(plan_json, records)
    data = dict(schema=SCHEMA, plan_json=plan_json, members=copy.deepcopy(list(records)))
    data["contract_digest"] = digest(data)
    return data


def restore(data):
    if (not isinstance(data, dict) or set(data) != {"schema", "plan_json", "members", "contract_digest"}
            or data["schema"] != SCHEMA
            or digest({k: v for k, v in data.items() if k != "contract_digest"}) != data["contract_digest"]):
        raise ValueError("source reverse portable integrity differs")
    data = copy.deepcopy(data)
    plan = validate_members(data["plan_json"], data["members"])
    native = NativeSourceReverseProgram(plan["source_graph_ir"], "", data["plan_json"],
                                        Path("/unavailable/compiler"), "")
    return native, tuple(data["members"])
