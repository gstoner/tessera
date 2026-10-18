"""Offline structural validation for native source-attention reverse programs.

Hashes detect corruption; they do not authenticate arbitrary native code.
Validation never invokes a compiler or acquires a CUDA provider.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
import math
import re
import struct

from .portable_source_attention import digest, sha

SCHEMA = "tessera.nvidia.source_attention_reverse_program.v1"
COMPILER_PATH = "nvidia_sm120_source_attention_reverse_compiled"
_FIELDS = ("graph_ir", "schedule_ir", "tile_ir", "target_ir", "native_ir")
_SYMBOL = r"[A-Za-z_][A-Za-z0-9_.$]*"


def _functions(text):
    if not isinstance(text, str):
        raise ValueError("source attention reverse Graph witness is not text")
    matches = list(re.finditer(
        r"^  func\.func (private )?@(" + _SYMBOL + r")\(([^\n]*)\) -> ([^\n]*) \{\n(.*?)^  \}",
        text, re.M | re.S))
    if not matches or len({m[2] for m in matches}) != len(matches):
        raise ValueError("source attention reverse typed functions differ")
    return matches


def _type(buffer):
    return "tensor<" + "x".join(map(str, buffer["shape"])) + "x" + buffer["storage"] + ">"


def _signature(fn, inputs, outputs, buffers):
    args = ", ".join(f"%arg{i}: {_type(buffers[index])}" for i, index in enumerate(inputs))
    result = ", ".join(_type(buffers[index]) for index in outputs)
    if len(outputs) > 1:
        result = "(" + result + ")"
    if fn[3] != args or fn[4].split(" attributes {", 1)[0] != result:
        raise ValueError("source attention reverse typed signature differs")


def _one(pattern, text, message):
    values = re.findall(pattern, text, re.M)
    if len(values) != 1:
        raise ValueError(message)
    return values[0]


def _norm_policy(tail, storage):
    attrs = _one(r'^\{(.*)\} : \(', tail, "source attention reverse normalization attributes differ")
    policy = _one(r'\bnumeric_policy = \{([^{}]*)\}', attrs,
                  "source attention reverse normalization numeric policy differs")
    fields = {}
    for pair in policy.split(", "):
        key, separator, value = pair.partition(" = ")
        if not separator or key in fields:
            raise ValueError("source attention reverse normalization numeric policy is malformed")
        fields[key] = value
    if (set(fields) - {"accum", "storage"} or fields.get("accum") != '"fp32"'
            or fields.get("storage", json.dumps({"f16": "fp16", "bf16": "bf16"}[storage]))
                != json.dumps({"f16": "fp16", "bf16": "bf16"}[storage])):
        raise ValueError("source attention reverse normalization accumulation/storage differs")
    attrs = re.sub(r'numeric_policy = \{[^{}]*\}', "numeric_policy = present", attrs)
    outer = {}
    for pair in re.split(r', (?=[A-Za-z_][\w.]* =)', attrs):
        key, separator, value = pair.partition(" = ")
        if not separator or key in outer:
            raise ValueError("source attention reverse normalization attributes are malformed")
        outer[key] = value
    if (set(outer) - {"eps", "axis", "numeric_policy", "tessera.effect_kind", "tessera.autodiff.activity"}
            or outer.get("axis", "-1 : i64") not in {"-1 : i64", "3 : i64"}
            or outer.get("tessera.effect_kind", '"pure"') != '"pure"'
            or ("eps" in outer and not re.fullmatch(r'[^ ]+ : f(?:32|64)', outer["eps"]))):
        raise ValueError("source attention reverse normalization typed attributes differ")


def _checkpoint(plan, index, record, dims):
    """Inspect native metadata without the compiler-replaying artifact decoder."""
    from .nvidia_native import _mlir_f32_bits
    storage = plan["buffers"][5]["storage"]
    tile = record["tile_ir"]
    raw = _one(r'tessera.native_contract = \{([^{}\n]*)\}', tile,
               "source attention reverse checkpoint native contract differs")
    fields = {}
    for pair in re.split(r', (?=[A-Za-z_]\w* =)', raw):
        key, separator, value = pair.partition(" = ")
        if not separator or key in fields:
            raise ValueError("source attention reverse checkpoint metadata is malformed")
        fields[key] = value
    scale = _one(r'^([^ ]+) : f32$', fields.get("scale", ""),
                 "source attention reverse checkpoint scale differs")
    scale_value = struct.unpack("f", _mlir_f32_bits(scale))[0]
    if not math.isfinite(scale_value) or scale_value <= 0 or fields.get("causal") not in {"true", "false"}:
        raise ValueError("source attention reverse checkpoint numeric policy differs")
    backward = index == 2
    inputs, outputs = (7, 3) if backward else (3, 2)
    expected = dict(accum='"f32"', arch='"sm_120"', target='"nvidia_sm120"',
                    arguments=json.dumps(["input_" + str(j) for j in range(inputs)]),
                    results=json.dumps(["output_" + str(j) for j in range(outputs)]),
                    bias="false", causal=fields["causal"],
                    family=json.dumps("attention_checkpoint_backward" if backward else "attention_checkpoint_forward"),
                    mask_alignment='"end_aligned_v1"', output_storage=json.dumps(storage),
                    row_lse_storage='"f32"', scale=fields["scale"],
                    shape="array<i64: " + ", ".join(map(str, dims)) + ">", storage=json.dumps(storage))
    if backward:
        expected.update(gradient_storage=json.dumps(storage), lse_cotangent="true",
                        row_correction='"probability_dot_v_f32_v1"')
    schedule = _one(r'tessera.schedule_hash = "([0-9a-f]{64})"', tile,
                    "source attention reverse checkpoint Schedule seal differs")
    prefix = "tessera_tile_attention_backward_lse_output_cotangent_" if backward else "tessera_tile_attention_lse_"
    op = "tile.attention_backward_kernel" if backward else "tile.attention_kernel"
    if (fields != expected or record["entry"] != prefix + storage + "_out_" + storage + "_" + schedule[:10]
            or tile.count(op) != 1 or 'lse_checkpoint = "saved"' not in tile
            or tile.count("scale = " + fields["scale"]) != 2
            or tile.count("causal = " + fields["causal"]) != 2
            or tile.count("storage = " + json.dumps(storage)) < 2
            or (backward and ('lse_cotangent = true' not in tile or 'route = "deterministic_direct"' not in tile))):
        raise ValueError("source attention reverse checkpoint physical policy differs")
    return scale_value, fields["causal"] == "true"


def validate_plan(plan):
    keys = {"schema", "kind", "execution_state", "target", "architecture", "root", "source_root",
            "source_graph_ir", "source_graph_sha256", "paired_graph_ir", "paired_graph_sha256",
            "program_graph_ir", "program_graph_sha256", "role_indices", "cotangent_indices",
            "requested_gradients", "buffers", "steps", "outputs", "member_graphs"}
    if (not isinstance(plan, dict) or set(plan) != keys
            or plan["schema"] != "tessera.native.source_attention_reverse.v1"
            or plan["kind"] != "norm_attention_saved_reverse" or plan["execution_state"] != "artifact_only"
            or plan["target"] != "nvidia_sm120" or plan["architecture"] != "sm_120"):
        raise ValueError("source attention reverse plan schema/target differs")
    for stage in ("source", "paired", "program"):
        text = plan[stage + "_graph_ir"]
        if not isinstance(text, str) or not text or sha(text) != plan[stage + "_graph_sha256"]:
            raise ValueError("source attention reverse Graph digest differs")
    original = plan["source_graph_ir"]
    if (not re.search(r'(?<=[{,] )tessera.native.source_attention_reverse_request(?=[,}])', original)
            or "tessera.attention_shape_bounds" in original or "tessera.attention_sequence_bounds" in original):
        raise ValueError("source attention reverse original export/shape policy differs")
    for field, allowed in (("role_indices", [0, 1, 2]), ("cotangent_indices", [3, 4])):
        value = plan[field]
        if (not isinstance(value, list) or len(value) != len(allowed)
                or any(type(v) is not int for v in value) or sorted(value) != allowed):
            raise ValueError("source attention reverse root/cotangent roles differ")
    requested = plan["requested_gradients"]
    if (not isinstance(requested, list) or not 1 <= len(requested) <= 3
            or any(type(v) is not int or v not in (0, 1, 2) for v in requested)
            or len(set(requested)) != len(requested)
            or not isinstance(plan["source_root"], str)
            or not re.fullmatch(_SYMBOL, plan["source_root"])
            or plan["root"] != plan["source_root"] + "__source_reverse"):
        raise ValueError("source attention reverse original root/gradient order differs")
    roles, seeds = plan["role_indices"], plan["cotangent_indices"]
    gradients = {roles[0]: 11, roles[1]: 9, roles[2]: 10}
    outputs = [gradients[i] for i in requested]
    if (plan["outputs"] != outputs or any(type(v) is not int for v in plan["outputs"])):
        raise ValueError("source attention reverse returned gradient roles differ")
    edges = (([roles[0]], [5], "tessera.rmsnorm"),
             ([5, roles[1], roles[2]], [6, 7], "tessera_attn.checkpoint_forward"),
             ([seeds[0], 5, roles[1], roles[2], 6, 7, seeds[1]], [8, 9, 10], "tessera_attn.checkpoint_backward"),
             ([roles[0], 8], [11], "native_row_region"))
    expected = [dict(step=i, member=plan["root"] + "__member_" + str(i),
                     inputs=ins, outputs=outs, operation=op)
                for i, (ins, outs, op) in enumerate(edges)]
    if (plan["steps"] != expected or any(type(s["step"]) is not int or any(
            type(v) is not int for v in (*s["inputs"], *s["outputs"])) for s in plan["steps"])):
        raise ValueError("source attention reverse native SSA edges differ")
    buffers = plan["buffers"]
    if not isinstance(buffers, list) or len(buffers) != 12:
        raise ValueError("source attention reverse allocation count differs")
    writes, reads = [-1] * 12, [-1] * 12
    for i, (ins, outs, _) in enumerate(edges):
        for index in ins:
            reads[index] = i
        for index in outs:
            writes[index] = reads[index] = i
    for index in outputs:
        reads[index] = 4
    for i, b in enumerate(buffers):
        if (not isinstance(b, dict) or set(b) != {
                "id", "shape", "storage", "bytes", "ownership", "first_write", "last_read"}
                or type(b["id"]) is not int or b["id"] != i
                or not isinstance(b["shape"], list) or not 3 <= len(b["shape"]) <= 4
                or any(type(d) is not int or not 0 < d < 2**31 for d in b["shape"])
                or b["storage"] not in {"f16", "bf16", "f32"}):
            raise ValueError("source attention reverse buffer schema differs")
        size = math.prod(b["shape"]) * (4 if b["storage"] == "f32" else 2)
        ownership = "readonly_input" if i < 5 else "returned_output" if i in outputs else "private_scratch"
        if (size >= 2**63 or type(b["bytes"]) is not int or b["bytes"] != size
                or type(b["first_write"]) is not int or b["first_write"] != writes[i]
                or type(b["last_read"]) is not int or b["last_read"] != reads[i]
                or b["ownership"] != ownership):
            raise ValueError("source attention reverse allocation lifetime differs")
    q, k, v = (buffers[i] for i in roles)
    if any(len(b["shape"]) != 4 for b in (q, k, v)) or q["storage"] not in {"f16", "bf16"}:
        raise ValueError("source attention reverse original storage/rank differs")
    b, hq, sq, d = q["shape"]
    bk, hkv, sk, dk = k["shape"]
    bv, hv, sv, dv = v["shape"]
    if b != bk or b != bv or hq % hkv or hv != hkv or sk != sv or d != dk or b * hq * sq >= 2**31:
        raise ValueError("source attention reverse physical dimensions differ")
    expected_shapes: list[list[int] | None] = [None] * 12
    for index, shape in zip(roles, (q["shape"], k["shape"], v["shape"]), strict=True):
        expected_shapes[index] = shape
    for index in (seeds[0], 6):
        expected_shapes[index] = [b, hq, sq, dv]
    for index in (seeds[1], 7):
        expected_shapes[index] = [b, hq, sq]
    for index in (5, 8, 11):
        expected_shapes[index] = q["shape"]
    expected_shapes[9], expected_shapes[10] = k["shape"], v["shape"]
    if any(item["shape"] != expected_shapes[i] or item["storage"] != (
            "f32" if i in (seeds[1], 7) else q["storage"]) for i, item in enumerate(buffers)):
        raise ValueError("source attention reverse private storage roles differ")
    _verify_calls(plan)
    return (b, hq, hkv, sq, sk, d, dv)


def _verify_calls(plan):
    buffers = plan["buffers"]
    original = _functions(plan["source_graph_ir"])
    if len(original) != 1 or original[0][2] != plan["source_root"]:
        raise ValueError("source attention reverse original function differs")
    original = original[0]
    seeds = plan["cotangent_indices"]
    _signature(original, range(3), [3, 4], buffers)
    indices = re.findall(r'tessera.autodiff.wrt_indices = (\[[^\]]+\])', original[4])
    if (len(indices) > 1 or (json.loads(indices[0]) if indices else [0, 1, 2]) != plan["requested_gradients"]
            or 'tessera.autodiff = "reverse"' not in original[4]):
        raise ValueError("source attention reverse original derivative intent differs")
    norm = _one(r'^    (%\w+) = tessera.rmsnorm %arg([0-2]) (.*)$', original[5],
                "source attention reverse original normalization differs")
    _norm_policy(norm[2], buffers[plan["role_indices"][0]]["storage"])
    attention = _one(r'^    (%[\w]+), (%[\w]+) = tessera.flash_attn (%\w+), %arg([0-2]), %arg([0-2]) (.*)$',
                     original[5], "source attention reverse original attention differs")
    attrs = _one(r'^\{(.*)\} : \(', attention[5], "source attention reverse original attributes differ")
    fields = {}
    for pair in re.split(r', (?=[A-Za-z_][\w.]* =)', attrs):
        key, separator, value = pair.partition(" = ")
        if not separator or key in fields:
            raise ValueError("source attention reverse original attributes are malformed")
        fields[key] = value
    allowed = {"head_dim", "causal", "scale", "lse_checkpoint", "operandSegmentSizes",
               "tessera.effect_kind", "tessera.autodiff.activity", "dropout_p", "dropout_seed",
               "softcap", "window_left", "window_right"}
    if (set(fields) - allowed or fields.get("lse_checkpoint") != '"saved"'
            or ("causal" in fields and fields["causal"] not in {"true", "false"})
            or ("scale" in fields and not re.fullmatch(r'[^ ]+ : f(?:32|64)', fields["scale"]))
            or ("dropout_seed" in fields and not re.fullmatch(r'-?[0-9]+ : i64', fields["dropout_seed"]))
            or fields.get("head_dim", str(buffers[plan["role_indices"][0]]["shape"][-1]) + " : i64")
                != str(buffers[plan["role_indices"][0]]["shape"][-1]) + " : i64"
            or fields.get("operandSegmentSizes", "array<i32: 1, 1, 1, 0>") != "array<i32: 1, 1, 1, 0>"):
        raise ValueError("source attention reverse original physical policy differs")
    from .nvidia_native import _mlir_f32_bits
    for field in ("dropout_p", "softcap"):
        if field in fields:
            value = _one(r'^([^ ]+) : f(?:32|64)$', fields[field], "source attention reverse original numeric policy differs")
            if struct.unpack("f", _mlir_f32_bits(value))[0] != 0:
                raise ValueError("source attention reverse original unsupported numeric policy")
    if any(fields.get(field, "-1 : i64") != "-1 : i64" for field in ("window_left", "window_right")):
        raise ValueError("source attention reverse original window policy differs")
    returned = _one(r'^    return ([^:]+) :', original[5], "source attention reverse original results differ")
    if (list(map(int, (norm[1], attention[3], attention[4]))) != plan["role_indices"]
            or attention[2] != norm[0] or 'lse_checkpoint = "saved"' not in attention[5]
            or returned.strip().split(", ") != ([attention[0], attention[1]] if seeds == [3, 4]
                                               else [attention[1], attention[0]])
            or len(original[5].strip().splitlines()) != 3):
        raise ValueError("source attention reverse original SSA roles differ")
    functions = _functions(plan["program_graph_ir"])
    graphs = plan["member_graphs"]
    if (len(functions) != 5 or functions[0][1] or functions[0][2] != plan["root"]
            or not isinstance(graphs, list) or len(graphs) != 4):
        raise ValueError("source attention reverse typed call program differs")
    root = functions[0]
    _signature(root, range(5), plan["outputs"], buffers)
    lines = root[5].strip().splitlines()
    if len(lines) != 5:
        raise ValueError("source attention reverse root must contain four calls and return")
    values = [f"%arg{i}" for i in range(5)]
    for i, (step, line) in enumerate(zip(plan["steps"], lines[:4], strict=True)):
        call = re.fullmatch(r'\s*(%\w+)(?::([0-9]+))? = call @(' + _SYMBOL + r')\(([^)]*)\) : (.*)', line)
        args = ", ".join(values[index] for index in step["inputs"])
        ins = "(" + ", ".join(_type(buffers[index]) for index in step["inputs"]) + ")"
        outs = ", ".join(_type(buffers[index]) for index in step["outputs"])
        count = len(step["outputs"])
        if count > 1:
            outs = "(" + outs + ")"
        if (call is None or call[3] != step["member"] or call[4] != args
                or int(call[2] or 1) != count or call[5] != ins + " -> " + outs):
            raise ValueError("source attention reverse root call binding differs")
        values.extend([call[1]] if count == 1 else [call[1] + "#" + str(j) for j in range(count)])
        member = _functions(graphs[i])
        if (len(member) != 1 or member[0][1] or member[0][2] != step["member"]
                or not functions[i + 1][1] or functions[i + 1][0] != member[0][0].replace(
                    "func.func @", "func.func private @", 1)):
            raise ValueError("source attention reverse isolated/called member differs")
        _signature(member[0], step["inputs"], step["outputs"], buffers)
        if i == 0:
            member_norm = _one(r'^    %\w+ = tessera.rmsnorm %arg0 (.*)$', member[0][5],
                               "source attention reverse member normalization differs")
            clean = lambda text: re.sub(r',? tessera.autodiff.activity = "[^"]*"', '', text)
            if clean(member_norm) != clean(norm[2]):
                raise ValueError("source attention reverse original/member normalization policy differs")
        for key, count in (("argument", len(step["inputs"])), ("result", len(step["outputs"]))):
            binding = _one(r'tessera\.' + key + r'_bindings = (\[[^\]]*\])', member[0][4],
                           "source attention reverse member names differ")
            prefix = "input_" if key == "argument" else "output_"
            if json.loads(binding) != [prefix + str(j) for j in range(count)]:
                raise ValueError("source attention reverse local member binding differs")
    returned = "return " + ", ".join(values[index] for index in plan["outputs"]) + " : "
    returned += ", ".join(_type(buffers[index]) for index in plan["outputs"])
    if lines[4].strip() != returned:
        raise ValueError("source attention reverse root return differs")
    for text, fns in [(plan["program_graph_ir"], functions)] + [
            (graph, _functions(graph)) for graph in graphs]:
        for fn in reversed(fns):
            text = text[:fn.start()] + text[fn.end():]
        if not re.fullmatch(r'module attributes \{tessera.arch = "sm_120", tessera.target = "nvidia_sm120"\} \{\s*\}\s*', text):
            raise ValueError("source attention reverse program contains unaccounted operations")


def _abi(plan, index, dims, entry):
    rows, columns = math.prod(plan["buffers"][5]["shape"][:-1]), dims[5]
    cooperative = "_cooperative_128_" in entry
    sizes = (rows if cooperative else (rows + 127) // 128,
             (math.prod(plan["buffers"][6]["shape"]) + 127) // 128,
             (sum(math.prod(plan["buffers"][i]["shape"]) for i in (8, 9, 10)) + 127) // 128,
             (rows + 127) // 128)
    parameters = (["ptr", "ptr", "i64", "i64"], ["ptr"] * 5 + ["i64"] * 7,
                  ["ptr"] * 10 + ["i64"] * 7, ["ptr"] * 3 + ["i64"] * 2)
    step = plan["steps"][index]
    return dict(schema="tessera.nvidia.source_attention_reverse_step.v1", index=index,
                entry=entry, block=[128, 1, 1], grid=[sizes[index], 1, 1],
                dimensions=list(dims) if index in (1, 2) else [rows, columns],
                inputs=step["inputs"], outputs=step["outputs"],
                parameters=parameters[index], storage=plan["buffers"][5]["storage"])


def seal(plan, members):
    dims = validate_plan(plan)
    records = []
    for index, member in enumerate(members):
        record = asdict(member)
        record["ptx"] = record.pop("image").decode("ascii")
        record["abi"] = _abi(plan, index, dims, member.entry)
        record["provenance"] = {field + "_sha256": sha(record[field]) for field in _FIELDS}
        record["member_digest"] = digest(record)
        records.append(record)
    data = dict(schema=SCHEMA, plan=copy.deepcopy(plan), members=records)
    data["contract_digest"] = digest(data)
    restore(data)
    return data


def restore(data):
    from .prepared_source_attention_reverse import ReverseMember
    if (not isinstance(data, dict) or set(data) != {"schema", "plan", "members", "contract_digest"}
            or data["schema"] != SCHEMA
            or data["contract_digest"] != digest({k: v for k, v in data.items() if k != "contract_digest"})):
        raise ValueError("source attention reverse portable integrity differs")
    data = copy.deepcopy(data)
    plan, records = data["plan"], data["members"]
    dims = validate_plan(plan)
    if not isinstance(records, list) or len(records) != 4:
        raise ValueError("source attention reverse image count differs")
    members = []
    checkpoints = []
    for i, record in enumerate(records):
        if (not isinstance(record, dict) or set(record) != {
                *_FIELDS, "entry", "ptx", "image_digest", "abi", "provenance", "member_digest"}
                or any(not isinstance(record[f], str) or not record[f] for f in (*_FIELDS, "entry", "ptx"))
                or record["member_digest"] != digest({k: v for k, v in record.items() if k != "member_digest"})
                or record["provenance"] != {f + "_sha256": sha(record[f]) for f in _FIELDS}
                or record["graph_ir"] != plan["member_graphs"][i]
                or record["image_digest"] != sha(record["ptx"])
                or digest(record["abi"]) != digest(_abi(plan, i, dims, record["entry"]))):
            raise ValueError("source attention reverse image provenance/ABI differs")
        entry, tile, graph = record["entry"], record["tile_ir"], record["graph_ir"]
        expected = record["abi"]["parameters"]
        signature = _one(r'llvm.func @' + re.escape(entry) + r'\(([^)]*)\)', tile,
                         "source attention reverse Tile entry differs")
        if (re.findall(r': (!llvm.ptr|i64)', signature) != ["!llvm.ptr" if v == "ptr" else v for v in expected]
                or re.findall(r'\.visible\s+\.entry\s+([A-Za-z0-9_]+)', record["ptx"]) != [entry]
                or not re.search(r'\.target\s+sm_120a\b', record["ptx"])
                or re.findall(r'\bllvm.func @(\w+)\(', record["native_ir"]) != [entry]):
            raise ValueError("source attention reverse native entry/parameter contract differs")
        params = _one(r'\.visible\s+\.entry\s+' + re.escape(entry) + r'\s*\(([^)]*)\)', record["ptx"],
                      "source attention reverse PTX entry differs")
        if len(re.findall(r'\.param\s+\.(?:u|s|b)64\s+', params)) != len(expected):
            raise ValueError("source attention reverse PTX parameter widths differ")
        storage = plan["buffers"][5]["storage"]
        if i in (1, 2):
            checkpoints.append(_checkpoint(plan, i, record, dims))
        elif i == 0:
            from .nvidia_native import _mlir_f32_bits
            schedule_hash = _one(r'tessera.schedule_hash = "([0-9a-f]{64})"', tile,
                                 "source attention reverse normalization Schedule seal differs")
            mode = _one(r'\bschedule = "(serial|cooperative_128)"', tile,
                        "source attention reverse normalization schedule differs")
            suffix = "_cooperative_128_" if mode == "cooperative_128" else "_"
            eps = _one(r'tessera.norm_epsilon = ([^ ]+) : f32', tile,
                       "source attention reverse normalization epsilon differs")
            source_eps = re.findall(r'\beps = ([^ ]+) : f(?:32|64)', graph)
            value = source_eps[0] if source_eps else "1e-5"
            numeric_eps = struct.unpack("f", _mlir_f32_bits(value))[0]
            if (entry != "tessera_tile_norm_rmsnorm_" + storage + suffix + schedule_hash[:10]
                    or not math.isfinite(numeric_eps) or numeric_eps <= 0
                    or _mlir_f32_bits(eps) != _mlir_f32_bits(value)
                    or tile.count("tile.norm_kernel") != 1 or 'kind = "rmsnorm"' not in tile
                    or 'storage = "' + storage + '"' not in tile
                    or 'accum = "f32"' not in tile or 'axis = -1 : i64' not in tile):
                raise ValueError("source attention reverse normalization physical policy differs")
        else:
            body_hash = _one(r'\bbody_hash = "([0-9a-f]{64})"', graph,
                             "source attention reverse pullback Graph seal differs")
            source_hash = _one(r'\bsource_hash = "([0-9a-f]{64})"', tile,
                               "source attention reverse pullback source seal differs")
            tile_hash = _one(r'(?<![\w_])body_hash = "([0-9a-f]{64})"', tile,
                             "source attention reverse pullback Tile seal differs")
            parent = _one(r'tile_body_hash = "([0-9a-f]{64})"', record["target_ir"],
                          "source attention reverse pullback Target seal differs")
            rows, cols = math.prod(plan["buffers"][5]["shape"][:-1]), dims[5]
            if (source_hash != body_hash or parent != tile_hash
                    or entry != "tessera_native_row_" + body_hash[:16]
                    or "schema = 6 : i64" not in tile or "tile.row_region_kernel" not in tile
                    or "tessera_nvidia.row_region_kernel" not in record["target_ir"]
                    or f"rows = {rows} : i64" not in tile or f"columns = {cols} : i64" not in tile):
                raise ValueError("source attention reverse pullback geometry/profile differs")
        members.append(ReverseMember(graph_ir=record["graph_ir"], schedule_ir=record["schedule_ir"],
                                     tile_ir=record["tile_ir"], target_ir=record["target_ir"],
                                     native_ir=record["native_ir"], entry=entry,
                                     image=record["ptx"].encode("ascii"), image_digest=record["image_digest"]))
    if checkpoints[0] != checkpoints[1]:
        raise ValueError("source attention reverse saved forward/backward policy differs")
    from .nvidia_native import _mlir_f32_bits
    original = _functions(plan["source_graph_ir"])[0][5]
    attention_line = _one(r'^    .* = tessera.flash_attn (.*)$', original,
                          "source attention reverse original attention policy differs")
    scale = re.findall(r'\bscale = ([^ ]+) : f(?:32|64)', attention_line)
    causal = re.findall(r'\bcausal = (true|false)', attention_line)
    if (len(scale) > 1 or len(causal) > 1 or 'numeric_policy =' in attention_line
            or _mlir_f32_bits(scale[0] if scale else str(dims[5] ** -.5)) != _mlir_f32_bits(str(checkpoints[0][0]))
            or checkpoints[0][1] != (bool(causal) and causal[0] == 'true')):
        raise ValueError("source attention reverse original/checkpoint numerical policy differs")
    return data, plan, tuple(members), dims
