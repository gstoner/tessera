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
from .attention_shape_contract import DYNAMIC_DIM, attention_dimensions

SCHEMA = "tessera.nvidia.source_attention_reverse_program.v1"
_ROWS_SCHEMA = "tessera.nvidia.source_attention_reverse_program.v2"
_BOUNDED_SCHEMA = "tessera.nvidia.source_attention_reverse_program.v3"
_BOUNDED_ROWS_SCHEMA = "tessera.nvidia.source_attention_reverse_program.v4"


def _rows(plan):
    return plan.get("schema") in ("tessera.native.source_attention_reverse.v2",
                                  "tessera.native.source_attention_reverse.v4")


def _bounded(plan):
    return isinstance(plan, dict) and plan.get("schema") in (
        "tessera.native.source_attention_reverse.v3", "tessera.native.source_attention_reverse.v4")


def _portable_schema(plan):
    return ((_BOUNDED_ROWS_SCHEMA if _rows(plan) else _BOUNDED_SCHEMA) if _bounded(plan)
            else (_ROWS_SCHEMA if _rows(plan) else SCHEMA))


def _physical_shape(buffer):
    return buffer.get("capacity_shape", buffer["shape"])


def _symbolic_dims(plan, dims):
    result = list(dims)
    if _bounded(plan):
        result[3] = result[4] = DYNAMIC_DIM
    return result

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


def _type(buffer, shape_key="shape"):
    dims = ("?" if x == DYNAMIC_DIM else str(x) for x in buffer[shape_key])
    return "tensor<" + "x".join(dims) + "x" + buffer["storage"] + ">"


def _signature(fn, inputs, outputs, buffers, shape_key="shape"):
    args = ", ".join(f"%arg{i}: {_type(buffers[index], shape_key)}" for i, index in enumerate(inputs))
    result = ", ".join(_type(buffers[index], shape_key) for index in outputs)
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
    if _bounded(plan):
        expected.update(shape="array<i64: " + ", ".join(map(str, _symbolic_dims(plan, dims))) + ">",
                        shape_bounds="array<i64: " + ", ".join(map(str, dims)) + ">",
                        shape_policy='"bounded_sequences_v1"')
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


def _row_checkpoint(plan, index, record, dims):
    from .native_attention_rows import contract, _entry
    raw, fields = contract(record["tile_ir"])
    consumer = index == 3
    storage = plan["buffers"][5]["storage"]
    count = 3 if consumer else 1
    expected = dict(accum="f32", arch="sm_120", arguments=["input_" + str(i) for i in range(6)],
                    results=["output_" + str(i) for i in range(count)], bias=False,
                    block_threads=128 if consumer else 32, causal=fields.get("causal"),
                    family="attention_backward_rows" if consumer else "attention_row_correction",
                    gradient_activity=[1, 1, 1], lse_cotangent=True,
                    mask_alignment="end_aligned_v1", output_storage=storage,
                    row_bytes=math.prod(_physical_shape(plan["buffers"][7])) * 4,
                    row_correction="probability_dot_v_f32_v1", row_storage="f32",
                    scale=fields.get("scale"), shape=list(dims), storage=storage,
                    target="nvidia_sm120")
    if _bounded(plan):
        expected.update(shape=_symbolic_dims(plan, dims), shape_bounds=list(dims),
                        shape_policy="bounded_sequences_v1")
    if consumer:
        expected.update(bias_gradient=False, gradient_output="compact_v1",
                        gradient_launch="packed_v1", row_correction_usage="external_input_v1")
    scale = fields.get("scale")
    if (digest(fields) != digest(expected) or type(fields.get("causal")) is not bool
            or type(scale) is not float or not math.isfinite(scale) or scale <= 0
            or record["entry"] != _entry(fields, sha(raw), consumer)):
        raise ValueError("source attention reverse row physical policy differs")
    operation = "tile.attention_backward_rows_kernel" if consumer else "tile.attention_row_correction_kernel"
    if record["tile_ir"].count(operation) != 1:
        raise ValueError("source attention reverse row Tile consumer differs")
    return scale, fields["causal"]


def validate_plan(plan):
    keys = {"schema", "kind", "execution_state", "target", "architecture", "root", "source_root",
            "source_graph_ir", "source_graph_sha256", "paired_graph_ir", "paired_graph_sha256",
            "program_graph_ir", "program_graph_sha256", "role_indices", "cotangent_indices",
            "requested_gradients", "buffers", "steps", "outputs", "member_graphs"}
    bounded = _bounded(plan)
    if bounded:
        keys |= {"shape_policy", "shape_bounds", "sequence_bounds"}
    if (not isinstance(plan, dict) or set(plan) != keys
            or plan["schema"] not in ("tessera.native.source_attention_reverse.v1",
                                     "tessera.native.source_attention_reverse.v2",
                                     "tessera.native.source_attention_reverse.v3",
                                     "tessera.native.source_attention_reverse.v4")
            or plan["kind"] != (("norm_attention_saved_reverse_rows" if _rows(plan)
                                 else "norm_attention_saved_reverse") + ("_bounded" if bounded else "")) or plan["execution_state"] != "artifact_only"
            or plan["target"] != "nvidia_sm120" or plan["architecture"] != "sm_120"):
        raise ValueError("source attention reverse plan schema/target differs")
    if bounded:
        for name, count in (("sequence_bounds", 2), ("shape_bounds", 7)):
            values = plan[name]
            if (not isinstance(values, list) or len(values) != count
                    or any(type(x) is not int or not 0 < x < 2**31 for x in values)):
                raise ValueError("source attention reverse capacity schema differs")
        if plan["shape_policy"] != "bounded_sequences_v1":
            raise ValueError("source attention reverse capacity policy differs")
    for stage in ("source", "paired", "program"):
        text = plan[stage + "_graph_ir"]
        if not isinstance(text, str) or not text or sha(text) != plan[stage + "_graph_sha256"]:
            raise ValueError("source attention reverse Graph digest differs")
    rows = _rows(plan)
    original = plan["source_graph_ir"]
    if (not re.search(r'(?<=[{,] )tessera.native.source_attention_reverse_request(?=[,}])', original)
            or "tessera.attention_shape_bounds" in original
            or (not bounded and "tessera.attention_sequence_bounds" in original)):
        raise ValueError("source attention reverse original export/shape policy differs")
    from .native_attention_rows import _parts
    header = _one(r'^module(?: @(?:"(?:[^"\\]|\\.)*"|' + _SYMBOL
                  + r'))? attributes \{(.*)\} \{$', original.splitlines()[0],
                  "source attention reverse canonical module attributes differ")
    fields = _parts(header)
    if bounded:
        key = "tessera.attention_sequence_bounds"
        request = [field for field in fields if field.partition(" = ")[0] == key]
        expected = key + " = array<i64: " + ", ".join(map(str, plan["sequence_bounds"])) + ">"
        if request != [expected]:
            raise ValueError("source attention reverse original sequence capacities differ")
    requests = {}
    for mode in ("reverse", "rows", "auto"):
        marker = ("tessera.native.source_attention_reverse_request" if mode == "reverse"
                  else "tessera.native.source_attention_reverse_" + mode + "_request")
        matches = [field for field in fields if field.partition(" = ")[0] == marker]
        if len(matches) > 1 or any(field != marker for field in matches):
            raise ValueError("source attention reverse policy request is not a unique unit marker")
        requests[mode] = bool(matches)
    if not requests.pop("reverse"):
        raise ValueError("source attention reverse original module request is missing")
    if (rows and not any(requests.values())) or (not rows and requests["rows"]):
        raise ValueError("source attention reverse row request/schema differs")
    # Auto selects a physical strategy in the compiler. Both sealed program
    # variants preserve the same mathematical source; replay checks the chosen
    # variant without reimplementing scheduling policy or invoking a compiler.
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
    gradients = {roles[0]: 12 if rows else 11, roles[1]: 10 if rows else 9, roles[2]: 11 if rows else 10}
    outputs = [gradients[i] for i in requested]
    if (plan["outputs"] != outputs or any(type(v) is not int for v in plan["outputs"])):
        raise ValueError("source attention reverse returned gradient roles differ")
    edges: tuple[tuple[list[int], list[int], str], ...] = (([roles[0]], [5], "tessera.rmsnorm"),
             ([5, roles[1], roles[2]], [6, 7], "tessera_attn.checkpoint_forward"),
             ([seeds[0], 5, roles[1], roles[2], 6, 7, seeds[1]], [8, 9, 10], "tessera_attn.checkpoint_backward"),
             ([roles[0], 8], [11], "native_row_region"))
    if rows:
        edges = (([roles[0]], [5], "tessera.rmsnorm"),
                 ([5, roles[1], roles[2]], [6, 7], "tessera_attn.checkpoint_forward"),
                 ([seeds[0], 5, roles[1], roles[2], 7, seeds[1]], [8],
                  "tessera_attn.checkpoint_row_correction"),
                 ([seeds[0], 5, roles[1], roles[2], 7, 8], [9, 10, 11],
                  "tessera_attn.checkpoint_backward_rows"),
                 ([roles[0], 9], [12], "native_row_region"))
    expected_steps = [dict(step=i, member=plan["root"] + "__member_" + str(i),
                     inputs=ins, outputs=outs, operation=op)
                for i, (ins, outs, op) in enumerate(edges)]
    if (plan["steps"] != expected_steps or any(type(s["step"]) is not int or any(
            type(v) is not int for v in (*s["inputs"], *s["outputs"])) for s in plan["steps"])):
        raise ValueError("source attention reverse native SSA edges differ")
    buffers = plan["buffers"]
    if not isinstance(buffers, list) or len(buffers) != (13 if rows else 12):
        raise ValueError("source attention reverse allocation count differs")
    writes, reads = [-1] * len(buffers), [-1] * len(buffers)
    for i, (ins, outs, _) in enumerate(edges):
        for index in ins:
            reads[index] = i
        for index in outs:
            writes[index] = reads[index] = i
    for index in outputs:
        reads[index] = len(edges)
    for i, b in enumerate(buffers):
        fields = {"id", "shape", "storage", "bytes", "ownership", "first_write", "last_read"}
        if bounded:
            fields |= {"trace_shape", "capacity_shape"}
        if (not isinstance(b, dict) or set(b) != fields or type(b["id"]) is not int or b["id"] != i
                or not isinstance(b["shape"], list) or not 3 <= len(b["shape"]) <= 4
                or b["storage"] not in {"f16", "bf16", "f32"}):
            raise ValueError("source attention reverse buffer schema differs")
        if bounded:
            for key in ("trace_shape", "capacity_shape"):
                shape = b[key]
                if (not isinstance(shape, list) or len(shape) != len(b["shape"])
                        or any(type(x) is not int or not 0 < x < 2**31 for x in shape)):
                    raise ValueError("source attention reverse trace/capacity shape differs")
            for axis, (logical, trace, capacity) in enumerate(zip(
                    b["shape"], b["trace_shape"], b["capacity_shape"], strict=True)):
                if (type(logical) is not int or
                        (axis == 2 and (logical != DYNAMIC_DIM or trace > capacity)) or
                        (axis != 2 and (logical != trace or logical != capacity))):
                    raise ValueError("source attention reverse logical/physical axes differ")
        elif any(type(d) is not int or not 0 < d < 2**31 for d in b["shape"]):
            raise ValueError("source attention reverse buffer schema differs")
        size = math.prod(_physical_shape(b)) * (4 if b["storage"] == "f32" else 2)
        ownership = "readonly_input" if i < 5 else "returned_output" if i in outputs else "private_scratch"
        if (size >= 2**63 or type(b["bytes"]) is not int or b["bytes"] != size
                or type(b["first_write"]) is not int or b["first_write"] != writes[i]
                or type(b["last_read"]) is not int or b["last_read"] != reads[i]
                or b["ownership"] != ownership):
            raise ValueError("source attention reverse allocation lifetime differs")
    q, k, v = (buffers[i] for i in roles)
    if any(len(b["shape"]) != 4 for b in (q, k, v)) or q["storage"] not in {"f16", "bf16"}:
        raise ValueError("source attention reverse original storage/rank differs")
    b, hq, sq, d = _physical_shape(q)
    bk, hkv, sk, dk = _physical_shape(k)
    bv, hv, sv, dv = _physical_shape(v)
    if b != bk or b != bv or hq % hkv or hv != hkv or sk != sv or d != dk or b * hq * sq >= 2**31:
        raise ValueError("source attention reverse physical dimensions differ")
    expected_shapes: list[list[int] | None] = [None] * len(buffers)
    for index, shape in zip(roles, (_physical_shape(q), _physical_shape(k), _physical_shape(v)), strict=True):
        expected_shapes[index] = shape
    for index in (seeds[0], 6):
        expected_shapes[index] = [b, hq, sq, dv]
    for index in (seeds[1], 7):
        expected_shapes[index] = [b, hq, sq]
    for index in (5, 8, 11):
        expected_shapes[index] = _physical_shape(q)
    expected_shapes[9], expected_shapes[10] = _physical_shape(k), _physical_shape(v)
    if rows:
        expected_shapes[8] = [b, hq, sq]
        expected_shapes[9] = _physical_shape(q)
        expected_shapes[10], expected_shapes[11], expected_shapes[12] = _physical_shape(k), _physical_shape(v), _physical_shape(q)
    if any(_physical_shape(item) != expected_shapes[i] or item["storage"] != (
            "f32" if i in (seeds[1], 7) or (rows and i == 8) else q["storage"]) for i, item in enumerate(buffers)):
        raise ValueError("source attention reverse private storage roles differ")
    dims = (b, hq, hkv, sq, sk, d, dv)
    if bounded:
        if plan["shape_bounds"] != list(dims) or plan["sequence_bounds"] != [sq, sk]:
            raise ValueError("source attention reverse root/capacity roles differ")
        attention_dimensions(_symbolic_dims(plan, dims), list(dims))
        key_ids = {roles[1], roles[2], gradients[roles[1]], gradients[roles[2]]}
        for i, item in enumerate(buffers):
            capacity_shape = expected_shapes[i]
            if capacity_shape is None:
                raise ValueError("source attention reverse capacity role is missing")
            expected_trace = list(capacity_shape)
            expected_trace[2] = k["trace_shape"][2] if i in key_ids else q["trace_shape"][2]
            if item["trace_shape"] != expected_trace:
                raise ValueError("source attention reverse trace sequence roles differ")
        if (math.prod(_physical_shape(buffers[6])) >= 2**31 or
                sum(math.prod(_physical_shape(buffers[gradients[i]])) for i in range(3)) > (2**31 - 1) * 128):
            raise ValueError("source attention reverse capacity launch exceeds native ABI")
        _verify_paired_trace(plan, gradients)
    _verify_calls(plan)
    return dims


def _verify_paired_trace(plan, gradients):
    paired = _functions(plan["paired_graph_ir"])
    if (len(paired) != 2 or paired[0][2] != plan["source_root"]
            or 'tessera.autodiff.role = "backward"' not in paired[1][4]
            or "tessera.autodiff.forward = @" + paired[0][2] not in paired[1][4]
            or "tessera.autodiff.paired = @" + paired[1][2] not in paired[0][4]
            or re.search(r"tensor<[^>]*\?", plan["paired_graph_ir"])):
        raise ValueError("source attention reverse original paired trace differs")
    _signature(paired[0], range(3), [3, 4, 6, 7], plan["buffers"], "trace_shape")
    _signature(paired[1], [0, 1, 2, 3, 4, 6, 7],
               [gradients[i] for i in range(3)], plan["buffers"], "trace_shape")
    expected = ", ".join(map(str, plan["sequence_bounds"]))
    if re.findall(r"tessera.attention_sequence_bounds = array<i64: ([^>]+)>",
                  plan["paired_graph_ir"]) != [expected]:
        raise ValueError("source attention reverse paired sequence capacities differ")


def _verify_calls(plan):
    buffers = plan["buffers"]
    original = _functions(plan["source_graph_ir"])
    if len(original) != 1 or original[0][2] != plan["source_root"]:
        raise ValueError("source attention reverse original function differs")
    original = original[0]
    seeds = plan["cotangent_indices"]
    _signature(original, range(3), [3, 4], buffers, "trace_shape" if _bounded(plan) else "shape")
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
    if _bounded(plan):
        expected = "(" + ", ".join(_type(buffers[i], "trace_shape") for i in (
            5, plan["role_indices"][1], plan["role_indices"][2])) + ") -> ("
        expected += ", ".join(_type(buffers[i], "trace_shape") for i in (6, 7)) + ")"
        if attention[5] != "{" + attrs + "} : " + expected:
            raise ValueError("source attention reverse original attention types differ")
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
    count = 5 if _rows(plan) else 4
    functions = _functions(plan["program_graph_ir"])
    graphs = plan["member_graphs"]
    if (len(functions) != count + 1 or functions[0][1] or functions[0][2] != plan["root"]
            or not isinstance(graphs, list) or len(graphs) != count):
        raise ValueError("source attention reverse typed call program differs")
    root = functions[0]
    _signature(root, range(5), plan["outputs"], buffers)
    lines = root[5].strip().splitlines()
    if len(lines) != count + 1:
        raise ValueError("source attention reverse root call count differs")
    values = [f"%arg{i}" for i in range(5)]
    for i, (step, line) in enumerate(zip(plan["steps"], lines[:-1], strict=True)):
        call = re.fullmatch(r'\s*(%\w+)(?::([0-9]+))? = call @(' + _SYMBOL + r')\(([^)]*)\) : (.*)', line)
        args = ", ".join(values[index] for index in step["inputs"])
        ins = "(" + ", ".join(_type(buffers[index]) for index in step["inputs"]) + ")"
        outs = ", ".join(_type(buffers[index]) for index in step["outputs"])
        result_count = len(step["outputs"])
        if result_count > 1:
            outs = "(" + outs + ")"
        if (call is None or call[3] != step["member"] or call[4] != args
                or int(call[2] or 1) != result_count or call[5] != ins + " -> " + outs):
            raise ValueError("source attention reverse root call binding differs")
        values.extend([call[1]] if result_count == 1 else [call[1] + "#" + str(j) for j in range(result_count)])
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
            expected_norm = clean(norm[2])
            if _bounded(plan):
                attrs = expected_norm.split(" : (", 1)[0]
                qid = plan["role_indices"][0]
                trace = attrs + " : (" + _type(buffers[qid], "trace_shape") + ") -> " + _type(buffers[5], "trace_shape")
                if expected_norm != trace:
                    raise ValueError("source attention reverse original normalization types differ")
                expected_norm = attrs + " : (" + _type(buffers[qid]) + ") -> " + _type(buffers[5])
                if re.findall(r"tessera.native.query_sequence_bound = ([0-9]+) : i64", graphs[i]) != [str(plan["sequence_bounds"][0])]:
                    raise ValueError("source attention reverse normalization query bound differs")
            if clean(member_norm) != expected_norm:
                raise ValueError("source attention reverse original/member normalization policy differs")
        for key, count in (("argument", len(step["inputs"])), ("result", len(step["outputs"]))):
            binding = _one(r'tessera\.' + key + r'_bindings = (\[[^\]]*\])', member[0][4],
                           "source attention reverse member names differ")
            prefix = "input_" if key == "argument" else "output_"
            if json.loads(binding) != [prefix + str(j) for j in range(count)]:
                raise ValueError("source attention reverse local member binding differs")
    returned = "return " + ", ".join(values[index] for index in plan["outputs"]) + " : "
    returned += ", ".join(_type(buffers[index]) for index in plan["outputs"])
    if lines[-1].strip() != returned:
        raise ValueError("source attention reverse root return differs")
    for text, fns in [(plan["program_graph_ir"], functions)] + [
            (graph, _functions(graph)) for graph in graphs]:
        for fn in reversed(fns):
            text = text[:fn.start()] + text[fn.end():]
        header = 'module attributes {tessera.arch = "sm_120", '
        if _bounded(plan):
            header += "tessera.attention_shape_bounds = array<i64: " + ", ".join(map(str, plan["shape_bounds"])) + ">, "
        header += 'tessera.target = "nvidia_sm120"} {'
        if not re.fullmatch(re.escape(header) + r"\s*\}\s*", text):
            raise ValueError("source attention reverse program contains unaccounted operations")


def _abi(plan, index, dims, entry):
    rows, columns = math.prod(_physical_shape(plan["buffers"][5])[:-1]), dims[5]
    cooperative = "_cooperative_128_" in entry
    sizes: tuple[int, ...] = (rows if cooperative else (rows + 127) // 128,
             (math.prod(_physical_shape(plan["buffers"][6])) + 127) // 128,
             (sum(math.prod(_physical_shape(plan["buffers"][i])) for i in (8, 9, 10)) + 127) // 128,
             (rows + 127) // 128)
    parameters: tuple[list[str], ...] = (["ptr", "ptr", "i64", "i64"], ["ptr"] * 5 + ["i64"] * 7,
                  ["ptr"] * 10 + ["i64"] * 7, ["ptr"] * 3 + ["i64"] * 2)
    if _rows(plan):
        sizes = (sizes[0], sizes[1], (math.prod(_physical_shape(plan["buffers"][7])) + 31) // 32,
                 (sum(math.prod(_physical_shape(plan["buffers"][i])) for i in (9, 10, 11)) + 127) // 128,
                 (rows + 127) // 128)
        parameters = (parameters[0], parameters[1], ["ptr"] * 7 + ["i64"] * 7,
                      ["ptr"] * 9 + ["i64"] * 7, parameters[3])
    step = plan["steps"][index]
    abi = dict(schema=("tessera.nvidia.source_attention_reverse_step.v2" if _rows(plan)
                        else "tessera.nvidia.source_attention_reverse_step.v1"), index=index,
                entry=entry, block=[32 if _rows(plan) and index == 2 else 128, 1, 1], grid=[sizes[index], 1, 1],
                dimensions=list(dims) if index in ((1, 2, 3) if _rows(plan) else (1, 2)) else [rows, columns],
                inputs=step["inputs"], outputs=step["outputs"],
                parameters=parameters[index], storage=plan["buffers"][5]["storage"])

    if _bounded(plan):
        abi.update(schema=("tessera.nvidia.source_attention_reverse_step.v4" if _rows(plan)
                           else "tessera.nvidia.source_attention_reverse_step.v3"),
                   shape_policy="bounded_sequences_v1", shape_bounds=list(dims),
                   grid_policy="actual_frame_v1", grid_capacity=abi.pop("grid"))
        abi["dimensions"] = (_symbolic_dims(plan, dims) if index in (
            (1, 2, 3) if _rows(plan) else (1, 2)) else [DYNAMIC_DIM, columns])
    return abi


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
    data = dict(schema=_portable_schema(plan), plan=copy.deepcopy(plan), members=records)
    data["contract_digest"] = digest(data)
    restore(data)
    return data


def restore(data):
    from .prepared_source_attention_reverse import ReverseMember
    if (not isinstance(data, dict) or set(data) != {"schema", "plan", "members", "contract_digest"}
            or data["schema"] not in (SCHEMA, _ROWS_SCHEMA, _BOUNDED_SCHEMA, _BOUNDED_ROWS_SCHEMA)
            or data["contract_digest"] != digest({k: v for k, v in data.items() if k != "contract_digest"})):
        raise ValueError("source attention reverse portable integrity differs")
    data = copy.deepcopy(data)
    plan, records = data["plan"], data["members"]
    dims = validate_plan(plan)
    if data["schema"] != _portable_schema(plan):
        raise ValueError("source attention reverse portable/native schema differs")
    if not isinstance(records, list) or len(records) != (5 if _rows(plan) else 4):
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
        if _bounded(plan):
            capacities = ", ".join(map(str, dims))
            for stage in _FIELDS:
                values = re.findall(r"tessera.attention_shape_bounds = array<i64: ([^>]+)>", record[stage])
                if ((stage in ("graph_ir", "schedule_ir", "tile_ir") and values != [capacities])
                        or any(value != capacities for value in values)):
                    raise ValueError("source attention reverse stage/module capacities differ")
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
        if _rows(plan) and i in (2, 3):
            checkpoints.append(_row_checkpoint(plan, i, record, dims))
        elif i in (1, 2):
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
            if _bounded(plan):
                operation = _one(r"[^\n]*tile\.norm_kernel[^\n]*", tile,
                                 "source attention reverse normalization capacity carrier differs")
                wanted = {"sequence_bound": dims[3], "row_capacity": dims[0] * dims[1] * dims[3],
                          "fixed_columns": dims[5], "row_multiple": dims[0] * dims[1]}
                if (re.findall(r'\bshape_policy = "([^"]*)"', operation) != ["bounded_query_v1"] or any(
                        re.findall(r"\b" + field + r" = ([0-9]+) : i64", operation) != [str(value)]
                        for field, value in wanted.items())):
                    raise ValueError("source attention reverse normalization capacity policy differs")
        else:
            body_hash = _one(r'\bbody_hash = "([0-9a-f]{64})"', graph,
                             "source attention reverse pullback Graph seal differs")
            source_hash = _one(r'\bsource_hash = "([0-9a-f]{64})"', tile,
                               "source attention reverse pullback source seal differs")
            tile_hash = _one(r'(?<![\w_])body_hash = "([0-9a-f]{64})"', tile,
                             "source attention reverse pullback Tile seal differs")
            parent = _one(r'tile_body_hash = "([0-9a-f]{64})"', record["target_ir"],
                          "source attention reverse pullback Target seal differs")
            rows, cols = math.prod(_physical_shape(plan["buffers"][5])[:-1]), dims[5]
            if (source_hash != body_hash or parent != tile_hash
                    or entry != "tessera_native_row_" + body_hash[:16]
                    or f"schema = {8 if _bounded(plan) else 6} : i64" not in tile or "tile.row_region_kernel" not in tile
                    or "tessera_nvidia.row_region_kernel" not in record["target_ir"]
                    or f"rows = {rows} : i64" not in tile or f"columns = {cols} : i64" not in tile):
                raise ValueError("source attention reverse pullback geometry/profile differs")
            if _bounded(plan):
                from .native_attention_rows import _parts
                raw = _one(r"tessera.native.sm120_row_producer = \{([^{}]*)\}", graph,
                           "source attention reverse bounded pullback producer differs")
                fields = {}
                for part in _parts(raw):
                    key, separator, value = part.partition(" = ")
                    if not separator or key in fields:
                        raise ValueError("source attention reverse pullback producer fields differ")
                    fields[key] = value
                expected_fields = dict(accum='"f32"', storage=json.dumps(storage),
                    body_hash=json.dumps(body_hash), rows=f"{rows} : i64",
                    columns=f"{cols} : i64", schema="8 : i64", sequence_bound=f"{dims[3]} : i64")
                if (fields != expected_fields or tile.count("tile.row_region_kernel") != 1
                        or f"row_multiple = {dims[0] * dims[1]} : i64" not in tile):
                    raise ValueError("source attention reverse pullback capacity seal differs")
                fn = _functions(graph)[0]
                text = "\n".join(line[2:] for line in fn[0].splitlines()).rstrip()
                text = text.replace("func.func @" + fn[2], "func.func private @__native_row_body", 1)
                text = re.sub(r"tessera.native.sm120_row_producer = \{[^{}]*\},? ?", "", text)
                if sha(text) != body_hash:
                    raise ValueError("source attention reverse projected pullback body differs")
        members.append(ReverseMember(graph_ir=record["graph_ir"], schedule_ir=record["schedule_ir"],
                                     tile_ir=record["tile_ir"], target_ir=record["target_ir"],
                                     native_ir=record["native_ir"], entry=entry,
                                     image=record["ptx"].encode("ascii"), image_digest=record["image_digest"]))
    if any(value != checkpoints[0] for value in checkpoints[1:]):
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
