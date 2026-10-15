"""Compiler-owned two-result normalization JVP member projection.

An outline is an artifact witness, not an executable package. Compilation must
consume these exact member Graphs through Schedule, Tile, and native lowering.
"""
from dataclasses import dataclass
import base64
import json
from pathlib import Path
import re

from .rocm_native import _tool_digest
from .scheduled_matmul import find_tessera_opt, run_tessera_opt

SCHEMA = "tessera.native.sm120_active_source_outline.v1"


def validate_plan(text):
    """Validate allocation/dataflow metadata without loading a GPU provider.

    This checks the portable frame, not authenticity of arbitrary caller code.
    A compiled package additionally binds each Graph and native image witness.
    """
    if not isinstance(text, str):
        raise ValueError("source-active plan requires serialized JSON")
    plan = json.loads(text)
    keys = {"schema", "kind", "execution_state", "source_graph_ir", "paired_graph_ir",
            "root", "role_indices", "buffers", "steps", "outputs", "member_graphs"}
    if (not isinstance(plan, dict) or set(plan) != keys or plan["schema"] != SCHEMA
            or plan["kind"] != "paired_active_source_jvp"
            or plan["execution_state"] != "artifact_only"):
        raise ValueError("source-active outline schema differs")
    roles = plan["role_indices"]
    if (not isinstance(roles, list) or len(roles) != 3
            or any(type(v) is not int for v in roles)
            or sorted(roles[:2]) != [0, 1] or roles[2] != 2):
        raise ValueError("source-active original argument roles differ")
    if (not isinstance(plan["root"], str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.$]*", plan["root"])
            or any(not isinstance(plan[k], str) or not plan[k]
                   for k in ("source_graph_ir", "paired_graph_ir"))):
        raise ValueError("source-active Graph witnesses differ")
    graphs = plan["member_graphs"]
    if (not isinstance(graphs, list) or len(graphs) != 3
            or any(not isinstance(g, str) or not g for g in graphs)):
        raise ValueError("source-active member Graph count differs")
    steps = plan["steps"]
    expected_inputs = [[roles[0], 2], [3, roles[1]], [4, roles[1]]]
    expected_outputs = [[3, 4], [5], [6]]
    if not isinstance(steps, list) or len(steps) != 3:
        raise ValueError("source-active step count differs")
    for i, step in enumerate(steps):
        if (not isinstance(step, dict)
                or set(step) != {"step", "operation", "member", "inputs", "outputs"}
                or type(step["step"]) is not int or step["step"] != i
                or step["operation"] != ("native_row_region" if i == 0 else "tessera.matmul")
                or step["member"] != plan["root"] + "__tensor_member_" + str(i)
                or step["inputs"] != expected_inputs[i] or step["outputs"] != expected_outputs[i]
                or any(type(v) is not int for v in [*step["inputs"], *step["outputs"]])):
            raise ValueError("source-active member dataflow differs")
    if (plan["outputs"] != [5, 6]
            or any(type(v) is not int for v in plan["outputs"])):
        raise ValueError("source-active returned pair differs")
    buffers = plan["buffers"]
    if not isinstance(buffers, list) or len(buffers) != 7:
        raise ValueError("source-active buffer count differs")
    for i, b in enumerate(buffers):
        if (not isinstance(b, dict)
                or set(b) != {"id", "bytes", "shape", "storage", "ownership", "first_write", "last_read"}
                or type(b["id"]) is not int or b["id"] != i
                or not isinstance(b["shape"], list) or len(b["shape"]) != 2
                or any(type(v) is not int or not 0 < v < 2**31 for v in b["shape"])
                or b["storage"] not in {"f16", "bf16", "f32"}):
            raise ValueError("source-active buffer shape/storage differs")
        size = b["shape"][0] * b["shape"][1] * (4 if b["storage"] == "f32" else 2)
        if type(b["bytes"]) is not int or b["bytes"] != size or size > 2**63 - 1:
            raise ValueError("source-active buffer bytes differ")
        write = next((s["step"] for s in steps if i in s["outputs"]), -1)
        read = max([s["step"] for s in steps if i in s["inputs"]] + [write])
        if i in (5, 6):
            read = len(steps)
        ownership = "readonly_input" if i < 3 else "returned_output" if i > 4 else "private_scratch"
        if (type(b["first_write"]) is not int or b["first_write"] != write
                or type(b["last_read"]) is not int or b["last_read"] != read
                or b["ownership"] != ownership):
            raise ValueError("source-active buffer lifetime differs")
    source, rhs = buffers[roles[0]], buffers[roles[1]]
    m, k = source["shape"]
    n = rhs["shape"][1]
    if (source["storage"] not in {"f16", "bf16"}
            or rhs["shape"][0] != k or rhs["storage"] != source["storage"]
            or m > 65535 * 16
            or any(buffers[i]["shape"] != [m, k] or buffers[i]["storage"] != source["storage"]
                   for i in (2, 3, 4))
            or any(buffers[i]["shape"] != [m, n] or buffers[i]["storage"] != "f32"
                   for i in (5, 6))):
        raise ValueError("source-active producer/consumer envelope differs")
    # The allocation owner aligns every private edge to 256 bytes.
    if sum((b["bytes"] + 255) // 256 * 256 for b in buffers) > 2**63 - 1:
        raise ValueError("source-active workspace capacity overflows")
    return plan


@dataclass(frozen=True)
class NativeActiveSourceProgram:
    source_ir: str
    outlined_ir: str
    plan_json: str
    tool: Path
    compiler_identity: str

    @property
    def manifest(self):
        return validate_plan(self.plan_json)

    def project_member(self, index):
        plan = self.manifest
        if type(index) is not int or not 0 <= index < 3:
            raise ValueError("source-active member index differs")
        if _tool_digest(self.tool) != self.compiler_identity:
            raise RuntimeError("source-active compiler changed after export")
        graph = run_tessera_opt(self.tool, self.outlined_ir,
            f"--tessera-autodiff-forward=export-scaled-program=true select-scaled-member={index}")
        if graph != plan["member_graphs"][index]:
            raise ValueError("source-active projection differs from native witness")
        if _tool_digest(self.tool) != self.compiler_identity:
            raise RuntimeError("source-active compiler changed during projection")
        return graph

    def lower_members(self):
        """Retain every actual compiler boundary; never synthesize Tile bodies."""
        result = []
        cache = {}
        for index in range(3):
            graph = self.project_member(index)
            if graph not in cache:
                schedule = run_tessera_opt(self.tool, graph, "--tessera-graph-to-schedule")
                tile = run_tessera_opt(self.tool, schedule, "--tessera-schedule-to-tile")
                target = run_tessera_opt(self.tool, tile, "--lower-tile-to-nvidia=sm=120")
                native = run_tessera_opt(self.tool, target, "--lower-tessera-nvidia-to-nvvm")
                cache[graph] = (graph, schedule, tile, target, native)
            result.append(cache[graph])
        if _tool_digest(self.tool) != self.compiler_identity:
            raise RuntimeError("source-active compiler changed during lowering")
        return tuple(result)


def export_native_active_source_jvp(source, *, tool=None):
    if not isinstance(source, str):
        raise TypeError("source-active export consumes frontend Graph MLIR")
    tool = find_tessera_opt() if tool is None else Path(tool)
    if tool is None:
        raise RuntimeError("source-active export requires matching compiler")
    identity = _tool_digest(tool)
    outlined = run_tessera_opt(tool, source, "--tessera-autodiff-forward=export-scaled-program=true")
    records = re.findall(r'tessera.native.sm120_tensor_program_json = "([^"]+)"', outlined)
    if len(records) != 1:
        raise ValueError("source-active export requires one native program record")
    text = base64.b64decode(records[0], validate=True).decode()
    validate_plan(text)
    if _tool_digest(tool) != identity:
        raise RuntimeError("source-active compiler changed during export")
    return NativeActiveSourceProgram(source, outlined, text, tool, identity)
