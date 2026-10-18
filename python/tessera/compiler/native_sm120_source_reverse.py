"""Native source-pullback outline metadata; not an executable VJP package."""
import base64
from dataclasses import dataclass
import json
from pathlib import Path
import re

from .rocm_native import _tool_digest
from .scheduled_matmul import find_tessera_opt, run_tessera_opt

SCHEMA = "tessera.native.sm120_source_reverse_outline.v1"


def validate_plan(text):
    plan = json.loads(text)
    keys = {"schema", "kind", "execution_state", "source_graph_ir", "paired_graph_ir",
            "root", "role_indices", "buffers", "steps", "outputs", "member_graphs"}
    if (not isinstance(plan, dict) or set(plan) != keys or plan["schema"] != SCHEMA
            or plan["kind"] != "paired_source_reverse" or plan["execution_state"] != "artifact_only"):
        raise ValueError("source reverse outline schema differs")
    roles = plan["role_indices"]
    if (not isinstance(roles, list) or len(roles) != 3
            or any(type(v) is not int for v in roles)
            or sorted(roles[:2]) != [0, 1] or roles[2] != 2):
        raise ValueError("source reverse root roles differ")
    if (not isinstance(plan["root"], str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.$]*", plan["root"])
            or any(not isinstance(plan[k], str) or not plan[k]
                   for k in ("source_graph_ir", "paired_graph_ir"))
            or not isinstance(plan["member_graphs"], list) or len(plan["member_graphs"]) != 2
            or any(not isinstance(g, str) or not g for g in plan["member_graphs"])):
        raise ValueError("source reverse Graph witnesses differ")
    steps = plan["steps"]
    if not isinstance(steps, list) or len(steps) != 2:
        raise ValueError("source reverse step count differs")
    for i, (inputs, outputs, operation) in enumerate((
        ([2, roles[1]], [3], "tessera.matmul"),
        ([roles[0], 3], [4], "native_row_region"),
    )):
        expected = {"step": i, "inputs": inputs, "outputs": outputs, "operation": operation,
                    "member": plan["root"] + "__reverse_member_" + str(i)}
        step = steps[i]
        if (not isinstance(step, dict) or step != expected or type(step.get("step")) is not int
                or any(type(v) is not int for v in (*step["inputs"], *step["outputs"]))):
            raise ValueError("source reverse member SSA edges differ")
    if plan["outputs"] != [4] or type(plan["outputs"][0]) is not int:
        raise ValueError("source reverse requested gradient differs")
    buffers = plan["buffers"]
    if not isinstance(buffers, list) or len(buffers) != 5:
        raise ValueError("source reverse buffer count differs")
    for i, record in enumerate(buffers):
        if (not isinstance(record, dict)
                or set(record) != {"id", "bytes", "shape", "storage", "ownership", "first_write", "last_read"}
                or type(record["id"]) is not int or record["id"] != i
                or not isinstance(record["shape"], list) or len(record["shape"]) != 2
                or any(type(d) is not int or not 0 < d < 2**31 for d in record["shape"])
                or record["storage"] not in {"f16", "bf16", "f32"}):
            raise ValueError("source reverse buffer metadata differs")
        size = record["shape"][0] * record["shape"][1] * (4 if record["storage"] == "f32" else 2)
        write = -1 if i < 3 else i - 3
        read = 1 if i == roles[0] or i == 3 else 0 if i < 3 else 2
        ownership = "readonly_input" if i < 3 else "private_scratch" if i == 3 else "returned_output"
        if (type(record["bytes"]) is not int or record["bytes"] != size or size > 2**63 - 1
                or type(record["first_write"]) is not int or record["first_write"] != write
                or type(record["last_read"]) is not int or record["last_read"] != read
                or record["ownership"] != ownership):
            raise ValueError("source reverse allocation lifetime differs")
    source, rhs, seed = buffers[roles[0]], buffers[roles[1]], buffers[2]
    m, k = source["shape"]
    n = rhs["shape"][1]
    if (source["storage"] not in {"f16", "bf16"} or rhs["storage"] != source["storage"]
            or rhs["shape"][0] != k or m > 65535 * 16 or seed["shape"] != [m, n]
            or seed["storage"] != "f32" or any(
                buffers[i]["shape"] != [m, k] or buffers[i]["storage"] != source["storage"]
                for i in (3, 4))
            or sum((b["bytes"] + 255) // 256 * 256 for b in buffers) > 2**63 - 1):
        raise ValueError("source reverse typed contraction envelope differs")
    return plan


@dataclass(frozen=True)
class NativeSourceReverseProgram:
    source_ir: str
    outlined_ir: str
    plan_json: str
    tool: Path
    compiler_identity: str

    @property
    def manifest(self):
        return validate_plan(self.plan_json)

    def lower_members(self):
        """Consume exact native-outlined Graphs; no Python Tile construction."""
        plan = self.manifest
        if _tool_digest(self.tool) != self.compiler_identity:
            raise RuntimeError("source reverse compiler changed after export")
        result = []
        for graph in plan["member_graphs"]:
            schedule = run_tessera_opt(self.tool, graph, "--tessera-graph-to-schedule")
            tile = run_tessera_opt(self.tool, schedule, "--tessera-schedule-to-tile")
            target = run_tessera_opt(self.tool, tile, "--lower-tile-to-nvidia=sm=120")
            native = run_tessera_opt(self.tool, target, "--lower-tessera-nvidia-to-nvvm")
            result.append((graph, schedule, tile, target, native))
        if _tool_digest(self.tool) != self.compiler_identity:
            raise RuntimeError("source reverse compiler changed during lowering")
        return tuple(result)


def export_native_source_reverse(source, *, tool=None):
    if not isinstance(source, str):
        raise TypeError("source reverse export consumes frontend Graph MLIR")
    tool = find_tessera_opt() if tool is None else Path(tool)
    if tool is None:
        raise RuntimeError("source reverse export requires matching compiler")
    identity = _tool_digest(tool)
    canonical_source = run_tessera_opt(tool, source, "--verify-each")
    outlined = run_tessera_opt(tool, source, "--tessera-autodiff-paired")
    records = re.findall(r'tessera.native.sm120_reverse_program_json = "([^"]+)"', outlined)
    if len(records) != 1:
        raise ValueError("source reverse export requires one native program record")
    text = base64.b64decode(records[0], validate=True).decode()
    plan = validate_plan(text)
    if plan["source_graph_ir"].strip() != canonical_source.strip():
        raise ValueError("source reverse outline does not bind this exact canonical input")
    if _tool_digest(tool) != identity:
        raise RuntimeError("source reverse compiler changed during export")
    return NativeSourceReverseProgram(plan["source_graph_ir"], outlined, text, tool, identity)
