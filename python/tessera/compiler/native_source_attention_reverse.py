"""Transport native original-source attention reverse plans without Graph synthesis."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .native_source_program import NativeSourceMember
from .rocm_native import _tool_digest
from .scheduled_matmul import find_tessera_opt, run_tessera_opt

SCHEMA = "tessera.native.source_attention_reverse.v1"
_SCHEMA_KINDS = {
    SCHEMA: "norm_attention_saved_reverse",
    "tessera.native.source_attention_reverse.v2": "norm_attention_saved_reverse_rows",
    "tessera.native.source_attention_reverse.v3": "norm_attention_saved_reverse_bounded",
    "tessera.native.source_attention_reverse.v4": "norm_attention_saved_reverse_rows_bounded",
}
_ATTRIBUTE = "tessera.native.source_attention_reverse_program_json"


@dataclass(frozen=True)
class NativeSourceAttentionReverse:
    source_ir: str
    outlined_ir: str
    tool: Path
    compiler_identity: str

    @property
    def manifest(self) -> dict:
        records = re.findall(re.escape(_ATTRIBUTE) + r' = "([^"]+)"', self.outlined_ir)
        if len(records) != 1:
            raise ValueError("native source-attention reverse requires one native plan")
        plan = json.loads(base64.b64decode(records[0], validate=True))
        if (not isinstance(plan, dict) or not isinstance(plan.get("schema"), str)
                or plan["schema"] not in _SCHEMA_KINDS
                or plan.get("kind") != _SCHEMA_KINDS[plan["schema"]]
                or plan.get("execution_state") != "artifact_only"
                or plan.get("target") != "nvidia_sm120" or plan.get("architecture") != "sm_120"):
            raise ValueError("native source-attention reverse plan contract differs")
        for name in ("source", "paired", "program"):
            text = plan.get(name + "_graph_ir")
            if (not isinstance(text, str) or not text.strip()
                    or hashlib.sha256(text.encode()).hexdigest() != plan.get(name + "_graph_sha256")):
                raise ValueError("native source-attention reverse Graph witness differs")
        if plan["source_graph_ir"] != self.source_ir:
            raise ValueError("native source-attention reverse binds a different original source")
        return plan

    def validate(self) -> None:
        if _tool_digest(self.tool) != self.compiler_identity:
            raise RuntimeError("native source-attention reverse compiler changed after export")
        _ = self.manifest
        replay = run_tessera_opt(self.tool, self.source_ir, "--tessera-autodiff-paired")
        if replay != self.outlined_ir:
            raise ValueError("native source-attention reverse plan differs from original-source replay")
        if _tool_digest(self.tool) != self.compiler_identity:
            raise RuntimeError("native source-attention reverse compiler changed during replay")

    def scheduled_members(self) -> tuple[NativeSourceMember, ...]:
        self.validate()
        members = []
        for graph in self.manifest["member_graphs"]:
            schedule = run_tessera_opt(self.tool, graph, "--tessera-graph-to-schedule")
            tile = run_tessera_opt(self.tool, schedule, "--tessera-schedule-to-tile")
            members.append(NativeSourceMember(graph, schedule, tile))
        if _tool_digest(self.tool) != self.compiler_identity:
            raise RuntimeError("native source-attention reverse compiler changed during scheduling")
        return tuple(members)

    def lower_members(self):
        result = []
        for member in self.scheduled_members():
            target = run_tessera_opt(self.tool, member.tile_ir, "--lower-tile-to-nvidia=sm=120")
            native = run_tessera_opt(self.tool, target, "--lower-tessera-nvidia-to-nvvm")
            result.append((member.graph_ir, member.schedule_ir, member.tile_ir, target, native))
        if _tool_digest(self.tool) != self.compiler_identity:
            raise RuntimeError("native source-attention reverse compiler changed during lowering")
        return tuple(result)


def export_native_source_attention_reverse(source: str, *, tool: Path | None = None) -> NativeSourceAttentionReverse:
    if not isinstance(source, str) or not source.strip():
        raise TypeError("native source-attention reverse consumes nonempty frontend Graph MLIR")
    compiler = tool if tool is not None else find_tessera_opt()
    if compiler is None:
        raise RuntimeError("native source-attention reverse requires the matching compiler")
    compiler = Path(compiler).resolve()
    identity = _tool_digest(compiler)
    canonical = run_tessera_opt(compiler, source, "--verify-each")
    outlined = run_tessera_opt(compiler, canonical, "--tessera-autodiff-paired")
    program = NativeSourceAttentionReverse(canonical, outlined, compiler, identity)
    program.validate()
    return program
