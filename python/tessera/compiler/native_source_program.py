"""Bind native SSA program exports without rebuilding their numerical Graph.

This is an artifact transport. It does not admit execution or own device buffers.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
import re

from .rocm_native import _tool_digest
from .scheduled_matmul import find_tessera_opt, run_tessera_opt


@dataclass(frozen=True)
class NativeSourceMember:
    graph_ir: str
    schedule_ir: str
    tile_ir: str


@dataclass(frozen=True)
class NativeSourceProgram:
    source_ir: str
    outlined_ir: str
    tool: Path
    compiler_identity: str

    @property
    def manifest(self) -> dict:
        records = re.findall(
            r'tessera.native.source_program_json = "([^"]+)"', self.outlined_ir)
        if len(records) != 1:
            raise ValueError("native source export requires one program record")
        plan = json.loads(base64.b64decode(records[0], validate=True))
        if (not isinstance(plan, dict)
                or plan.get("schema") != "tessera.native.source_program.v1"
                or plan.get("execution_state") != "artifact_only"):
            raise ValueError("native source export schema or execution state differs")
        return plan

    def validate(self) -> None:
        if _tool_digest(self.tool) != self.compiler_identity:
            raise RuntimeError("native source compiler changed after export")
        # Replay verifies member SSA, original policies, roles, result order and
        # lifetimes together. No Python reconstruction is an alternate authority.
        replay = run_tessera_opt(
            self.tool, self.source_ir, "--tessera-outline-source-program")
        if replay != self.outlined_ir:
            raise ValueError("native source export differs from retained source replay")
        _ = self.manifest
        if _tool_digest(self.tool) != self.compiler_identity:
            raise RuntimeError("native source compiler changed during replay")

    def scheduled_members(self) -> tuple[NativeSourceMember, ...]:
        self.validate()
        members = []
        for graph in self.manifest["member_graphs"]:
            schedule = run_tessera_opt(
                self.tool, graph, "--tessera-graph-to-schedule")
            tile = run_tessera_opt(
                self.tool, schedule, "--tessera-schedule-to-tile")
            members.append(NativeSourceMember(graph, schedule, tile))
        return tuple(members)


    def scheduled_artifacts(self):
        """Project metadata from retained native members, never reauthor Graph."""
        import math
        import struct
        from .nvidia_native import _mlir_f32_bits
        from .scheduled_kernel import ScheduledKernelArtifact
        from .scheduled_checkpoint import _decode_checkpoint

        norm, attention = self.scheduled_members()
        plan = self.manifest

        def one(pattern: str) -> str:
            values = re.findall(pattern, norm.tile_ir)
            if len(values) != 1:
                raise ValueError("native norm member lost its unique physical field")
            return values[0]

        source = plan["buffers"][plan["steps"][0]["inputs"][0]]
        result = plan["buffers"][3]
        shape = tuple(source["shape"])
        storage = source["storage"]
        epsilon = struct.unpack("f", _mlir_f32_bits(
            one(r'tessera.norm_epsilon = ([^ ]+) : f32')))[0]
        artifact = ScheduledKernelArtifact(
            graph_ir=norm.graph_ir, schedule_ir=norm.schedule_ir,
            tile_ir=norm.tile_ir, target=plan["target"],
            architecture=plan["architecture"],
            function_name=one(r'\bllvm.func @(\w+)\('),
            family="norm", kind="rmsnorm", input_name="input_0",
            output_name="output_0", input_shape=shape,
            output_shape=tuple(result["shape"]),
            dtype={"f16": "fp16", "bf16": "bf16"}[storage],
            storage=storage, accum="f32", axis=-1, keepdims=False,
            rows=math.prod(shape[:-1]), columns=shape[-1],
            outer=math.prod(shape[:-1]), axis_extent=shape[-1], inner=1,
            workgroup_size=int(one(r'tessera.workgroup_size = (\d+) : i64')),
            schedule_digest=one(r'tessera.schedule_hash = "([0-9a-f]{64})"'),
            schedule=one(r'\bschedule = "([^"]+)"'), epsilon=epsilon)
        artifact.validate()
        checkpoint = _decode_checkpoint(
            attention.graph_ir, attention.schedule_ir, attention.tile_ir,
            backward=False)
        checkpoint.validate()
        return artifact, checkpoint

    def prepare(self, *, argument_names=("input_0", "input_1", "input_2")):
        """Bind both compiler-owned images to one native private-buffer frame."""
        from .prepared_source_attention import PreparedSourceAttention
        return PreparedSourceAttention(self, argument_names=argument_names)

    def package_members(self, *, pipeline_name: str):
        """Compile images only; a checked composed native owner is still required."""
        from .nvidia_native import (
            package_scheduled_kernel, package_scheduled_checkpoint)
        norm, attention = self.scheduled_artifacts()
        packages = (
            package_scheduled_kernel(norm, pipeline_name=pipeline_name),
            package_scheduled_checkpoint(attention, pipeline_name=pipeline_name),
        )
        if _tool_digest(self.tool) != self.compiler_identity:
            raise RuntimeError("native source compiler changed during member packaging")
        return packages


def outline_source_program(source_ir: str, *, tool: Path | None = None) -> NativeSourceProgram:
    compiler = tool if tool is not None else find_tessera_opt()
    if compiler is None:
        raise RuntimeError("native source program requires production tessera-opt")
    compiler = Path(compiler).resolve()
    identity = _tool_digest(compiler)
    outlined = run_tessera_opt(
        compiler, source_ir, "--tessera-outline-source-program")
    program = NativeSourceProgram(source_ir, outlined, compiler, identity)
    program.validate()
    return program
