"""Thin adapter for compiler-owned NVFP4 Graph partition; no Graph reconstruction."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
import re

from .rocm_native import _tool_digest
from .scheduled_matmul import find_tessera_opt, run_tessera_opt


def _plan_json(ir: str) -> str:
    matches = re.findall(r'tessera.native.nvfp4_program_json = "([^"]+)"', ir)
    if len(matches) != 1:
        raise ValueError("native NVFP4 export requires one compiler program record")
    text = base64.b64decode(matches[0], validate=True).decode()
    if json.loads(text).get("schema") not in {"tessera.native.nvfp4_program.v1", "tessera.native.nvfp4_program.v2"}:
        raise ValueError("unsupported native NVFP4 program schema")
    return text


@dataclass(frozen=True)
class NativeNVFP4GraphProgram:
    source_ir: str
    outlined_ir: str
    tool: Path
    compiler_identity: str
    plan_json: str

    @property
    def manifest(self) -> dict:
        # Detached metadata; caller mutation cannot change the captured export.
        return json.loads(self.plan_json)

    def project_member(self, index: int) -> str:
        if type(index) is not int or index not in (0, 1, 2):
            raise ValueError("native NVFP4 member index must be 0, 1 or 2")
        if _tool_digest(self.tool) != self.compiler_identity:
            raise RuntimeError("native NVFP4 compiler changed after program export")
        ir = run_tessera_opt(
            self.tool, self.source_ir,
            f"--tessera-autodiff-forward=export-scaled-primal=true select-scaled-member={index}")
        if ir != self.manifest["member_graphs"][index]:
            raise ValueError("native NVFP4 member differs from original compiler program")
        return ir


def export_native_nvfp4_program(source_ir: str) -> NativeNVFP4GraphProgram:
    if not isinstance(source_ir, str):
        raise TypeError("native NVFP4 export consumes frontend-authored MLIR text")
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("native NVFP4 export requires the matching compiler")
    identity = _tool_digest(tool)
    ir = run_tessera_opt(
        tool, source_ir, "--tessera-autodiff-forward=export-scaled-primal=true")
    if _tool_digest(tool) != identity:
        raise RuntimeError("native NVFP4 compiler changed during program export")
    return NativeNVFP4GraphProgram(source_ir, ir, tool, identity, _plan_json(ir))
