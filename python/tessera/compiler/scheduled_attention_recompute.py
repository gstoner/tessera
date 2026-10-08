"""Original NVIDIA recompute Graph lowered by native MLIR passes."""
from __future__ import annotations

import copy
from dataclasses import dataclass, fields
import hashlib
import json
import math
import re
import struct

from .scheduled_matmul import find_tessera_opt, run_tessera_opt


def _native_fields(tile):
    match = re.search(r'tessera.native_contract = \{([^\n]*)\}', tile)
    if match is None:
        raise ValueError("native recompute contract is missing")
    text = match[1]
    def string(name):
        found = re.search(r'\b' + name + r' = "([^"]*)"', text)
        if found is None:
            raise ValueError(f"native recompute {name} is missing")
        return found[1]
    def number(name, floating=False):
        found = re.search(r'\b' + name + r' = ([^ ,}]+) : ' + ("f32" if floating else "i64"), text)
        if found is None:
            raise ValueError(f"native recompute {name} is missing")
        value = found[1]
        if not floating:
            return int(value)
        return struct.unpack(">d", int(value, 16).to_bytes(8, "big"))[0] if value.startswith("0x") else float(value)
    def boolean(name):
        found = re.search(r'\b' + name + r' = (true|false)', text)
        if found is None:
            raise ValueError(f"native recompute {name} is missing")
        return found[1] == "true"
    def names(name):
        found = re.search(r'\b' + name + r' = (\[[^\]]*\])', text)
        if found is None:
            raise ValueError(f"native recompute {name} is missing")
        return tuple(json.loads(found[1]))
    if (string("family") != "attention_backward_recompute" or string("target") != "nvidia_sm120"
            or string("arch") != "sm_120" or string("route") != "deterministic_direct"
            or string("lse_checkpoint") != "recompute" or string("workspace_owner") != "output_element"
            or string("mask_alignment") != "end_aligned_v1" or number("workspace_bytes") != 0
            or not boolean("deterministic")):
        raise ValueError("native recompute target/route contract differs")
    shape = re.search(r'\bshape = array<i64: ([0-9, ]+)>', text)
    entries = re.findall(r'llvm.func @([\w]+)\(', tile)
    hashes = re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', tile)
    if shape is None or len(entries) != 1 or len(hashes) != 1:
        raise ValueError("native recompute requires one entry, shape and Schedule hash")
    return dict(entry=entries[0], schedule_digest=hashes[0],
        input_names=names("arguments"), output_names=names("results"),
        dims=tuple(map(int, shape[1].split(","))), storage=string("storage"),
        scale=number("scale", True), causal=boolean("causal"), bias=boolean("bias"),
        window_left=number("window_left"), window_right=number("window_right"),
        softcap=number("softcap", True), dropout_p=number("dropout_p", True),
        dropout_seed=number("dropout_seed"))


@dataclass(frozen=True)
class NativeAttentionRecomputeArtifact:
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    entry: str
    schedule_digest: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    dims: tuple[int, ...]
    storage: str
    scale: float
    causal: bool
    bias: bool
    window_left: int
    window_right: int
    softcap: float
    dropout_p: float
    dropout_seed: int

    def validate(self):
        if (self.storage not in {"f16", "bf16", "f32"} or len(self.dims) != 7
                or any(type(d) is not int or d <= 0 for d in self.dims)
                or type(self.bias) is not bool or type(self.causal) is not bool
                or len(self.input_names) != 4 + int(self.bias) or len(self.output_names) != 3
                or any(not isinstance(n, str) or not n for n in self.input_names + self.output_names)
                or len(set(self.input_names + self.output_names)) != len(self.input_names) + 3):
            raise ValueError("native recompute binding, dtype or shape metadata differs")
        for value in (self.scale, self.softcap, self.dropout_p):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("native recompute numeric policy must be finite")
        if self.scale <= 0 or self.softcap < 0 or not 0 <= self.dropout_p < 1:
            raise ValueError("native recompute numeric policy is outside its bounds")
        if any(type(n) is not int for n in (self.window_left, self.window_right, self.dropout_seed)):
            raise ValueError("native recompute integer policy differs")
        if self.window_left < -1 or self.window_right < -1:
            raise ValueError("native recompute window policy is outside its bounds")
        tool = find_tessera_opt()
        if tool is None:
            raise RuntimeError("native recompute validation requires production tessera-opt")
        if run_tessera_opt(tool, self.graph_ir, "--tessera-graph-to-schedule") != self.schedule_ir:
            raise ValueError("native recompute Schedule disagrees with original Graph replay")
        if run_tessera_opt(tool, self.schedule_ir, "--tessera-schedule-to-tile") != self.tile_ir:
            raise ValueError("native recompute Tile disagrees with Schedule replay")
        expected = {field.name: getattr(self, field.name) for field in fields(self)
                    if field.name not in {"graph_ir", "schedule_ir", "tile_ir"}}
        if _native_fields(self.tile_ir) != expected:
            raise ValueError("native recompute projection differs from sealed contract")

    @property
    def graph_digest(self):
        return hashlib.sha256(self.graph_ir.encode()).hexdigest()

    @property
    def schedule_ir_digest(self):
        return hashlib.sha256(self.schedule_ir.encode()).hexdigest()


def lower_native_attention_recompute(module):
    if len(module.functions) != 1:
        raise ValueError("native recompute requires one Graph function")
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("native recompute lowering requires production tessera-opt")
    targeted = copy.deepcopy(module)
    fn = targeted.functions[0]
    fn.fn_attrs["tessera.argument_bindings"] = json.dumps([arg.name for arg in fn.args])
    fn.fn_attrs["tessera.result_bindings"] = json.dumps(
        [name.removeprefix("%") for name in fn.return_values])
    targeted.module_attrs["tessera.target"] = '"nvidia_sm120"'
    targeted.module_attrs["tessera.arch"] = '"sm_120"'
    graph = targeted.to_mlir(target="nvidia_sm120", canonical=True)
    schedule = run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    artifact = NativeAttentionRecomputeArtifact(graph, schedule, tile, **_native_fields(tile))
    artifact.validate()
    return artifact
