"""Replay-checked native gfx1151 MoE token-gather Schedule contract."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import re

from .scheduled_matmul import find_tessera_opt, run_tessera_opt


@dataclass(frozen=True)
class ScheduledMoeDispatchArtifact:
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    schedule_digest: str
    names: tuple[str, str, str]
    dims: tuple[int, int, int]

    def validate(self) -> None:
        tool = find_tessera_opt()
        if tool is None:
            raise RuntimeError("MoE dispatch needs production tessera-opt")
        if run_tessera_opt(tool, self.graph_ir, "--tessera-graph-to-schedule") != self.schedule_ir:
            raise ValueError("MoE Schedule disagrees with Graph replay")
        if run_tessera_opt(tool, self.schedule_ir, "--tessera-schedule-to-tile") != self.tile_ir:
            raise ValueError("MoE Tile disagrees with Schedule replay")
        fields = ("shape = array<i64: " + ", ".join(map(str, self.dims)) + ">",
                  "bindings = " + json.dumps(self.names))
        if any(field not in self.tile_ir for field in fields):
            raise ValueError("MoE descriptor disagrees with native contract")
        if re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', self.tile_ir) != [self.schedule_digest]:
            raise ValueError("MoE schedule hash disagrees")
        if re.findall(r"llvm.func @([\w]+)\(", self.tile_ir) != [
            "tessera_tile_moe_dispatch_f32_direct"
        ]:
            raise ValueError("MoE Tile entry disagrees with runtime ABI")


def project_scheduled_moe_dispatch_graph(module, *, target: str = "rocm_gfx1151") -> str:
    if target != "rocm_gfx1151":
        raise ValueError("MoE token-gather Schedule only admits gfx1151")
    from .rocm_native import _moe_dispatch_contract

    contract = _moe_dispatch_contract(module)
    if contract is None:
        raise ValueError("MoE token gather needs static f32[T,H], i32[S] -> f32[S,H]")
    x, token, output, dims = contract
    source = copy.deepcopy(module)
    # An explicit None is the public API default, not a transport policy.
    if source.functions[0].body[0].kwargs.get("transport") is None:
        source.functions[0].body[0].kwargs.pop("transport", None)
    source.module_attrs.update({"tessera.target": '"rocm_gfx1151"',
                                "tessera.arch": '"gfx1151"'})
    source.functions[0].fn_attrs["tessera.bindings"] = json.dumps((x, token, output))
    return source.to_mlir(target=target, canonical=True)


def lower_scheduled_moe_dispatch(module, *, target: str = "rocm_gfx1151") -> ScheduledMoeDispatchArtifact:
    graph = project_scheduled_moe_dispatch_graph(module, target=target)
    from .rocm_native import _moe_dispatch_contract

    contract = _moe_dispatch_contract(module)
    if contract is None:
        raise ValueError("MoE dispatch lost its admitted Graph contract")
    x, token, output, dims = contract
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("MoE dispatch needs production tessera-opt")
    schedule = run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    hashes = re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', tile)
    if len(hashes) != 1:
        raise RuntimeError("MoE lowering lost its unique Schedule hash")
    artifact = ScheduledMoeDispatchArtifact(graph, schedule, tile, hashes[0],
                                            (x, token, output), dims)
    artifact.validate()
    return artifact
