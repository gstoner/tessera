"""Native paged read contract; Python owns bindings, not Tile construction."""

from dataclasses import dataclass
import json
import re

from .scheduled_matmul import find_tessera_opt, run_tessera_opt


@dataclass(frozen=True)
class ScheduledPagedKVArtifact:
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    schedule_digest: str
    names: tuple[str, str, str]
    dims: tuple[int, ...]
    entry: str = "tessera_tile_paged_kv_read_f32_direct"

    def validate(self) -> None:
        tool = find_tessera_opt()
        if tool is None:
            raise RuntimeError("paged read requires native Schedule replay")
        if run_tessera_opt(tool, self.schedule_ir, "--tessera-schedule-to-tile") != self.tile_ir:
            raise ValueError("paged Tile IR disagrees with native Schedule replay")
        if len(self.dims) != 7 or any(type(d) is not int for d in self.dims):
            raise ValueError("paged dimensions must be integers")
        fields = ("shape = array<i64: " + ", ".join(map(str, self.dims)) + ">", "bindings = " + json.dumps(self.names))
        if any(field not in self.tile_ir for field in fields):
            raise ValueError("paged descriptor disagrees with native contract")
        if re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', self.tile_ir) != [self.schedule_digest]:
            raise ValueError("paged schedule hash disagrees")
        if self.entry != "tessera_tile_paged_kv_read_f32_direct" or re.findall(
            r"llvm.func @([\w]+)\(", self.tile_ir
        ) != [self.entry]:
            raise ValueError("paged entry disagrees with runtime ABI")


def lower_scheduled_paged_kv(names: tuple[str, str, str], dims: tuple[int, ...]) -> ScheduledPagedKVArtifact:
    if len(dims) != 7 or any(type(d) is not int for d in dims):
        raise ValueError("paged dimensions must be integers")
    if len(names) != 3 or len(set(names)) != 3 or any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", n) for n in names):
        raise ValueError("paged bindings must be unique identifiers")
    p, lp, ps, h, d, start, tokens = dims
    graph = f"""module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"}} {{
      func.func @paged_read(%pages: tensor<{p}x{ps}x{h}x{d}xf32>, %table: tensor<{lp}xi32>) -> tensor<{tokens}x{h}x{d}xf32>
          attributes {{tessera.bindings = {json.dumps(names)}}} {{
        %out = tessera.kv_cache.read %pages, %table {{start = {start} : i64, end = {start + tokens} : i64}}
          : (tensor<{p}x{ps}x{h}x{d}xf32>, tensor<{lp}xi32>) -> tensor<{tokens}x{h}x{d}xf32>
        return %out : tensor<{tokens}x{h}x{d}xf32>
      }}
    }}"""
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("paged read requires production tessera-opt")
    schedule = run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    hashes = re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', tile)
    if len(hashes) != 1:
        raise RuntimeError("paged lowering lost its unique schedule hash")
    artifact = ScheduledPagedKVArtifact(graph, schedule, tile, hashes[0], names, dims)
    artifact.validate()
    return artifact
