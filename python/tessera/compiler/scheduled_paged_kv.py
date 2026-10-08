"""Native paged read contract; Python owns bindings, not Tile construction."""

from dataclasses import dataclass
import copy
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
    page_layout: str = "row_major"

    def validate(self) -> None:
        tool = find_tessera_opt()
        if tool is None:
            raise RuntimeError("paged read requires native Schedule replay")
        if run_tessera_opt(tool, self.graph_ir, "--tessera-graph-to-schedule") != self.schedule_ir:
            raise ValueError("paged Schedule IR disagrees with Graph replay")
        if run_tessera_opt(tool, self.schedule_ir, "--tessera-schedule-to-tile") != self.tile_ir:
            raise ValueError("paged Tile IR disagrees with native Schedule replay")
        if len(self.dims) != 7 or any(type(d) is not int for d in self.dims):
            raise ValueError("paged dimensions must be integers")
        fields = ("shape = array<i64: " + ", ".join(map(str, self.dims)) + ">", "bindings = " + json.dumps(self.names))
        if any(field not in self.tile_ir for field in fields):
            raise ValueError("paged descriptor disagrees with native contract")
        if re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', self.tile_ir) != [self.schedule_digest]:
            raise ValueError("paged schedule hash disagrees")
        expected_entry = {
            "row_major": "tessera_tile_paged_kv_read_f32_direct",
            "strided": "tessera_tile_paged_kv_read_f32_strided",
        }.get(self.page_layout)
        if self.entry != expected_entry or re.findall(
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
        %out = tessera.paged_kv_read %pages, %table {{start = {start} : i64, end = {start + tokens} : i64}}
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


def project_scheduled_paged_kv_graph(module, *, target: str) -> str:
    """Project a copy of the typed Graph; native passes own subsequent IR."""
    if target not in {"rocm_gfx1151", "rocm_gfx1201"}:
        raise ValueError("ROCm paged read Schedule admission requires gfx1151 or gfx1201")
    from .rocm_native import _paged_kv_contract

    contract = _paged_kv_contract(module)
    if contract is None:
        raise ValueError("paged read requires the static f32/i32 Graph contract")
    pages, table, output, dims = contract
    source = copy.deepcopy(module)
    # The public frontend spells the bounded physical-page read as
    # tessera.kv_cache.read. Once admission has proved a single tensor result
    # and explicit pages/table, use the registered typed Graph op that owns
    # this exact read contract; the native pass owns every later IR layer.
    source.functions[0].body[0].op_name = "tessera.paged_kv_read"
    architecture = target.removeprefix("rocm_")
    source.module_attrs.update({"tessera.target": json.dumps(target),
                                "tessera.arch": json.dumps(architecture)})
    source.functions[0].fn_attrs["tessera.bindings"] = json.dumps((pages, table, output))
    return source.to_mlir(target=target, canonical=True)


def lower_scheduled_paged_kv_graph(module, *, target: str) -> ScheduledPagedKVArtifact:
    graph = project_scheduled_paged_kv_graph(module, target=target)
    from .rocm_native import _paged_kv_contract

    contract = _paged_kv_contract(module)
    if contract is None:
        raise ValueError("paged read lost its admitted Graph contract")
    pages, table, output, dims = contract
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("paged read requires production tessera-opt")
    schedule = run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    hashes = re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', tile)
    if len(hashes) != 1:
        raise RuntimeError("paged lowering lost its unique schedule hash")
    artifact = ScheduledPagedKVArtifact(graph, schedule, tile, hashes[0],
                                        (pages, table, output), dims,
                                        entry=("tessera_tile_paged_kv_read_f32_strided"
                                               if paged_storage_layout(module) == "strided"
                                               else "tessera_tile_paged_kv_read_f32_direct"),
                                        page_layout=paged_storage_layout(module))
    artifact.validate()
    return artifact


def paged_storage_layout(module) -> str:
    from .rocm_native import _paged_kv_contract
    contract = _paged_kv_contract(module)
    if contract is None:
        raise ValueError("paged storage requires the admitted typed Graph")
    arg = next(arg for arg in module.functions[0].args if arg.name == contract[0])
    layout = arg.layout or arg.ir_type.layout or "row_major"
    if layout not in {"row_major", "strided"}:
        raise ValueError("paged storage requires row_major or strided pages")
    return layout


def project_paged_host_storage(module, ordered):
    """Project memory facts onto a copy; native passes own every IR lowering."""
    from .rocm_native import _paged_kv_contract
    from .paged_host_span import checked_page_span
    contract = _paged_kv_contract(module)
    if contract is None:
        return module
    arguments = module.functions[0].args
    position = next(i for i, arg in enumerate(arguments) if arg.name == contract[0])
    pages = ordered[position]
    checked_page_span(pages)
    layout = paged_storage_layout(module)
    if pages.flags.c_contiguous and layout == "row_major":
        return module
    source = copy.deepcopy(module)
    source.functions[0].args[position].layout = "strided"
    return source
