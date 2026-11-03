"""Native paged read contract; Python owns bindings, not Tile construction.

E2E-REAL-6 (sync ``E2E-REAL-6-rocm-paged-kv-2026-11-03``): the ROCm consumer
reads every launch field from :meth:`ScheduledPagedKVArtifact.native_contract`,
the verified projection of the replayed Tile ``tessera.native_contract``.
Python here owns only admission (:func:`admit_paged_kv`) and the projection of
the frontend ``tessera.kv_cache.read`` onto the typed Graph op; the retired
Graph-authored ROCm descriptor survives as the test-only oracle
``tests/_support/rocm_paged_kv_baseline.py``.
"""

from dataclasses import dataclass
import copy
import json
import re
from typing import NamedTuple

from .scheduled_matmul import find_tessera_opt, run_tessera_opt

#: Targets whose paged read has an owning native ROCm consumer.
ROCM_PAGED_KV_TARGETS = frozenset({"rocm_gfx1151", "rocm_gfx1201"})

_ENTRY_BY_LAYOUT = {
    "row_major": "tessera_tile_paged_kv_read_f32_direct",
    "strided": "tessera_tile_paged_kv_read_f32_strided",
}
_COMMON_CONTRACT_KEYS = frozenset({
    "arch", "bindings", "layout", "page_ownership", "shape", "table_bounds", "target"})
_STRIDED_CONTRACT_KEYS = frozenset({"page_stride_policy", "source_extent_policy"})
_CONTRACT_FIELD = re.compile(r'(\w+) = ("[^"]*"|\[[^\]]*\]|array<i64: [-0-9, ]*>)(?:, |$)')


class PagedKVAdmission(NamedTuple):
    """What admission reads off the frontend Graph: names and the static extent.

    Admission refuses anything outside the bounded f32/i32 envelope so route
    selection falls through rather than raising. No launch field is derived
    from it; the ROCm descriptor comes from :class:`PagedKVNativeContract`.
    """

    pages: str
    table: str
    output: str
    dims: tuple[int, int, int, int, int, int, int]


@dataclass(frozen=True)
class PagedKVNativeContract:
    """The paged read as the replayed native Tile IR states it."""

    target: str
    architecture: str
    names: tuple[str, str, str]
    dims: tuple[int, int, int, int, int, int, int]
    layout: str
    entry: str
    storage: str
    table_storage: str
    kernel_route: str
    policies: tuple[tuple[str, str], ...]
    scalar_count: int


def _parse_native_contract(text: str) -> dict[str, object]:
    fields: dict[str, object] = {}
    position = 0
    for match in _CONTRACT_FIELD.finditer(text):
        if match.start() != position:
            raise ValueError("paged native contract has an unparsable field")
        key, raw = match.group(1), match.group(2)
        if key in fields:
            raise ValueError("paged native contract repeats a field")
        if raw.startswith('"'):
            fields[key] = raw[1:-1]
        elif raw.startswith("["):
            fields[key] = tuple(json.loads(raw))
        else:
            body = raw.removeprefix("array<i64: ").removesuffix(">")
            fields[key] = tuple(int(item) for item in body.split(", ")) if body else ()
        position = match.end()
    if position != len(text):
        raise ValueError("paged native contract has an unparsable field")
    return fields


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

    def native_contract(self) -> PagedKVNativeContract:
        """Project the launch contract from the Tile IR; refuse any drift.

        Every field comes from the replayed ``tessera.native_contract``, the
        one ``llvm.func`` and its one ``tile.paged_kv_read_kernel``. The
        artifact's own ``names``/``dims``/``page_layout``/``entry`` must agree
        with that text, so a relabelled artifact fails here, not at launch.
        """
        contracts = re.findall(r"tessera\.native_contract = \{([^{}]*)\}", self.tile_ir)
        functions = re.findall(r"llvm\.func @(\w+)\(([^)]*)\)", self.tile_ir)
        kernels = re.findall(
            r"tile\.paged_kv_read_kernel ([^{]*)\{([^}]*)\}", self.tile_ir)
        modules = re.findall(r'^module attributes \{([^}]*)\}', self.tile_ir, re.MULTILINE)
        if len(contracts) != 1 or len(functions) != 1 or len(kernels) != 1 or len(modules) != 1:
            raise ValueError("paged Tile IR must carry exactly one native contract and kernel")
        fields = _parse_native_contract(contracts[0])
        layout = fields.get("layout")
        if layout not in _ENTRY_BY_LAYOUT:
            raise ValueError("paged native contract names no supported page layout")
        expected_keys = _COMMON_CONTRACT_KEYS | (_STRIDED_CONTRACT_KEYS if layout == "strided" else set())
        if set(fields) != expected_keys:
            raise ValueError("paged native contract fields disagree with its layout")
        target, arch = fields["target"], fields["arch"]
        names, dims = fields["bindings"], fields["shape"]
        if (not isinstance(target, str) or not isinstance(arch, str)
                or not isinstance(names, tuple) or len(names) != 3
                or not all(isinstance(n, str) for n in names)
                or not isinstance(dims, tuple) or len(dims) != 7
                or any(type(d) is not int or d < 0 for d in dims) or any(d <= 0 for d in dims[:5])
                or dims[6] <= 0 or dims[5] + dims[6] > dims[1] * dims[2]):
            raise ValueError("paged native contract has an invalid binding or extent")
        module_attrs = dict(re.findall(r'([\w.]+) = "([^"]*)"', modules[0]))
        if module_attrs.get("tessera.target") != target or module_attrs.get("tessera.arch") != arch:
            raise ValueError("paged native contract disagrees with its Tile module target")
        entry, parameters = functions[0]
        scalar_count = parameters.count(": i64")
        expected_scalars = 11 if layout == "strided" else 7
        operands, attr_text = kernels[0]
        attrs = dict(re.findall(r'(\w+) = "([^"]*)"', attr_text))
        expected_attrs = {"route": "direct", "storage": "f32", "table_storage": "i32"}
        if layout == "strided":
            expected_attrs["page_layout"] = "strided"
        if (entry != _ENTRY_BY_LAYOUT[layout] or parameters.count(": !llvm.ptr") != 3
                or scalar_count != expected_scalars
                or operands.count("%") != 3 + expected_scalars or attrs != expected_attrs):
            raise ValueError("paged Tile kernel disagrees with its native contract")
        if (tuple(self.names) != names or tuple(self.dims) != dims
                or self.page_layout != layout or self.entry != entry):
            raise ValueError("paged artifact fields disagree with its native Tile contract")
        policies = tuple(sorted((key, str(fields[key])) for key in fields
                                if key not in {"arch", "bindings", "layout", "shape", "target"}))
        return PagedKVNativeContract(
            target=target, architecture=arch, names=names, dims=dims, layout=layout,
            entry=entry, storage=attrs["storage"], table_storage=attrs["table_storage"],
            kernel_route=attrs["route"], policies=policies, scalar_count=scalar_count,
        )


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


def _static_shape(arg) -> tuple[int, ...] | None:
    if arg is None or arg.ir_type.rank is None or arg.ir_type.rank < 1:
        return None
    try:
        shape = tuple(int(dim) for dim in arg.ir_type.shape)
    except (TypeError, ValueError):
        return None
    return shape if all(dim > 0 for dim in shape) else None


def admit_paged_kv(module) -> PagedKVAdmission | None:
    """Admit one isolated bounded f32/i32 ``tessera.kv_cache.read`` or refuse.

    The registered ``tessera.paged_kv_read`` verifier re-checks every fact
    here during native replay; this is the fall-through admission that keeps
    route selection from raising on Graphs outside the envelope.
    """
    if not (len(module.functions) == 1 and len(module.functions[0].body) == 1
            and module.functions[0].body[0].op_name == "tessera.kv_cache.read"):
        return None
    function = module.functions[0]
    op = function.body[0]
    if len(op.operands) != 2 or len(function.result_types) != 1:
        return None
    pages_name, table_name = (value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in function.args}
    pages, table = args.get(pages_name), args.get(table_name)
    pages_shape, table_shape = _static_shape(pages), _static_shape(table)
    if (pages is None or table is None or pages.ir_type.dtype != "fp32"
            or table.ir_type.dtype != "int32" or pages_shape is None or len(pages_shape) != 4
            or table_shape is None or len(table_shape) != 1):
        return None
    physical_pages, page_size, heads, dim = pages_shape
    logical_pages = table_shape[0]
    start, end = op.kwargs.get("start"), op.kwargs.get("end")
    if type(start) is not int or type(end) is not int:
        return None
    tokens = end - start
    result = function.result_types[0]
    try:
        result_shape = tuple(int(value) for value in result.shape)
    except (TypeError, ValueError):
        return None
    if (start < 0 or tokens <= 0 or end > logical_pages * page_size
            or result.dtype != "fp32" or result_shape != (tokens, heads, dim)):
        return None
    return PagedKVAdmission(pages_name, table_name, op.result or "output",
                            (physical_pages, logical_pages, page_size, heads, dim, start, tokens))


def supports_scheduled_paged_kv(module, *, target: str) -> bool:
    """Whether the native ROCm paged-read Schedule route admits ``module``."""
    if target not in ROCM_PAGED_KV_TARGETS or admit_paged_kv(module) is None:
        return False
    try:
        paged_storage_layout(module)
    except ValueError:
        return False
    return True


def project_scheduled_paged_kv_graph(module, *, target: str) -> str:
    """Project a copy of the typed Graph; native passes own subsequent IR."""
    if target not in ROCM_PAGED_KV_TARGETS:
        raise ValueError("ROCm paged read Schedule admission requires gfx1151 or gfx1201")
    admission = admit_paged_kv(module)
    if admission is None:
        raise ValueError("paged read requires the static f32/i32 Graph contract")
    source = copy.deepcopy(module)
    # The public frontend spells the bounded physical-page read as
    # tessera.kv_cache.read. Once admission has proved a single tensor result
    # and explicit pages/table, use the registered typed Graph op that owns
    # this exact read contract; the native pass owns every later IR layer.
    source.functions[0].body[0].op_name = "tessera.paged_kv_read"
    architecture = target.removeprefix("rocm_")
    source.module_attrs.update({"tessera.target": json.dumps(target),
                                "tessera.arch": json.dumps(architecture)})
    source.functions[0].fn_attrs["tessera.bindings"] = json.dumps(admission[:3])
    return source.to_mlir(target=target, canonical=True)


def lower_scheduled_paged_kv_graph(module, *, target: str) -> ScheduledPagedKVArtifact:
    """Replay the projected Graph natively; the artifact's fields are read back
    from the Tile contract it produced, never from the Python Graph."""
    graph = project_scheduled_paged_kv_graph(module, target=target)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("paged read requires production tessera-opt")
    schedule = run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    hashes = re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', tile)
    if len(hashes) != 1:
        raise RuntimeError("paged lowering lost its unique schedule hash")
    contracts = re.findall(r"tessera\.native_contract = \{([^{}]*)\}", tile)
    if len(contracts) != 1:
        raise RuntimeError("paged lowering lost its unique native contract")
    fields = _parse_native_contract(contracts[0])
    layout, names, dims = fields.get("layout"), fields.get("bindings"), fields.get("shape")
    if layout not in _ENTRY_BY_LAYOUT or not isinstance(names, tuple) or not isinstance(dims, tuple):
        raise RuntimeError("paged lowering produced no supported page contract")
    artifact = ScheduledPagedKVArtifact(
        graph, schedule, tile, hashes[0], names, dims,
        entry=_ENTRY_BY_LAYOUT[str(layout)], page_layout=str(layout))
    artifact.validate()
    artifact.native_contract()
    return artifact


def paged_storage_layout(module) -> str:
    admission = admit_paged_kv(module)
    if admission is None:
        raise ValueError("paged storage requires the admitted typed Graph")
    arg = next(arg for arg in module.functions[0].args if arg.name == admission.pages)
    layout = arg.layout or arg.ir_type.layout or "row_major"
    if layout not in {"row_major", "strided"}:
        raise ValueError("paged storage requires row_major or strided pages")
    return layout


def project_paged_host_storage(module, ordered):
    """Project memory facts onto a copy; native passes own every IR lowering."""
    from .paged_host_span import checked_page_span
    admission = admit_paged_kv(module)
    if admission is None:
        return module
    arguments = module.functions[0].args
    position = next(i for i, arg in enumerate(arguments) if arg.name == admission.pages)
    pages = ordered[position]
    checked_page_span(pages)
    layout = paged_storage_layout(module)
    if pages.flags.c_contiguous and layout == "row_major":
        return module
    source = copy.deepcopy(module)
    source.functions[0].args[position].layout = "strided"
    return source
