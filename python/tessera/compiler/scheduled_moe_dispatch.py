"""Replay-checked native gfx1151 MoE token-gather Schedule contract.

E2E-REAL-6 (sync ``E2E-REAL-6-rocm-moe-dispatch-2026-11-03``): the ROCm
consumer reads every launch field from
:meth:`ScheduledMoeDispatchArtifact.native_contract`, the verified projection
of the replayed Tile ``tessera.native_contract`` (``NativeMoeDispatch.h``).
Python here owns only admission (:func:`admit_moe_dispatch`) and the binding
names projected onto the Graph; the retired Graph-authored ROCm descriptor
survives as the test-only oracle ``tests/_support/rocm_moe_dispatch_baseline.py``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import re
from typing import NamedTuple, cast

from .scheduled_matmul import find_tessera_opt, run_tessera_opt

#: Targets whose MoE token gather has an owning native ROCm consumer. The
#: native Schedule owner (``NativeMoeDispatch.h``) admits gfx1151 only.
ROCM_MOE_DISPATCH_TARGETS = frozenset({"rocm_gfx1151"})

MOE_DISPATCH_ENTRY = "tessera_tile_moe_dispatch_f32_direct"
_CONTRACT_KEYS = frozenset({"arch", "bindings", "layout", "route", "shape", "target"})
_CONTRACT_FIELD = re.compile(r'(\w+) = ("[^"]*"|\[[^\]]*\]|array<i64: [-0-9, ]*>)(?:, |$)')


class MoeDispatchAdmission(NamedTuple):
    """What admission reads off the frontend Graph: names and the static extent.

    Admission refuses anything outside the bounded f32/i32 envelope so route
    selection falls through rather than raising. No launch field is derived
    from it; the ROCm descriptor comes from :class:`MoeDispatchNativeContract`.
    """

    x: str
    token: str
    output: str
    dims: tuple[int, int, int]


@dataclass(frozen=True)
class MoeDispatchNativeContract:
    """The token gather as the replayed native Tile IR states it."""

    target: str
    architecture: str
    names: tuple[str, str, str]
    dims: tuple[int, int, int]
    layout: str
    entry: str
    storage: str
    index_storage: str
    kernel_route: str
    scalar_count: int


def _parse_native_contract(text: str) -> dict[str, object]:
    fields: dict[str, object] = {}
    position = 0
    for match in _CONTRACT_FIELD.finditer(text):
        if match.start() != position:
            raise ValueError("MoE native contract has an unparsable field")
        key, raw = match.group(1), match.group(2)
        if key in fields:
            raise ValueError("MoE native contract repeats a field")
        if raw.startswith('"'):
            fields[key] = raw[1:-1]
        elif raw.startswith("["):
            fields[key] = tuple(json.loads(raw))
        else:
            body = raw.removeprefix("array<i64: ").removesuffix(">")
            fields[key] = tuple(int(item) for item in body.split(", ")) if body else ()
        position = match.end()
    if position != len(text):
        raise ValueError("MoE native contract has an unparsable field")
    return fields


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
        if re.findall(r"llvm.func @([\w]+)\(", self.tile_ir) != [MOE_DISPATCH_ENTRY]:
            raise ValueError("MoE Tile entry disagrees with runtime ABI")

    def native_contract(self) -> MoeDispatchNativeContract:
        """Project the launch contract from the Tile IR; refuse any drift.

        Every field comes from the replayed ``tessera.native_contract``, the
        one ``llvm.func`` and its one ``tile.moe_dispatch_kernel``. The
        artifact's own ``names``/``dims`` must agree with that text, so a
        relabelled artifact fails here, not at launch.
        """
        contracts = re.findall(r"tessera\.native_contract = \{([^{}]*)\}", self.tile_ir)
        functions = re.findall(r"llvm\.func @(\w+)\(([^)]*)\)", self.tile_ir)
        kernels = re.findall(r"tile\.moe_dispatch_kernel ([^{]*)\{([^}]*)\}", self.tile_ir)
        modules = re.findall(r"^module attributes \{([^}]*)\}", self.tile_ir, re.MULTILINE)
        if len(contracts) != 1 or len(functions) != 1 or len(kernels) != 1 or len(modules) != 1:
            raise ValueError("MoE Tile IR must carry exactly one native contract and kernel")
        fields = _parse_native_contract(contracts[0])
        if set(fields) != _CONTRACT_KEYS:
            raise ValueError("MoE native contract fields disagree with the token-gather contract")
        target, arch = fields["target"], fields["arch"]
        names, dims = fields["bindings"], fields["shape"]
        layout, route = fields["layout"], fields["route"]
        if (not isinstance(target, str) or not isinstance(arch, str)
                or target not in ROCM_MOE_DISPATCH_TARGETS or target != f"rocm_{arch}"
                or layout != "row_major" or route != "direct_gather"):
            raise ValueError("MoE native contract names no supported target, layout or route")
        if (not isinstance(names, tuple) or len(names) != 3
                or not all(isinstance(n, str) and n for n in names) or len(set(names)) != 3
                or not isinstance(dims, tuple) or len(dims) != 3
                or any(type(d) is not int or d <= 0 for d in dims)):
            raise ValueError("MoE native contract has an invalid binding or extent")
        module_attrs = dict(re.findall(r'([\w.]+) = "([^"]*)"', modules[0]))
        if module_attrs.get("tessera.target") != target or module_attrs.get("tessera.arch") != arch:
            raise ValueError("MoE native contract disagrees with its Tile module target")
        entry, parameters = functions[0]
        scalar_count = parameters.count(": i64")
        operands, attr_text = kernels[0]
        attrs = dict(re.findall(r'(\w+) = "([^"]*)"', attr_text))
        if (entry != MOE_DISPATCH_ENTRY or parameters.count(": !llvm.ptr") != 3
                or scalar_count != 3 or operands.count("%") != 6
                or attrs != {"index_storage": "i32", "storage": "f32"}):
            raise ValueError("MoE Tile kernel disagrees with its native contract")
        if tuple(self.names) != names or tuple(self.dims) != dims:
            raise ValueError("MoE artifact fields disagree with its native Tile contract")
        return MoeDispatchNativeContract(
            target=target, architecture=arch, names=names, dims=dims, layout=layout,
            entry=entry, storage=attrs["storage"], index_storage=attrs["index_storage"],
            kernel_route=route, scalar_count=scalar_count,
        )


def _static_shape(arg) -> tuple[int, ...] | None:
    if arg is None or arg.ir_type.rank is None or arg.ir_type.rank < 1:
        return None
    try:
        shape = tuple(int(dim) for dim in arg.ir_type.shape)
    except (TypeError, ValueError):
        return None
    return shape if all(dim > 0 for dim in shape) else None


def admit_moe_dispatch(module) -> MoeDispatchAdmission | None:
    """Admit one isolated local f32/i32 ``tessera.moe_dispatch`` or refuse.

    ``NativeMoeDispatch.h`` re-checks every fact here during native replay;
    this is the fall-through admission that keeps route selection from
    raising on Graphs outside the envelope.
    """
    if not (len(module.functions) == 1 and len(module.functions[0].body) == 1
            and module.functions[0].body[0].op_name == "tessera.moe_dispatch"):
        return None
    function = module.functions[0]
    op = function.body[0]
    if len(op.operands) != 2 or len(function.result_types) != 1:
        return None
    if op.kwargs.get("transport") is not None:
        return None
    x_name, token_name = (value.removeprefix("%") for value in op.operands)
    args = {arg.name: arg for arg in function.args}
    x, token = args.get(x_name), args.get(token_name)
    x_shape, token_shape = _static_shape(x), _static_shape(token)
    if (x is None or token is None or x.ir_type.dtype != "fp32"
            or token.ir_type.dtype != "int32" or x_shape is None or len(x_shape) != 2
            or token_shape is None or len(token_shape) != 1):
        return None
    tokens, hidden = x_shape
    slots = token_shape[0]
    result = function.result_types[0]
    try:
        result_shape = tuple(int(value) for value in result.shape)
    except (TypeError, ValueError):
        return None
    if result.dtype != "fp32" or result_shape != (slots, hidden):
        return None
    return MoeDispatchAdmission(x_name, token_name, op.result or "output", (tokens, slots, hidden))


def supports_scheduled_moe_dispatch(module, *, target: str) -> bool:
    """Whether the native ROCm token-gather Schedule route admits ``module``."""
    return target in ROCM_MOE_DISPATCH_TARGETS and admit_moe_dispatch(module) is not None


def project_scheduled_moe_dispatch_graph(module, *, target: str = "rocm_gfx1151") -> str:
    if target not in ROCM_MOE_DISPATCH_TARGETS:
        raise ValueError("MoE token-gather Schedule only admits gfx1151")
    admission = admit_moe_dispatch(module)
    if admission is None:
        raise ValueError("MoE token gather needs static f32[T,H], i32[S] -> f32[S,H]")
    source = copy.deepcopy(module)
    # An explicit None is the public API default, not a transport policy.
    if source.functions[0].body[0].kwargs.get("transport") is None:
        source.functions[0].body[0].kwargs.pop("transport", None)
    architecture = target.removeprefix("rocm_")
    source.module_attrs.update({"tessera.target": json.dumps(target),
                                "tessera.arch": json.dumps(architecture)})
    source.functions[0].fn_attrs["tessera.bindings"] = json.dumps(admission[:3])
    return source.to_mlir(target=target, canonical=True)


def lower_scheduled_moe_dispatch(module, *, target: str = "rocm_gfx1151") -> ScheduledMoeDispatchArtifact:
    """Replay the projected Graph natively; the artifact's fields are read back
    from the Tile contract it produced, never from the Python Graph."""
    graph = project_scheduled_moe_dispatch_graph(module, target=target)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("MoE dispatch needs production tessera-opt")
    schedule = run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    hashes = re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', tile)
    if len(hashes) != 1:
        raise RuntimeError("MoE lowering lost its unique Schedule hash")
    contracts = re.findall(r"tessera\.native_contract = \{([^{}]*)\}", tile)
    if len(contracts) != 1:
        raise RuntimeError("MoE lowering lost its unique native contract")
    fields = _parse_native_contract(contracts[0])
    names, dims = fields.get("bindings"), fields.get("shape")
    if not isinstance(names, tuple) or len(names) != 3 or not isinstance(dims, tuple) or len(dims) != 3:
        raise RuntimeError("MoE lowering produced no token-gather contract")
    # native_contract() below re-checks both against the Tile text.
    artifact = ScheduledMoeDispatchArtifact(
        graph, schedule, tile, hashes[0],
        cast("tuple[str, str, str]", names), cast("tuple[int, int, int]", dims))
    artifact.validate()
    artifact.native_contract()
    return artifact
