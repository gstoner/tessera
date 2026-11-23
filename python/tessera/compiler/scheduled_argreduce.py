"""Native sm_120 row argmax/argmin (AR-DECODE-PROGRAM-1 S1b).

The contract lives in C++ (``NativeArgReduce.h``): an isolated
``tessera.argmax|argmin`` in an ``nvidia_sm120`` module becomes a
content-addressed ``schedule.argreduce`` and then the shared
``tile.argreduce_kernel`` envelope. Python builds the Graph text, replays the
native passes and re-reads every launch field from the replayed
``tessera.native_contract``; it decides nothing about the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re

from .scheduled_matmul import find_tessera_opt, run_tessera_opt

ARGREDUCE_KINDS = frozenset({"argmax", "argmin"})
_CONTRACT_KEYS = frozenset({
    "arch", "axis_mode", "bindings", "keepdims", "kind", "logical_shape", "nan_mode",
    "output_storage", "result_shape", "schedule", "shape", "storage", "target", "tie_break"})
_CONTRACT_FIELD = re.compile(
    r'(\w+) = ("[^"]*"|\[[^\]]*\]|array<i64(?:: [-0-9, ]*)?>|true|false)(?:, |$)')
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_INT32_MAX = 2**31 - 1


def _parse_native_contract(text: str) -> dict[str, object]:
    fields: dict[str, object] = {}
    position = 0
    for match in _CONTRACT_FIELD.finditer(text):
        if match.start() != position:
            raise ValueError("argreduce native contract has an unparsable field")
        key, raw = match.group(1), match.group(2)
        if key in fields:
            raise ValueError("argreduce native contract repeats a field")
        if raw.startswith('"'):
            fields[key] = raw[1:-1]
        elif raw.startswith("["):
            fields[key] = tuple(json.loads(raw))
        elif raw in ("true", "false"):
            fields[key] = raw == "true"
        else:
            body = raw.removeprefix("array<i64").removeprefix(": ").removesuffix(">")
            fields[key] = tuple(int(item) for item in body.split(", ")) if body else ()
        position = match.end()
    if position != len(text):
        raise ValueError("argreduce native contract has an unparsable field")
    return fields


def argreduce_result_shape(shape: tuple[int, ...], *, axis: int | None,
                           keepdims: bool) -> tuple[int, ...]:
    """The Graph result shape (NumPy rules: ``axis=None`` flattens)."""
    if axis is None:
        return (1,) * len(shape) if keepdims else ()
    return tuple(shape[:-1]) + ((1,) if keepdims else ())


def _tensor(shape: tuple[int, ...], element: str) -> str:
    return "tensor<" + "".join(f"{dim}x" for dim in shape) + element + ">"


@dataclass(frozen=True)
class ScheduledArgReduceArtifact:
    """Graph -> Schedule -> Tile replay of one sm_120 row argreduce.

    ``rows``/``columns`` are the launch extents (``columns`` is the reduced
    width). :meth:`validate` replays both passes and re-reads every field.
    """

    graph_ir: str
    schedule_ir: str
    tile_ir: str
    schedule_digest: str
    kind: str
    names: tuple[str, str]
    shape: tuple[int, ...]
    result_shape: tuple[int, ...]
    axis_mode: str
    keepdims: bool
    rows: int
    columns: int
    entry: str

    def validate(self) -> None:
        tool = find_tessera_opt()
        if tool is None:
            raise RuntimeError("sm_120 argreduce requires native Schedule replay")
        if run_tessera_opt(tool, self.graph_ir, "--tessera-graph-to-schedule") != self.schedule_ir:
            raise ValueError("argreduce Schedule IR disagrees with Graph replay")
        if run_tessera_opt(tool, self.schedule_ir, "--tessera-schedule-to-tile") != self.tile_ir:
            raise ValueError("argreduce Tile IR disagrees with native Schedule replay")
        contracts = re.findall(r"tessera\.native_contract = \{([^{}]*)\}", self.tile_ir)
        functions = re.findall(r"llvm\.func @(\w+)\(([^)]*)\)", self.tile_ir)
        kernels = re.findall(r"tile\.argreduce_kernel ([^{]*)\{([^}]*)\}", self.tile_ir)
        if len(contracts) != 1 or len(functions) != 1 or len(kernels) != 1:
            raise ValueError("argreduce Tile IR must carry one native contract and kernel")
        fields = _parse_native_contract(contracts[0])
        if set(fields) != _CONTRACT_KEYS:
            raise ValueError("argreduce native contract has unexpected fields")
        semantic = {
            "target": "nvidia_sm120", "arch": "sm_120", "storage": "f32",
            "output_storage": "i32", "tie_break": "first", "nan_mode": "first_nan",
            "schedule": "cta_128_warp_shuffle"}
        if any(fields[key] != value for key, value in semantic.items()):
            raise ValueError("argreduce native contract leaves the sm_120 envelope")
        if (fields["kind"] != self.kind or fields["bindings"] != tuple(self.names)
                or fields["logical_shape"] != tuple(self.shape)
                or fields["result_shape"] != tuple(self.result_shape)
                or fields["axis_mode"] != self.axis_mode or fields["keepdims"] != self.keepdims
                or fields["shape"] != (self.rows, self.columns)):
            raise ValueError("argreduce artifact fields disagree with its native Tile contract")
        entry, parameters = functions[0]
        operands, attr_text = kernels[0]
        attrs = dict(re.findall(r'(\w+) = "([^"]*)"', attr_text))
        if (entry != self.entry or parameters.count(": !llvm.ptr") != 2
                or parameters.count(": i64") != 2 or operands.count("%") != 4
                or attrs != {key: str(fields[key]) for key in (
                    "kind", "nan_mode", "output_storage", "schedule", "storage", "tie_break")}):
            raise ValueError("argreduce Tile kernel disagrees with its runtime ABI")
        if re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', self.tile_ir) != [self.schedule_digest]:
            raise ValueError("argreduce schedule hash disagrees")
        if self.entry != f"tessera_tile_argreduce_{self.kind}_f32_{self.schedule_digest[:10]}":
            raise ValueError("argreduce entry symbol disagrees with its schedule digest")


def lower_scheduled_argreduce(
    kind: str, names: tuple[str, str], shape: tuple[int, ...], *,
    axis: int | None = -1, keepdims: bool = False,
) -> ScheduledArgReduceArtifact:
    """Replay one isolated sm_120 ``tessera.argmax|argmin`` through native passes."""
    if kind not in ARGREDUCE_KINDS:
        raise ValueError(f"sm_120 argreduce kind must be one of {sorted(ARGREDUCE_KINDS)}")
    if len(names) != 2 or len(set(names)) != 2 or any(not _IDENTIFIER.fullmatch(n) for n in names):
        raise ValueError("sm_120 argreduce bindings must be two unique identifiers")
    if not shape or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError("sm_120 argreduce shape must be positive integers of rank >= 1")
    if math.prod(shape) > _INT32_MAX:
        raise ValueError("sm_120 argreduce shape exceeds its i32 index ABI")
    if axis is not None and axis not in (-1, len(shape) - 1):
        raise ValueError("sm_120 argreduce reduces only the last axis (or the flattened tensor)")
    if type(keepdims) is not bool:
        raise ValueError("sm_120 argreduce keepdims must be a bool")
    result_shape = argreduce_result_shape(tuple(shape), axis=axis, keepdims=keepdims)
    attrs = ([f"axis = {axis} : i64"] if axis is not None else []) + [f"keepdims = {str(keepdims).lower()}"]
    source_t, result_t = _tensor(tuple(shape), "f32"), _tensor(result_shape, "i32")
    graph = f"""module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"}} {{
      func.func @argreduce(%x: {source_t}) -> {result_t}
          attributes {{tessera.bindings = {json.dumps(list(names))}}} {{
        %i = tessera.{kind} %x {{{", ".join(attrs)}}} : ({source_t}) -> {result_t}
        return %i : {result_t}
      }}
    }}"""
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("sm_120 argreduce requires production tessera-opt")
    schedule = run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    hashes = re.findall(r'tessera.schedule_hash = "([0-9a-f]{64})"', tile)
    if len(hashes) != 1:
        raise RuntimeError("sm_120 argreduce lowering lost its unique schedule hash")
    columns = math.prod(shape) if axis is None else shape[-1]
    artifact = ScheduledArgReduceArtifact(
        graph_ir=graph, schedule_ir=schedule, tile_ir=tile, schedule_digest=hashes[0],
        kind=kind, names=(names[0], names[1]), shape=tuple(shape), result_shape=result_shape,
        axis_mode="flatten" if axis is None else "last", keepdims=keepdims,
        rows=math.prod(shape) // columns, columns=columns,
        entry=f"tessera_tile_argreduce_{kind}_f32_{hashes[0][:10]}",
    )
    artifact.validate()
    return artifact
