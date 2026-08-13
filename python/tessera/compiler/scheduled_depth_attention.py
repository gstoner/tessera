"""BLOCK-ATTNRES-1 Graph -> Schedule -> launch-Tile artifact boundary."""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass

from .graph_ir import GraphIRModule
from .scheduled_matmul import digest_text, find_tessera_opt, run_tessera_opt


_HASH_RE = re.compile(r'tessera\.schedule_hash = "([0-9a-f]{64})"')
_STATISTICS = "rms_key_online_softmax_stats_v1"
_MERGE = "max_shifted_pairwise_merge_v1"


@dataclass(frozen=True)
class ScheduledDepthAttentionArtifact:
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target: str
    architecture: str
    function_name: str
    query_name: str
    sources_name: str
    output_name: str
    query_shape: tuple[int, ...]
    sources_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    source_count: int
    rows: int
    width: int
    storage: str
    softmax: str
    accum: str
    eps: float
    source_tile: int
    workgroup_size: int
    statistics_recurrence: str
    merge_recurrence: str
    reassociable: bool
    schedule_digest: str

    @property
    def graph_digest(self) -> str:
        return digest_text(self.graph_ir)

    @property
    def schedule_ir_digest(self) -> str:
        return digest_text(self.schedule_ir)

    @property
    def tile_digest(self) -> str:
        return digest_text(self.tile_ir)

    def validate(self) -> None:
        if self.schedule_ir.count("schedule.depth_attention") != 1:
            raise ValueError("depth attention requires one scheduled SSA operation")
        if len(re.findall(r"(?m)^\s*schedule\.artifact\b", self.schedule_ir)) != 1:
            raise ValueError("depth attention requires one durable schedule artifact")
        if self.schedule_ir.count(self.schedule_digest) != 3:
            raise ValueError("depth attention has incomplete content identity")
        if self.tile_ir.count("tile.depth_attention_kernel") != 1:
            raise ValueError("depth attention requires one launch-level Tile operation")
        if "tessera.depth_attn" in self.tile_ir or "schedule." in self.tile_ir:
            raise ValueError("depth-attention Tile artifact retains Graph or Schedule IR")
        if _HASH_RE.findall(self.tile_ir) != [self.schedule_digest]:
            raise ValueError("depth-attention Tile artifact has stale schedule identity")
        required = (
            f'statistics_recurrence = "{self.statistics_recurrence}"',
            f'merge_recurrence = "{self.merge_recurrence}"',
            "reassociable = true",
            f"source_tile = {self.source_tile} : i64",
            f"tessera.workgroup_size = {self.workgroup_size} : i64",
        )
        if any(marker not in self.tile_ir for marker in required):
            raise ValueError("depth-attention Tile artifact lost its algorithm contract")
        if self.schedule_ir_digest == self.tile_digest:
            raise ValueError("Schedule and Tile depth-attention artifacts must differ")


def supports_scheduled_depth_attention(
    module: GraphIRModule, *, target: str
) -> bool:
    try:
        _graph_contract(module, target)
    except ValueError:
        return False
    return True


def lower_scheduled_depth_attention(
    module: GraphIRModule, *, target: str
) -> ScheduledDepthAttentionArtifact:
    contract = _graph_contract(module, target)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("scheduled depth attention requires production tessera-opt")

    targeted = copy.deepcopy(module)
    targeted.module_attrs["tessera.target"] = f'"{contract[0]}"'
    targeted.module_attrs["tessera.arch"] = f'"{contract[1]}"'
    graph_ir = targeted.to_mlir(target=target, canonical=True)
    schedule_ir = run_tessera_opt(tool, graph_ir, "--tessera-graph-to-schedule")
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    hashes = _HASH_RE.findall(tile_ir)
    if len(hashes) != 1:
        raise RuntimeError("scheduled depth attention lost its schedule identity")
    artifact = ScheduledDepthAttentionArtifact(
        graph_ir=graph_ir,
        schedule_ir=schedule_ir,
        tile_ir=tile_ir,
        target=contract[0],
        architecture=contract[1],
        function_name=contract[2],
        query_name=contract[3],
        sources_name=contract[4],
        output_name=contract[5],
        query_shape=contract[11],
        sources_shape=contract[12],
        output_shape=contract[13],
        source_count=contract[6],
        rows=contract[7],
        width=contract[8],
        storage="f32",
        softmax="f32",
        accum="f32",
        eps=contract[9],
        source_tile=min(contract[6], 16),
        workgroup_size=contract[10],
        statistics_recurrence=_STATISTICS,
        merge_recurrence=_MERGE,
        reassociable=True,
        schedule_digest=hashes[0],
    )
    artifact.validate()
    return artifact


def _graph_contract(module: GraphIRModule, target: str) -> tuple:
    if len(module.functions) != 1:
        raise ValueError("scheduled depth attention requires one Graph function")
    function = module.functions[0]
    if len(function.body) != 1 or len(function.result_types) != 1:
        raise ValueError("scheduled depth attention requires one Graph operation")
    op = function.body[0]
    if op.op_name != "tessera.depth_attn" or len(op.operands) != 2:
        raise ValueError("scheduled depth attention requires tessera.depth_attn")
    args = {argument.name: argument for argument in function.args}
    query_name = op.operands[0].removeprefix("%")
    sources_name = op.operands[1].removeprefix("%")
    if query_name not in args or sources_name not in args:
        raise ValueError("depth-attention inputs must be function arguments")
    try:
        query_shape = tuple(int(value) for value in args[query_name].ir_type.shape)
        sources_shape = tuple(int(value) for value in args[sources_name].ir_type.shape)
        output_shape = tuple(int(value) for value in function.result_types[0].shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("scheduled depth attention requires static shapes") from exc
    if (
        len(query_shape) != 1
        or len(sources_shape) < 2
        or any(value <= 0 for value in query_shape + sources_shape)
        or query_shape[0] != sources_shape[-1]
        or output_shape != sources_shape[1:]
    ):
        raise ValueError("invalid static depth-attention shape contract")
    if (
        args[query_name].ir_type.dtype != "fp32"
        or args[sources_name].ir_type.dtype != "fp32"
        or function.result_types[0].dtype != "fp32"
    ):
        raise ValueError("initial depth-attention artifact requires fp32")
    policy = op.kwargs.get("numeric_policy")
    if policy != {"storage": "fp32", "softmax": "fp32", "accum": "fp32"}:
        raise ValueError("depth attention requires the canonical all-fp32 policy")
    eps = float(op.kwargs.get("eps", 0.0))
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("depth attention requires finite eps > 0")
    if target == "x86":
        compiler_target, architecture, workgroup_size = "x86", "zen5-avx512", 1
    elif target == "rocm_gfx1151":
        compiler_target, architecture, workgroup_size = "rocm", "gfx1151", 256
    else:
        raise ValueError(
            "depth attention supports only artifact construction for x86 and "
            "rocm_gfx1151; sibling architectures require owned profiles"
        )
    return (
        compiler_target,
        architecture,
        function.name,
        query_name,
        sources_name,
        op.result or function.return_values[0].removeprefix("%"),
        sources_shape[0],
        math.prod(sources_shape[1:-1]),
        sources_shape[-1],
        eps,
        workgroup_size,
        query_shape,
        sources_shape,
        output_shape,
    )


__all__ = [
    "ScheduledDepthAttentionArtifact",
    "lower_scheduled_depth_attention",
    "supports_scheduled_depth_attention",
]
