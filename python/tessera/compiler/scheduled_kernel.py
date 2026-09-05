"""Canonical Graph -> Schedule -> launch-Tile handoff for E2E-REAL-5."""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass

from .graph_ir import GraphIRModule
from .scheduled_matmul import digest_text, find_tessera_opt, run_tessera_opt


_HASH_RE = re.compile(r'tessera\.schedule_hash = "([0-9a-f]{64})"')


@dataclass(frozen=True)
class ScheduledKernelArtifact:
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target: str
    architecture: str
    function_name: str
    family: str
    kind: str
    input_name: str
    output_name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: str
    storage: str
    accum: str
    axis: int
    keepdims: bool
    rows: int
    columns: int
    outer: int
    axis_extent: int
    inner: int
    workgroup_size: int
    schedule_digest: str
    schedule: str = "serial"

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
        schedule_op = f"schedule.{self.family}"
        tile_op = f"tile.{self.family}_kernel" if self.family == "softmax" else "tile.reduce_kernel"
        if len(re.findall(rf"(?m)^\s*%[^=]+ = {re.escape(schedule_op)}\b", self.schedule_ir)) != 1:
            raise ValueError(f"scheduled {self.family} artifact requires one scheduled SSA operation")
        if len(re.findall(r"(?m)^\s*schedule\.artifact\b", self.schedule_ir)) != 1:
            raise ValueError(f"scheduled {self.family} artifact requires one durable schedule record")
        if self.schedule_ir.count(self.schedule_digest) != 3:
            raise ValueError(f"scheduled {self.family} artifact has incomplete Schedule digest identity")
        if self.tile_ir.count(tile_op) != 1:
            raise ValueError(f"scheduled {self.family} artifact requires exactly one Tile launch op")
        if "tessera.softmax" in self.tile_ir or "tessera.reduce" in self.tile_ir or "schedule." in self.tile_ir:
            raise ValueError("scheduled semantic-kernel Tile artifact retains Graph or Schedule ops")
        if _HASH_RE.findall(self.tile_ir) != [self.schedule_digest]:
            raise ValueError("scheduled semantic-kernel Tile artifact has a stale schedule digest")
        if not re.search(
            rf"tessera\.workgroup_size = {self.workgroup_size} : i64", self.tile_ir
        ):
            raise ValueError("scheduled semantic-kernel Tile artifact has stale workgroup size")
        if self.schedule_ir_digest == self.tile_digest:
            raise ValueError("Schedule and Tile artifacts must be distinct boundary outputs")


def supports_scheduled_kernel(module: GraphIRModule, *, target: str) -> bool:
    try:
        _graph_contract(module, target)
    except ValueError:
        return False
    return True


def lower_scheduled_kernel(
    module: GraphIRModule,
    *,
    target: str,
    schedule: str | None = None,
) -> ScheduledKernelArtifact:
    if schedule is not None:
        module = copy.deepcopy(module)
        if (module.functions and module.functions[0].body
                and module.functions[0].body[0].op_name != "tessera.softmax"):
            module.functions[0].body[0].kwargs["schedule"] = schedule
    contract = _graph_contract(module, target)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("scheduled softmax/reduction lowering requires production tessera-opt")

    targeted = copy.deepcopy(module)
    op = targeted.functions[0].body[0]
    if contract[5] == "reduce":
        op.op_name = "tessera.reduce"
        op.kwargs = {"kind": contract[6], "axis": contract[14]}
        if target == "nvidia_sm120":
            op.kwargs.update(keepdims=contract[15], schedule=contract[20])
    targeted.module_attrs["tessera.target"] = f'"{contract[0]}"'
    targeted.module_attrs["tessera.arch"] = f'"{contract[1]}"'
    graph_ir = targeted.to_mlir(target=target, canonical=True)
    schedule_ir = run_tessera_opt(tool, graph_ir, "--tessera-graph-to-schedule")
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    hashes = _HASH_RE.findall(tile_ir)
    if len(hashes) != 1:
        raise RuntimeError("scheduled semantic-kernel lowering did not preserve one schedule digest")
    artifact = ScheduledKernelArtifact(
        graph_ir=graph_ir,
        schedule_ir=schedule_ir,
        tile_ir=tile_ir,
        target=contract[0],
        architecture=contract[1],
        function_name=contract[2],
        input_name=contract[3],
        output_name=contract[4],
        family=contract[5],
        kind=contract[6],
        input_shape=contract[7],
        output_shape=contract[8],
        dtype=contract[9],
        storage=contract[10],
        accum=contract[11],
        rows=contract[12],
        columns=contract[13],
        axis=contract[14],
        keepdims=contract[15],
        outer=contract[16],
        axis_extent=contract[17],
        inner=contract[18],
        workgroup_size=contract[19],
        schedule_digest=hashes[0],
        schedule=contract[20],
    )
    artifact.validate()
    return artifact


def _graph_contract(module: GraphIRModule, target: str) -> tuple:
    if len(module.functions) != 1:
        raise ValueError("scheduled semantic kernel requires one Graph function")
    function = module.functions[0]
    if len(function.body) != 1 or len(function.result_types) != 1:
        raise ValueError("scheduled semantic kernel requires one Graph operation and result")
    op = function.body[0]
    if len(op.operands) != 1:
        raise ValueError("scheduled semantic kernel requires one operand")
    args = {arg.name: arg for arg in function.args}
    input_name = op.operands[0].removeprefix("%")
    if input_name not in args:
        raise ValueError("scheduled semantic-kernel operand must be a function argument")
    try:
        input_shape = tuple(int(value) for value in args[input_name].ir_type.shape)
        output_shape = tuple(int(value) for value in function.result_types[0].shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("scheduled semantic kernel requires static shapes") from exc
    if not input_shape or any(value <= 0 for value in input_shape):
        raise ValueError("scheduled semantic kernel requires a non-empty positive shape")
    dtype = args[input_name].ir_type.dtype
    output_dtype = function.result_types[0].dtype
    if target == "nvidia_sm120":
        expected_dtype = dtype if op.op_name == "tessera.softmax" else "fp32"
        if dtype not in {"fp16", "bf16", "fp32"} or output_dtype != expected_dtype:
            raise ValueError("NVIDIA scheduled unary storage contract is unsupported")
    elif dtype != "fp32" or output_dtype != "fp32":
        raise ValueError("initial scheduled semantic-kernel contract requires f32")
    mode = str(op.kwargs.get("schedule", "serial"))
    if mode != "serial" and (target != "nvidia_sm120" or mode != "cooperative_128"):
        raise ValueError("unsupported scheduled reduction policy")
    output_name = op.result or function.return_values[0].removeprefix("%")
    if target == "x86":
        compiler_target, architecture, workgroup_size = "x86", "zen5-avx512", 1
    elif target == "rocm_gfx1151":
        compiler_target, architecture, workgroup_size = "rocm", "gfx1151", 256
    elif target == "nvidia_sm120":
        compiler_target, architecture, workgroup_size = "nvidia_sm120", "sm_120", 128
    elif target == "apple_gpu":
        compiler_target, architecture, workgroup_size = "apple_gpu", "apple7", 1
    else:
        raise ValueError("unsupported scheduled semantic-kernel target")

    if op.op_name == "tessera.softmax":
        if mode != "serial":
            raise ValueError("reduction scheduling policy is not applicable to softmax")
        if op.kwargs.get("axis", -1) != -1 or output_shape != input_shape:
            raise ValueError("scheduled softmax requires shape-preserving last-axis semantics")
        family, kind, axis, keepdims = "softmax", "softmax", -1, False
        rows, columns = math.prod(input_shape[:-1]), input_shape[-1]
        outer = axis_extent = inner = 1
    elif op.op_name in {
        "tessera.reduce", "tessera.sum", "tessera.mean", "tessera.max", "tessera.amax", "tessera.min", "tessera.amin"
    }:
        kind = str(op.kwargs.get("kind", "")) if op.op_name == "tessera.reduce" else (
            "mean" if op.op_name == "tessera.mean" else
            "max" if op.op_name in {"tessera.max", "tessera.amax"} else
            "min" if op.op_name in {"tessera.min", "tessera.amin"} else "sum"
        )
        keepdims = op.kwargs.get("keepdims", False)
        if not isinstance(keepdims, bool):
            raise ValueError("scheduled reduction keepdims must be boolean")
        allowed = {"sum", "mean", "max", "min"} if target == "nvidia_sm120" else {"sum", "mean", "max"}
        if kind not in allowed or (keepdims and target != "nvidia_sm120"):
            raise ValueError("scheduled reduction requires rank-reducing sum/mean/max")
        raw_axis = op.kwargs.get("axis", -1)
        if not isinstance(raw_axis, int) or isinstance(raw_axis, bool):
            raise ValueError("scheduled reduction requires one integer axis")
        axis = raw_axis + len(input_shape) if raw_axis < 0 else raw_axis
        if not 0 <= axis < len(input_shape):
            raise ValueError("scheduled reduction axis is out of range")
        if target == "x86" and axis != len(input_shape) - 1:
            raise ValueError("initial x86 scheduled reduction requires the last axis")
        # Apple's synthesized reduce kernel gives one thread per row and folds
        # over the trailing extent, so it expresses last-axis reductions only.
        if target == "apple_gpu" and axis != len(input_shape) - 1:
            raise ValueError("Apple GPU scheduled reduction requires the last axis")
        expected = input_shape[:axis] + ((1,) if keepdims else ()) + input_shape[axis + 1 :]
        if output_shape != expected:
            raise ValueError("scheduled reduction output shape does not match its axis")
        family = "reduce"
        rows = columns = 1
        outer = math.prod(input_shape[:axis])
        axis_extent = input_shape[axis]
        inner = math.prod(input_shape[axis + 1 :])
    else:
        raise ValueError("unsupported scheduled semantic-kernel operation")

    assert dtype is not None  # Rejected by the storage contract above.
    storage = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}[dtype]
    entry = function.name
    if target == "nvidia_sm120":
        entry = f"tessera_tile_softmax_{storage}" if family == "softmax" else f"tessera_tile_reduce_{kind}_{storage}_{mode}"
    return (
        compiler_target, architecture, entry, input_name, output_name,
        family, kind, input_shape, output_shape, dtype, storage, "f32", rows,
        columns, axis, keepdims, outer, axis_extent, inner, workgroup_size, mode,
    )
