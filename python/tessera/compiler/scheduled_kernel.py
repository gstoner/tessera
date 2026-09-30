"""Canonical Graph -> Schedule -> launch-Tile handoff for E2E-REAL-5."""

from __future__ import annotations

import copy
import json
import math
import re
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from . import native_x86_kernel
from .graph_ir import GraphIRModule
from .scheduled_matmul import digest_text, find_tessera_opt, run_tessera_opt


_HASH_RE = re.compile(r'tessera\.schedule_hash = "([0-9a-f]{64})"')
_X86_GRAPH_CACHE: OrderedDict[tuple[object, ...], ScheduledKernelArtifact] = OrderedDict()
_X86_GRAPH_CACHE_LOCK = RLock()
_X86_GRAPH_CACHE_LIMIT = 64


def _tool_identity(path: Path) -> tuple[object, ...] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


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
    epsilon: float = 0.0
    #: E2E-REAL-6 x86 elementwise / cohort-2 / breadth: the serialized
    #: contract stem (``x86_kernel`` or a NativeAbsolute ``absolute`` /
    #: ``floor`` / ``ceil`` / ``cumsum``) and every input binding in operand
    #: order. Empty for the semantic-kernel families (softmax/reduce/norm).
    record: str = ""
    input_names: tuple[str, ...] = ()

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
        if self.record:
            self._validate_record()
            return
        schedule_op = f"schedule.{self.family}"
        tile_op = f"tile.{self.family}_kernel"
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

    def _validate_record(self) -> None:
        """A native contract carried on the durable ``schedule.artifact`` record."""
        from .native_x86_kernel import TILE_OPS

        if self.target != "x86" or self.architecture != "zen5-avx512":
            raise ValueError("x86 native kernel artifact requires the zen5-avx512 target")
        if len(re.findall(r"(?m)^\s*schedule\.artifact\b", self.schedule_ir)) != 1:
            raise ValueError("x86 native kernel artifact requires one durable schedule record")
        record_hash = re.findall(r'(?<![\w.])hash = "' + re.escape(self.schedule_digest) + '"', self.schedule_ir)
        if (len(record_hash) != 1
                or self.schedule_ir.count(f'schedule.artifact_hash = "{self.schedule_digest}"') != 1):
            raise ValueError("x86 native kernel artifact has incomplete Schedule digest identity")
        if self.tile_ir.count(TILE_OPS.get(self.family, "\0")) != 1:
            raise ValueError(f"x86 native kernel artifact requires exactly one {self.family} launch op")
        if re.search(r"=\s*tessera\.[a-z]", self.tile_ir) or re.search(r"(?m)^\s*schedule\.", self.tile_ir):
            raise ValueError("x86 native kernel Tile artifact retains Graph or Schedule ops")
        if _HASH_RE.findall(self.tile_ir) != [self.schedule_digest]:
            raise ValueError("x86 native kernel Tile artifact has a stale schedule digest")
        if f"tessera.{self.record}_contract = {{" not in self.tile_ir:
            raise ValueError("x86 native kernel Tile artifact lost its serialized contract")
        if not self.input_names or self.input_names[0] != self.input_name:
            raise ValueError("x86 native kernel artifact requires its ordered input bindings")
        if self.schedule_ir_digest == self.tile_digest:
            raise ValueError("Schedule and Tile artifacts must be distinct boundary outputs")


def supports_scheduled_kernel(module: GraphIRModule, *, target: str) -> bool:
    try:
        if target == "x86" and native_x86_kernel.owns(module):
            native_x86_kernel.admit(module)
        else:
            _graph_contract(module, target)
    except ValueError:
        return False
    return True


def _lower_x86_kernel(module: GraphIRModule, architecture: str | None) -> ScheduledKernelArtifact:
    """E2E-REAL-6 x86 elementwise / cohort-2 / breadth: the native contract."""
    if architecture not in (None, "zen5-avx512"):
        raise ValueError("x86 native kernels exist only for the zen5-avx512 image")
    request = native_x86_kernel.admit(module)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("scheduled x86 kernel lowering requires production tessera-opt")
    graph_ir, schedule_ir, tile_ir = native_x86_kernel.lower(module, request, tool)
    hashes = _HASH_RE.findall(tile_ir)
    if len(hashes) != 1:
        raise RuntimeError("scheduled x86 kernel lowering did not preserve one schedule digest")
    storage = {"fp32": "f32", "bool": "i8", "int32": "i32", "int64": "i64"}
    input_shape, output_shape = request.shapes[0], request.shapes[-1]
    artifact = ScheduledKernelArtifact(
        graph_ir=graph_ir, schedule_ir=schedule_ir, tile_ir=tile_ir,
        target="x86", architecture="zen5-avx512",
        function_name=module.functions[0].name,
        family=request.family, kind=request.kind,
        input_name=request.bindings[0], output_name=request.bindings[-1],
        input_shape=input_shape, output_shape=output_shape,
        dtype=request.dtypes[0], storage=storage.get(request.dtypes[0], ""), accum="",
        axis=-1, keepdims=bool(request.kwargs.get("keepdims", False)),
        rows=math.prod(input_shape[:-1]), columns=input_shape[-1],
        outer=1, axis_extent=1, inner=1, workgroup_size=1,
        schedule_digest=hashes[0], record=request.record,
        input_names=request.bindings[:-1],
    )
    artifact.validate()
    return artifact


def lower_scheduled_kernel(
    module: GraphIRModule,
    *,
    target: str,
    schedule: str | None = None,
    architecture: str | None = None,
) -> ScheduledKernelArtifact:
    if target == "x86" and native_x86_kernel.owns(module):
        if schedule is not None:
            raise ValueError("x86 native kernels take no reduction schedule")
        return _lower_x86_kernel(module, architecture)
    if schedule is not None:
        module = copy.deepcopy(module)
        if (module.functions and module.functions[0].body
                and module.functions[0].body[0].op_name in {"tessera.reduce", "tessera.sum", "tessera.mean", "tessera.max", "tessera.min", "tessera.amax", "tessera.amin"}):
            module.functions[0].body[0].kwargs["schedule"] = schedule
    contract = _graph_contract(module, target)
    if architecture is not None:
        if target != 'x86' or architecture not in ('zen5-avx512','x86_64_base'):
            raise ValueError('unsupported scheduled unary architecture')
        contract = (contract[0], architecture, *contract[2:])
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("scheduled softmax/reduction lowering requires production tessera-opt")

    targeted = copy.deepcopy(module)
    op = targeted.functions[0].body[0]
    if op.op_name == "tessera.softmax_safe":
        # The public ``softmax_safe`` *is* ``softmax`` (``tessera.softmax_safe``
        # is defined as the max-subtracted softmax). The contract admits it only
        # where a Graph-owned packager used to serve it, and the Schedule record
        # names the one semantic it lowers.
        op.op_name = "tessera.softmax"
        op.kwargs = {**op.kwargs, "axis": -1}
    if contract[5] == "norm":
        op.kwargs = {"eps": contract[21]}
        if target == "x86":
            # The x86 native-package request marker (NativeX86Kernel.h): an
            # isolated norm is claimed for `schedule.norm` only when the module
            # names its host bindings; a norm in any other x86 program is not.
            targeted.module_attrs["tessera.launch_bindings"] = json.dumps([contract[3], contract[4]])
    if contract[5] == "reduce":
        op.op_name = "tessera.reduce"
        op.kwargs = {"kind": contract[6], "axis": contract[14]}
        if target in {"nvidia_sm120", "x86"}:
            op.kwargs.update(keepdims=contract[15], schedule=contract[20])
        elif contract[15]:
            # gfx1151 keepdims (E2E-REAL-6). Only a true value is spelled so the
            # established rank-reducing f32 Graph text is byte-identical.
            op.kwargs["keepdims"] = True
    targeted.module_attrs["tessera.target"] = f'"{contract[0]}"'
    targeted.module_attrs["tessera.arch"] = f'"{contract[1]}"'
    graph_ir = targeted.to_mlir(target=target, canonical=True)
    tool_identity = _tool_identity(tool)
    cache_key = None
    if target == "x86" and tool_identity is not None:
        cache_key = (graph_ir, tool_identity, run_tessera_opt)
        with _X86_GRAPH_CACHE_LOCK:
            if cached := _X86_GRAPH_CACHE.get(cache_key):
                _X86_GRAPH_CACHE.move_to_end(cache_key)
                return cached
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
        function_name=(f"tessera_tile_norm_{contract[6]}_{contract[10]}_{hashes[0][:10]}"
                       if contract[5] == "norm" and contract[0] == "nvidia_sm120" else contract[2]),
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
        epsilon=contract[21],
    )
    artifact.validate()
    if cache_key is not None and tool_identity == _tool_identity(tool):
        with _X86_GRAPH_CACHE_LOCK:
            _X86_GRAPH_CACHE[cache_key] = artifact
            _X86_GRAPH_CACHE.move_to_end(cache_key)
            if len(_X86_GRAPH_CACHE) > _X86_GRAPH_CACHE_LIMIT:
                _X86_GRAPH_CACHE.popitem(last=False)
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
    # E2E-REAL-6 (ROCm unary family): gfx1151 carries the envelope the retired
    # Graph-owned ``rocm_native.package_{softmax,reduction}`` constructors served
    # with device proof -- f16/f32 softmax (incl. ``softmax_safe``), f16/bf16/f32
    # sum/mean/max with f32 output, keepdims. gfx1201 keeps its proved f32
    # rank-reducing envelope: evidence never transfers between the two chips.
    rocm_unary = target == "rocm_gfx1151"
    # The admitted targets package the same stable row-softmax semantic
    # from a native Schedule/Tile consumer. Keep gfx1201 and Apple withheld
    # until their own Graph lane and exact-device rows exist.
    if op.op_name == "tessera.softmax_safe" and target not in {"rocm_gfx1151", "x86"}:
        raise ValueError("scheduled softmax_safe is admitted only where it has a proved consumer")
    if rocm_unary:
        softmax_like = op.op_name in {"tessera.softmax", "tessera.softmax_safe"}
        if softmax_like and (dtype not in {"fp16", "fp32"} or output_dtype != dtype):
            raise ValueError("gfx1151 scheduled softmax requires f16/f32 storage preserved to the output")
        if not softmax_like and (dtype not in {"fp16", "bf16", "fp32"} or output_dtype != "fp32"):
            raise ValueError("gfx1151 scheduled reduction requires f16/bf16/f32 storage and f32 output")
    elif target == "nvidia_sm120" or (target == "apple_gpu" and op.op_name == "tessera.softmax") or (
        target == "rocm_gfx1201" and op.op_name in {"tessera.rmsnorm", "tessera.rmsnorm_safe"}
    ):
        expected_dtype = dtype if op.op_name in {"tessera.softmax", "tessera.softmax_safe", "tessera.rmsnorm", "tessera.rmsnorm_safe", "tessera.layer_norm"} else "fp32"
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
    elif target in {"rocm_gfx1151", "rocm_gfx1201"}:
        compiler_target, architecture, workgroup_size = "rocm", target.removeprefix("rocm_"), 256
    elif target == "nvidia_sm120":
        compiler_target, architecture, workgroup_size = "nvidia_sm120", "sm_120", 128
    elif target == "apple_gpu":
        compiler_target, architecture, workgroup_size = "apple_gpu", "apple7", 1
    else:
        raise ValueError("unsupported scheduled semantic-kernel target")

    epsilon = 0.0
    if op.op_name in {"tessera.rmsnorm", "tessera.rmsnorm_safe", "tessera.layer_norm"}:
        from .nvidia_native import _norm_contract

        # E2E-REAL-6 x86 (2026-09-28): Zen 5 carries the static f32
        # unweighted row normalization the retired `package_cohort2` served.
        norm = _norm_contract(module) if target in {"nvidia_sm120", "x86", "rocm_gfx1201"} else None
        if norm is not None and target == "x86" and norm[0] != "fp32":
            norm = None
        if norm is not None and target == "rocm_gfx1201" and (
            norm[0] not in {"fp16", "bf16", "fp32"} or norm[1] != "rmsnorm"
        ):
            norm = None
        if norm is None or op.kwargs.get("numeric_policy") is not None or mode != "serial":
            raise ValueError("unsupported scheduled normalization contract")
        try:
            epsilon = struct.unpack("f", struct.pack("f", norm[2]))[0]
        except OverflowError as exc:
            raise ValueError("normalization epsilon must fit f32") from exc
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("normalization epsilon must be positive finite f32")
        family, kind, axis, keepdims = "norm", norm[1], -1, False
        rows, columns = math.prod(input_shape[:-1]), input_shape[-1]
        outer = axis_extent = inner = 1
    elif op.op_name in {"tessera.softmax", "tessera.softmax_safe"}:
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
        if kind not in allowed or (keepdims and target not in {"nvidia_sm120", "x86", "rocm_gfx1151"}):
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
        columns, axis, keepdims, outer, axis_extent, inner, workgroup_size, mode, epsilon,
    )
