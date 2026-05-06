"""Tile IR object model and Schedule IR lowering.

Tile IR binds scheduled graph computation to tile-level movement and execution
ops. This model covers the current compiler surface: tile.* execution/movement
ops, tessera.attn.* FA-4 helpers, and tessera.queue.* producer/consumer
barriers used by warp-specialized pipelines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..diagnostics import DiagnosticLevel, DiagnosticWhere, TesseraDiagnostic, TesseraErrorCode
from .schedule_ir import ScheduleIRModule, ScheduleIRVerificationError, ScheduleOp


TILE_MEMORY_OPS = {"tile.async_copy", "tile.wait_async"}
QUEUE_OPS = {"tessera.queue.create", "tessera.queue.push", "tessera.queue.pop", "tessera.queue.barrier"}
ATTN_OPS = {"tessera.attn.scaled_dot_product", "tessera.attn.online_softmax", "tessera.attn.lse_save", "tessera.attn.attend_v"}


def _diagnostic_level(severity: str) -> DiagnosticLevel:
    return {
        "fatal": DiagnosticLevel.FATAL,
        "error": DiagnosticLevel.ERROR,
        "warning": DiagnosticLevel.WARNING,
        "info": DiagnosticLevel.INFO,
        "note": DiagnosticLevel.NOTE,
    }.get(severity.lower(), DiagnosticLevel.ERROR)


@dataclass(frozen=True)
class TileIRDiagnostic:
    severity: str
    message: str
    code: str = "TILE_IR"

    def format(self) -> str:
        structured = self.to_tessera_diagnostic()
        return f"{structured.level.value.upper()} [{structured.code.value}] [{self.code}]: {self.message}\n  where: {structured.where}"

    def to_tessera_diagnostic(self) -> TesseraDiagnostic:
        return TesseraDiagnostic(
            level=_diagnostic_level(self.severity),
            message=self.message,
            code=TesseraErrorCode.TILE_LOWERING,
            where=DiagnosticWhere(ir_level="tile-ir", pass_name="verifier"),
            hints=["inspect Tile IR async copies, queues, attention ops, and barriers"],
        )


@dataclass(frozen=True)
class TileIRVerificationResult:
    diagnostics: tuple[TileIRDiagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)

    def format(self) -> str:
        return "\n".join(d.format() for d in self.diagnostics)

    def structured_diagnostics(self) -> tuple[TesseraDiagnostic, ...]:
        return tuple(d.to_tessera_diagnostic() for d in self.diagnostics)


class TileIRVerificationError(ValueError):
    pass


@dataclass
class TileOp:
    op_name: str
    attrs: dict[str, Any] = field(default_factory=dict)
    operands: list[str] = field(default_factory=list)
    result: Optional[str] = None
    body: list["TileOp"] = field(default_factory=list)

    def to_mlir(self, indent: str = "  ") -> str:
        result_text = f"%{self.result} = " if self.result else ""
        operands = ", ".join(self.operands)
        attr_text = _format_attr_dict(self.attrs)
        if self.body:
            lines = [f"{indent}{result_text}\"{self.op_name}\"({operands}) ({{"]
            for child in self.body:
                lines.append(child.to_mlir(indent + "  "))
            lines.append(f"{indent}}}) {attr_text} : () -> ()")
            return "\n".join(lines)
        return f"{indent}{result_text}\"{self.op_name}\"({operands}) {attr_text} : () -> ()"


@dataclass
class TileFunction:
    name: str
    body: list[TileOp] = field(default_factory=list)
    target: str = "cpu"

    def to_mlir(self, indent: str = "  ") -> str:
        lines = [f"{indent}\"tessera.tile.func\"() ({{"]
        for op in self.body:
            lines.append(op.to_mlir(indent + "  "))
        lines.append(f"{indent}}}) {{sym_name = {json.dumps(self.name)}, target = {json.dumps(self.target)}}} : () -> ()")
        return "\n".join(lines)


@dataclass
class TileIRModule:
    functions: list[TileFunction] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=lambda: {"tessera.ir.level": "tile"})

    def verify(self) -> TileIRVerificationResult:
        return TileIRVerifier().verify_module(self)

    def to_mlir(self, *, verify: bool = True) -> str:
        if verify:
            result = self.verify()
            if not result.ok:
                raise TileIRVerificationError(result.format())
        lines = [f"module attributes {_format_attr_dict(self.attrs)} {{"]
        for fn in self.functions:
            lines.append(fn.to_mlir())
        lines.append("}")
        return "\n".join(lines)


class TileIRVerifier:
    def verify_module(self, module: TileIRModule) -> TileIRVerificationResult:
        diagnostics: list[TileIRDiagnostic] = []
        for fn in module.functions:
            diagnostics.extend(self.verify_function(fn).diagnostics)
        return TileIRVerificationResult(tuple(diagnostics))

    def verify_function(self, fn: TileFunction) -> TileIRVerificationResult:
        diagnostics: list[TileIRDiagnostic] = []
        queues: set[int] = set()
        self._verify_ops(fn.body, diagnostics, queues)
        return TileIRVerificationResult(tuple(diagnostics))

    def _verify_ops(self, ops: Iterable[TileOp], diagnostics: list[TileIRDiagnostic], queues: set[int]) -> None:
        for op in ops:
            if op.op_name == "tile.async_copy":
                self._verify_async_copy(op, diagnostics)
            elif op.op_name == "tile.wait_async":
                if int(op.attrs.get("stage", -1)) < 0:
                    diagnostics.append(TileIRDiagnostic("error", "wait_async stage must be >= 0", "TILE_IR_WAIT_STAGE"))
            elif op.op_name == "tile.mma":
                self._require_attrs(op, diagnostics, "source", "result", "ordinal")
            elif op.op_name == "tile.reduce":
                self._require_attrs(op, diagnostics, "op", "order")
            elif op.op_name == "tessera.attn.online_softmax":
                if not op.attrs.get("policy"):
                    diagnostics.append(TileIRDiagnostic("error", "online_softmax requires policy", "TILE_IR_ATTN_POLICY"))
            elif op.op_name == "tessera.queue.create":
                queue_id = int(op.attrs.get("queue_id", -1))
                depth = int(op.attrs.get("depth", 0))
                if queue_id < 0:
                    diagnostics.append(TileIRDiagnostic("error", "queue.create queue_id must be >= 0", "TILE_IR_QUEUE_ID"))
                elif queue_id in queues:
                    diagnostics.append(TileIRDiagnostic("error", f"duplicate queue {queue_id}", "TILE_IR_DUP_QUEUE"))
                else:
                    queues.add(queue_id)
                if depth < 1:
                    diagnostics.append(TileIRDiagnostic("error", "queue.create depth must be >= 1", "TILE_IR_QUEUE_DEPTH"))
            elif op.op_name in {"tessera.queue.push", "tessera.queue.pop", "tessera.queue.barrier"}:
                queue_id = int(op.attrs.get("queue_id", -1))
                if queue_id not in queues:
                    diagnostics.append(TileIRDiagnostic("error", f"{op.op_name} references undefined queue {queue_id}", "TILE_IR_UNDEFINED_QUEUE"))
            self._verify_ops(op.body, diagnostics, queues)

    def _verify_async_copy(self, op: TileOp, diagnostics: list[TileIRDiagnostic]) -> None:
        if int(op.attrs.get("stage", -1)) < 0:
            diagnostics.append(TileIRDiagnostic("error", "async_copy stage must be >= 0", "TILE_IR_ASYNC_STAGE"))
        if int(op.attrs.get("vector", 0)) < 1:
            diagnostics.append(TileIRDiagnostic("error", "async_copy vector must be >= 1", "TILE_IR_ASYNC_VECTOR"))

    def _require_attrs(self, op: TileOp, diagnostics: list[TileIRDiagnostic], *attrs: str) -> None:
        missing = [attr for attr in attrs if attr not in op.attrs]
        if missing:
            diagnostics.append(TileIRDiagnostic("error", f"{op.op_name} missing attrs: {', '.join(missing)}", "TILE_IR_MISSING_ATTR"))


def lower_schedule_to_tile_ir(schedule_module: ScheduleIRModule, *, target_kind: str = "cpu") -> TileIRModule:
    schedule_result = schedule_module.verify()
    if not schedule_result.ok:
        raise ScheduleIRVerificationError(schedule_result.format())
    tile_module = TileIRModule(attrs={"tessera.ir.level": "tile", "target": target_kind})
    for schedule_fn in schedule_module.functions:
        body = _lower_schedule_ops(schedule_fn.body)
        tile_module.functions.append(TileFunction(schedule_fn.name, body=body, target=target_kind))
    return tile_module


def _lower_schedule_ops(ops: list[ScheduleOp]) -> list[TileOp]:
    lowered: list[TileOp] = []
    for op in ops:
        if op.op_name in {"schedule.mesh.define", "schedule.layout", "schedule.artifact"}:
            continue
        if op.op_name == "schedule.debug_artifact":
            lowered.append(TileOp("tile.debug_artifact", dict(op.attrs)))
            continue
        if op.op_name == "schedule.mesh.region":
            lowered.extend(_lower_schedule_ops(op.body[:-1]))
            continue
        if op.op_name == "schedule.pipeline.region":
            lowered.extend(_lower_pipeline_region(op))
            continue
        if op.op_name == "schedule.tile":
            lowered.append(_tile_compute_op(op))
            continue
        if op.op_name == "schedule.elementwise":
            lowered.append(_elementwise_op(op))
            continue
        if op.op_name == "schedule.prefetch":
            lowered.append(TileOp("tile.async_copy", _copy_attrs(op)))
            continue
        if op.op_name == "schedule.collective":
            lowered.append(TileOp("tile.collective", dict(op.attrs)))
            continue
        if op.op_name == "schedule.marker":
            marker = op.attrs.get("marker")
            if marker == "tessera.barrier":
                queue_id = int(op.attrs.get("ordinal", 0))
                lowered.append(TileOp("tessera.queue.create", {"queue_id": queue_id, "depth": 1, "producer_warps": 1, "consumer_warps": 1}))
                lowered.append(TileOp("tessera.queue.barrier", {"queue_id": queue_id, "scope": "block"}))
                lowered.append(TileOp("tile.debug_barrier", {"queue_id": queue_id, "scope": "block", "source": marker, "ordinal": queue_id}))
            continue
    return lowered


def _lower_pipeline_region(op: ScheduleOp) -> list[TileOp]:
    attrs = dict(op.attrs)
    source = str(attrs.get("source", "schedule.pipeline"))
    result = attrs.get("result", "pipeline")
    queue_id = int(attrs.get("ordinal", 0))
    lowered = [
        TileOp("tessera.queue.create", {"queue_id": queue_id, "depth": max(1, int(attrs.get("micro_batches", 1))), "producer_warps": 1, "consumer_warps": 1}),
        TileOp("tile.async_copy", {"source": source, "result": result, "ordinal": attrs.get("ordinal", 0), "stage": 0, "vector": 16}),
        TileOp("tessera.queue.push", {"queue_id": queue_id, "stage": 0}),
        TileOp("tessera.queue.pop", {"queue_id": queue_id, "stage": 0}),
    ]
    if source == "tessera.flash_attn" or attrs.get("schedule") == "fa4":
        lowered.extend([
            TileOp("tessera.attn.scaled_dot_product", {"source": source, "result": result, "ordinal": attrs.get("ordinal", 0), "causal": bool(attrs.get("causal", True))}),
            TileOp("tessera.attn.online_softmax", {"source": source, "result": result, "ordinal": attrs.get("ordinal", 0), "policy": "safe"}),
            TileOp("tessera.attn.lse_save", {"source": source, "result": result, "ordinal": attrs.get("ordinal", 0)}),
            TileOp("tessera.attn.attend_v", {"source": source, "result": result, "ordinal": attrs.get("ordinal", 0)}),
        ])
    lowered.extend([
        TileOp("tile.wait_async", {"stage": 0}),
        TileOp("tessera.queue.barrier", {"queue_id": queue_id, "scope": "warpgroup"}),
        TileOp("tile.debug_barrier", {"queue_id": queue_id, "scope": "warpgroup", "source": source, "ordinal": attrs.get("ordinal", 0)}),
    ])
    return lowered


def _tile_compute_op(op: ScheduleOp) -> TileOp:
    source = str(op.attrs.get("source", ""))
    if source in {"tessera.matmul", "tessera.gemm"}:
        tile_name = "tile.mma"
    elif source in {"tessera.conv2d", "tessera.conv2d_nhwc"}:
        tile_name = "tile.conv2d"
    else:
        tile_name = "tile.generic"
    attrs = {**dict(op.attrs), "lowering": _lowering_kind(source), "vectorize": True}
    if tile_name == "tile.mma":
        attrs.update(_mma_resource_estimate(attrs))
    return TileOp(tile_name, attrs)


def _elementwise_op(op: ScheduleOp) -> TileOp:
    source = str(op.attrs.get("source", ""))
    bare = source.removeprefix("tessera.")
    tile_name = "tile.rope" if source == "tessera.rope" else f"tile.{bare or 'elementwise'}"
    attrs = {
        **dict(op.attrs),
        "lowering": _lowering_kind(source),
        "vectorize": op.attrs.get("vectorize", True),
        "resource": _elementwise_resource_estimate(source),
    }
    ops = [TileOp(tile_name, attrs)]
    if source == "tessera.rope":
        ops.append(TileOp("tile.rotary_pair", {**dict(op.attrs), "vector": 2}))
        return TileOp("tile.group", {"source": source, "ordinal": op.attrs.get("ordinal", 0)}, body=ops)
    if source.startswith("tessera.kv_cache."):
        ops.append(TileOp("tile.kv_cache", {**dict(op.attrs), "storage": "paged"}))
        return TileOp("tile.group", {"source": source, "ordinal": op.attrs.get("ordinal", 0)}, body=ops)
    return ops[0]


def _copy_attrs(op: ScheduleOp) -> dict[str, Any]:
    vector = int(op.attrs.get("vector", 16))
    return {
        "source": op.attrs.get("source", "schedule.prefetch"),
        "result": op.attrs.get("result", "prefetch"),
        "ordinal": op.attrs.get("ordinal", 0),
        "stage": int(op.attrs.get("stage", 0)),
        "vector": vector,
        "bytes": int(op.attrs.get("bytes", vector * 4)),
        "resource": {"async_copy_bytes": int(op.attrs.get("bytes", vector * 4)), "barrier_count": 1},
    }


def _mma_resource_estimate(attrs: dict[str, Any]) -> dict[str, Any]:
    tile_m = int(attrs.get("tile_m", 128))
    tile_n = int(attrs.get("tile_n", 128))
    tile_k = int(attrs.get("tile_k", 64))
    num_stages = int(attrs.get("num_stages", 2))
    smem = 2 * (tile_m * tile_k + tile_k * tile_n) * num_stages
    registers = max(32, (tile_m * tile_n) // 256)
    return {
        "resource": {
            "shared_memory_bytes": smem,
            "register_estimate": registers,
            "async_copy_bytes": int(attrs.get("bytes_moved", 0)),
            "queue_depth": num_stages,
            "barrier_count": num_stages,
        }
    }


def _elementwise_resource_estimate(source: str) -> dict[str, Any]:
    if source in {"tessera.softmax", "tessera.softmax_safe"}:
        return {"shared_memory_bytes": 1024, "register_estimate": 24, "async_copy_bytes": 0, "queue_depth": 0, "barrier_count": 1}
    return {"shared_memory_bytes": 0, "register_estimate": 16, "async_copy_bytes": 0, "queue_depth": 0, "barrier_count": 0}


def _lowering_kind(op_name: str) -> str:
    if op_name in {"tessera.matmul", "tessera.gemm"}:
        return "tensor_core_mma"
    if op_name == "tessera.flash_attn":
        return "fa4_pipeline"
    if op_name.startswith("tessera.kv_cache."):
        return "paged_state"
    if op_name in {"tessera.transpose", "tessera.cast"}:
        return "layout_transform"
    if op_name in {"tessera.softmax", "tessera.softmax_safe"}:
        return "stable_reduction"
    if op_name == "tessera.adam":
        return "functional_optimizer_step"
    return "elementwise"


def _format_attr_dict(attrs: dict[str, Any]) -> str:
    if not attrs:
        return "{}"
    return "{" + ", ".join(f"{key} = {_format_attr_value(value)}" for key, value in attrs.items()) + "}"


def _format_attr_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int):
        return f"{value} : i64"
    if isinstance(value, float):
        return repr(value)
    if value is None:
        return "none"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_format_attr_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return json.dumps(json.dumps(value, sort_keys=True))
    return json.dumps(str(value))
