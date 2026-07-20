"""Schedule IR object model and Graph IR lowering.

This layer captures the schedule.* dialect shape used between Graph IR and Tile
IR: mesh definitions, mesh regions, pipeline regions, pipeline stages, and
schedule terminators. It intentionally mirrors the MLIR dialect enough for
frontend tests and developer inspection while remaining usable without native
MLIR bindings.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..diagnostics import TesseraDiagnostic
from .graph_ir import GraphIRDiagnostic, GraphIRModule, GraphIRVerificationError, IROp
from .op_catalog import canonical_graph_op_name


SCHEDULE_MEMORY_SPACES = {"register", "shared", "lds", "global", "managed", "host", "tmem"}
SCHEDULE_OVERLAPS = {"none", "compute", "collective"}
MATMUL_OPS = {"tessera.matmul", "tessera.gemm"}
CONV2D_OPS = {"tessera.conv2d_nhwc", "tessera.conv2d"}
ROPE_OPS = {"tessera.rope"}
MEDIA_OPS = {
    "tessera.image_preprocess",
    "tessera.video_frame_sample",
    "tessera.patch_embed",
    "tessera.patch_merge",
    "tessera.media_project",
    "tessera.splice_embeddings",
}
JEPA_OPS = {
    "tessera.jepa.mask_blocks_2d",
    "tessera.jepa.mask_tubes_3d",
    "tessera.jepa.gather_context",
    "tessera.jepa.gather_targets",
    "tessera.jepa.stop_gradient",
    "tessera.jepa.ema_update",
    "tessera.jepa.latent_predict",
    "tessera.jepa.l2_loss",
    "tessera.jepa.selective_decode",
    "tessera.jepa.train_step",
    "tessera.jepa_mask_blocks_2d",
    "tessera.jepa_mask_tubes_3d",
    "tessera.jepa_gather_context",
    "tessera.jepa_gather_targets",
    "tessera.jepa_stop_gradient",
    "tessera.jepa_ema_update",
    "tessera.jepa_latent_predict",
    "tessera.jepa_l2_loss",
    "tessera.jepa_selective_decode",
    "tessera.jepa_train_step",
}


@dataclass(frozen=True)
class ScheduleIRVerificationResult:
    diagnostics: tuple[GraphIRDiagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)

    def format(self) -> str:
        return "\n".join(d.format() for d in self.diagnostics)

    def structured_diagnostics(self) -> tuple[TesseraDiagnostic, ...]:
        return tuple(d.to_tessera_diagnostic() for d in self.diagnostics)


class ScheduleIRVerificationError(ValueError):
    pass


@dataclass
class ScheduleOp:
    op_name: str
    attrs: dict[str, Any] = field(default_factory=dict)
    operands: list[str] = field(default_factory=list)
    result: Optional[str] = None
    body: list["ScheduleOp"] = field(default_factory=list)
    source_op: Optional[IROp] = None

    def to_mlir(self, indent: str = "  ") -> str:
        attr_text = _format_attr_dict(self.attrs)
        operand_text = ", ".join(self.operands)
        result_text = f"%{self.result} = " if self.result else ""
        if self.body:
            lines = [f"{indent}{result_text}\"{self.op_name}\"({operand_text}) ({{"]
            for child in self.body:
                lines.append(child.to_mlir(indent + "  "))
            lines.append(f"{indent}}}) {attr_text} : () -> ()")
            return "\n".join(lines)
        return f"{indent}{result_text}\"{self.op_name}\"({operand_text}) {attr_text} : () -> ()"


@dataclass
class ScheduleFunction:
    name: str
    body: list[ScheduleOp] = field(default_factory=list)
    target: str = "cpu"

    def to_mlir(self, indent: str = "  ") -> str:
        lines = [f"{indent}\"tessera.schedule.func\"() ({{"]
        for op in self.body:
            lines.append(op.to_mlir(indent + "  "))
        lines.append(f"{indent}}}) {{sym_name = {json.dumps(self.name)}, target = {json.dumps(self.target)}}} : () -> ()")
        return "\n".join(lines)


@dataclass
class ScheduleIRModule:
    functions: list[ScheduleFunction] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=lambda: {"tessera.ir.level": "schedule"})

    def verify(self) -> ScheduleIRVerificationResult:
        return ScheduleIRVerifier().verify_module(self)

    def to_mlir(self, *, verify: bool = True) -> str:
        if verify:
            result = self.verify()
            if not result.ok:
                raise ScheduleIRVerificationError(result.format())
        attr_text = _format_attr_dict(self.attrs)
        lines = [f"module attributes {attr_text} {{"]
        for fn in self.functions:
            lines.append(fn.to_mlir())
        lines.append("}")
        return "\n".join(lines)


class ScheduleIRVerifier:
    def verify_module(self, module: ScheduleIRModule) -> ScheduleIRVerificationResult:
        diagnostics: list[GraphIRDiagnostic] = []
        for fn in module.functions:
            diagnostics.extend(self.verify_function(fn).diagnostics)
        return ScheduleIRVerificationResult(tuple(diagnostics))

    def verify_function(self, fn: ScheduleFunction) -> ScheduleIRVerificationResult:
        diagnostics: list[GraphIRDiagnostic] = []
        meshes: dict[str, ScheduleOp] = {}
        self._verify_ops(fn.body, diagnostics, meshes=meshes, region_stack=[])
        return ScheduleIRVerificationResult(tuple(diagnostics))

    def _verify_ops(
        self,
        ops: Iterable[ScheduleOp],
        diagnostics: list[GraphIRDiagnostic],
        *,
        meshes: dict[str, ScheduleOp],
        region_stack: list[str],
    ) -> None:
        for op in ops:
            if op.op_name == "schedule.mesh.define":
                self._verify_mesh_define(op, diagnostics, meshes)
            elif op.op_name == "schedule.mesh.region":
                self._verify_mesh_region(op, diagnostics, meshes, region_stack)
            elif op.op_name == "schedule.pipeline.region":
                self._verify_pipeline_region(op, diagnostics, meshes, region_stack)
            elif op.op_name == "schedule.stage":
                self._verify_stage(op, diagnostics, meshes, region_stack)
            elif op.op_name == "schedule.prefetch":
                self._verify_prefetch(op, diagnostics)
            elif op.op_name == "schedule.state.read":
                self._verify_state_read(op, diagnostics)
            elif op.op_name == "schedule.yield":
                if not region_stack:
                    diagnostics.append(GraphIRDiagnostic(
                        "error",
                        "schedule.yield must terminate a schedule region",
                        code="SCHEDULE_IR_YIELD_OUTSIDE_REGION",
                    ))

    def _verify_mesh_define(
        self,
        op: ScheduleOp,
        diagnostics: list[GraphIRDiagnostic],
        meshes: dict[str, ScheduleOp],
    ) -> None:
        name = op.attrs.get("sym_name") or op.attrs.get("name")
        dims = op.attrs.get("dims")
        axes = op.attrs.get("axis_names")
        if not name:
            diagnostics.append(GraphIRDiagnostic("error", "mesh.define requires sym_name", code="SCHEDULE_IR_MESH_NAME"))
        elif name in meshes:
            diagnostics.append(GraphIRDiagnostic("error", f"duplicate mesh {name!r}", code="SCHEDULE_IR_DUP_MESH"))
        else:
            meshes[str(name)] = op
        if not isinstance(dims, list) or not dims:
            diagnostics.append(GraphIRDiagnostic("error", "mesh.define requires non-empty dims", code="SCHEDULE_IR_MESH_DIMS"))
        elif any(int(dim) <= 0 for dim in dims):
            diagnostics.append(GraphIRDiagnostic("error", "mesh dimensions must be positive", code="SCHEDULE_IR_MESH_DIMS"))
        if not isinstance(axes, list) or not axes:
            diagnostics.append(GraphIRDiagnostic("error", "mesh.define requires axis_names", code="SCHEDULE_IR_MESH_AXES"))
        elif isinstance(dims, list) and len(dims) != len(axes):
            diagnostics.append(GraphIRDiagnostic("error", "mesh dims and axis_names length mismatch", code="SCHEDULE_IR_MESH_RANK"))

    def _verify_mesh_region(
        self,
        op: ScheduleOp,
        diagnostics: list[GraphIRDiagnostic],
        meshes: dict[str, ScheduleOp],
        region_stack: list[str],
    ) -> None:
        mesh = str(op.attrs.get("mesh", ""))
        axis = op.attrs.get("axis")
        if mesh not in meshes:
            diagnostics.append(GraphIRDiagnostic("error", f"mesh.region references undefined mesh {mesh!r}", code="SCHEDULE_IR_UNDEFINED_MESH"))
        else:
            axes = meshes[mesh].attrs.get("axis_names", [])
            if axis not in axes:
                diagnostics.append(GraphIRDiagnostic("error", f"axis {axis!r} is not defined on mesh {mesh!r}", code="SCHEDULE_IR_UNKNOWN_AXIS"))
        if not op.body or op.body[-1].op_name != "schedule.yield":
            diagnostics.append(GraphIRDiagnostic("error", "mesh.region must end with schedule.yield", code="SCHEDULE_IR_REGION_TERMINATOR"))
        self._verify_ops(op.body, diagnostics, meshes=meshes, region_stack=region_stack + ["mesh.region"])

    def _verify_pipeline_region(
        self,
        op: ScheduleOp,
        diagnostics: list[GraphIRDiagnostic],
        meshes: dict[str, ScheduleOp],
        region_stack: list[str],
    ) -> None:
        if not op.attrs.get("schedule"):
            diagnostics.append(GraphIRDiagnostic("error", "pipeline.region requires schedule", code="SCHEDULE_IR_PIPELINE_SCHEDULE"))
        if int(op.attrs.get("micro_batches", 0)) < 1:
            diagnostics.append(GraphIRDiagnostic("error", "pipeline.region micro_batches must be >= 1", code="SCHEDULE_IR_PIPELINE_MICRO_BATCHES"))
        if not any(child.op_name == "schedule.stage" for child in op.body):
            diagnostics.append(GraphIRDiagnostic("error", "pipeline.region requires at least one stage", code="SCHEDULE_IR_PIPELINE_STAGE"))
        if not op.body or op.body[-1].op_name != "schedule.yield":
            diagnostics.append(GraphIRDiagnostic("error", "pipeline.region must end with schedule.yield", code="SCHEDULE_IR_REGION_TERMINATOR"))
        self._verify_ops(op.body, diagnostics, meshes=meshes, region_stack=region_stack + ["pipeline.region"])

    def _verify_stage(
        self,
        op: ScheduleOp,
        diagnostics: list[GraphIRDiagnostic],
        meshes: dict[str, ScheduleOp],
        region_stack: list[str],
    ) -> None:
        if "pipeline.region" not in region_stack:
            diagnostics.append(GraphIRDiagnostic("error", "stage must be inside pipeline.region", code="SCHEDULE_IR_STAGE_PARENT"))
        devices = op.attrs.get("devices")
        if not isinstance(devices, list) or not devices:
            diagnostics.append(GraphIRDiagnostic("error", "stage requires non-empty devices", code="SCHEDULE_IR_STAGE_DEVICES"))
        if not op.body or op.body[-1].op_name != "schedule.yield":
            diagnostics.append(GraphIRDiagnostic("error", "stage must end with schedule.yield", code="SCHEDULE_IR_REGION_TERMINATOR"))
        self._verify_ops(op.body, diagnostics, meshes=meshes, region_stack=region_stack + ["stage"])

    def _verify_prefetch(self, op: ScheduleOp, diagnostics: list[GraphIRDiagnostic]) -> None:
        if op.attrs.get("into") not in SCHEDULE_MEMORY_SPACES:
            diagnostics.append(GraphIRDiagnostic("error", "prefetch has invalid memory space", code="SCHEDULE_IR_MEMORY_SPACE"))
        if op.attrs.get("overlap") not in SCHEDULE_OVERLAPS:
            diagnostics.append(GraphIRDiagnostic("error", "prefetch has invalid overlap policy", code="SCHEDULE_IR_OVERLAP"))

    def _verify_state_read(self, op: ScheduleOp, diagnostics: list[GraphIRDiagnostic]) -> None:
        missing = [key for key in ("source", "result", "ordinal", "effect", "access")
                   if key not in op.attrs]
        if missing:
            diagnostics.append(GraphIRDiagnostic(
                "error", f"state.read missing attrs: {', '.join(missing)}",
                code="SCHEDULE_IR_STATE_READ_ATTR"))
        if op.attrs.get("effect") != "read":
            diagnostics.append(GraphIRDiagnostic(
                "error", "state.read must declare effect=read",
                code="SCHEDULE_IR_STATE_READ_EFFECT"))


def lower_graph_to_schedule_ir(
    graph_module: GraphIRModule,
    *,
    tile: tuple[int, int, int] = (128, 128, 64),
    target_kind: str = "cpu",
) -> ScheduleIRModule:
    graph_result = graph_module.verify(target=target_kind)
    if not graph_result.ok:
        raise GraphIRVerificationError(graph_result.format())
    schedule_module = ScheduleIRModule(attrs={"tessera.ir.level": "schedule", "target": target_kind})
    for graph_fn in graph_module.functions:
        body: list[ScheduleOp] = []
        mesh_ops = [
            ScheduleOp(
                "schedule.mesh.define",
                {
                    "sym_name": mesh.name,
                    "dims": list(mesh.shape),
                    "axis_names": list(mesh.axes),
                },
            )
            for mesh in graph_module.meshes
        ]
        body.extend(mesh_ops)
        scheduled_ops = _lower_graph_ops(graph_fn.body, tile=tile)
        if graph_module.meshes:
            mesh = graph_module.meshes[0]
            for axis in mesh.axes:
                scheduled_ops = [
                    ScheduleOp(
                        "schedule.mesh.region",
                        {"mesh": mesh.name, "axis": axis},
                        body=[*scheduled_ops, ScheduleOp("schedule.yield")],
                    )
                ]
        body.extend(scheduled_ops)
        body.append(ScheduleOp("schedule.artifact", {
            "hash": f"{graph_fn.name}:{len(graph_fn.body)}:{target_kind}",
            "arch": target_kind,
            "shape_key": _shape_key(graph_fn.body),
            "tile": {"m": tile[0], "n": tile[1], "k": tile[2]},
            "movement": {"prefetch": "auto", "overlap": "compute", "stages": 2},
            "numeric_policy": "f32@accum(f32)",
            "cost_model": "roofline",
        }))
        schedule_module.functions.append(ScheduleFunction(graph_fn.name, body=body, target=target_kind))
    return schedule_module


def _lower_graph_ops(ops: list[IROp], *, tile: tuple[int, int, int]) -> list[ScheduleOp]:
    scheduled: list[ScheduleOp] = []
    tile_m, tile_n, tile_k = tile
    for idx, op in enumerate(ops):
        op_name = canonical_graph_op_name(op.op_name)
        if op_name == "tessera.graph.debug_value":
            scheduled.append(ScheduleOp("schedule.debug_artifact", {
                **_base_attrs(op, idx),
                "name": op.kwargs.get("name", op.result or f"debug_{idx}"),
                "capture": "value_summary",
            }))
            continue
        if op_name.startswith("tessera.schedule."):
            scheduled.extend(_lower_schedule_directive(op, idx))
            continue
        if op_name.startswith("tessera.dist."):
            scheduled.append(ScheduleOp("schedule.collective", _base_attrs(op, idx)))
            continue
        if op_name in MATMUL_OPS:
            scheduled.extend([
                ScheduleOp("schedule.knob", {**_base_attrs(op, idx), "name": "tile_m", "choices": [32, 64, 128, 256], "frozen": False}),
                ScheduleOp("schedule.knob", {**_base_attrs(op, idx), "name": "tile_n", "choices": [32, 64, 128, 256], "frozen": False}),
                ScheduleOp("schedule.knob", {**_base_attrs(op, idx), "name": "tile_k", "choices": [32, 64, 128, 256], "frozen": False}),
                ScheduleOp("schedule.knob", {**_base_attrs(op, idx), "name": "num_warps", "choices": [1, 2, 4, 8], "frozen": False}),
                ScheduleOp("schedule.knob", {**_base_attrs(op, idx), "name": "num_stages", "choices": [1, 2, 3, 4], "frozen": False}),
            ])
            scheduled.append(ScheduleOp(
                "schedule.tile",
                {
                    **_base_attrs(op, idx),
                    "tile_m": tile_m,
                    "tile_n": tile_n,
                    "tile_k": tile_k,
                    "num_warps": 4,
                    "num_stages": 2,
                    "cost_model": "roofline",
                    "flops": _matmul_flops(op),
                    "bytes_moved": _matmul_bytes(op),
                },
            ))
        elif op_name in CONV2D_OPS:
            scheduled.append(ScheduleOp(
                "schedule.tile",
                {**_base_attrs(op, idx), "tile_h": 16, "tile_w": 16, "tile_c": 32},
            ))
        elif op_name == "tessera.flash_attn":
            scheduled.append(_flash_attention_pipeline(op, idx))
        elif op_name == "tessera.msa_sparse_attention":
            scheduled.append(_msa_kv_outer_sparse(op, idx))
        elif op_name in MEDIA_OPS:
            scheduled.append(_media_op(op, idx))
        elif op_name in JEPA_OPS:
            scheduled.append(_jepa_op(op, idx))
        elif op_name == "tessera.kv_cache.read":
            scheduled.append(ScheduleOp(
                "schedule.state.read",
                {
                    **_base_attrs(op, idx),
                    "effect": "read",
                    "access": "paged_slice",
                    "bounds": "start_end",
                },
                operands=list(op.operands),
                result=op.result,
                source_op=op,
            ))
        elif op_name in ROPE_OPS:
            scheduled.append(ScheduleOp("schedule.elementwise", {**_base_attrs(op, idx), "vectorize": True, "pattern": "rotary_pairs"}))
        elif op_name.startswith("tessera.scf.") or op_name in {"tessera.barrier", "tessera.assert"}:
            scheduled.append(ScheduleOp("schedule.marker", {**_base_attrs(op, idx), **op.kwargs, "marker": op_name}))
        elif op.result is not None:
            scheduled.append(ScheduleOp("schedule.elementwise", {**_base_attrs(op, idx), "vectorize": True}))
        operand_names = [operand.removeprefix("%") for operand in op.operands]
        if operand_names:
            scheduled.append(ScheduleOp("schedule.layout", {"operands": operand_names, "layout": "row_major", "ordinal": idx}))
        if op_name.startswith("tessera.kv_cache."):
            scheduled.append(ScheduleOp("schedule.prefetch", {**_base_attrs(op, idx), "into": "shared", "overlap": "compute"}))
    return scheduled


def _lower_schedule_directive(op: IROp, ordinal: int) -> list[ScheduleOp]:
    name = op.op_name.removeprefix("tessera.")
    attrs = {**_base_attrs(op, ordinal), **op.kwargs}
    if name == "schedule.pipeline":
        depth = int(attrs.get("depth", attrs.get("stages", 1)))
        stage_count = max(1, depth)
        stages = [
            ScheduleOp("schedule.stage", {"devices": [stage]}, body=[ScheduleOp("schedule.yield")])
            for stage in range(stage_count)
        ]
        return [
            ScheduleOp(
                "schedule.pipeline.region",
                {"schedule": attrs.get("schedule", "gpipe"), "micro_batches": int(attrs.get("micro_batches", 1))},
                body=[*stages, ScheduleOp("schedule.yield")],
            )
        ]
    if name == "schedule.tile":
        return [ScheduleOp("schedule.tile", attrs)]
    if name == "schedule.prefetch":
        return [ScheduleOp("schedule.prefetch", {**attrs, "into": attrs.get("into", attrs.get("scope", "shared")), "overlap": attrs.get("overlap", "compute")})]
    return [ScheduleOp(name, attrs)]


def _flash_attention_pipeline(op: IROp, ordinal: int) -> ScheduleOp:
    return ScheduleOp(
        "schedule.pipeline.region",
        {"schedule": "fa4", "micro_batches": 1, **_base_attrs(op, ordinal)},
        body=[
            ScheduleOp(
                "schedule.stage",
                {"devices": [0]},
                body=[
                    ScheduleOp("schedule.prefetch", {**_base_attrs(op, ordinal), "into": "shared", "overlap": "compute", "tile_q": 64, "tile_kv": 64}),
                    ScheduleOp("schedule.yield"),
                ],
            ),
            ScheduleOp("schedule.yield"),
        ],
    )


def _msa_kv_outer_sparse(op: IROp, ordinal: int) -> ScheduleOp:
    attrs = {**_base_attrs(op, ordinal), **dict(op.kwargs)}
    top_k = int(attrs.get("top_k", attrs.get("top_k_blocks", 1)))
    block_size = int(attrs.get("block_size", 64))
    num_heads = int(attrs.get("num_heads", attrs.get("num_attention_heads", 1)))
    num_kv_heads = int(attrs.get("num_kv_heads", attrs.get("num_key_value_heads", 1)))
    gqa_group_size = int(attrs.get("gqa_group_size", max(1, num_heads // max(1, num_kv_heads))))
    mode = str(attrs.get("mode", "decode" if int(attrs.get("tile_q", 64)) == 1 else "prefill"))
    tile_q = int(attrs.get("tile_q", 1 if mode == "decode" else 64))
    tile_kv = int(attrs.get("tile_kv", max(block_size, 128)))
    head_dim = int(attrs.get("head_dim", 128))
    return ScheduleOp(
        "schedule.attn.kv_outer_sparse",
        {
            **attrs,
            "target_op": "tessera_attn.msa_kv_outer_sparse",
            "block_ids_layout": "B,Hkv,Sq,top_k",
            "block_size": block_size,
            "top_k": top_k,
            "gqa_group_size": gqa_group_size,
            "tile_q": tile_q,
            "tile_kv": tile_kv,
            "head_dim": head_dim,
            "mode": mode,
            "acc_dtype": attrs.get("acc_dtype", "fp32"),
            "dense_equivalence_oracle": bool(attrs.get("dense_equivalence_oracle", False)),
            "kv_traversal": "kv_outer",
            "online_softmax": True,
        },
        operands=list(op.operands),
        result=op.result,
        source_op=op,
    )


def _media_op(op: IROp, ordinal: int) -> ScheduleOp:
    op_name = canonical_graph_op_name(op.op_name)
    attrs = {**_base_attrs(op, ordinal), **dict(op.kwargs)}
    op_kind = op_name.removeprefix("tessera.")
    return ScheduleOp(
        f"schedule.media.{op_kind}",
        {
            **attrs,
            "contract": op_kind,
            "execution": attrs.get("execution", "projected_embeddings"),
            "status": attrs.get("status", "artifact_only"),
        },
        operands=list(op.operands),
        result=op.result,
        source_op=op,
    )


def _jepa_op(op: IROp, ordinal: int) -> ScheduleOp:
    op_name = canonical_graph_op_name(op.op_name)
    attrs = {**_base_attrs(op, ordinal), **dict(op.kwargs)}
    op_kind = (
        op_name.removeprefix("tessera.jepa.")
        if op_name.startswith("tessera.jepa.")
        else op_name.removeprefix("tessera.jepa_")
    )
    attrs["source"] = f"tessera.jepa.{op_kind}"
    return ScheduleOp(
        f"schedule.jepa.{op_kind}",
        {
            **attrs,
            "contract": op_kind,
            "latent_space": attrs.get("latent_space", "continuous"),
            "status": attrs.get("status", "artifact_only"),
        },
        operands=list(op.operands),
        result=op.result,
        source_op=op,
    )


def _base_attrs(op: IROp, ordinal: int) -> dict[str, Any]:
    attrs = {
        "source": canonical_graph_op_name(op.op_name),
        "ordinal": ordinal,
    }
    if op.result is not None:
        attrs["result"] = op.result
    # Phase 8.4.4 — surface the operand element type so downstream Tile and
    # Target IR layers can pick dtype-specific runtime symbols (e.g. mps_matmul
    # f32/f16/bf16). The Graph IR encodes types as strings like "tensor<*xf16>";
    # we extract the trailing element-type token. Defaults to f32 when the
    # operand types aren't parseable, preserving the pre-Phase 8.4.4 contract.
    dtype = _resolve_element_dtype(op)
    if dtype:
        attrs["dtype"] = dtype
    return attrs


def _resolve_element_dtype(op: IROp) -> str | None:
    operand_types = list(getattr(op, "operand_types", ()) or ())
    if not operand_types:
        return None
    # Pick the dtype from the first operand. The IROp invariant for compute
    # ops is uniform element type across operands (e.g. matmul A and B match).
    t = operand_types[0]
    if "bf16" in t:
        return "bf16"
    if "f16" in t and "bf16" not in t:
        return "f16"
    if "f32" in t:
        return "f32"
    if "f64" in t:
        return "f64"
    return None


def _shape_key(ops: list[IROp]) -> str:
    parts = []
    for op in ops:
        if op.result_type:
            parts.append(op.result_type)
    return "|".join(parts) or "unknown"


def _tensor_shape(type_text: str | None) -> tuple[int, ...]:
    if not type_text or not type_text.startswith("tensor<") or "x" not in type_text:
        return ()
    body = type_text.removeprefix("tensor<").removesuffix(">")
    parts = body.split("x")[:-1]
    dims: list[int] = []
    for part in parts:
        try:
            dims.append(int(part))
        except ValueError:
            return ()
    return tuple(dims)


def _matmul_flops(op: IROp) -> int:
    if len(op.operand_types) < 2:
        return 0
    lhs = _tensor_shape(str(op.operand_types[0]))
    rhs = _tensor_shape(str(op.operand_types[1]))
    if len(lhs) != 2 or len(rhs) != 2:
        return 0
    return 2 * lhs[0] * rhs[1] * lhs[1]


def _matmul_bytes(op: IROp) -> int:
    shapes = [_tensor_shape(str(t)) for t in [*op.operand_types[:2], op.result_type]]
    if any(not shape for shape in shapes):
        return 0
    return 4 * sum(math.prod(shape) for shape in shapes)


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
