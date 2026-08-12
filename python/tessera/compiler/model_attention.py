"""Content-addressed physical model-attention packages.

This boundary is intentionally narrower than generic attention: it binds one
frontier-model family contract to the exact Schedule, Tile, and Target artifact
consumed by an architecture-owned runtime lane. Runtime launch reads the bound
launch contract and never reconstructs or re-lowers Graph IR.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from .graph_ir import GraphIRModule
from .schedule_ir import ScheduleOp, lower_graph_to_schedule_ir
from .target_ir import lower_tile_to_target_ir
from .tile_ir import lower_schedule_to_tile_ir


_SCHEMA = "tessera.model_attention_package.v1"
_TARGETS = {
    "x86": ("x86", "zen5_avx512", "x86_msa_compiled"),
    "rocm_gfx1151": ("rocm", "gfx1151", "rocm_sparse_attn_compiled"),
}


def _digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stamp_schedule_lineage(ops: list[ScheduleOp], graph_digest: str) -> None:
    for op in ops:
        if op.op_name == "schedule.artifact":
            op.attrs["parent_graph_digest"] = graph_digest
            op.attrs["work_item"] = "MODEL-FUSED-PHYS-1"
        _stamp_schedule_lineage(op.body, graph_digest)


@dataclass(frozen=True)
class ModelAttentionPackage:
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target_ir: str
    contract: Mapping[str, Any]

    @property
    def artifact_hash(self) -> str:
        return str(self.contract["artifact_hash"])

    def validate(self) -> None:
        validate_model_attention_contract(
            self.contract,
            graph_ir=self.graph_ir,
            schedule_ir=self.schedule_ir,
            tile_ir=self.tile_ir,
            target_ir=self.target_ir,
        )

    def runtime_artifact(self) -> Any:
        from tessera.runtime import RuntimeArtifact

        self.validate()
        launch = dict(self.contract["launch_contract"])
        return RuntimeArtifact(
            graph_ir=self.graph_ir,
            schedule_ir=self.schedule_ir,
            tile_ir=self.tile_ir,
            target_ir=self.target_ir,
            metadata={
                "target": self.contract["runtime_target"],
                "architecture": self.contract["architecture"],
                "compiler_path": self.contract["runtime_lane"],
                # Runtime dispatch is matrix-owned, but launch deliberately
                # requires the package producer to assert that the bound
                # artifact is intended to execute.  This is not inferred from
                # the presence of legacy Graph ``ops`` metadata.
                "executable": True,
                "arg_names": list(launch["arg_names"]),
                "model_attention_package": dict(self.contract),
            },
        )


def build_msa_physical_package(
    module: GraphIRModule, *, target: str
) -> ModelAttentionPackage:
    """Bind MiniMax MSA to one exact x86 or gfx1151 physical artifact."""
    target_info = _TARGETS.get(target)
    if target_info is None:
        raise ValueError(
            "MSA physical packages support only x86 and rocm_gfx1151; "
            "unproven targets fail closed"
        )
    if len(module.functions) != 1 or len(module.functions[0].body) != 1:
        raise ValueError("MSA physical package requires one Graph operation")
    function = module.functions[0]
    op = function.body[0]
    if op.op_name != "tessera.msa_sparse_attention" or len(op.operands) < 3:
        raise ValueError("MSA physical package requires tessera.msa_sparse_attention")
    if op.result is None:
        raise ValueError("MSA physical package requires one result")

    graph_ir = module.to_mlir(target=target, canonical=True)
    graph_digest = _digest_text(graph_ir)
    schedule = lower_graph_to_schedule_ir(module, target_kind=target)
    for schedule_function in schedule.functions:
        _stamp_schedule_lineage(schedule_function.body, graph_digest)
    schedule_ir = schedule.to_mlir()
    tile = lower_schedule_to_tile_ir(schedule, target_kind=target)
    tile_ir = tile.to_mlir()
    target_module = lower_tile_to_target_ir(tile, target_kind=target)
    target_ir = target_module.to_mlir()
    runtime_target, architecture, runtime_lane = target_info
    required_target_marker = (
        "tessera_rocm.msa_block_sparse"
        if target == "rocm_gfx1151"
        else "tessera_x86.kernel"
    )
    if required_target_marker not in target_ir or runtime_lane not in target_ir:
        raise ValueError("MSA physical artifact did not reach its declared runtime lane")

    launch_contract = {
        "operation": op.op_name,
        "arg_names": [argument.name for argument in function.args],
        "operands": [name.removeprefix("%") for name in op.operands],
        "kwargs": dict(op.kwargs),
        "output": op.result.removeprefix("%"),
    }
    body: dict[str, Any] = {
        "schema": _SCHEMA,
        "family": "minimax_msa",
        "runtime_target": runtime_target,
        "compiler_target": target,
        "architecture": architecture,
        "runtime_lane": runtime_lane,
        "graph_digest": graph_digest,
        "schedule_digest": _digest_text(schedule_ir),
        "tile_digest": _digest_text(tile_ir),
        "target_digest": _digest_text(target_ir),
        "launch_contract": launch_contract,
        "graph_redispatch": "prohibited",
        "differential_oracle": "stdlib.attention.msa_sparse_attention",
    }
    package = ModelAttentionPackage(
        graph_ir, schedule_ir, tile_ir, target_ir,
        {**body, "artifact_hash": _digest(body)},
    )
    package.validate()
    return package


def validate_model_attention_contract(
    contract: Mapping[str, Any], *, graph_ir: str, schedule_ir: str,
    tile_ir: str, target_ir: str,
) -> None:
    body = dict(contract)
    artifact_hash = str(body.pop("artifact_hash", ""))
    if body.get("schema") != _SCHEMA or artifact_hash != _digest(body):
        raise ValueError("model-attention package has stale content identity")
    for field, text in (
        ("graph_digest", graph_ir),
        ("schedule_digest", schedule_ir),
        ("tile_digest", tile_ir),
        ("target_digest", target_ir),
    ):
        if body.get(field) != _digest_text(text):
            raise ValueError(f"model-attention package has stale {field}")
    target = str(body.get("compiler_target", ""))
    expected = _TARGETS.get(target)
    if expected is None or tuple(body.get(key) for key in (
        "runtime_target", "architecture", "runtime_lane"
    )) != expected:
        raise ValueError("model-attention package target ownership is invalid")
    if body.get("graph_redispatch") != "prohibited":
        raise ValueError("model-attention package permits Graph redispatch")
    launch = body.get("launch_contract")
    if not isinstance(launch, dict) or launch.get("operation") != "tessera.msa_sparse_attention":
        raise ValueError("model-attention package has an invalid launch contract")


__all__ = [
    "ModelAttentionPackage",
    "build_msa_physical_package",
    "validate_model_attention_contract",
]
