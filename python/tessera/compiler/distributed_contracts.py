"""Canonical lowering map for the public sharding/collective vocabulary.

The public S6 API contains nine names, but they do not represent nine physical
backend kernels.  Mesh/spec/shard-map entries are compile-time placement
contracts.  Reduction aliases and broadcast map onto the registered collective
Target IR.  Collective permute deliberately remains a distinct point-to-point
contract; lowering it as all-to-all would change its semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DistributedPrimitiveContract:
    name: str
    ownership: str
    canonical_op: str | None = None
    reduction: str = "none"
    normalize: bool = False
    adjoint: str | None = None

    def __post_init__(self) -> None:
        if self.ownership not in {
            "compile_time_metadata",
            "compile_time_region",
            "collective_target",
            "point_to_point_target",
        }:
            raise ValueError(f"invalid distributed ownership {self.ownership!r}")
        if self.ownership == "collective_target" and self.canonical_op not in {
            "all_reduce",
            "all_gather",
        }:
            raise ValueError("collective aliases require a registered canonical op")


DISTRIBUTED_PRIMITIVE_CONTRACTS: dict[str, DistributedPrimitiveContract] = {
    "named_sharding": DistributedPrimitiveContract("named_sharding", "compile_time_metadata"),
    "partition_spec": DistributedPrimitiveContract("partition_spec", "compile_time_metadata"),
    "shard_map": DistributedPrimitiveContract("shard_map", "compile_time_region"),
    "psum": DistributedPrimitiveContract(
        "psum",
        "collective_target",
        "all_reduce",
        "sum",
        adjoint="broadcast_to_axis",
    ),
    "pmean": DistributedPrimitiveContract(
        "pmean",
        "collective_target",
        "all_reduce",
        "sum",
        True,
        "broadcast_to_axis_scaled",
    ),
    "pmax": DistributedPrimitiveContract(
        "pmax",
        "collective_target",
        "all_reduce",
        "max",
        adjoint="argmax_routed",
    ),
    "pmin": DistributedPrimitiveContract(
        "pmin",
        "collective_target",
        "all_reduce",
        "min",
        adjoint="argmin_routed",
    ),
    "broadcast_to_axis": DistributedPrimitiveContract(
        "broadcast_to_axis",
        "collective_target",
        "all_gather",
        adjoint="psum",
    ),
    "collective_permute": DistributedPrimitiveContract(
        "collective_permute",
        "point_to_point_target",
        adjoint="inverse_collective_permute",
    ),
}


def collective_record_for_primitive(
    name: str,
    *,
    tensor: str,
    mesh_axis: str,
    tensor_axis: int = 0,
    world_size: int | None = None,
    result: str | None = None,
) -> dict[str, Any]:
    """Lower one executable public alias to a registered Target-IR record.

    Compile-time placement objects cannot reach transport.  Collective permute
    also fails closed until its ordered peer map has a first-class Target op;
    silently spelling it as all-to-all would send different data.
    """
    try:
        contract = DISTRIBUTED_PRIMITIVE_CONTRACTS[name]
    except KeyError as exc:
        raise ValueError(f"unknown distributed primitive {name!r}") from exc
    if contract.ownership not in {"collective_target", "point_to_point_target"}:
        raise ValueError(f"{name} is owned by {contract.ownership}, not collective transport")
    if not tensor or not mesh_axis:
        raise ValueError("collective aliases require tensor and mesh-axis identity")
    if tensor_axis < 0:
        raise ValueError("collective tensor axis must be non-negative")
    if world_size is not None and world_size <= 0:
        raise ValueError("collective world size must be positive")
    if contract.ownership == "point_to_point_target":
        raise ValueError("collective_permute requires an explicit peer map; use collective_permute_record")
    record: dict[str, Any] = {
        "op": f"tessera_collective.{contract.canonical_op}",
        "tensor": tensor,
        "input": tensor,
        "output": result or tensor,
        "mesh_axis": mesh_axis,
        "tensor_axis": tensor_axis,
        "reduction": "mean" if contract.normalize else contract.reduction,
    }
    if world_size is not None:
        record["world_size"] = world_size
    return record


def collective_permute_record(
    *,
    tensor: str,
    result: str,
    mesh_axis: str,
    pairs: tuple[tuple[int, int], ...],
    world_size: int,
) -> dict[str, Any]:
    """Lower an ordered one-source/one-destination permutation to Target IR."""
    if not tensor or not result or not mesh_axis:
        raise ValueError("collective_permute requires tensor, result, and mesh axis")
    if world_size < 2:
        raise ValueError("collective_permute requires world_size >= 2")
    sources = tuple(int(source) for source, _ in pairs)
    targets = tuple(int(target) for _, target in pairs)
    if len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
        raise ValueError("collective_permute peer sources and targets must be unique")
    if any(peer < 0 or peer >= world_size for peer in (*sources, *targets)):
        raise ValueError("collective_permute peer is outside the communicator")
    return {
        "op": "tessera_collective.collective_permute",
        "tensor": result,
        "input": tensor,
        "output": result,
        "mesh_axis": mesh_axis,
        "tensor_axis": 0,
        "reduction": "none",
        "world_size": world_size,
        "source_peers": list(sources),
        "target_peers": list(targets),
    }


__all__ = [
    "DISTRIBUTED_PRIMITIVE_CONTRACTS",
    "DistributedPrimitiveContract",
    "collective_permute_record",
    "collective_record_for_primitive",
]
