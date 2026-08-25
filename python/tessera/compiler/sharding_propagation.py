"""Fail-closed fixed-point placement propagation for Graph IR.

This module owns placement facts only.  It neither launches transport nor
chooses a reshard operation; explicit incompatibilities are returned to the
caller so collective insertion remains a separate, reviewable transformation.
"""

from __future__ import annotations

import hashlib
import math
import json
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Mapping, Sequence

from .effects import Effect, registered_op_effect
from .graph_ir import GraphIRFunction, IROp, IRType, tensor_ir_type
from .op_catalog import get_op_spec

PlacementKind = Literal["unknown", "replicated", "tiled", "partial_reduction"]


@dataclass(frozen=True)
class Placement:
    kind: PlacementKind
    tiled_axes: tuple[tuple[int, str], ...] = ()
    reduction_axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        tiled = tuple(sorted((int(dim), str(axis)) for dim, axis in self.tiled_axes))
        reductions = tuple(sorted(map(str, self.reduction_axes)))
        if len({dim for dim, _ in tiled}) != len(tiled):
            raise ValueError("a tensor dimension can map to only one mesh axis")
        if any(dim < 0 or not axis for dim, axis in tiled) or any(
            not axis for axis in reductions
        ):
            raise ValueError("placement dimensions and mesh axes must be valid")
        if self.kind == "replicated" and (tiled or reductions):
            raise ValueError("replicated placement cannot carry shard axes")
        if self.kind == "tiled" and (not tiled or reductions):
            raise ValueError("tiled placement requires only tiled axes")
        if self.kind == "partial_reduction" and not reductions:
            raise ValueError("partial reduction placement requires reduction axes")
        if self.kind == "unknown" and (tiled or reductions):
            raise ValueError("unknown placement cannot claim shard axes")
        object.__setattr__(self, "tiled_axes", tiled)
        object.__setattr__(self, "reduction_axes", reductions)

    @classmethod
    def unknown(cls) -> "Placement":
        return cls("unknown")

    @classmethod
    def replicated(cls) -> "Placement":
        return cls("replicated")

    @classmethod
    def tiled(cls, axes: Mapping[int, str]) -> "Placement":
        return cls("tiled", tuple(axes.items()))

    @classmethod
    def partial_reduction(
        cls,
        reduction_axes: Sequence[str],
        *,
        tiled_axes: Mapping[int, str] | None = None,
    ) -> "Placement":
        return cls(
            "partial_reduction",
            tuple((tiled_axes or {}).items()),
            tuple(reduction_axes),
        )


@dataclass(frozen=True)
class PlacementConflict:
    op_name: str
    operands: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ShardingPropagationResult:
    placements: Mapping[str, Placement]
    conflicts: tuple[PlacementConflict, ...]
    iterations: int
    digest: str


PlacementRule = Callable[[IROp, tuple[Placement, ...]], Placement]


@dataclass(frozen=True)
class ReshardAction:
    """One explicit placement conversion required at an operand boundary."""

    value: str
    consumer: str
    consumer_index: int
    operand_index: int
    source: Placement
    target: Placement
    collective: Literal[
        "all_gather", "all_reduce", "all_to_all", "reduce_scatter", "local_shard"
    ]
    mesh_axes: tuple[str, ...]
    subgroup: tuple[int, ...] = ()
    region_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReshardPlan:
    actions: tuple[ReshardAction, ...]
    matching_rounds: tuple[tuple[tuple[int, int], ...], ...]
    digest: str


@dataclass(frozen=True)
class MockMeshExecutionResult:
    """Deterministic execution evidence for one materialized placement graph."""

    values: Mapping[str, tuple[Any, ...]]
    returned: Mapping[str, tuple[Any, ...]]
    executed_reshards: tuple[str, ...]
    backend: str = "deterministic_mock_mesh"


def _ssa(value: str) -> str:
    return str(value).strip().lstrip("%")


def _elementwise_rule(_op: IROp, operands: tuple[Placement, ...]) -> Placement:
    non_replicated = tuple(item for item in operands if item.kind != "replicated")
    if not non_replicated:
        return Placement.replicated()
    first = non_replicated[0]
    return (
        first if all(item == first for item in non_replicated) else Placement.unknown()
    )


def _reduction_rule(op: IROp, operands: tuple[Placement, ...]) -> Placement:
    if len(operands) != 1:
        return Placement.unknown()
    source = operands[0]
    if source.kind == "replicated":
        return source
    if source.kind != "tiled":
        return Placement.unknown()
    axis = op.kwargs.get("axis")
    if not isinstance(axis, int):
        return Placement.unknown()
    tiled = dict(source.tiled_axes)
    reduced_mesh_axis = tiled.pop(axis, None)
    if reduced_mesh_axis is None:
        return source
    shifted = {
        (dim - 1 if dim > axis else dim): mesh_axis for dim, mesh_axis in tiled.items()
    }
    return Placement.partial_reduction((reduced_mesh_axis,), tiled_axes=shifted)


def _collective_rule(op: IROp, operands: tuple[Placement, ...]) -> Placement:
    if len(operands) != 1:
        return Placement.unknown()
    source = operands[0]
    name = op.op_name.removeprefix("tessera.")
    if name == "all_reduce":
        if source.kind == "partial_reduction":
            return (
                Placement.tiled(dict(source.tiled_axes))
                if source.tiled_axes
                else Placement.replicated()
            )
        return source
    mesh_axis = op.kwargs.get("mesh_axis", op.kwargs.get("axis_name"))
    tensor_axis = op.kwargs.get("tensor_axis", op.kwargs.get("axis"))
    if (
        not isinstance(mesh_axis, str)
        or not mesh_axis
        or not isinstance(tensor_axis, int)
    ):
        return Placement.unknown()
    if name == "all_gather" and source.kind == "tiled":
        tiled = dict(source.tiled_axes)
        if tiled.get(tensor_axis) != mesh_axis:
            return Placement.unknown()
        tiled.pop(tensor_axis)
        return Placement.tiled(tiled) if tiled else Placement.replicated()
    if name == "reduce_scatter" and source.kind in {"replicated", "partial_reduction"}:
        if (
            source.kind == "partial_reduction"
            and mesh_axis not in source.reduction_axes
        ):
            return Placement.unknown()
        return Placement.tiled({tensor_axis: mesh_axis})
    if name == "all_to_all" and source.kind == "tiled":
        scatter_axis = op.kwargs.get("scatter_axis")
        gather_axis = op.kwargs.get("gather_axis", tensor_axis)
        if not isinstance(scatter_axis, int) or not isinstance(gather_axis, int):
            return Placement.unknown()
        tiled = dict(source.tiled_axes)
        if tiled.pop(scatter_axis, None) != mesh_axis or gather_axis in tiled:
            return Placement.unknown()
        tiled[gather_axis] = mesh_axis
        return Placement.tiled(tiled)
    return Placement.unknown()


DEFAULT_PLACEMENT_RULES: Mapping[str, PlacementRule] = {
    **{
        name: _elementwise_rule
        for name in (
            "tessera.add",
            "tessera.sub",
            "tessera.mul",
            "tessera.div",
            "tessera.maximum",
            "tessera.minimum",
            "tessera.relu",
            "tessera.exp",
            "tessera.log",
            "tessera.tanh",
        )
    },
    "tessera.reduce": _reduction_rule,
    "tessera.all_reduce": _collective_rule,
    "tessera.all_gather": _collective_rule,
    "tessera.reduce_scatter": _collective_rule,
    "tessera.all_to_all": _collective_rule,
}


def _registered_rule(
    op: IROp, rules: Mapping[str, PlacementRule]
) -> PlacementRule | None:
    direct = rules.get(op.op_name)
    if direct is not None:
        return direct
    # Consume the canonical operation catalog only for shape-preserving,
    # placement-polymorphic families. A dashboard's broad "complete" label is
    # not enough to infer axis-changing contracts such as reshape, transpose,
    # contraction, attention, or stencil halo exchange.
    spec = get_op_spec(op.op_name)
    if spec is not None and spec.lowering in {
        "elementwise",
        "numeric_helper",
        "comparison",
        "logical",
        "rotary_embedding",
        "quantize",
    }:
        return _elementwise_rule
    if spec is not None and spec.lowering == "stable_reduction":
        return _reduction_rule
    return None


def propagate_sharding(
    function: GraphIRFunction,
    argument_placements: Mapping[str, Placement],
    *,
    rules: Mapping[str, PlacementRule] = DEFAULT_PLACEMENT_RULES,
) -> ShardingPropagationResult:
    """Propagate registered placements to a fixed point; unknown stays safe."""

    placements: dict[str, Placement] = {
        _ssa(arg.name): argument_placements.get(_ssa(arg.name), Placement.unknown())
        for arg in function.args
    }
    conflicts: dict[tuple[str, tuple[str, ...], str], PlacementConflict] = {}
    max_iterations = max(1, len(function.body) + 1)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        changed = False
        for op in function.body:
            operands = tuple(
                placements.get(_ssa(value), Placement.unknown())
                for value in op.operands
            )
            effect = registered_op_effect(op.op_name, op.kwargs)
            has_region = op.op_name.startswith(("scf.", "tessera.scf.")) or any(
                key in op.kwargs
                for key in ("region", "regions", "body", "then_region", "else_region")
            )
            rule = _registered_rule(op, rules)
            effect_admitted = effect == Effect.pure or rule is _collective_rule
            if (
                not effect_admitted
                or has_region
                or rule is None
                or any(item.kind == "unknown" for item in operands)
            ):
                inferred = Placement.unknown()
            else:
                inferred = rule(op, operands)
                if inferred.kind == "unknown":
                    conflict = PlacementConflict(
                        op.op_name,
                        tuple(map(_ssa, op.operands)),
                        "incompatible_or_underspecified_placement",
                    )
                    conflicts[
                        (conflict.op_name, conflict.operands, conflict.reason)
                    ] = conflict
            for result in op.result_names:
                name = _ssa(result)
                if placements.get(name) != inferred:
                    placements[name] = inferred
                    changed = True
        if not changed:
            break

    payload = {
        "function": function.name,
        "placements": {
            name: {
                "kind": placement.kind,
                "tiled_axes": placement.tiled_axes,
                "reduction_axes": placement.reduction_axes,
            }
            for name, placement in sorted(placements.items())
        },
        "conflicts": [
            (item.op_name, item.operands, item.reason)
            for item in sorted(
                conflicts.values(), key=lambda item: (item.op_name, item.operands)
            )
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ShardingPropagationResult(
        dict(placements), tuple(conflicts.values()), iterations, digest
    )


def plan_explicit_reshards(
    function: GraphIRFunction,
    propagation: ShardingPropagationResult,
    operand_requirements: Mapping[int, Sequence[Placement]],
    *,
    subgroup: Sequence[int] = (),
) -> ReshardPlan:
    """Plan reviewable collectives for explicit producer/consumer mismatches.

    Requirements are keyed by Graph operation index and must be total over that
    operation's operands. Unknown placements fail closed; this routine never
    guesses a mesh axis or silently changes a layout.
    """
    normalized_subgroup = tuple(int(rank) for rank in subgroup)
    if normalized_subgroup and (
        len(normalized_subgroup) < 2
        or len(set(normalized_subgroup)) != len(normalized_subgroup)
        or any(rank < 0 for rank in normalized_subgroup)
    ):
        raise ValueError("reshard subgroup must contain distinct non-negative ranks")
    producer_regions: dict[str, tuple[str, ...]] = {
        result: _region_path(op)
        for op in function.body
        for result in map(_ssa, op.result_names)
    }
    actions: list[ReshardAction] = []
    for op_index, requirements in sorted(operand_requirements.items()):
        if op_index < 0 or op_index >= len(function.body):
            raise ValueError("reshard requirement references an invalid operation")
        op = function.body[op_index]
        if len(requirements) != len(op.operands):
            raise ValueError("reshard requirements must cover every operation operand")
        for operand_index, (value, target) in enumerate(zip(op.operands, requirements)):
            source = propagation.placements.get(_ssa(value), Placement.unknown())
            if source == target:
                continue
            value_name = _ssa(value)
            consumer_region = _region_path(op)
            producer_region = producer_regions.get(value_name, ())
            if producer_region and not _is_region_ancestor(
                producer_region, consumer_region
            ):
                raise ValueError(
                    "cannot reshard a value across sibling or escaping regions"
                )
            action = _reshard_action(
                value_name,
                op.op_name,
                op_index,
                operand_index,
                source,
                target,
                subgroup=normalized_subgroup,
                region_path=consumer_region,
            )
            actions.append(action)
    matching_rounds = (
        _matching_rounds(normalized_subgroup)
        if any(action.collective == "all_to_all" for action in actions)
        else ()
    )
    payload = [
        {
            "value": action.value,
            "consumer": action.consumer,
            "consumer_index": action.consumer_index,
            "operand_index": action.operand_index,
            "source": _placement_payload(action.source),
            "target": _placement_payload(action.target),
            "collective": action.collective,
            "mesh_axes": action.mesh_axes,
            "subgroup": action.subgroup,
            "region_path": action.region_path,
        }
        for action in actions
    ]
    payload.append({"matching_rounds": matching_rounds})
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ReshardPlan(tuple(actions), matching_rounds, digest)


def _region_path(op: IROp) -> tuple[str, ...]:
    raw = op.kwargs.get("_region_path", op.kwargs.get("region_path", ()))
    if not raw:
        raw = op.kwargs.get("_region", op.kwargs.get("region", ()))
    if isinstance(raw, str):
        return tuple(part for part in raw.split("/") if part)
    if isinstance(raw, Sequence):
        return tuple(str(part) for part in raw)
    raise ValueError("region path must be a string or sequence")


def _is_region_ancestor(producer: tuple[str, ...], consumer: tuple[str, ...]) -> bool:
    return len(producer) <= len(consumer) and consumer[: len(producer)] == producer


def _matching_rounds(
    subgroup: tuple[int, ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Decompose a complete directed exchange into deterministic matchings.

    Round ``offset`` maps every subgroup rank to one distinct peer.  This is the
    cyclic 1-factorization used by the transport-independent all-to-all plan;
    each source/target pair occurs exactly once and no rank sends or receives
    twice in a round.
    """
    if not subgroup:
        raise ValueError("all-to-all reshard requires an explicit subgroup")
    size = len(subgroup)
    return tuple(
        tuple(
            (subgroup[index], subgroup[(index + offset) % size])
            for index in range(size)
        )
        for offset in range(1, size)
    )


def materialize_reshard_plan(
    function: GraphIRFunction,
    plan: ReshardPlan,
    *,
    mesh_shape: Mapping[str, int] | None = None,
) -> GraphIRFunction:
    """Insert typed placement conversions as real Graph SSA operations.

    Collective results use the exact local shape implied by the target
    placement. Replicated-to-tiled movement is an operation-owned
    ``tessera.slice`` with explicit reshard identity, not a same-shaped fake
    collective. A missing, non-divisible, or dynamic mesh extent fails closed.
    """

    mesh = _mesh_contract(plan, mesh_shape)
    by_consumer: dict[int, list[ReshardAction]] = {}
    for action in plan.actions:
        by_consumer.setdefault(action.consumer_index, []).append(action)
    body: list[IROp] = []
    used_names = {_ssa(name) for op in function.body for name in op.result_names} | {
        _ssa(arg.name) for arg in function.args
    }
    value_types: dict[str, IRType] = {
        _ssa(arg.name): arg.ir_type for arg in function.args
    }
    for source_op in function.body:
        for result_name in source_op.result_names:
            if source_op.inferred_type is not None:
                value_types[_ssa(result_name)] = source_op.inferred_type

    for op_index, source_op in enumerate(function.body):
        op = replace(
            source_op,
            operands=list(source_op.operands),
            operand_types=list(source_op.operand_types),
            kwargs=dict(source_op.kwargs),
        )
        for ordinal, action in enumerate(
            sorted(
                by_consumer.get(op_index, ()),
                key=lambda item: item.operand_index,
            )
        ):
            name = f"reshard_{op_index}_{action.operand_index}_{ordinal}"
            suffix = 0
            while name in used_names:
                suffix += 1
                name = f"reshard_{op_index}_{action.operand_index}_{ordinal}_{suffix}"
            used_names.add(name)
            source_type = value_types.get(action.value)
            if source_type is None:
                raise ValueError(
                    f"reshard value {action.value!r} lacks a structured tensor type"
                )
            target_type = _local_shard_type(source_type, action.target, mesh)
            tensor_axis = _collective_tensor_axis(action)
            kwargs: dict[str, object] = {
                "mesh_axis": action.mesh_axes[0],
                "mesh_size": mesh[action.mesh_axes[0]],
                "axis": tensor_axis,
                "op": (
                    "sum"
                    if action.collective in {"all_reduce", "reduce_scatter"}
                    else "none"
                ),
                "reshard_kind": action.collective,
                "reshard_plan_digest": plan.digest,
                "source_global_shape": list(source_type.shape),
                "target_local_shape": list(target_type.shape),
                "subgroup": list(action.subgroup),
                "region_path": list(action.region_path),
            }
            if action.collective == "all_to_all":
                kwargs["scatter_axis"] = _single_tiled_dim(action.source)
                kwargs["gather_axis"] = _single_tiled_dim(action.target)
                kwargs["matching_rounds"] = [
                    [list(edge) for edge in round_] for round_ in plan.matching_rounds
                ]
            operand_type = op.operand_types[action.operand_index]
            inserted = IROp(
                result=name,
                op_name=(
                    "tessera.slice"
                    if action.collective == "local_shard"
                    else f"tessera.{action.collective}"
                ),
                operands=[op.operands[action.operand_index]],
                operand_types=[operand_type],
                result_type=str(target_type),
                inferred_type=target_type,
                kwargs=kwargs,
            )
            body.append(inserted)
            value_types[name] = target_type
            op.operands[action.operand_index] = f"%{name}"
            op.operand_types[action.operand_index] = str(target_type)
        body.append(op)
    return replace(function, body=body)


def _mesh_contract(
    plan: ReshardPlan, mesh_shape: Mapping[str, int] | None
) -> dict[str, int]:
    mesh = {str(axis): int(size) for axis, size in (mesh_shape or {}).items()}
    for action in plan.actions:
        for axis in action.mesh_axes:
            if axis not in mesh and action.subgroup:
                mesh[axis] = len(action.subgroup)
            if axis not in mesh or mesh[axis] < 2:
                raise ValueError(
                    f"reshard mesh axis {axis!r} requires an explicit size >= 2"
                )
            if action.subgroup and len(action.subgroup) != mesh[axis]:
                raise ValueError(
                    f"reshard subgroup size does not match mesh axis {axis!r}"
                )
    return mesh


def _local_shard_type(
    global_type: IRType, placement: Placement, mesh: Mapping[str, int]
) -> IRType:
    if global_type.rank is None:
        raise ValueError("typed reshard requires a statically ranked tensor")
    shape = list(global_type.shape)
    for dimension, mesh_axis in placement.tiled_axes:
        if dimension >= len(shape):
            raise ValueError("placement tensor axis is outside the tensor rank")
        try:
            extent = int(shape[dimension])
        except ValueError as error:
            raise ValueError("typed reshard requires static shard extents") from error
        size = mesh.get(mesh_axis)
        if size is None or extent % size:
            raise ValueError(
                f"tensor extent {extent} is not divisible by mesh axis {mesh_axis!r}"
            )
        shape[dimension] = str(extent // size)
    return tensor_ir_type(tuple(shape), global_type.dtype, layout=global_type.layout)


def execute_resharded_graph_on_mock_mesh(
    function: GraphIRFunction,
    rank_inputs: Mapping[str, Sequence[Any]],
    *,
    mesh_shape: Mapping[str, int],
) -> MockMeshExecutionResult:
    """Execute explicit reshard SSA on the deterministic in-process mesh.

    This interpreter is deliberately small and fail-closed: it handles the
    five distributed movement forms plus the pointwise consumers used by the
    placement gate. Every movement is visible in ``executed_reshards``.
    """

    import numpy as np

    from tessera.collectives import CollectiveAdapter

    mesh = {str(axis): int(size) for axis, size in mesh_shape.items()}
    world_size = math.prod(mesh.values()) if mesh else 0
    if world_size < 2:
        raise ValueError("mock reshard execution requires a multi-rank mesh")
    adapter = CollectiveAdapter(backend="mock", world_size=world_size, mesh_axes=mesh)
    values: dict[str, tuple[Any, ...]] = {}
    for arg in function.args:
        name = _ssa(arg.name)
        per_rank = tuple(rank_inputs.get(name, ()))
        if len(per_rank) != world_size:
            raise ValueError(f"mock mesh input {name!r} requires one value per rank")
        values[name] = tuple(np.asarray(value) for value in per_rank)

    executed: list[str] = []
    for op in function.body:
        if op.result is None or len(op.result_names) != 1:
            raise ValueError("mock reshard execution requires single-result SSA ops")
        operands = [values.get(_ssa(name)) for name in op.operands]
        if any(value is None for value in operands):
            raise ValueError(f"mock mesh operation {op.op_name} has an unknown operand")
        inputs = [tuple(value) for value in operands if value is not None]
        name = _ssa(op.result_names[0])
        if (
            op.op_name == "tessera.slice"
            and op.kwargs.get("reshard_kind") == "local_shard"
        ):
            axis = int(op.kwargs["axis"])
            mesh_axis = str(op.kwargs["mesh_axis"])
            size = mesh[mesh_axis]
            outputs = []
            for rank, value in enumerate(inputs[0]):
                if value.shape[axis] % size:
                    raise ValueError("mock local shard extent is not divisible")
                coordinate = _mesh_axis_coordinate(rank, mesh, mesh_axis)
                outputs.append(np.split(value, size, axis=axis)[coordinate])
            result = tuple(outputs)
            executed.append("local_shard")
        elif op.op_name == "tessera.all_reduce":
            result = tuple(
                adapter.all_reduce(inputs[0], op=str(op.kwargs.get("op", "sum")))
            )
            executed.append("all_reduce")
        elif op.op_name == "tessera.reduce_scatter":
            result = tuple(
                adapter.reduce_scatter(
                    inputs[0],
                    axis=int(op.kwargs.get("axis", 0)),
                    op=str(op.kwargs.get("op", "sum")),
                )
            )
            executed.append("reduce_scatter")
        elif op.op_name == "tessera.all_gather":
            result = tuple(
                adapter.all_gather(inputs[0], axis=int(op.kwargs.get("axis", 0)))
            )
            executed.append("all_gather")
        elif op.op_name == "tessera.all_to_all":
            result = tuple(
                adapter.all_to_all(
                    inputs[0],
                    scatter_axis=int(op.kwargs.get("scatter_axis", 0)),
                    gather_axis=int(op.kwargs.get("gather_axis", 0)),
                )
            )
            executed.append("all_to_all")
        elif op.op_name == "tessera.collective_permute":
            sources = tuple(int(item) for item in op.kwargs.get("source_peers", ()))
            targets = tuple(int(item) for item in op.kwargs.get("target_peers", ()))
            result = tuple(
                adapter.collective_permute(
                    inputs[0], pairs=tuple(zip(sources, targets))
                )
            )
            executed.append("collective_permute")
        elif op.op_name in {"tessera.add", "tessera.sub", "tessera.mul", "tessera.div"}:
            operation: Any = {
                "tessera.add": np.add,
                "tessera.sub": np.subtract,
                "tessera.mul": np.multiply,
                "tessera.div": np.divide,
            }[op.op_name]
            result = tuple(operation(*rank_values) for rank_values in zip(*inputs))
        elif op.op_name in {"tessera.sigmoid", "tessera.tanh", "tessera.relu"}:
            operation = {
                "tessera.sigmoid": lambda value: 1.0 / (1.0 + np.exp(-value)),
                "tessera.tanh": np.tanh,
                "tessera.relu": lambda value: np.maximum(value, 0),
            }[op.op_name]
            result = tuple(operation(value) for value in inputs[0])
        else:
            raise ValueError(f"mock mesh has no execution rule for {op.op_name!r}")
        values[name] = tuple(np.asarray(value) for value in result)

    returned = {_ssa(name): values[_ssa(name)] for name in function.return_values}
    return MockMeshExecutionResult(
        values=dict(values),
        returned=returned,
        executed_reshards=tuple(executed),
    )


def _mesh_axis_coordinate(
    rank: int, mesh_shape: Mapping[str, int], mesh_axis: str
) -> int:
    axes = tuple(mesh_shape)
    if mesh_axis not in axes:
        raise ValueError(f"unknown mesh axis {mesh_axis!r}")
    index = axes.index(mesh_axis)
    stride = math.prod(tuple(mesh_shape[axis] for axis in axes[index + 1 :]))
    return (rank // stride) % mesh_shape[mesh_axis]


def _single_tiled_dim(placement: Placement) -> int:
    if len(placement.tiled_axes) != 1:
        raise ValueError(
            "collective SSA currently requires exactly one tiled tensor axis"
        )
    return placement.tiled_axes[0][0]


def _collective_tensor_axis(action: ReshardAction) -> int:
    if action.collective == "all_gather":
        return _single_tiled_dim(action.source)
    if action.collective == "reduce_scatter":
        return _single_tiled_dim(action.target)
    if action.collective == "all_to_all":
        return _single_tiled_dim(action.target)
    return 0


def _placement_payload(placement: Placement) -> Mapping[str, object]:
    return {
        "kind": placement.kind,
        "tiled_axes": placement.tiled_axes,
        "reduction_axes": placement.reduction_axes,
    }


def _reshard_action(
    value: str,
    consumer: str,
    consumer_index: int,
    operand_index: int,
    source: Placement,
    target: Placement,
    *,
    subgroup: tuple[int, ...],
    region_path: tuple[str, ...],
) -> ReshardAction:
    if source.kind == "unknown" or target.kind == "unknown":
        raise ValueError("cannot insert a reshard across unknown placement")
    source_tiles = dict(source.tiled_axes)
    target_tiles = dict(target.tiled_axes)
    collective: Literal[
        "all_gather", "all_reduce", "all_to_all", "reduce_scatter", "local_shard"
    ]
    axes: tuple[str, ...]
    if source.kind == "partial_reduction" and target.kind == "replicated":
        if source_tiles:
            raise ValueError("replication would also require gathering surviving tiles")
        collective = "all_reduce"
        axes = source.reduction_axes
    elif source.kind == "partial_reduction" and target.kind == "tiled":
        if target_tiles == source_tiles:
            collective = "all_reduce"
        else:
            added = set(target_tiles.items()) - set(source_tiles.items())
            if len(added) != 1 or next(iter(added))[1] not in source.reduction_axes:
                raise ValueError(
                    "reduce-scatter target must add exactly the reduced mesh axis"
                )
            collective = "reduce_scatter"
        axes = source.reduction_axes
    elif source.kind == "tiled" and target.kind == "replicated":
        if len(source_tiles) != 1:
            raise ValueError("all-gather reshard currently requires one tiled axis")
        collective = "all_gather"
        axes = tuple(sorted(source_tiles.values()))
    elif source.kind == "replicated" and target.kind == "tiled":
        collective = "local_shard"
        axes = tuple(sorted(target_tiles.values()))
    elif source.kind == "tiled" and target.kind == "tiled":
        source_axes = set(source_tiles.values())
        target_axes = set(target_tiles.values())
        if source_axes != target_axes or len(source_axes) != 1:
            raise ValueError("all-to-all reshard requires one identical mesh axis")
        collective = "all_to_all"
        axes = tuple(source_axes)
    else:
        raise ValueError(f"unsupported reshard transition {source.kind}->{target.kind}")
    return ReshardAction(
        value,
        consumer,
        consumer_index,
        operand_index,
        source,
        target,
        collective,
        axes,
        subgroup,
        region_path,
    )


__all__ = [
    "DEFAULT_PLACEMENT_RULES",
    "Placement",
    "PlacementConflict",
    "PlacementKind",
    "PlacementRule",
    "ReshardAction",
    "ReshardPlan",
    "ShardingPropagationResult",
    "MockMeshExecutionResult",
    "execute_resharded_graph_on_mock_mesh",
    "materialize_reshard_plan",
    "plan_explicit_reshards",
    "propagate_sharding",
]
