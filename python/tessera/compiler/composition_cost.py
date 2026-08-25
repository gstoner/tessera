"""Prune-only cost analysis for measured Tile action DAGs.

R3 consumes the measured resource vectors introduced by R2 and searches legal
topological orders of a Tile action DAG.  It deliberately has no selection or
promotion API: the result may reduce the set of candidates sent to exact-device
measurement, but scalar measured latency remains the final authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .benchmark_row import SCALAR_SELECTOR_AUTHORITY
from .effects import Effect, registered_op_effect
from .graph_dataflow import analyze_graph_dataflow
from .graph_ir import GraphIRFunction, IROp
from .op_catalog import get_op_spec
from .schedule_object import ScheduleAction, ScheduleEdge, ScheduleObject


COMPOSITION_MODEL = "tessera.tile_action_dag_cost.v2"
EXHAUSTIVE_ORACLE_ACTION_LIMIT = 8
EXHAUSTIVE_ORACLE_MAX_ORDERS = math.factorial(EXHAUSTIVE_ORACLE_ACTION_LIMIT)


@dataclass(frozen=True)
class CompositionCalibration:
    """Measured bandwidths used to convert byte counts into lane time."""

    memory_bytes_per_ms: float
    communication_bytes_per_ms: float
    provenance: Mapping[str, Any]
    digest: str

    def __post_init__(self) -> None:
        for name, value in (
            ("memory_bytes_per_ms", self.memory_bytes_per_ms),
            ("communication_bytes_per_ms", self.communication_bytes_per_ms),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("calibration provenance must be a mapping")
        for name in ("source", "domain"):
            provenance_value = self.provenance.get(name)
            if not isinstance(provenance_value, str) or not provenance_value:
                raise ValueError(f"calibration provenance requires non-empty {name}")
        _require_digest(self.digest, "calibration digest")


TileAction = ScheduleAction


@dataclass(frozen=True)
class InferredDependency:
    """One compiler-derived action edge and every reason supporting it."""

    predecessor: str
    successor: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class InferredActionDAG:
    """Fail-closed result of lowering Graph dataflow into Tile dependencies."""

    actions: tuple[TileAction, ...]
    dependencies: tuple[InferredDependency, ...]
    graph_analysis_digest: str
    schedule_object: ScheduleObject


@dataclass(frozen=True)
class ActionDAGParity:
    """Generated-edge coverage of an existing hand-authored R3 fixture."""

    missing_reference_edges: tuple[tuple[str, str], ...]
    additional_conservative_edges: tuple[tuple[str, str], ...]

    @property
    def conservative(self) -> bool:
        return not self.missing_reference_edges

    @property
    def exact(self) -> bool:
        return self.conservative and not self.additional_conservative_edges


def infer_action_dag(
    function: GraphIRFunction,
    resource_vectors: Sequence[Mapping[str, Any]],
    *,
    action_ids: Sequence[str] | None = None,
) -> InferredActionDAG:
    """Generate legal Tile action edges from registered Graph semantics.

    The construction is intentionally conservative. Unknown effects, nested
    region boundaries, stochastic identities, ordered collectives, aliases,
    and memory dependence serialize against surrounding actions. Pure,
    independent SSA subgraphs remain reorderable. This makes the R3 model a
    consumer of W2.1 facts instead of accepting an unsafe hand-authored DAG.
    """

    ops = tuple(function.body)
    if len(resource_vectors) != len(ops):
        raise ValueError("resource vector count must match Graph operation count")
    ids = tuple(action_ids or tuple(f"graph_action_{i}" for i in range(len(ops))))
    if len(ids) != len(ops) or len(set(ids)) != len(ids) or any(not item for item in ids):
        raise ValueError("action_ids must be unique, non-empty, and total over Graph ops")

    analysis = analyze_graph_dataflow(function)
    if not analysis.valid or not analysis.digest:
        raise ValueError("Graph dataflow analysis is stale or unavailable")
    producer: dict[str, int] = {}
    for index, op in enumerate(ops):
        for result in op.result_names:
            producer[_ssa(result)] = index

    reasons: dict[tuple[int, int], set[str]] = {}

    def add(before: int, after: int, reason: str) -> None:
        if before != after:
            reasons.setdefault((before, after), set()).add(reason)

    for index, op in enumerate(ops):
        for operand in op.operands:
            before = producer.get(_ssa(operand))
            if before is not None:
                add(before, index, "ssa_value_flow")

        current_barriers = _barrier_reasons(op)
        current_unknown = _unknown_dataflow_reasons(analysis, op)
        for before in range(index):
            previous = ops[before]
            previous_barriers = _barrier_reasons(previous)
            for reason in current_barriers:
                if reason != ORDERED_COLLECTIVE_REASON:
                    add(before, index, reason)
            for reason in previous_barriers:
                if reason != ORDERED_COLLECTIVE_REASON:
                    add(before, index, reason)
            # An ordered collective constrains the ORDER OF COLLECTIVES — every
            # rank must issue them in the same relative order — not the order of
            # unrelated local work. Serializing a collective against every
            # surrounding operation makes any pipeline with per-chunk transport
            # a total chain, which erases exactly the compute/communication
            # overlap a schedule exists to express (measured on the MegaMoE
            # producer: 12 actions, 66 inferred edges = the complete order;
            # PR #625 review). Data flowing through a collective is still
            # ordered by the SSA/alias/memory-dependence edges below, and any
            # collective that also carries mutation/state/I/O keeps its
            # all-pairs barrier above.
            if (
                ORDERED_COLLECTIVE_REASON in current_barriers
                and ORDERED_COLLECTIVE_REASON in previous_barriers
            ):
                add(before, index, ORDERED_COLLECTIVE_REASON)
            for reason in current_unknown:
                add(before, index, reason)
            for reason in _unknown_dataflow_reasons(analysis, previous):
                add(before, index, reason)
            if analysis.has_memory_dependence(previous, op):
                add(before, index, "memory_dependence")
            # Two REGISTERED pure ops cannot conflict: neither writes, so
            # sharing a read-only operand (a weight tensor every expert reads)
            # is not a dependence. Ordering them on that alias made every
            # consumer of a common input sequential (PR #625 review). Any
            # write-through-alias case has at least one effectful side and
            # still takes the edge below.
            both_pure = (
                registered_op_effect(previous.op_name, previous.kwargs)
                == Effect.pure
                and registered_op_effect(op.op_name, op.kwargs) == Effect.pure
            )
            if not both_pure and _may_share_alias(analysis, previous, op):
                add(before, index, "alias_set")

    dependencies = tuple(
        InferredDependency(ids[before], ids[after], tuple(sorted(edge_reasons)))
        for (before, after), edge_reasons in sorted(reasons.items())
    )
    predecessors: dict[int, list[str]] = {index: [] for index in range(len(ops))}
    for (before, after) in reasons:
        predecessors[after].append(ids[before])
    actions = tuple(
        TileAction(ids[index], resource_vectors[index], tuple(predecessors[index]))
        for index in range(len(ops))
    )
    data_reasons = {
        "ssa_value_flow",
        "alias_set",
        "memory_dependence",
        "unknown_alias_fact",
    }
    schedule_edges: list[ScheduleEdge] = []
    for dependency in dependencies:
        data = tuple(reason for reason in dependency.reasons if reason in data_reasons)
        sync = tuple(reason for reason in dependency.reasons if reason not in data_reasons)
        if data:
            schedule_edges.append(
                ScheduleEdge(
                    dependency.predecessor,
                    dependency.successor,
                    "data",
                    data,
                )
            )
        if sync:
            schedule_edges.append(
                ScheduleEdge(
                    dependency.predecessor,
                    dependency.successor,
                    "sync",
                    sync,
                )
            )
    schedule = ScheduleObject(
        object_id=function.name,
        actions=actions,
        edges=tuple(schedule_edges),
    )
    return InferredActionDAG(actions, dependencies, analysis.digest, schedule)


def compare_inferred_action_dag(
    inferred: InferredActionDAG,
    reference_actions: Sequence[TileAction],
) -> ActionDAGParity:
    """Compare generated dependencies with a reference without weakening it."""

    generated_ids = {action.action_id for action in inferred.actions}
    reference_ids = {action.action_id for action in reference_actions}
    if generated_ids != reference_ids:
        raise ValueError(
            "generated/reference action identities differ: "
            f"missing={sorted(reference_ids - generated_ids)}, "
            f"extra={sorted(generated_ids - reference_ids)}"
        )
    generated = {
        (dependency.predecessor, dependency.successor)
        for dependency in inferred.dependencies
    }
    reference = {
        (predecessor, action.action_id)
        for action in reference_actions
        for predecessor in action.depends_on
    }
    return ActionDAGParity(
        tuple(sorted(reference - generated)),
        tuple(sorted(generated - reference)),
    )


ORDERED_COLLECTIVE_REASON = "ordered_collective"
"""Edge reason whose scope is collective-to-collective, not all-pairs.

Every other barrier reason (unregistered effect, mutation/state/I/O,
stochastic identity, region boundary) serializes against ALL surrounding
work and stays fail-closed; this one is an ordering relation among the
ordered collectives themselves."""


def _ssa(name: str) -> str:
    return str(name).strip().lstrip("%")


def _barrier_reasons(op: IROp) -> tuple[str, ...]:
    effect = registered_op_effect(op.op_name, op.kwargs)
    spec = get_op_spec(op.op_name)
    result: set[str] = set()
    if effect == Effect.top:
        result.add("unregistered_effect")
    if effect == Effect.random or (
        spec is not None and spec.stochastic_identity != "none"
    ) or op.kwargs.get("tessera.stochastic_identity") not in (None, "none"):
        result.add("stochastic_identity")
    if effect == Effect.collective and op.kwargs.get("ordered", True):
        result.add(ORDERED_COLLECTIVE_REASON)
    if effect in {Effect.state, Effect.memory, Effect.io}:
        result.add("mutation_or_effect")
    if op.op_name.startswith(("tessera.scf.", "scf.")) or any(
        key in op.kwargs for key in ("region", "regions", "body", "then_region", "else_region")
    ):
        result.add("region_boundary")
    return tuple(sorted(result))


def _may_share_alias(analysis: Any, lhs: IROp, rhs: IROp) -> bool:
    lhs_values = tuple(lhs.operands) + tuple(lhs.result_names)
    rhs_values = tuple(rhs.operands) + tuple(rhs.result_names)
    return any(analysis.may_alias(left, right) for left in lhs_values for right in rhs_values)


def _unknown_dataflow_reasons(analysis: Any, op: IROp) -> tuple[str, ...]:
    """Make unknown alias information an explicit fail-closed edge reason."""

    values = tuple(op.operands) + tuple(op.result_names)
    if any(analysis.fact(value).alias_roots is None for value in values):
        return ("unknown_alias_fact",)
    return ()


@dataclass(frozen=True)
class CompositionCandidate:
    candidate_id: str
    actions: tuple[TileAction, ...]
    schedule_object: ScheduleObject | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("composition candidate_id must be non-empty")
        _validate_dag(self.actions)
        schedule = self.schedule_object or ScheduleObject(
            self.candidate_id, self.actions
        )
        if schedule.actions != self.actions:
            raise ValueError("composition candidate actions disagree with schedule object")
        object.__setattr__(self, "schedule_object", schedule)

    @classmethod
    def from_graph(
        cls,
        candidate_id: str,
        function: GraphIRFunction,
        resource_vectors: Sequence[Mapping[str, Any]],
        *,
        action_ids: Sequence[str] | None = None,
    ) -> tuple["CompositionCandidate", InferredActionDAG]:
        """Construct the R3 candidate from W2.1 facts, never handwritten edges."""
        inferred = infer_action_dag(
            function, resource_vectors, action_ids=action_ids
        )
        schedule = ScheduleObject(
            candidate_id,
            inferred.schedule_object.actions,
            inferred.schedule_object.edges,
            inferred.schedule_object.roles,
            inferred.schedule_object.residency,
        )
        return cls(candidate_id, inferred.actions, schedule), inferred


@dataclass(frozen=True)
class CompositionEstimate:
    candidate_id: str
    predicted_makespan_ms: float
    action_order: tuple[str, ...]
    orders_examined: int
    exhaustive: bool
    lower_bound_ms: float
    search_method: str
    list_schedule_makespan_ms: float
    calibration_digest: str
    analysis_digest: str
    method: str = COMPOSITION_MODEL
    selector_authority: str = SCALAR_SELECTOR_AUTHORITY
    eligible_for_promotion: bool = False


@dataclass(frozen=True)
class CompositionPruningResult:
    """Candidate filter result, intentionally without a selected winner."""

    retained: tuple[str, ...]
    pruned: tuple[str, ...]
    estimates: tuple[CompositionEstimate, ...]
    relative_margin: float
    method: str = COMPOSITION_MODEL
    selector_authority: str = SCALAR_SELECTOR_AUTHORITY
    eligible_for_promotion: bool = False


def estimate_composition(
    candidate: CompositionCandidate,
    calibration: CompositionCalibration,
    *,
    max_orders: int = EXHAUSTIVE_ORACLE_MAX_ORDERS,
) -> CompositionEstimate:
    """Return a deterministic, bounded estimate for one legal action DAG.

    Production search uses critical-path/list scheduling.  DAGs containing at
    most eight actions additionally run the exhaustive enumerator as a declared
    oracle; the oracle result remains the estimate for that deliberately small
    domain.  Wider DAGs never enter factorial enumeration.
    """

    if max_orders < 1:
        raise ValueError("max_orders must be >= 1")
    list_order = _critical_path_list_order(candidate.actions, calibration)
    list_ms = _simulate_order(candidate.actions, list_order, calibration)
    lower_bound = _composition_lower_bound(candidate.actions, calibration)
    if lower_bound > list_ms + 1e-12:
        raise RuntimeError("composition lower bound exceeds a feasible schedule")

    best_order = list_order
    best_ms = list_ms
    orders_examined = 1
    exhaustive = False
    search_method = "critical_path_list"
    if len(candidate.actions) <= EXHAUSTIVE_ORACLE_ACTION_LIMIT:
        orders, exhaustive = _topological_orders(
            candidate.actions, max_orders=max_orders
        )
        if exhaustive:
            orders_examined = len(orders)
            search_method = "exhaustive_small_dag_oracle"
            for order in orders:
                makespan = _simulate_order(candidate.actions, order, calibration)
                if makespan < best_ms or (
                    math.isclose(makespan, best_ms) and order < best_order
                ):
                    best_ms = makespan
                    best_order = order
    payload = {
        "candidate_id": candidate.candidate_id,
        "action_order": list(best_order),
        "artifact_digests": [
            action.resource_vector["artifact_digest"] for action in candidate.actions
        ],
        "calibration_digest": calibration.digest,
        "lower_bound_ms": lower_bound,
        "search_method": search_method,
        "method": COMPOSITION_MODEL,
    }
    analysis_digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CompositionEstimate(
        candidate_id=candidate.candidate_id,
        predicted_makespan_ms=best_ms,
        action_order=best_order,
        orders_examined=orders_examined,
        exhaustive=exhaustive,
        lower_bound_ms=lower_bound,
        search_method=search_method,
        list_schedule_makespan_ms=list_ms,
        calibration_digest=calibration.digest,
        analysis_digest=analysis_digest,
    )


def prune_composition_candidates(
    candidates: Sequence[CompositionCandidate],
    calibration: CompositionCalibration,
    *,
    relative_margin: float = 0.25,
    max_orders: int = EXHAUSTIVE_ORACLE_MAX_ORDERS,
) -> CompositionPruningResult:
    """Prune only candidates with a mathematical proof of inferiority.

    An exhaustive estimate is exact.  A scalable list estimate is only an upper
    bound, but its admissible lower bound can still prove it cannot beat another
    candidate's feasible upper bound.  All other inexact candidates remain.
    This is pruning, not selection: exact-device scalar latency remains the
    authority among retained candidates.
    """

    if not candidates:
        raise ValueError("at least one composition candidate is required")
    if not math.isfinite(relative_margin) or relative_margin < 0.0:
        raise ValueError("relative_margin must be finite and non-negative")
    ids = [candidate.candidate_id for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("composition candidate ids must be unique")
    estimates = tuple(
        estimate_composition(candidate, calibration, max_orders=max_orders)
        for candidate in candidates
    )
    feasible_floor = min(estimate.predicted_makespan_ms for estimate in estimates)
    threshold = feasible_floor * (1.0 + relative_margin)
    by_id = {estimate.candidate_id: estimate for estimate in estimates}
    pruned = tuple(
        candidate_id for candidate_id in ids
        if (
            by_id[candidate_id].predicted_makespan_ms
            if by_id[candidate_id].exhaustive
            else by_id[candidate_id].lower_bound_ms
        ) > threshold
    )
    pruned_set = set(pruned)
    retained = tuple(candidate_id for candidate_id in ids if candidate_id not in pruned_set)
    return CompositionPruningResult(retained, pruned, estimates, relative_margin)


def _validate_dag(actions: Sequence[TileAction]) -> None:
    if not actions:
        raise ValueError("composition candidate requires at least one Tile action")
    ids = [action.action_id for action in actions]
    if len(set(ids)) != len(ids):
        raise ValueError("Tile action ids must be unique")
    known = set(ids)
    for action in actions:
        missing = set(action.depends_on) - known
        if missing:
            raise ValueError(
                f"Tile action {action.action_id!r} has unknown dependencies: "
                f"{sorted(missing)}"
            )
    if len(_kahn_order(actions)) != len(actions):
        raise ValueError("Tile action dependencies contain a cycle")


def _kahn_order(actions: Sequence[TileAction]) -> tuple[str, ...]:
    """Return one deterministic order, or a strict prefix for a cyclic DAG."""

    dependents: dict[str, list[str]] = {action.action_id: [] for action in actions}
    indegree = {action.action_id: len(action.depends_on) for action in actions}
    for action in actions:
        for predecessor in action.depends_on:
            dependents[predecessor].append(action.action_id)
    ready = sorted(action_id for action_id, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        action_id = ready.pop(0)
        order.append(action_id)
        for successor in sorted(dependents[action_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    return tuple(order)


def _topological_orders(
    actions: Sequence[TileAction], *, max_orders: int
) -> tuple[tuple[tuple[str, ...], ...], bool]:
    ids = tuple(action.action_id for action in actions)
    deps = {action.action_id: set(action.depends_on) for action in actions}
    found: list[tuple[str, ...]] = []
    overflow = False

    def visit(prefix: tuple[str, ...], remaining: set[str]) -> None:
        nonlocal overflow
        if overflow:
            return
        if not remaining:
            found.append(prefix)
            if len(found) > max_orders:
                overflow = True
            return
        ready = sorted(node for node in remaining if deps[node].issubset(prefix))
        for node in ready:
            visit(prefix + (node,), remaining - {node})

    visit((), set(ids))
    if overflow:
        return tuple(found[:max_orders]), False
    return tuple(found), True


def _action_lane_durations(
    action: TileAction, calibration: CompositionCalibration
) -> dict[str, float]:
    vector = action.resource_vector
    return {
        "compute": float(vector["compute_time_ms"]),
        "memory": int(vector["bytes_moved"]) / calibration.memory_bytes_per_ms,
        "communication": int(vector["communication_bytes"])
        / calibration.communication_bytes_per_ms,
    }


def _action_duration(
    action: TileAction, calibration: CompositionCalibration
) -> float:
    return max(_action_lane_durations(action, calibration).values(), default=0.0)


def _critical_path_priorities(
    actions: Sequence[TileAction], calibration: CompositionCalibration
) -> dict[str, float]:
    """Compute deterministic bottom levels in reverse topological order."""

    order = _kahn_order(actions)
    if len(order) != len(actions):
        raise ValueError("Tile action dependencies contain a cycle")
    by_id = {action.action_id: action for action in actions}
    successors: dict[str, list[str]] = {action_id: [] for action_id in order}
    for action in actions:
        for predecessor in action.depends_on:
            successors[predecessor].append(action.action_id)
    bottom: dict[str, float] = {}
    for action_id in reversed(order):
        tail = max((bottom[item] for item in successors[action_id]), default=0.0)
        bottom[action_id] = _action_duration(by_id[action_id], calibration) + tail
    return bottom


def _critical_path_list_order(
    actions: Sequence[TileAction], calibration: CompositionCalibration
) -> tuple[str, ...]:
    """Schedule ready actions by descending bottom level, then stable identity."""

    priorities = _critical_path_priorities(actions, calibration)
    dependencies = {
        action.action_id: frozenset(action.depends_on) for action in actions
    }
    remaining = set(dependencies)
    emitted: set[str] = set()
    order: list[str] = []
    while remaining:
        ready = [
            action_id
            for action_id in remaining
            if dependencies[action_id].issubset(emitted)
        ]
        if not ready:
            raise ValueError("Tile action dependencies contain a cycle")
        action_id = min(ready, key=lambda item: (-priorities[item], item))
        order.append(action_id)
        emitted.add(action_id)
        remaining.remove(action_id)
    return tuple(order)


def _composition_lower_bound(
    actions: Sequence[TileAction], calibration: CompositionCalibration
) -> float:
    """Admissible max(critical path, resource-lane work, queue work) bound."""

    priorities = _critical_path_priorities(actions, calibration)
    critical_path = max(priorities.values(), default=0.0)
    lane_work: dict[tuple[str, str], float] = {}
    queue_work: dict[str, float] = {}
    for action in actions:
        vector = action.resource_vector
        resource = str(vector["resource_identity"])
        queue = str(vector["queue_identity"])
        durations = _action_lane_durations(action, calibration)
        queue_work[queue] = queue_work.get(queue, 0.0) + max(
            durations.values(), default=0.0
        )
        for lane, duration in durations.items():
            lane_key = (resource, lane)
            lane_work[lane_key] = lane_work.get(lane_key, 0.0) + duration
    return max(
        critical_path,
        max(lane_work.values(), default=0.0),
        max(queue_work.values(), default=0.0),
    )


def _simulate_order(
    actions: Sequence[TileAction],
    order: Sequence[str],
    calibration: CompositionCalibration,
) -> float:
    by_id = {action.action_id: action for action in actions}
    completed: dict[str, float] = {}
    queue_free: dict[str, float] = {}
    lane_free: dict[tuple[str, str], float] = {}
    for action_id in order:
        action = by_id[action_id]
        vector = action.resource_vector
        queue = str(vector["queue_identity"])
        resource = str(vector["resource_identity"])
        durations = _action_lane_durations(action, calibration)
        active = {name: duration for name, duration in durations.items() if duration > 0.0}
        dependency_ready = max(
            (completed[dependency] for dependency in action.depends_on),
            default=0.0,
        )
        start = max(
            dependency_ready,
            queue_free.get(queue, 0.0),
            *(lane_free.get((resource, lane), 0.0) for lane in active),
        )
        finish = start + max(active.values(), default=0.0)
        completed[action_id] = finish
        queue_free[queue] = finish
        for lane, duration in active.items():
            lane_free[(resource, lane)] = start + duration
    return max(completed.values(), default=0.0)


def _require_digest(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be 64 lowercase hex characters")


__all__ = [
    "COMPOSITION_MODEL",
    "EXHAUSTIVE_ORACLE_ACTION_LIMIT",
    "EXHAUSTIVE_ORACLE_MAX_ORDERS",
    "ActionDAGParity",
    "CompositionCalibration",
    "CompositionCandidate",
    "CompositionEstimate",
    "CompositionPruningResult",
    "TileAction",
    "compare_inferred_action_dag",
    "estimate_composition",
    "infer_action_dag",
    "prune_composition_candidates",
]
